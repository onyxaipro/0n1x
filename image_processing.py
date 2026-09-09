"""
ComfyUI nodes for image post-processing.
- Remove Image Metadata: strip all metadata and re-save as clean JPEG
- Save as Phone Photo: re-save with realistic phone camera EXIF
"""

import logging
import os
import math
import random
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
from PIL.ExifTags import IFD
from PIL.TiffImagePlugin import IFDRational
import folder_paths


# ---------- Directory helpers ----------

_VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _load_images_from_folder(folder):
    """Load all images from a subfolder under ComfyUI's input directory."""
    input_dir = os.path.join(folder_paths.get_input_directory(), folder)
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"Folder not found: {input_dir}")
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in _VALID_IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")
    images = []
    for f in files:
        img = Image.open(os.path.join(input_dir, f)).convert("RGB")
        images.append((f, img))
    return images


# ---------- Remove Image Metadata ----------

class MetadataRemoveNode:
    """Batch-strip all metadata from images in a folder (EXIF, XMP, PNG chunks, ICC, C2PA)."""

    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (folder_paths.get_input_subfolders(),),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "output_folder": ("STRING", {"default": "cleaned"}),
                "quality": ("INT", {"default": 93, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "add_noise": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "remove_metadata"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def remove_metadata(self, folder, filename_prefix="ComfyUI", output_folder="cleaned", quality=93, add_noise=True):
        source_images = _load_images_from_folder(folder)
        out_dir = os.path.join(folder_paths.get_output_directory(), output_folder)
        os.makedirs(out_dir, exist_ok=True)
        subfolder = output_folder

        results = []
        for counter, (orig_name, img) in enumerate(source_images, start=1):
            img_np = np.array(img).astype(np.uint8)
            if add_noise:
                noise = np.random.choice([-1, 0, 1], size=img_np.shape).astype(np.int16)
                img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            clean = Image.fromarray(img_np, mode="RGB")

            out_name = f"{filename_prefix}_{counter:05d}.jpg"
            clean.save(os.path.join(out_dir, out_name), "JPEG", quality=quality)
            results.append({"filename": out_name, "subfolder": subfolder, "type": self.type})

        logging.info("MetadataRemoveNode: processed %d images from '%s' -> output/%s", len(results), folder, output_folder)
        return {"ui": {"images": results}}


# ---------- Save as Phone Photo ----------

_PHONE_PROFILES = {
    "iPhone 15 Pro Max": {
        "make": "Apple", "model": "iPhone 15 Pro Max",
        "software": "17.5.1",
        "focal_length": (6765, 1000), "f_number": (178, 100),
        "iso_range": (50, 800), "exposure_range": (1, 4000),
        "lens_model": "iPhone 15 Pro Max back triple camera 6.765mm f/1.78",
        "resolutions": {"12MP": (4032, 3024), "48MP": (8064, 6048)},
    },
    "iPhone 16 Pro Max": {
        "make": "Apple", "model": "iPhone 16 Pro Max",
        "software": "18.3.1",
        "focal_length": (6765, 1000), "f_number": (178, 100),
        "iso_range": (50, 1600), "exposure_range": (1, 4000),
        "lens_model": "iPhone 16 Pro Max back quad camera 6.765mm f/1.78",
        "resolutions": {"12MP": (4032, 3024), "48MP": (8064, 6048)},
    },
    "Samsung Galaxy S24 Ultra": {
        "make": "samsung", "model": "SM-S928B",
        "software": "S928BXXS4AXL1",
        "focal_length": (6300, 1000), "f_number": (170, 100),
        "iso_range": (50, 800), "exposure_range": (1, 3000),
        "lens_model": "SM-S928B",
        "resolutions": {"12MP": (4000, 3000), "50MP": (8160, 6120)},
    },
    "Samsung Galaxy S25 Ultra": {
        "make": "samsung", "model": "SM-S938B",
        "software": "S938BXXU1AXK5",
        "focal_length": (6300, 1000), "f_number": (170, 100),
        "iso_range": (50, 1000), "exposure_range": (1, 4000),
        "lens_model": "SM-S938B",
        "resolutions": {"12MP": (4000, 3000), "50MP": (8160, 6120)},
    },
}

_PHONE_CHOICES = list(_PHONE_PROFILES.keys())

_RESOLUTION_CHOICES = ["12MP", "48MP/50MP", "Original"]

_GPS_LOCATIONS = {
    "None": None,
    "New York": (40.7128, -74.0060, 10),
    "Los Angeles": (34.0522, -118.2437, 71),
    "London": (51.5074, -0.1278, 11),
    "Tokyo": (35.6762, 139.6503, 40),
    "Paris": (48.8566, 2.3522, 35),
    "Sydney": (-33.8688, 151.2093, 58),
    "Dubai": (25.2048, 55.2708, 5),
    "Random US City": "random",
}

_RANDOM_US_CITIES = [
    (40.7128, -74.0060, 10),    # New York
    (34.0522, -118.2437, 71),   # Los Angeles
    (41.8781, -87.6298, 181),   # Chicago
    (29.7604, -95.3698, 15),    # Houston
    (33.4484, -112.0740, 331),  # Phoenix
    (29.4241, -98.4936, 198),   # San Antonio
    (32.7157, -117.1611, 20),   # San Diego
    (30.2672, -97.7431, 149),   # Austin
    (37.7749, -122.4194, 16),   # San Francisco
    (47.6062, -122.3321, 56),   # Seattle
    (39.7392, -104.9903, 1609), # Denver
    (35.2271, -80.8431, 229),   # Charlotte
    (36.1627, -86.7816, 182),   # Nashville
    (25.7617, -80.1918, 2),     # Miami
    (38.9072, -77.0369, 22),    # Washington DC
]

_GPS_CHOICES = list(_GPS_LOCATIONS.keys())


def _decimal_to_dms(decimal_deg):
    """Convert decimal degrees to (degrees, minutes, seconds) as IFDRational tuples."""
    d = abs(decimal_deg)
    degrees = int(d)
    minutes = int((d - degrees) * 60)
    seconds = round((d - degrees - minutes / 60) * 3600 * 100)
    return (IFDRational(degrees, 1), IFDRational(minutes, 1), IFDRational(seconds, 100))


def _build_phone_exif(profile, width, height, gps_coords=None):
    """Build EXIF bytes mimicking a phone camera shot."""
    now = datetime.now() - timedelta(seconds=random.randint(0, 300))
    dt_str = now.strftime("%Y:%m:%d %H:%M:%S")

    iso = random.randint(profile["iso_range"][0], profile["iso_range"][1])
    exp_denom = random.randint(profile["exposure_range"][0], profile["exposure_range"][1])
    ssv_num = int(round(math.log2(exp_denom) * 100))
    ssv = (ssv_num, 100)

    exif = Image.Exif()
    # IFD0
    exif[0x010F] = profile["make"]
    exif[0x0110] = profile["model"]
    exif[0x0112] = 1  # Orientation: normal
    exif[0x011A] = 72  # XResolution
    exif[0x011B] = 72  # YResolution
    exif[0x0128] = 2   # ResolutionUnit: inches
    exif[0x0131] = profile["software"]
    exif[0x0132] = dt_str
    exif[0x0213] = 1   # YCbCrPositioning: centered

    # Exif sub-IFD
    exif_ifd = exif.get_ifd(IFD.Exif)
    exif_ifd[0x829A] = (1, exp_denom)          # ExposureTime
    exif_ifd[0x829D] = profile["f_number"]      # FNumber
    exif_ifd[0x8822] = 2                         # ExposureProgram: normal
    exif_ifd[0x8827] = iso                       # ISOSpeedRatings
    exif_ifd[0x9000] = b"0232"                   # ExifVersion
    exif_ifd[0x9003] = dt_str                    # DateTimeOriginal
    exif_ifd[0x9004] = dt_str                    # DateTimeDigitized
    exif_ifd[0x9201] = ssv                       # ShutterSpeedValue (APEX)
    exif_ifd[0x9202] = profile["f_number"]       # ApertureValue
    exif_ifd[0x9203] = (0, 1)                    # BrightnessValue
    exif_ifd[0x9204] = (0, 1)                    # ExposureBiasValue
    exif_ifd[0x9207] = 5                         # MeteringMode: pattern
    exif_ifd[0x9209] = 0x10                      # Flash: no flash
    exif_ifd[0x920A] = profile["focal_length"]   # FocalLength
    exif_ifd[0xA001] = 1                         # ColorSpace: sRGB
    exif_ifd[0xA002] = width                     # PixelXDimension
    exif_ifd[0xA003] = height                    # PixelYDimension
    exif_ifd[0xA405] = 26                        # FocalLengthIn35mmFilm (~26mm equiv)
    exif_ifd[0xA406] = 0                         # SceneCaptureType: standard
    exif_ifd[0xA433] = profile["make"]           # LensMake
    exif_ifd[0xA434] = profile["lens_model"]     # LensModel

    # GPS IFD
    if gps_coords:
        lat, lon, alt = gps_coords
        # Add slight random jitter (±0.001° ≈ ±100m)
        lat += random.uniform(-0.001, 0.001)
        lon += random.uniform(-0.001, 0.001)

        gps_ifd = exif.get_ifd(IFD.GPSInfo)
        gps_ifd[0] = b"\x02\x03\x00\x00"          # GPSVersionID
        gps_ifd[1] = "N" if lat >= 0 else "S"      # GPSLatitudeRef
        gps_ifd[2] = _decimal_to_dms(lat)           # GPSLatitude
        gps_ifd[3] = "E" if lon >= 0 else "W"      # GPSLongitudeRef
        gps_ifd[4] = _decimal_to_dms(lon)           # GPSLongitude
        gps_ifd[5] = 0                              # GPSAltitudeRef: above sea level
        gps_ifd[6] = IFDRational(int(alt * 100), 100)  # GPSAltitude
        gps_ifd[7] = (IFDRational(now.hour, 1), IFDRational(now.minute, 1), IFDRational(now.second, 1))  # GPSTimeStamp
        gps_ifd[29] = now.strftime("%Y:%m:%d")      # GPSDateStamp

    return exif.tobytes()


class SaveAsPhonePhotoNode:
    """Batch-save images from a folder as JPEG with realistic phone camera EXIF metadata."""

    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (folder_paths.get_input_subfolders(),),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "phone_model": (_PHONE_CHOICES,),
                "resolution": (_RESOLUTION_CHOICES, {"default": "12MP"}),
                "gps_location": (_GPS_CHOICES, {"default": "Random US City"}),
                "output_folder": ("STRING", {"default": "phone_photos"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "add_noise": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_phone_photo"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_phone_photo(self, folder, filename_prefix="ComfyUI", phone_model="iPhone 15 Pro Max", resolution="12MP", gps_location="Random US City",
                         output_folder="phone_photos", quality=95, add_noise=True):
        source_images = _load_images_from_folder(folder)
        out_dir = os.path.join(folder_paths.get_output_directory(), output_folder)
        os.makedirs(out_dir, exist_ok=True)
        subfolder = output_folder

        profile = _PHONE_PROFILES[phone_model]

        # Resolve GPS coordinates
        gps_coords = None
        gps_val = _GPS_LOCATIONS.get(gps_location)
        if gps_val == "random":
            gps_coords = random.choice(_RANDOM_US_CITIES)
        elif gps_val is not None:
            gps_coords = gps_val

        # Resolve target resolution
        target_size = None
        if resolution != "Original":
            res_map = profile["resolutions"]
            if resolution == "12MP":
                target_size = res_map.get("12MP")
            elif resolution == "48MP/50MP":
                target_size = res_map.get("48MP") or res_map.get("50MP")

        results = []
        for counter, (orig_name, img) in enumerate(source_images, start=1):
            # Resize to phone resolution if needed
            if target_size:
                src_w, src_h = img.size
                tgt_w, tgt_h = target_size
                # Match orientation: if source is portrait, swap target to portrait
                if src_h > src_w and tgt_w > tgt_h:
                    tgt_w, tgt_h = tgt_h, tgt_w
                elif src_w > src_h and tgt_h > tgt_w:
                    tgt_w, tgt_h = tgt_h, tgt_w
                img = img.resize((tgt_w, tgt_h), Image.LANCZOS)

            img_np = np.array(img).astype(np.uint8)
            if add_noise:
                noise = np.random.choice([-1, 0, 1], size=img_np.shape).astype(np.int16)
                img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            clean = Image.fromarray(img_np, mode="RGB")
            w, h = clean.size
            exif_bytes = _build_phone_exif(profile, w, h, gps_coords=gps_coords)

            out_name = f"{filename_prefix}_{counter:05d}.jpg"
            clean.save(os.path.join(out_dir, out_name), "JPEG", quality=quality, exif=exif_bytes)
            results.append({"filename": out_name, "subfolder": subfolder, "type": self.type})

        logging.info("SaveAsPhonePhotoNode: processed %d images from '%s' -> output/%s", len(results), folder, output_folder)
        return {"ui": {"images": results}}
