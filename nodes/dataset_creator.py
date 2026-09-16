"""
Onyx Dataset Creator
Generates a face-swapped dataset from pre-made body preset samples.
Presets (Flat / Skinny / Normal / Curvy / Large) live under Resources/.
Model chain: Nano Banana Pro (5 retries) → Nano Banana 2 (5 retries).
Resume support via processed.json in the output folder.
"""

import os
import json
import time
import base64
import threading
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor
from .instagram_faceswap import (
    _load_skip_log,
    _save_skip_log,
    _do_face_swap,
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-image prompt overrides (applied in addition to custom_prompt).
# Key = source filename without extension.
from .onyx_render_profile import ensure_profile_ready
_IMAGE_PROMPT_OVERRIDES = {
    "swap_01": "neutral facial expression, perfect profile, she's looking at the left",
    "swap_03": "she must be laughing",
    "swap_06": "she's looking in front of her, we can see the underside of her chin",
    "swap_08": "neutral facial expression, she's looking at the left, perfect profile view, keep midriff visible",
    "swap_09": "neutral facial expression, she's looking at the right, perfect profile view",
    "swap_10": "She has the exact same pose. She still facing to the left.",
    "swap_11": "She's looking at the left, 3/4 angle",
    "swap_12": "she's looking at the ceiling, she tilts her head back, we can see the underside of her chin",
    "swap_13": "exact same pose, 3/4 front angle, she still facing the right side.",
    "swap_14": "She looks at the top left, she tilts her head slightly back and looks up to the upper left",
    "swap_15": "She looks at the right, perfect profile",
    "swap_21": "She looks in front of her, slightly below the camera",
    "swap_23": "She looks down at the ground, her head is tilted forward, we can see the top of her skull",
    "swap_25": "She looks straight at the camera, but her body keep facing to the right with the exact same pose",
    "swap_18": "Exact same pose. She still facing to the right, don't keep the hair color of the picture 2.",
    "swap_27": "She smiles and shows her teeth. Change the hair color to match the picture 1.",
    "swap_29": "The hair color is the same as image 1. Don't keep the hair color of the picture 2, same hair color as the picture 1.",
    "swap_31": "she's looking in front of her, 3/4 left face angle",
    "swap_35": "She smiles and shows her teeth, she's looking down, she tilts her head forward",
    "swap_45": "neutral facial expression",
}

# Appended to every face-swap prompt to handle multiple masked faces
_FACESWAP_GLOBAL_SUFFIX = "regenerate black masked areas"

# Per-swap YuNet face detection threshold overrides
# Default for all swaps: 0.63
_YUNET_THRESHOLD_OVERRIDES = {
    "swap_07": 0.64,
    "swap_27": 0.56,
    "swap_30": 0.64,
    "swap_42": 0.64,
    "swap_45": 0.58,
}
_YUNET_DEFAULT = 0.63

# Per-preset AND per-image prompt additions (key = (preset, swap_key))
_PRESET_IMAGE_OVERRIDES = {
    ("Curvy", "swap_22"): (
        "Her chest is extremely large with heavy breast volume, strong projection, and visible softness. "
        "exaggerated proportions only focus on the chest area. "
        "The clothing follows and emphasizes this thick, soft, curvy silhouette, "
        "suggesting heavy natural volume underneath."
    ),
    ("Curvy", "swap_24"): (
        "Her chest is extremely large with heavy breast volume, strong projection, and visible softness. "
        "exaggerated proportions only focus on the chest area. "
        "The clothing follows and emphasizes this thick, soft, curvy silhouette, "
        "suggesting heavy natural volume underneath."
    ),
    ("Curvy", "swap_35"): (
        "Her chest is extremely large with heavy breast volume, strong projection, and visible softness. "
        "exaggerated proportions only focus on the chest area. "
        "The clothing follows and emphasizes this thick, soft, curvy silhouette, "
        "suggesting heavy natural volume underneath."
    ),
    ("Athletic", "swap_09"): (
        "her face is looking at the right and show a perfect 90 degres profile view, only one eye is visible."
    ),
}

# Per-preset prompt additions (appended to every image of that preset)
_PRESET_PROMPT_OVERRIDES = {
    "Curvy": (
        "Keep the exact same curvy body shape and proportions as in the original image. "
        "Do not slim down or alter the figure in any way. "
        "The breasts and buttocks must remain exactly the same size, shape, and volume as the original picture. "
        "The chest area must stay completely unchanged, preserving the original fullness, contour, projection, and clothing fit. "
        "NEVER minimize any body curves. "
        "Maintain the same hourglass appearance, and overall body volume throughout the torso, hips, thighs, and buttocks."
    ),
}

# ── Large-preset per-swap prompt routing ────────────────────────────────────
_LARGE_NO_PROMPT     = {1, 12, 17, 27, 41, 43}
_LARGE_FULL_SWAPS    = {4, 5, 7, 10, 11, 13, 14, 16, 19, 20, 22, 24, 25, 33, 35, 36, 37, 44}
_LARGE_BUTTOCKS_SWAPS = {2, 42}
# (all other swap numbers get the chest-only variation)

_LARGE_FULL_PROMPT = (
    "Her body is the same from the picture 2, according to this caracteristics. "
    "She has an extremely thick, heavily curvy body with abundant soft body fat distributed across her entire physique. "
    "Her chest is extremely large with heavy breast volume, strong projection, and visible softness. "
    "Her stomach is slightly flat, blending naturally into a normal waist. "
    "Her buttocks are huge, round, and very prominent with substantial visible volume. "
    "The overall silhouette is intensely curvy, dense, soft, and chubby rather than slim, "
    "with exaggerated proportions spread naturally across her chest and buttocks. "
    "The clothing follows and emphasizes this thick, soft, ultra-curvy silhouette, "
    "suggesting heavy natural volume underneath."
)
_LARGE_BUTTOCKS_PROMPT = (
    "Her body is the same from the picture 2, according to this caracteristics. "
    "Her buttocks are huge, round, and very prominent with substantial visible volume. "
    "The overall silhouette is intensely curvy, dense, soft, and chubby rather than slim, "
    "with exaggerated proportions spread naturally across her buttocks."
)
_LARGE_CHEST_PROMPT = (
    "Her body is the same from the picture 2, according to this caracteristics. "
    "Her chest is extremely large with heavy breast volume, strong projection, and visible softness. "
    "The overall silhouette is intensely curvy, dense, soft, and chubby rather than slim, "
    "with exaggerated proportions spread naturally across her chest. "
    "The clothing follows and emphasizes this thick, soft, ultra-curvy silhouette, "
    "suggesting heavy natural volume underneath."
)

# ─────────────────────────────────────────────────────────────────────────────
PRESET_NAMES   = ["Flat", "Skinny", "Normal", "Curvy", "Large", "Athletic", "Test"]

# Presets volontairement sans legendes : le dossier ne sert qu'a fournir des photos.
_NO_CAPTION_PRESETS = {"TEST"}
_RETRY_COUNT   = 5
_BATCH_OTHERS  = 10   # batch size for non-Vertex providers
_LOCK          = threading.Lock()

# ── Background similarity check ───────────────────────────────────────────────
# Compares corner regions (top-left, top-right, bottom-left, bottom-right)
# between the source swap image and the generated result.
# Uses Bhattacharyya coefficient on per-channel histograms — robust to small
# texture shifts and minor lighting variations between successive generations.
#
# Threshold calibration:
#   ~0.85-0.95  → typical face swap with intact background (passes ✅)
#   ~0.70-0.84  → some texture variation, still acceptable (passes ✅)
#   ~0.50-0.69  → significant background change (fails ❌)
#   ~0.20-0.49  → reframing / completely different background (fails ❌)
_BG_CHECK_THRESHOLD = 0.75   # plus strict (etait 0.68)
_BG_CORNER_RATIO    = 0.18   # 18 % of image width/height sampled per corner
_BG_HIST_BINS       = 32     # histogram resolution


def _check_background(source_pil: Image.Image, result_pil: Image.Image):
    """Compare corner regions via per-channel Bhattacharyya histogram similarity.

    Returns:
        ok    (bool)  : True if background is considered preserved.
        score (float) : Mean similarity across all 4 corners × 3 channels [0..1].
    """
    SIZE = (256, 256)
    src = np.array(source_pil.resize(SIZE, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.uint8)
    res = np.array(result_pil.resize(SIZE, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.uint8)

    H, W = SIZE[1], SIZE[0]
    ch   = max(1, int(H * _BG_CORNER_RATIO))
    cw   = max(1, int(W * _BG_CORNER_RATIO))

    # Four corner bounding boxes: (row_start, row_end, col_start, col_end)
    corners = [
        (0,    ch,    0,    cw),    # top-left
        (0,    ch,    W-cw, W),     # top-right
        (H-ch, H,     0,    cw),    # bottom-left
        (H-ch, H,     W-cw, W),     # bottom-right
    ]

    scores = []
    for r0, r1, c0, c1 in corners:
        src_p = src[r0:r1, c0:c1]
        res_p = res[r0:r1, c0:c1]
        for c in range(3):
            h1 = np.histogram(src_p[:, :, c], bins=_BG_HIST_BINS, range=(0, 256))[0].astype(float)
            h2 = np.histogram(res_p[:, :, c], bins=_BG_HIST_BINS, range=(0, 256))[0].astype(float)
            h1 /= (h1.sum() + 1e-8)
            h2 /= (h2.sum() + 1e-8)
            # Bhattacharyya coefficient ∈ [0, 1]
            scores.append(float(np.sum(np.sqrt(h1 * h2))))

    score = float(np.mean(scores))
    return score >= _BG_CHECK_THRESHOLD, score

# ─────────────────────────────────────────────────────────────────────────────
# Pose preservation check
# Primary : MediaPipe Pose (install: pip install mediapipe)
# If MediaPipe is unavailable or detects no pose → check is skipped (pass-through).
# ─────────────────────────────────────────────────────────────────────────────
_POSE_CHECK_ENABLED   = True
_POSE_MP_THRESHOLD    = 0.25   # max mean normalised landmark distance [0..1]
                                # lower = stricter. Tune down if bad poses pass through.
_POSE_KEY_LANDMARKS   = [11, 12, 23, 24, 25, 26]  # shoulders + hips + knees
_POSE_SIZE            = (512, 512)

_mp_pose_instance = None   # lazy singleton


def _get_mp_pose():
    """Lazy-load MediaPipe Pose singleton. Returns None if not available."""
    global _mp_pose_instance
    if _mp_pose_instance is not None:
        return _mp_pose_instance
    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            print("[Pose] mediapipe version incompatible (no 'solutions' API). "
                  "Install mediapipe==0.10.3 — pose check skipped.")
            _mp_pose_instance = False
            return None
        _mp_pose_instance = mp.solutions.pose.Pose(
            static_image_mode        = True,
            model_complexity         = 1,
            enable_segmentation      = False,
            min_detection_confidence = 0.5,
        )
        print("[Pose] MediaPipe Pose loaded ✅")
        return _mp_pose_instance
    except (ImportError, AttributeError, Exception) as _e:
        print(f"[Pose] MediaPipe unavailable ({_e}) — pose check skipped.")
        _mp_pose_instance = False
        return None


def _check_pose(source_pil: "Image.Image", result_pil: "Image.Image"):
    """Compare body pose between source and generated result using MediaPipe.

    Returns:
        ok     (bool)  : True if pose is preserved (or check skipped).
        score  (float) : Similarity score [0..1], 1.0 when skipped.
        method (str)   : 'mediapipe' or 'skip'.
    """
    if not _POSE_CHECK_ENABLED:
        return True, 1.0, "disabled"

    mp_pose = _get_mp_pose()
    if not mp_pose:
        return True, 1.0, "skip"

    src_np = np.array(
        source_pil.resize(_POSE_SIZE, Image.Resampling.LANCZOS).convert("RGB"),
        dtype=np.uint8,
    )
    res_np = np.array(
        result_pil.resize(_POSE_SIZE, Image.Resampling.LANCZOS).convert("RGB"),
        dtype=np.uint8,
    )

    try:
        r_src = mp_pose.process(src_np)
        r_res = mp_pose.process(res_np)
        if r_src.pose_landmarks and r_res.pose_landmarks:
            lm_src = r_src.pose_landmarks.landmark
            lm_res = r_res.pose_landmarks.landmark
            dists = []
            for lm_idx in _POSE_KEY_LANDMARKS:
                dx = lm_src[lm_idx].x - lm_res[lm_idx].x
                dy = lm_src[lm_idx].y - lm_res[lm_idx].y
                dists.append((dx * dx + dy * dy) ** 0.5)
            mean_dist = float(np.mean(dists))
            ok    = mean_dist <= _POSE_MP_THRESHOLD
            score = max(0.0, 1.0 - mean_dist / max(_POSE_MP_THRESHOLD, 1e-6))
            return ok, score, "mediapipe"
        # Pose not detected in one of the images → pass-through
        return True, 1.0, "skip"
    except Exception:
        return True, 1.0, "skip"


def _count_faces(pil_img: "Image.Image", score_threshold: float = 0.5) -> int:
    """Count faces in a PIL image using YuNet. Returns -1 if OpenCV unavailable."""
    try:
        import cv2
        import urllib.request as _ur
        img_rgb = np.array(pil_img.convert("RGB"))
        img_bgr = img_rgb[:, :, ::-1].copy()
        h, w    = img_bgr.shape[:2]
        model_dir  = os.path.join(os.path.dirname(__file__), ".yunet_cache")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        # Meme garde que dans nano_banana_aio : os.path.exists() seul laisse un
        # telechargement rate (page HTML, pointeur git-LFS, fichier tronque) en
        # cache pour toujours, et le detecteur ne trouve alors plus aucun visage.
        from .nano_banana_aio import _ensure_yunet_model
        if not _ensure_yunet_model(model_path):
            print("⚠️  [Dataset] No usable detection model — 0 face will be reported.")
            return 0
        detector = cv2.FaceDetectorYN.create(
            model_path, "", (w, h),
            score_threshold=score_threshold, nms_threshold=0.3, top_k=5000,
        )
        min_face = min(w, h) * 0.04
        count = 0
        _, detections = detector.detect(img_bgr)
        if detections is not None:
            for det in detections:
                fw, fh = int(det[2]), int(det[3])
                if fw >= min_face and fh >= min_face:
                    count += 1
        return count
    except Exception:
        return -1  # -1 = detection unavailable


def _assign_output_numbers(image_files):
    """Map each source image to the number used in its output filename.

    The presets are named swap_01 ... swap_45, and both the CAPTIONS files and
    the resume log key off that number, so it has to be preserved. But the rule
    "take the trailing digits" only holds for names built that way. On a folder
    of arbitrary filenames it collapses: in Resources/test, five numbers are
    shared by twelve files and nine files end with no digit at all. Outputs
    would overwrite each other, and the resume log would report images as
    already processed that had never run.

    So: keep the trailing digits when they are present AND unique across the
    batch, and fall back to the position in the batch otherwise. A preset that
    was already numbered correctly is untouched.
    """
    import re as _re
    keys = [os.path.splitext(os.path.basename(p))[0] for p in image_files]
    nums = []
    for k in keys:
        m = _re.search(r"\d+$", k)
        nums.append(m.group(0) if m else None)

    usable = all(n is not None for n in nums) and len(set(nums)) == len(nums)
    if usable:
        return {p: n for p, n in zip(image_files, nums)}, False
    return {p: f"{i:03d}" for i, p in enumerate(image_files, 1)}, True


def _stabilize_renumbering(dataset_folder, image_files, out_numbers):
    """Only used when _assign_output_numbers fell back to positional numbering.

    That fallback numbers images by their position in Resources/{preset}/,
    sorted. Add, remove or rename ANY file in that folder between two runs and
    every image after the change point gets a different number — a source
    image already rendered as j4y_015.png silently becomes "pending" again on
    the next run, because the resume check (further down) looks for a
    j4y_016.png that doesn't exist. Nothing was wrong with that image; the
    numbering just moved under it.

    Fix: persist the key -> number mapping the first time each source image
    gets one, in numbering.json next to processed.json. A source image keeps
    its number for the life of this dataset folder no matter what else
    changes in the preset folder afterwards; only genuinely new images get a
    new number, appended after the current highest.
    """
    manifest_path = os.path.join(dataset_folder, "numbering.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        manifest = {}

    used = {int(v) for v in manifest.values() if str(v).isdigit()}
    next_num = (max(used) + 1) if used else 1
    changed = False

    for p in image_files:
        key = os.path.splitext(os.path.basename(p))[0]
        if key in manifest:
            out_numbers[p] = manifest[key]
        else:
            num = f"{next_num:03d}"
            manifest[key] = num
            out_numbers[p] = num
            next_num += 1
            changed = True

    if changed:
        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Dataset] Warning: could not save numbering.json: {e}")

    return out_numbers


def _find_resources_dir():
    # Primary: standard location (nodes/dataset_creator.py -> .. -> Resources)
    primary = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Resources",
    )
    if os.path.isdir(primary):
        return primary

    # Fallback: scan ComfyUI custom_nodes for any onyx/ofm folder with Resources/
    try:
        import folder_paths
        custom_nodes_dir = os.path.join(folder_paths.base_path, "custom_nodes")
        for folder_name in os.listdir(custom_nodes_dir):
            name_lower = folder_name.lower()
            if "onyx" in name_lower or "ofm" in name_lower:
                candidate = os.path.join(custom_nodes_dir, folder_name, "Resources")
                if os.path.isdir(candidate):
                    print(f"[Dataset] Resources found via fallback: {candidate}")
                    return candidate
    except Exception:
        pass

    # Last resort: return primary anyway (error will surface at runtime)
    return primary

_RESOURCES_DIR = _find_resources_dir()


# ─────────────────────────────────────────────────────────────────────────────
# Captions loader
#
# CAPTIONS.txt format :
#   Image 1
#   TRIGGER, side profile, medium shot...
#
#   Image 2
#   TRIGGER, mirror selfie, waist-up shot...
#
# On parse en dict {int: caption_text}, lazy + cached.
# ─────────────────────────────────────────────────────────────────────────────
_CAPTIONS_CACHE: dict = {}  # {preset_lower: {int: caption_text}}

def _load_captions_dict(preset: str = "") -> dict:
    """Returns {image_number: caption_text} parsed from Resources/CAPTIONS - {PRESET}.txt.
    Cache en memoire par preset pour eviter de re-parser a chaque image."""
    _key = preset.upper()
    if _key in _NO_CAPTION_PRESETS:
        return {}
    if _key in _CAPTIONS_CACHE:
        return _CAPTIONS_CACHE[_key]

    # Tente le fichier preset-specifique, puis fallback sur l'ancien CAPTIONS.txt
    _candidates = [
        f"CAPTIONS - {preset.upper()}.txt",
        f"CAPTIONS - {preset.capitalize()}.txt",
        f"captions - {preset.lower()}.txt",
        "CAPTIONS.txt", "captions.txt", "Captions.txt",
    ]
    for fname in _candidates:
        path = os.path.join(_RESOURCES_DIR, fname)
        if os.path.isfile(path):
            break
    else:
        print(f"[Captions] No caption file found for preset '{preset}' in {_RESOURCES_DIR}")
        _CAPTIONS_CACHE[_key] = {}
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as e:
        print(f"[Captions] Read error: {e}")
        return {}

    import re as _re
    # On split sur des lignes "Image N" (header) puis on prend la suite.
    # Format: "Image 1\nTRIGGER, ...\n\nImage 2\nTRIGGER, ...\n..."
    blocks = _re.split(r"^\s*Image\s+(\d+)\s*$", text, flags=_re.MULTILINE)
    # blocks = ["", "1", "caption1", "2", "caption2", ...]
    data = {}
    for i in range(1, len(blocks) - 1, 2):
        try:
            num = int(blocks[i])
        except (ValueError, TypeError):
            continue
        cap = blocks[i + 1].strip()
        if cap:
            data[num] = cap

    print(f"[Captions] Loaded {len(data)} caption(s) from {path}")
    _CAPTIONS_CACHE[_key] = data
    return data



# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class OnyxDatasetCreatorNode:
    """
    Onyx Dataset Creator
    Generates a face-swapped dataset using pre-made body preset samples.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference": ("IMAGE",),

                "preset": (PRESET_NAMES, {
                    "default": "Normal",
                    "tooltip": "Body preset. Matching folder must exist under Resources/.",
                }),
                "resolution": (["1K", "2K", "4K"], {
                    "default": "2K",
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Extra instructions appended to the face-swap prompt.",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder where the dataset images will be saved.",
                }),
                "trigger_word": ("STRING", {
                    "default": "MyChar",
                    "tooltip": "Used for file naming: TriggerWord_001.jpg, TriggerWord_002.jpg…",
                }),
                "test_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When ON: process only the single test image from Resources/{preset}/_test/. "
                               "Use this to preview the preset before launching the full dataset.",
                }),

                # ── Provider ──────────────────────────────────────────────────
                "provider": (["WAVESPEED", "GOOGLE", "FAL", "KIE", "VERTEX"], {
                    "default": "WAVESPEED",
                }),
                "disable_safety": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable safety filters where supported.",
                }),

                # ── API keys ──────────────────────────────────────────────────
                "gemini_api_key":     ("STRING", {"default": ""}),
                "wavespeed_api_key":  ("STRING", {"default": ""}),
                "fal_api_key":        ("STRING", {"default": ""}),
                "kie_api_key":        ("STRING", {"default": ""}),
                "vertex_json_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder containing Vertex AI service-account JSON files. "
                               "1 JSON = 1 project = 1 parallel worker slot.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("last_image", "summary")
    OUTPUT_NODE   = False
    FUNCTION      = "run"
    CATEGORY      = "Onyx/NanoBanana"
    DESCRIPTION   = (
        "Onyx Dataset Creator\n"
        "Generates a face-swapped dataset from pre-made body preset samples.\n"
        "Chain: Nano Banana Pro (×5) → Nano Banana 2 (×5).\n"
        "Resume support: already-processed images are automatically skipped."
    )

    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        face_reference,
        preset,
        resolution,
        custom_prompt,
        output_folder,
        trigger_word,
        test_mode,
        provider,
        disable_safety,
        gemini_api_key,
        wavespeed_api_key,
        fal_api_key,
        kie_api_key,
        vertex_json_folder,
    ):
        ensure_profile_ready()
        try:
            from .nano_banana_aio import OnyxNanoBananaAIO, _load_vertex_json_folder
        except ImportError as e:
            return (torch.zeros(1, 64, 64, 3), f"❌ Could not import OnyxNanoBananaAIO: {e}")

        aio = OnyxNanoBananaAIO()

        # ── Resolve preset folder ─────────────────────────────────────────────
        preset_dir = os.path.join(_RESOURCES_DIR, preset)
        if not os.path.isdir(preset_dir):
            # Le dossier peut ne pas avoir la casse exacte du preset. Windows s'en
            # moque, pas Linux - et un dossier "test" introuvable sous le nom
            # "Test" ne produirait aucun message utile.
            try:
                for _d in os.listdir(_RESOURCES_DIR):
                    if _d.lower() == preset.lower() and os.path.isdir(
                            os.path.join(_RESOURCES_DIR, _d)):
                        preset_dir = os.path.join(_RESOURCES_DIR, _d)
                        break
            except Exception:
                pass
        if not os.path.isdir(preset_dir):
            return (
                torch.zeros(1, 64, 64, 3),
                f"❌ Preset folder not found: {preset_dir}\n"
                f"Create Resources/{preset}/ inside the custom node folder.",
            )

        # ── Collect images ────────────────────────────────────────────────────
        _img_exts = {".jpg", ".jpeg", ".png", ".webp"}
        _test_dir = os.path.join(preset_dir, "_test")

        if test_mode:
            if not os.path.isdir(_test_dir):
                return (
                    torch.zeros(1, 64, 64, 3),
                    f"❌ Test folder not found: {_test_dir}\n"
                    f"Create Resources/{preset}/_test/ and place one sample image inside.",
                )
            image_files = sorted([
                os.path.join(_test_dir, f)
                for f in os.listdir(_test_dir)
                if os.path.splitext(f)[1].lower() in _img_exts
            ])
            if not image_files:
                return (torch.zeros(1, 64, 64, 3), f"❌ No image found in {_test_dir}")
            image_files = [image_files[0]]   # use only the first test image
        else:
            _excluded_dirs = {os.path.normpath(_test_dir)}
            image_files = []
            for root, dirs, files in os.walk(preset_dir):
                # Exclude _test and any folder named "old" (case-insensitive)
                dirs[:] = [
                    d for d in dirs
                    if os.path.normpath(os.path.join(root, d)) not in _excluded_dirs
                    and d.lower() != "old"
                ]
                for f in files:
                    if os.path.splitext(f)[1].lower() in _img_exts:
                        image_files.append(os.path.join(root, f))
            image_files = sorted(image_files)

        if not image_files:
            return (torch.zeros(1, 64, 64, 3), f"❌ No images found in {preset_dir}")

        total = len(image_files)

        # Une seule table, partagee par la detection de reprise et la sauvegarde :
        # les faire calculer chacune de leur cote, c'est la garantie qu'elles
        # divergeront un jour.
        _out_numbers, _renumbered = _assign_output_numbers(image_files)
        if _renumbered:
            print(f"[Dataset] '{preset}': filenames carry no usable numbering — "
                  f"outputs are numbered 001..{total:03d} in alphabetical order.")

        # ── Output folder ─────────────────────────────────────────────────────
        if not output_folder or not output_folder.strip():
            output_folder = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dataset_output",
            )
        dataset_folder = os.path.join(output_folder, f"{trigger_word}_{preset}")
        os.makedirs(dataset_folder, exist_ok=True)

        if _renumbered:
            _out_numbers = _stabilize_renumbering(dataset_folder, image_files, _out_numbers)

        # ── Resume: skip already processed ───────────────────────────────────
        skip_log = _load_skip_log(dataset_folder)
        print(f"\n{'═'*60}")
        print(f"🗂️   ONYX DATASET CREATOR")
        print(f"    Preset     : {preset}")
        print(f"    Trigger    : {trigger_word}")
        print(f"    Provider   : {provider}")
        print(f"    Resolution : {resolution}")
        print(f"    Images     : {total}")
        _existing = sum(
            1 for f in os.listdir(dataset_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"    Already done: {_existing} file(s) in output folder")
        print(f"    Test mode  : {'ON' if test_mode else 'OFF'}")
        print(f"    Output     : {dataset_folder}")
        print(f"{'═'*60}\n")

        # ── Vertex JSON files ─────────────────────────────────────────────────
        _vj_files = []
        if provider == "VERTEX" and vertex_json_folder:
            _vj_files = _load_vertex_json_folder(vertex_json_folder.strip())
            print(f"[Dataset] 🔑 {len(_vj_files)} Vertex project(s) found.")

        _batch_workers = len(_vj_files) if (provider == "VERTEX" and _vj_files) else _BATCH_OTHERS

        # ── Shared kwargs for _do_face_swap ───────────────────────────────────
        swap_kwargs = dict(
            provider           = provider,
            image_size         = resolution,
            disable_safety     = disable_safety,
            face_expression    = "Neutral",
            gemini_api_key     = (gemini_api_key    or "").strip(),
            wavespeed_api_key  = (wavespeed_api_key or "").strip(),
            kie_api_key        = (kie_api_key       or "").strip(),
            fal_api_key        = (fal_api_key       or "").strip(),
            vertex_json_folder = (vertex_json_folder or "").strip(),
            vertex_location    = "us-central1",
            custom_prompt      = (custom_prompt or "").strip(),
        )

        # ── Counters ──────────────────────────────────────────────────────────
        processed_count = 0
        skipped_count   = 0
        failed_count    = 0
        last_tensor     = torch.zeros(1, 64, 64, 3)
        summary_lines   = []

        # ── Separate pending from already-done (file-based check) ──────────────
        # An image is considered done if its output file already exists.
        # Delete the output file to force a regeneration.
        import re as _re
        # Pre-build a set of (trigger_word, numeric_value) from existing output files
        # so detection works regardless of zero-padding (viola_01 vs viola_001)
        _existing_nums = set()
        for _fn in os.listdir(dataset_folder):
            _fn_lower = _fn.lower()
            if not _fn_lower.endswith((".jpg", ".jpeg", ".png")):
                continue
            _stem = os.path.splitext(_fn)[0]
            if _stem.startswith(trigger_word + "_"):
                _suffix = _stem[len(trigger_word) + 1:]
                if _suffix.isdigit():
                    _existing_nums.add(int(_suffix))
        pending_imgs = []
        for idx, img_path in enumerate(image_files, 1):
            key     = os.path.splitext(os.path.basename(img_path))[0]
            num_str = _out_numbers[img_path]
            num_int = int(num_str) if num_str.isdigit() else -1
            if num_int in _existing_nums:
                skipped_count += 1
                print(f"⏭️   [{idx:>4}/{total}] {key} — output exists, skipping.")
            else:
                pending_imgs.append((idx, img_path))

        if not pending_imgs:
            return (last_tensor, f"✅ All {total} images already processed.")

        # File numbers are derived from the swap filename (swap_01 → 01, swap_45 → 45)

        # ── _process_one ──────────────────────────────────────────────────────
        def _process_one(idx, img_path, vj_override):
            key = os.path.splitext(os.path.basename(img_path))[0]
            try:
                pil_img = Image.open(img_path).convert("RGB")
                target_tensor = pil_to_tensor(pil_img)
            except Exception as e:
                return (key, None, None, "load_failed", str(e))

            # Count faces in source image (for summary)
            _yunet_thr = _YUNET_THRESHOLD_OVERRIDES.get(key, _YUNET_DEFAULT)
            face_count = _count_faces(pil_img, score_threshold=_yunet_thr)

            # Build per-image prompt (global custom_prompt + per-image override)
            _per_img_extra = _IMAGE_PROMPT_OVERRIDES.get(key, "")
            _base_prompt   = swap_kwargs.get("custom_prompt", "")
            _final_prompt  = (_base_prompt + " " + _per_img_extra).strip()
            _preset_extra  = _PRESET_PROMPT_OVERRIDES.get(preset, "")
            if _preset_extra:
                _final_prompt = (_final_prompt + " " + _preset_extra).strip()
            _preset_img_extra = _PRESET_IMAGE_OVERRIDES.get((preset, key), "")
            if _preset_img_extra:
                _final_prompt = (_final_prompt + " " + _preset_img_extra).strip()
            # Large-preset: per-swap body description routing
            if preset == "Large":
                import re as _re2
                _swap_num = int(_m.group()) if (_m := _re2.search(r'\d+', key)) else -1
                if _swap_num in _LARGE_NO_PROMPT:
                    _large_extra = ""
                elif _swap_num in _LARGE_FULL_SWAPS:
                    _large_extra = _LARGE_FULL_PROMPT
                elif _swap_num in _LARGE_BUTTOCKS_SWAPS:
                    _large_extra = _LARGE_BUTTOCKS_PROMPT
                else:
                    _large_extra = _LARGE_CHEST_PROMPT
                if _large_extra:
                    _final_prompt = (_final_prompt + " " + _large_extra).strip()
            if test_mode:
                _final_prompt = (_final_prompt + " exact same body shape").strip()
                _final_prompt = (
                    _final_prompt + " "
                    "Her chest is extremely large with heavy breast volume, strong projection, and visible softness. "
                    "exaggerated proportions only focus on the chest area. "
                    "The clothing follows and emphasizes this thick, soft, curvy silhouette, "
                    "suggesting heavy natural volume underneath."
                ).strip()
                if preset == "Large":
                    _final_prompt = (
                        _final_prompt + " "
                        "Her body is the same from the picture 2, according to this caracteristics. "
                        "She has an extremely thick, heavily curvy body with abundant soft body fat distributed across her entire physique. "
                        "Her chest is extremely large with heavy breast volume, strong projection, and visible softness. "
                        "Her stomach is slightly flat, blending naturally into a normal waist. "
                        "Her buttocks are huge, round, and very prominent with substantial visible volume. "
                        "The overall silhouette is intensely curvy, dense, soft, and chubby rather than slim, "
                        "with exaggerated proportions spread naturally across her chest and buttocks. "
                        "The clothing follows and emphasizes this thick, soft, ultra-curvy silhouette, "
                        "suggesting heavy natural volume underneath."
                    ).strip()
            _final_prompt  = (_final_prompt + " " + _FACESWAP_GLOBAL_SUFFIX).strip()
            if _per_img_extra:
                print(f"   📝 [{idx}/{total}] Per-image prompt override: \"{_per_img_extra}\"")

            model_order = ["Nano Banana Pro", "Nano Banana 2"]
            # Per-preset retry counts
            if preset == "Large":
                _retry_for = {"Nano Banana Pro": 4, "Nano Banana 2": 8}
            else:
                _retry_for = {"Nano Banana Pro": _RETRY_COUNT, "Nano Banana 2": _RETRY_COUNT}
            result_tensor = None
            used_model    = None

            for model_name in model_order:
                _retries = _retry_for.get(model_name, _RETRY_COUNT)
                for attempt in range(1, _retries + 1):
                    print(f"   🚀 [{idx}/{total}] {model_name} attempt {attempt}/{_retries}...")
                    try:
                        result  = _do_face_swap(
                            aio, model_name,
                            face_reference, target_tensor,
                            yunet_score_threshold=_yunet_thr,
                            temperature=0.3,
                            **{**swap_kwargs, "custom_prompt": _final_prompt},
                            vertex_json_file_override=vj_override,
                        )
                        img_out = result[0]
                        if img_out.shape[1] > 64 and img_out.shape[2] > 64:
                            # Background similarity check
                            result_pil = Image.fromarray(
                                (img_out[0].cpu().numpy() * 255)
                                .clip(0, 255).astype(np.uint8)
                            )
                            bg_ok, bg_score = _check_background(pil_img, result_pil)
                            if bg_ok:
                                try:
                                    pose_ok, pose_score, pose_method = _check_pose(pil_img, result_pil)
                                except Exception as _pe:
                                    print(f"   [Pose] Exception: {_pe} — skipping check")
                                    pose_ok, pose_score, pose_method = True, 1.0, "error"
                                # score==0.0 → MediaPipe detection error, pass-through
                                if pose_score == 0.0 and pose_method == "mediapipe":
                                    print(f"   [Pose] score=0.000 [mediapipe] — detection error, skipping check")
                                    pose_ok = True
                                elif pose_method != "error":
                                    _pose_status = "OK" if pose_ok else "FAIL -- retrying"
                                    print(f"   [Pose] score={pose_score:.3f} [{pose_method}] "
                                          f"threshold={_POSE_MP_THRESHOLD} -> {_pose_status}")
                                if pose_ok:
                                    result_tensor = img_out
                                    used_model    = model_name
                                    print(f"   ✅ [{idx}/{total}] {model_name} — success "
                                          f"(attempt {attempt}) | bg={bg_score:.2f} | "
                                          f"pose={pose_score:.3f} [{pose_method}]")
                                    break
                            else:
                                print(f"   🔍 [{idx}/{total}] Background check failed "
                                      f"(score={bg_score:.2f} < {_BG_CHECK_THRESHOLD}) "
                                      f"— retrying…")
                        else:
                            print(f"   ⚠️  [{idx}/{total}] {model_name} returned placeholder "
                                  f"(attempt {attempt}/{_retries}).")
                    except Exception as e:
                        print(f"   ❌ [{idx}/{total}] {model_name} attempt {attempt}/{_retries} failed: {e}")
                    # Vertex: delay between retries to avoid 429
                    if result_tensor is None and attempt < _retries and provider == "VERTEX":
                        print(f"   ⏳ [{idx}/{total}] Vertex retry delay 10s...")
                        time.sleep(10)
                if result_tensor is not None:
                    break
                print(f"   ↩️  [{idx}/{total}] {model_name} exhausted — trying next model...")

            return (key, result_tensor, used_model,
                    "done" if result_tensor is not None else "all_failed", None, face_count)

        # ── _commit_result ────────────────────────────────────────────────────
        def _commit_result(idx, img_path, key, result_tensor, used_model, status, error, face_count=0):
            nonlocal processed_count, failed_count, skipped_count, last_tensor

            if status == "load_failed":
                failed_count += 1
                with _LOCK:
                    skip_log[key] = {"status": "load_failed", "error": error}
                    _save_skip_log(dataset_folder, skip_log)
                return

            if result_tensor is None:
                print(f"   ❌ All models failed for {key}.")
                failed_count += 1
                with _LOCK:
                    skip_log[key] = {"status": "all_failed"}
                    _save_skip_log(dataset_folder, skip_log)
                return

            # Numerotation issue de la table calculee une fois pour tout le lot.
            num_str   = _out_numbers[img_path]
            save_name = f"{trigger_word}_{num_str}.png"
            save_path = os.path.join(dataset_folder, save_name)
            try:
                result_pil = Image.fromarray(
                    (result_tensor[0].cpu().numpy() * 255)
                    .clip(0, 255).astype(np.uint8)
                )
                result_pil.save(save_path, "PNG", optimize=True)
                print(f"   💾 Saved → {save_path}")

                # ── Caption auto (Resources/CAPTIONS.txt) ──────────────
                # Pour chaque image generee, on cree un .txt avec la
                # caption correspondante au numero de swap, en remplacant
                # le placeholder "TRIGGER" par le trigger_word choisi.
                _captions = _load_captions_dict(preset)
                if _captions:
                    try:
                        _num_int = int(num_str)
                    except ValueError:
                        _num_int = -1
                    _cap = _captions.get(_num_int)
                    if _cap:
                        # Remplace TRIGGER (forme la plus stricte) par trigger_word
                        _cap_filled = _cap.replace("TRIGGER", trigger_word)
                        _txt_path = os.path.join(
                            dataset_folder, f"{trigger_word}_{num_str}.txt"
                        )
                        try:
                            with open(_txt_path, "w", encoding="utf-8") as _fh:
                                _fh.write(_cap_filled)
                            print(f"   📝 Caption → {_txt_path}")
                        except Exception as _ce:
                            print(f"   ⚠️  Caption save failed: {_ce}")
                    else:
                        print(f"   ℹ️  No caption for image {_num_int} in CAPTIONS - {preset.upper()}.txt")
            except Exception as e:
                print(f"   ⚠️  Save failed: {e}")

            last_tensor     = result_tensor
            processed_count += 1
            with _LOCK:
                skip_log[key] = {"status": "done", "model": used_model, "file": save_name}
                _save_skip_log(dataset_folder, skip_log)
            _fc = f" | {face_count} face(s)" if face_count >= 0 else ""
            summary_lines.append(f"  ✅ {save_name} ← {key} ({used_model}){_fc}")
            # Delay before next image to avoid Vertex 429 on first attempt
            if provider == "VERTEX":
                print(f"   ⏳ [Dataset] Vertex cooldown 10s before next image...")
                time.sleep(10)

        # ── Run ───────────────────────────────────────────────────────────────
        use_parallel = len(pending_imgs) > 1 and _batch_workers > 1
        if use_parallel:
            print(f"\n[Dataset] ⚡ Pool mode: {len(pending_imgs)} images — "
                  f"{_batch_workers} workers (provider={provider})")
        else:
            print(f"\n[Dataset] 🔁 Sequential mode: {len(pending_imgs)} image(s).")

        if use_parallel:
            with ThreadPoolExecutor(max_workers=_batch_workers) as executor:
                future_map = {}
                for i, (idx, ip) in enumerate(pending_imgs):
                    vj_override = _vj_files[i % len(_vj_files)] if _vj_files else ""
                    fut = executor.submit(_process_one, idx, ip, vj_override)
                    future_map[fut] = (idx, ip)
                done_count = 0
                for future in as_completed(future_map):
                    idx, img_path = future_map[future]
                    done_count   += 1
                    remaining     = len(pending_imgs) - done_count
                    try:
                        key, rt, um, status, err, fc = future.result()
                        _commit_result(idx, img_path, key, rt, um, status, err, fc)
                    except Exception as e:
                        key = os.path.splitext(os.path.basename(img_path))[0]
                        print(f"   ❌ Thread error [{idx}/{total}] {key}: {e}")
                        failed_count += 1
                    if remaining > 0:
                        print(f"[Dataset] ⏳ {remaining} image(s) remaining...")
        else:
            for i, (idx, img_path) in enumerate(pending_imgs):
                vj_override = _vj_files[i % len(_vj_files)] if _vj_files else ""
                try:
                    key, rt, um, status, err, fc = _process_one(idx, img_path, vj_override)
                    _commit_result(idx, img_path, key, rt, um, status, err, fc)
                except Exception as e:
                    key = os.path.splitext(os.path.basename(img_path))[0]
                    print(f"   ❌ Error [{idx}/{total}] {key}: {e}")
                    failed_count += 1

        # ── Summary ───────────────────────────────────────────────────────────
        summary = (
            f"🗂️   Dataset Creator — {trigger_word} / {preset}\n"
            f"   Processed : {processed_count}\n"
            f"   Skipped   : {skipped_count} (already done)\n"
            f"   Failed    : {failed_count}\n"
            f"   Output    : {dataset_folder}\n"
        )
        if summary_lines:
            summary += "\nGenerated:\n" + "\n".join(summary_lines)

        print(f"\n{'='*60}")
        print(summary)
        print(f"{'='*60}\n")

        return (last_tensor, summary)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "OnyxDatasetCreatorNode": OnyxDatasetCreatorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxDatasetCreatorNode": "Onyx Dataset Creator",
}
