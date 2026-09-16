"""
Onyx Camera Look — pipeline complet, zero dependance externe (numpy + PIL uniquement).
"""
import io
import random
import math
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

import os as _onyx_os, importlib.util as _onyx_ilu
_onyx_prof_path = _onyx_os.path.join(_onyx_os.path.dirname(_onyx_os.path.dirname(_onyx_os.path.abspath(__file__))), "nodes", "onyx_render_profile.py")
_onyx_prof_spec = _onyx_ilu.spec_from_file_location("onyx_render_profile_pp", _onyx_prof_path)
_onyx_prof_mod = _onyx_ilu.module_from_spec(_onyx_prof_spec)
_onyx_prof_spec.loader.exec_module(_onyx_prof_mod)
ensure_profile_ready = _onyx_prof_mod.ensure_profile_ready
def _noise_params(strength: float):
    # Quadratic curve: low values barely perceptible, smooth ramp to heavy at max.
    # strength=0->0 | 0.1->~0.0004 | 3->0.36 | 5->1.0 (normalized)
    t = (strength / 5.0) ** 2
    iso_scale  = t * 2.0   # 0 -> 2.0
    read_noise = t * 4.0   # 0 -> 4.0
    return iso_scale, read_noise


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur_np(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via numpy convolution (no cv2)."""
    if sigma <= 0:
        return img
    radius = max(1, math.ceil(3 * sigma))
    k = _gaussian_kernel_1d(sigma, radius)
    out = img.astype(np.float32)
    # horizontal
    for c in range(out.shape[2]):
        for row in range(out.shape[0]):
            out[row, :, c] = np.convolve(out[row, :, c], k, mode='same')
    # vertical
    for c in range(out.shape[2]):
        for col in range(out.shape[1]):
            out[:, col, c] = np.convolve(out[:, col, c], k, mode='same')
    return np.clip(out, 0, 255)


def _box_blur_1d(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Original horizontal motion blur algorithm — uniform box kernel, mirror-padded at
    the edges. Matches camera_pipeline.py's _motion_blur (scipy convolve mode='mirror')
    exactly — np.pad(mode='reflect') is numpy's equivalent of scipy's 'mirror' mode
    (edge pixel itself not repeated). Plain np.convolve(mode='same') would implicitly
    zero-pad instead, darkening the image edges — that mismatch is fixed here.
    Returns unclipped float32; caller clips."""
    if kernel_size <= 1:
        return img.astype(np.float32)
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    pad = ks // 2
    k = np.ones(ks, dtype=np.float32) / ks
    out = img.astype(np.float32)
    padded = np.pad(out, ((0, 0), (pad, pad), (0, 0)), mode='reflect')
    result = np.zeros_like(out)
    for c in range(out.shape[2]):
        for row in range(out.shape[0]):
            result[row, :, c] = np.convolve(padded[row, :, c], k, mode='valid')
    return result


def _shift_channel(ch: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel shift of a single channel via separable bilinear interpolation,
    mirror-padded edges, plain numpy (no scipy). Used to displace R/G/B channels
    against each other for the Bayer pixel-displacement effect below."""
    h, w = ch.shape
    out = ch.astype(np.float32)

    # Horizontal shift
    if dx != 0:
        ix = int(np.floor(dx))
        fx = dx - ix
        pad = abs(ix) + 2
        padded = np.pad(out, ((0, 0), (pad, pad)), mode='reflect')
        lo = np.roll(padded, ix, axis=1)
        hi = np.roll(padded, ix + 1, axis=1)
        blended = lo * (1.0 - fx) + hi * fx
        out = blended[:, pad:pad + w]

    # Vertical shift
    if dy != 0:
        iy = int(np.floor(dy))
        fy = dy - iy
        pad = abs(iy) + 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode='reflect')
        lo = np.roll(padded, iy, axis=0)
        hi = np.roll(padded, iy + 1, axis=0)
        blended = lo * (1.0 - fy) + hi * fy
        out = blended[pad:pad + h, :]

    return out


def _bayer_demosaic_effect(img: np.ndarray, strength: float) -> np.ndarray:
    """Bayer-pattern RGB pixel displacement — not a blur. In a real RGGB sensor, R is
    sampled from the top-left of each 2x2 photosite block and B from the bottom-right
    (G from the two remaining corners), so after demosaicing the R and B channels sit on
    sampling grids diagonally offset by ~1px from each other. That geometric mismatch is
    what actually reads as color fringing / channel misalignment at edges — the earlier
    mosaic+bilinear-interpolate port also blurred each channel while filling in the
    missing samples, which is why it looked/felt like blur instead. This version skips
    the interpolation entirely and just shifts R and B diagonally in opposite directions
    by `strength` px (G stays put), which isolates the actual displacement look.
    strength=1.0 matches the ~1px diagonal offset a real Bayer sensor produces;
    higher values push the effect further for a stronger, stylized look."""
    if strength <= 0:
        return img.astype(np.float32)

    img_f = img.astype(np.float32)
    R, G, B = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    offset = strength * 0.5
    R_shifted = _shift_channel(R, -offset, -offset)
    B_shifted = _shift_channel(B, offset, offset)

    return np.stack([R_shifted, G, B_shifted], axis=2)


def _motion_blur(img: np.ndarray, kernel_size: float) -> np.ndarray:
    """Horizontal motion blur — same uniform box-kernel averaging as the original
    implementation (_box_blur_1d), not a Gaussian. `kernel_size` can now be fractional:
    the result is cross-faded between the two nearest odd integer kernel sizes so the
    strength dials smoothly instead of jumping in whole-pixel steps, while the actual
    blur process stays identical to before."""
    if kernel_size <= 1:
        return img.astype(np.float32)

    lo = int(math.floor(kernel_size))
    if lo % 2 == 0:
        lo -= 1
    lo = max(1, lo)
    hi = lo + 2
    frac = min(max((kernel_size - lo) / 2.0, 0.0), 1.0)

    blurred_lo = _box_blur_1d(img, lo)
    if frac <= 0.0:
        return blurred_lo
    blurred_hi = _box_blur_1d(img, hi)
    return blurred_lo * (1.0 - frac) + blurred_hi * frac


def _simulate_camera_pipeline(
    img_arr,
    demosaic_strength     = 1.0,
    iso_scale             = 1.0,
    read_noise_std        = 2.0,
    motion_blur_kernel    = 1.0,
    jpeg_quality          = 98,
    seed                  = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = img_arr.astype(np.float32)

    # 1. Sensor noise: Poisson shot noise + Gaussian read noise
    if iso_scale > 0:
        # Fixed physical scale (80 photons = full brightness) keeps the Poisson model valid.
        # iso_scale / 2.0 is a pure amplitude factor: at max (iso_scale=2.0) -> factor 1.0.
        # This avoids the divergence caused by clamping scale to a near-zero minimum.
        ref_scale = 80.0
        shot_raw  = rng.poisson(np.clip(img, 0, 255) / 255.0 * ref_scale).astype(np.float32) / ref_scale * 255.0 - img
        img       = np.clip(img + shot_raw * (iso_scale / 2.0), 0.0, 255.0)
    if read_noise_std > 0:
        img = np.clip(img + rng.normal(0.0, read_noise_std, img.shape).astype(np.float32), 0.0, 255.0)

    # 2. Bayer pixel displacement (deplacement R/B diagonal, pas un flou) — puissance reglable en continu
    if demosaic_strength > 0:
        img = np.clip(_bayer_demosaic_effect(np.clip(img, 0, 255).astype(np.uint8), demosaic_strength), 0, 255)

    # 3. Motion blur — meme process qu'avant (box kernel), puissance reglable en continu
    if motion_blur_kernel > 1:
        img = np.clip(_motion_blur(np.clip(img, 0, 255).astype(np.uint8), motion_blur_kernel), 0, 255)

    # 4. Compression JPEG via PIL
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return np.array(Image.open(buf))


# ---------------------------------------------------------------------------
# Node ComfyUI
# ---------------------------------------------------------------------------

class Onyx_Camera_Look:
    """
    Onyx Camera Look — pipeline camera physiquement inspire.
    Ordre : bruit capteur -> demosaic -> motion blur -> JPEG.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on":  "Enabled",
                    "label_off": "Bypassed",
                }),

                "demosaic_pixel_blur": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Bayer pixel displacement (R/B channels shifted diagonally, not a blur). 0 = disabled, 1 = natural ~1px sensor offset, higher = stronger.",
                }),
                "noise_strength": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Sensor noise level. 0 = none | 3 = standard | 5 = heavy.",
                }),
                "kernel_motion_blur": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 51.0, "step": 0.1,
                    "tooltip": "Motion blur kernel width (px, box blur — same process as before). 1 = none.",
                }),
                "jpeg_compression": ("INT", {
                    "default": 98, "min": 85, "max": 100, "step": 1,
                    "tooltip": "JPEG quality. 100 = near-lossless, 85 = visible artifacts.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "execute"
    CATEGORY      = "Onyx/Post-Processing"
    DESCRIPTION   = (
        "Onyx Camera Look\n"
        "Complete camera pipeline: sensor noise -> demosaic -> motion blur -> JPEG.\n"
        "Random seed per frame. No external dependencies (numpy + PIL only)."
    )

    def execute(
        self,
        image:                torch.Tensor,
        enabled:              bool,
        demosaic_pixel_blur:  float,
        noise_strength:       float,
        kernel_motion_blur:   float,
        jpeg_compression:     int,
    ):
        ensure_profile_ready()
        if not enabled:
            return (image,)

        iso_scale, read_noise = _noise_params(noise_strength)
        out_frames = []

        for i in range(image.shape[0]):
            frame_seed = random.randint(0, 2_147_483_647)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)

            processed = _simulate_camera_pipeline(
                img_arr               = img_np,
                demosaic_strength     = demosaic_pixel_blur,
                iso_scale             = iso_scale,
                read_noise_std        = read_noise,
                motion_blur_kernel    = kernel_motion_blur,
                jpeg_quality          = jpeg_compression,
                seed                  = frame_seed,
            )
            tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0)
            out_frames.append(tensor)

        if not out_frames:
            return (image,)

        return (torch.cat(out_frames, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Onyx_Camera_Look": Onyx_Camera_Look,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Onyx_Camera_Look": "Onyx Camera Look",
}
