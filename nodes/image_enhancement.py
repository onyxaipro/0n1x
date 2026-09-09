"""
Onyx Image Enhancement
Enhances image quality by leveraging Nano Banana Pro/2's tendency to fully
regenerate an image when the input and output aspect ratios differ.

Soft mode: crop to a mismatched ratio + 1% pixel crop, then generate back to original ratio.
Hard mode: two-pass generation (opposite ratio first, then back to original ratio).
"""

import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MODELS = ["Nano Banana Pro", "Nano Banana 2"]

# SOFT mode: input ratio -> ratio we crop the image to before generation
_SOFT_CROP_MAP = {
    "3:4":  "9:16",
    "9:16": "3:4",
    "4:5":  "9:16",
    "1:1":  "9:16",
}

# Ratio float values for detection
_RATIO_VALUES = {
    "3:4":  3 / 4,
    "9:16": 9 / 16,
    "4:5":  4 / 5,
    "1:1":  1.0,
}

_ENHANCE_PROMPT = (
    "Maintain the pose, framing, clothing and environment from this picture.\n"
    "Generate a newly reconstructed version of the subject from picture 1 naturally photographed in the scene, "
    "pose and camera framing of this picture.\n"
    "\n"
    "Freely regenerate everything from scratch.\n"
    "The subject must look like a single naturally photographed person captured in-camera.\n"
    "The lighting on the face and hair must perfectly match the overall lighting of picture 1.\n"
    "Preserve the identity consistency and facial proportions.\n"
    "No floating face effect or incorrect head placement.\n"
    "Face must inherit the same natural RAW photo look as the body.\n"
    "\n"
    "Exact same framing, and exact same pose from the original picture. Do not drift from the original picture.\n"
    "\n"
    "Preserve the exact original composition with zero deviation. "
    "Maintain the exact same camera framing, perspective, body positioning, pose, limb placement, head angle and subject scale from the original image. "
    "Do not modify the scene composition in any way. "
    "The generated image must remain perfectly aligned to the original photograph. "
    "Absolutely no cropping, reframing, zooming, rotation or camera shift. "
    "Keep the entire original image fully intact and fully visible. "
    "Only extend or regenerate the missing surrounding areas required to match the target aspect ratio. "
    "The original image content must remain unchanged in position and scale.\n"
    "\n"
    "Camera/look: DSLR-raw crispness, sharp focus, no smoothing. "
    "Crisp skin texture, light natural sensor grain. "
    "ULTRA DETAILED SKIN TEXTURE with visible pores, NO ACNE, pas de boutons, EXTREME DETAIL, "
    "Native 8K resolution. Photorealism, Add a very subtle film grain"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_soft_ratio(h, w):
    """Return the closest accepted ratio string, or None if not close enough."""
    r = w / h
    best, best_diff = None, float("inf")
    for name, val in _RATIO_VALUES.items():
        diff = abs(r - val)
        if diff < best_diff:
            best_diff = diff
            best = name
    return best if best_diff < 0.08 else None


def _center_crop_to_ratio(tensor, ratio_str):
    """Center-crop [1, H, W, C] tensor to the given 'W:H' ratio string."""
    num, den = map(int, ratio_str.split(":"))
    target_r = num / den          # width / height
    _, H, W, _ = tensor.shape
    current_r = W / H
    if current_r > target_r:     # too wide -> crop width
        new_W = int(H * target_r)
        off = (W - new_W) // 2
        return tensor[:, :, off:off + new_W, :]
    elif current_r < target_r:   # too tall -> crop height
        new_H = int(W / target_r)
        off = (H - new_H) // 2
        return tensor[:, off:off + new_H, :, :]
    return tensor


def _tiny_crop(tensor, factor=0.99):
    """Shrink both dimensions by factor via center crop to force pixel mismatch."""
    _, H, W, _ = tensor.shape
    new_H = int(H * factor)
    new_W = int(W * factor)
    oh = (H - new_H) // 2
    ow = (W - new_W) // 2
    return tensor[:, oh:oh + new_H, ow:ow + new_W, :]


def _extract_tensors(result):
    imgs = []
    t = result[0]
    if t is None or t.shape[-1] != 3:
        return imgs
    if t.dim() == 4:
        for b in range(t.shape[0]):
            if float(t[b].max()) > 0.01:
                imgs.append(t[b])
    elif float(t.max()) > 0.01:
        imgs.append(t)
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class OnyxImageEnhancementNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "mode":       (["Soft", "Hard"], {
                    "default": "Soft",
                    "tooltip": (
                        "Soft: crop to mismatched ratio + tiny crop, then generate back to original ratio.\n"
                        "Hard: two-pass generation (opposite ratio first, then back to original)."
                    ),
                }),
                "model":      (_MODELS, {"default": "Nano Banana Pro"}),
                "provider":   (["GOOGLE", "WAVESPEED", "KIE", "FAL", "VERTEX"], {"default": "GOOGLE"}),
                "resolution": (["2K", "4K"], {"default": "2K"}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of enhanced images to generate.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Optional extra instructions (appended to the enhancement prompt).",
                }),
            },
            "optional": {
                "gemini_api_key":     ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "[GOOGLE] Google AI Studio API key."}),
                "wavespeed_api_key":  ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "[WAVESPEED] WaveSpeed API key."}),
                "kie_api_key":        ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "[KIE] Kie.ai API key."}),
                "fal_api_key":        ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "[FAL] Fal.ai API key."}),
                "vertex_json_folder": ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "[VERTEX] Path to service account JSON folder."}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
    OUTPUT_NODE   = False
    FUNCTION      = "run"
    CATEGORY      = "Onyx/Automation"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        image,
        mode               = "Soft",
        model              = "Nano Banana Pro",
        provider           = "GOOGLE",
        resolution         = "2K",
        batch_size         = 1,
        prompt             = "",
        gemini_api_key     = "",
        wavespeed_api_key  = "",
        kie_api_key        = "",
        fal_api_key        = "",
        vertex_json_folder = "",
    ):
        from .nano_banana_aio import OnyxNanoBananaAIO
        aio = OnyxNanoBananaAIO()

        batch_size = max(1, min(batch_size, 10))

        extra        = prompt.strip()
        final_prompt = (_ENHANCE_PROMPT + "\n\n" + extra) if extra else _ENHANCE_PROMPT

        _, H, W, _ = image.shape

        shared = dict(
            provider                 = provider,
            negative_prompt          = "",
            image_size               = resolution,
            gemini_api_key           = gemini_api_key,
            wavespeed_api_key        = wavespeed_api_key,
            kie_api_key              = kie_api_key,
            fal_api_key              = fal_api_key,
            vertex_json_folder       = vertex_json_folder,
            disable_safety_threshold = True,
            model                    = model,
            batch_size               = batch_size,
            use_search               = False,
            system_instructions      = None,
            temperature              = 1.0,
            top_p                    = 0.95,
            fal_safety_tolerance     = "6",
            fal_enable_web_search    = False,
            image_2                  = None,
            video_mode_enabled       = False,
            face_swap_enabled        = False,
            breast_refiner_enabled   = False,
            low_neck_enabled         = False,
            face_expression          = "Neutral",
            gpt2_image_quality       = "high",
        )

        # ── SOFT mode ─────────────────────────────────────────────────────────
        if mode == "Soft":
            ratio_str = _detect_soft_ratio(H, W)
            if ratio_str is None:
                raise ValueError(
                    f"[Enhancement] Unsupported input ratio ({W}x{H}). "
                    "Accepted ratios: 3:4, 9:16, 4:5, 1:1."
                )
            crop_ratio = _SOFT_CROP_MAP[ratio_str]
            print(f"[Enhancement] Soft | detected={ratio_str} | crop_to={crop_ratio} | generate_as={ratio_str}")

            # Crop to mismatched ratio, then apply 1% pixel crop
            prepped = _center_crop_to_ratio(image, crop_ratio)
            prepped = _tiny_crop(prepped, 0.99)
            print(f"[Enhancement] Prepped shape: {prepped.shape}")

            result = aio.generate_unified(
                prompt       = final_prompt,
                aspect_ratio = ratio_str,
                image_1      = prepped,
                **shared,
            )
            imgs = _extract_tensors(result)

        # ── HARD mode ─────────────────────────────────────────────────────────
        else:
            original_ar = OnyxNanoBananaAIO._detect_aspect_ratio(image)

            if W > H:
                orientation = "landscape"
                first_ar    = "9:16"
            elif W == H:
                orientation = "square"
                first_ar    = "16:9"
                original_ar = "1:1"
            else:
                orientation = "portrait"
                first_ar    = "16:9"

            print(f"[Enhancement] Hard | orientation={orientation} | pass1={first_ar} | pass2={original_ar}")

            # Pass 1: generate in opposite ratio
            result1 = aio.generate_unified(
                prompt       = final_prompt,
                aspect_ratio = first_ar,
                image_1      = image,
                **shared,
            )
            pass1 = _extract_tensors(result1)
            if not pass1:
                raise RuntimeError("[Enhancement] Hard mode pass 1 produced no images.")

            intermediate = torch.stack(pass1, dim=0)
            print(f"[Enhancement] Pass 1 done ({len(pass1)} image(s)). Starting pass 2...")

            # Pass 2: generate back in original ratio
            result2 = aio.generate_unified(
                prompt       = final_prompt,
                aspect_ratio = original_ar,
                image_1      = intermediate,
                **shared,
            )
            imgs = _extract_tensors(result2)

        # ── Output ────────────────────────────────────────────────────────────
        if not imgs:
            print("[Enhancement] No images generated.")
            return (torch.zeros(1, 64, 64, 3),)

        batch = torch.stack(imgs, dim=0)
        print(f"[Enhancement] Done — {len(imgs)} image(s).")
        return (batch,)
