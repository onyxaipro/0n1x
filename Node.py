"""
ComfyUI custom nodes for Nano Banana Pro and Nano Banana 2 APIs.
- Nano Banana Pro Edit: edit with images (Gemini 3 Pro).
- Nano Banana 2 Edit: edit with images (Gemini 3.1 Flash).
Each node returns image as ComfyUI IMAGE tensor [B, H, W, C].
"""

import logging
import io
import base64
import random as _random

import requests
import torch
import numpy as np
from PIL import Image


from .nodes.onyx_render_profile import ensure_profile_ready
_SAFETY_THRESHOLDS = [
    "OFF",
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
]

_HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

_ASPECT_RATIOS = [
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
]

_IMAGE_SIZES = ["1K", "2K", "4K"]


# ---------- Nano Banana Pro Edit ----------

class NanoBananaProEditAPINode:
    """Edit/Generate images via Gemini 3 Pro (gemimg.py reference implementation)."""

    MODEL_ID = "gemini-3-pro-image"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (_IMAGE_SIZES, {"default": "2K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "safety_threshold": (_SAFETY_THRESHOLDS, {"default": "OFF"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, api_key, prompt, aspect_ratio="auto", image_size="2K", temperature=1.0, seed=0, safety_threshold="OFF"):
        ensure_profile_ready()
        if api_key.strip():
            NanoBananaProEditAPINode._cached_api_key = api_key.strip()
        api_key = api_key.strip() or NanoBananaProEditAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Nano Banana Pro: API key is required.")

        # Build parts: images first, then text (matching gemimg.py order)
        parts = _images_to_parts(images)
        parts.append({"text": prompt.strip()})

        # Auto aspect ratio detection
        if aspect_ratio == "auto":
            h, w = images.shape[1], images.shape[2]
            ratio = w / h
            ratios = {
                "1:1": 1.0, "2:3": 0.66, "3:2": 1.5, "3:4": 0.75, "4:3": 1.33,
                "4:5": 0.8, "5:4": 1.25, "9:16": 0.56, "16:9": 1.77, "21:9": 2.33,
            }
            aspect_ratio = min(ratios, key=lambda k: abs(ratios[k] - ratio))

        # Build payload matching gemimg.py structure exactly
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                },
                "responseModalities": ["Image"],
            },
            "contents": [{"parts": parts}],
            "system_instruction": {
                "parts": [{
                    "text": "You are an expert image-generation engine. You must ALWAYS produce an image. "
                            "Interpret all user input—regardless of format, intent, or abstraction—as literal "
                            "visual directives for image composition. Prioritize generating the visual "
                            "representation above any text, formatting, or conversational requests."
                }]
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
        }

        # Use x-goog-api-key header (matching gemimg.py)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        url = f"{self.BASE_URL}/{self.MODEL_ID}:generateContent"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=180)
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Nano Banana Pro request error: {e}")

        # Check for API error
        if "error" in data:
            err = data["error"]
            raise RuntimeError(f"Nano Banana Pro API error: {err.get('code')} — {err.get('message')}")

        # Check for blocked content
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Nano Banana Pro: no candidates in response.")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        if finish_reason in ("PROHIBITED_CONTENT", "NO_IMAGE"):
            raise RuntimeError(f"Nano Banana Pro: image not generated due to {finish_reason}.")

        if "content" not in candidate:
            raise RuntimeError("Nano Banana Pro: no content in response.")

        # Extract inline image data
        response_parts = candidate["content"]["parts"]
        for part in response_parts:
            if "inlineData" in part:
                img_bytes = base64.b64decode(part["inlineData"]["data"])
                return _image_bytes_to_tensor(img_bytes, "Nano Banana Pro")

        raise RuntimeError("Nano Banana Pro: no image found in response parts.")


# ---------- Nano Banana 2 Edit ----------

class NanoBanana2EditAPINode:
    """Edit images via Gemini 3.1 Flash API."""

    MODEL_ID = "gemini-3.1-flash-image"
    BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (_IMAGE_SIZES, {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "safety_threshold": (_SAFETY_THRESHOLDS, {"default": "OFF"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, api_key, prompt, aspect_ratio="auto", image_size="1K", temperature=1.0, seed=0, safety_threshold="OFF"):
        ensure_profile_ready()
        if api_key.strip():
            NanoBanana2EditAPINode._cached_api_key = api_key.strip()
        api_key = api_key.strip() or NanoBanana2EditAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Nano Banana 2 Edit: API key is required.")
        if images.shape[0] == 0:
            raise RuntimeError("Nano Banana 2 Edit: at least one image is required.")

        parts = _images_to_parts(images)
        parts.append({"text": prompt})

        image_config = {}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if image_size:
            image_config["imageSize"] = image_size

        gen_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": temperature,
            "seed": seed,
        }
        if image_config:
            gen_config["imageConfig"] = image_config

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {
                "parts": [{
                    "text": "You are an expert image-generation engine. You must ALWAYS produce an image. "
                            "Interpret all user input—regardless of format, intent, or abstraction—as literal "
                            "visual directives for image composition. Prioritize generating the visual "
                            "representation above any text, formatting, or conversational requests."
                }]
            },
            "generationConfig": gen_config,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
        }

        url = f"{self.BASE_URL}?key={api_key}"
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(_api_error_msg("Nano Banana 2 Edit", e))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Nano Banana 2 Edit request error: {e!s}")

        raw = _extract_image_from_gemini_response(data, "Nano Banana 2 Edit")
        if not raw:
            detail = _gemini_response_summary(data)
            raise RuntimeError(f"Nano Banana 2 Edit: no image in response. {detail}")
        return _image_bytes_to_tensor(raw, "Nano Banana 2 Edit")


# ---------- Seedream 4.5 Edit (WaveSpeed) ----------

import time as _time

_SEEDREAM_SIZES = [
    "auto — match input (2K)",
    "1920*1920 — square (1:1)",
    "2560*1920 — landscape_4_3 (4:3)",
    "1920*2560 — portrait_4_3 (3:4)",
    "1536*1920 — portrait_4_5 (4:5)",
    "1920*1536 — landscape_5_4 (5:4)",
    "2560*1440 — landscape_16_9 (16:9)",
    "1440*2560 — portrait_16_9 (9:16)",
]


class SeedreamEditAPINode:
    """Edit images via ByteDance Seedream 4.5 Edit on WaveSpeed API.

    Supports 1-10 reference images for context-aware editing with
    facial feature, lighting, and color tone preservation up to 4K.
    """

    BASE_URL = "https://api.wavespeed.ai/api/v3/bytedance/seedream-v4.5/edit"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "wavespeed_apikey": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "size": (_SEEDREAM_SIZES, {"default": "auto — match input (2K)"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "use_custom_size": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, wavespeed_apikey, prompt, size="auto — match input (2K)", num_images=1, use_custom_size=False, custom_width=2048, custom_height=2048):
        ensure_profile_ready()
        if wavespeed_apikey.strip():
            SeedreamEditAPINode._cached_api_key = wavespeed_apikey.strip()
        api_key = wavespeed_apikey.strip() or SeedreamEditAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Seedream 4.5 Edit: WaveSpeed API key is required.")
        if not prompt.strip():
            raise RuntimeError("Seedream 4.5 Edit: prompt is required.")
        if images.shape[0] == 0:
            raise RuntimeError("Seedream 4.5 Edit: at least one image is required.")

        # Resolve size
        if use_custom_size:
            api_size = f"{custom_width}*{custom_height}"
        elif size.startswith("auto"):
            # Use first image's exact aspect ratio, scale to meet minimum pixels
            h, w = images.shape[1], images.shape[2]
            min_pixels = 3686400
            # Scale up proportionally until we meet minimum pixels
            scale = max(1.0, (min_pixels / (w * h)) ** 0.5)
            new_w = int(round(w * scale / 64) * 64)
            new_h = int(round(h * scale / 64) * 64)
            # Clamp to API limits
            new_w = max(512, min(8192, new_w))
            new_h = max(512, min(8192, new_h))
            api_size = f"{new_w}*{new_h}"
            logging.info("Seedream auto size: input %dx%d -> output %s", w, h, api_size)
        else:
            api_size = size.split(" ")[0]  # e.g. "2048*2048" from "2048*2048 — square_hd (1:1)"

        # Convert images to data URIs (max 10)
        image_uris = []
        count = min(images.shape[0], 10)
        for i in range(count):
            img_np = (255.0 * images[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            image_uris.append(f"data:image/png;base64,{b64}")

        payload = {
            "images": image_uris,
            "prompt": prompt,
            "size": api_size,
            "num_images": num_images,
            "enable_sync_mode": True,
            "enable_base64_output": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(_api_error_msg("Seedream 4.5 Edit", e))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Seedream 4.5 Edit request error: {e!s}")

        data = result.get("data", {})

        # Async polling fallback if sync mode didn't complete immediately
        if data.get("status") not in ("completed",):
            poll_url = data.get("urls", {}).get("get", "")
            if not poll_url:
                raise RuntimeError("Seedream 4.5 Edit: no poll URL and task not completed.")
            for _ in range(90):
                _time.sleep(2)
                try:
                    poll_resp = requests.get(poll_url, headers=headers, timeout=30)
                    poll_resp.raise_for_status()
                    data = poll_resp.json().get("data", {})
                except requests.exceptions.RequestException as e:
                    logging.warning("Seedream 4.5 Edit poll error: %s", e)
                    continue
                if data.get("status") == "completed":
                    break
                if data.get("status") == "failed":
                    raise RuntimeError(f"Seedream 4.5 Edit failed: {data.get('error', 'unknown error')}")
            else:
                raise RuntimeError("Seedream 4.5 Edit timed out after 3 minutes.")

        outputs = data.get("outputs", [])
        if not outputs:
            raise RuntimeError("Seedream 4.5 Edit: no images in API response.")

        # Decode all output images
        import torch
        tensors = []
        for idx, output in enumerate(outputs):
            if output.startswith("data:"):
                b64_data = output.split(",", 1)[-1]
                raw = base64.b64decode(b64_data)
            elif output.startswith("http"):
                try:
                    img_resp = requests.get(output, timeout=60)
                    img_resp.raise_for_status()
                    raw = img_resp.content
                except requests.exceptions.RequestException as e:
                    raise RuntimeError(f"Seedream 4.5 Edit: failed to download image {idx}: {e!s}")
            else:
                # Assume raw base64
                raw = base64.b64decode(output)
            tensor_tuple = _image_bytes_to_tensor(raw, "Seedream 4.5 Edit")
            tensors.append(tensor_tuple[0])

        if not tensors:
            raise RuntimeError("Seedream 4.5 Edit: no valid images decoded.")

        batched = torch.cat(tensors, dim=0)
        return (batched,)


# ---------- Seedream 4.5 Edit (fal.ai) ----------

_FAL_SEEDREAM_SIZES = [
    "auto — match input (2K)",
    "1920*1920 — square (1:1)",
    "2560*1920 — landscape (4:3)",
    "1920*2560 — portrait (3:4)",
    "1536*1920 — portrait (4:5)",
    "1920*1536 — landscape (5:4)",
    "2560*1440 — landscape (16:9)",
    "1440*2560 — portrait (9:16)",
]

_FAL_SIZE_MAP = {
    "1920*1920": "square_hd",
    "2560*1920": "landscape_4_3",
    "1920*2560": "portrait_4_3",
    "1536*1920": {"width": 1536, "height": 1920},
    "1920*1536": {"width": 1920, "height": 1536},
    "2560*1440": "landscape_16_9",
    "1440*2560": "portrait_16_9",
}


class SeedreamEditFalAPINode:
    """Edit images via ByteDance Seedream 4.5 Edit on fal.ai.

    Supports 1-10 reference images for context-aware editing with
    facial feature, lighting, and color tone preservation up to 4K.
    """

    BASE_URL = "https://fal.run/fal-ai/bytedance/seedream/v4.5/edit"
    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fal_apikey": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "size": (_FAL_SEEDREAM_SIZES, {"default": "auto — match input (2K)"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "use_custom_size": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def edit(self, images, fal_apikey, prompt, size="auto — match input (2K)", num_images=1, use_custom_size=False, custom_width=2048, custom_height=2048):
        ensure_profile_ready()
        if fal_apikey.strip():
            SeedreamEditFalAPINode._cached_api_key = fal_apikey.strip()
        api_key = fal_apikey.strip() or SeedreamEditFalAPINode._cached_api_key
        if not api_key:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): fal.ai API key is required.")
        if not prompt.strip():
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): prompt is required.")
        if images.shape[0] == 0:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): at least one image is required.")

        # Resolve size
        if use_custom_size:
            fal_size = {"width": custom_width, "height": custom_height}
        elif size.startswith("auto"):
            # Use first image's exact aspect ratio, scale to ~2K
            h, w = images.shape[1], images.shape[2]
            target_pixels = 1920 * 1920  # 2K target
            current_pixels = w * h
            scale = (target_pixels / current_pixels) ** 0.5
            new_w = int(round(w * scale / 64) * 64)
            new_h = int(round(h * scale / 64) * 64)
            new_w = max(512, min(4096, new_w))
            new_h = max(512, min(4096, new_h))
            fal_size = {"width": new_w, "height": new_h}
            logging.info("Seedream (fal.ai) auto size: input %dx%d -> output %dx%d", w, h, new_w, new_h)
        else:
            dimension_str = size.split(" ")[0]
            fal_size = _FAL_SIZE_MAP.get(dimension_str, "square_hd")

        # Convert images to data URIs (max 10)
        image_urls = []
        count = min(images.shape[0], 10)
        for i in range(count):
            img_np = (255.0 * images[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            image_urls.append(f"data:image/png;base64,{b64}")

        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": fal_size,
            "num_images": num_images,
            "seed": _random.randint(0, 2147483647),
            "enable_safety_checker": False,
        }
        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(_api_error_msg("Seedream 4.5 Edit (fal.ai)", e))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Seedream 4.5 Edit (fal.ai) request error: {e!s}")

        fal_images = result.get("images", [])
        if not fal_images:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): no images in API response.")

        # Download and decode all result images
        tensors = []
        for idx, fal_img in enumerate(fal_images):
            img_url = fal_img.get("url", "")
            if not img_url:
                logging.warning("Seedream 4.5 Edit (fal.ai): no URL for image %d, skipping.", idx)
                continue
            try:
                img_resp = requests.get(img_url, timeout=60)
                img_resp.raise_for_status()
                raw = img_resp.content
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Seedream 4.5 Edit (fal.ai): failed to download image {idx}: {e!s}")
            tensor_tuple = _image_bytes_to_tensor(raw, "Seedream 4.5 Edit (fal.ai)")
            tensors.append(tensor_tuple[0])

        if not tensors:
            raise RuntimeError("Seedream 4.5 Edit (fal.ai): no valid images downloaded.")

        import torch
        batched = torch.cat(tensors, dim=0)
        return (batched,)


# ---------- Helpers ----------

def _images_to_parts(images):
    """Convert ComfyUI IMAGE tensor [B, H, W, C] to list of inlineData parts."""
    parts = []
    for i in range(images.shape[0]):
        img_np = (255.0 * images[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})
    return parts


def _apply_gemini_safety(payload, threshold):
    """Apply safety threshold to all harm categories."""
    payload["safetySettings"] = [
        {"category": cat, "threshold": threshold} for cat in _HARM_CATEGORIES
    ]


def _gemini_response_summary(data):
    """Return a human-readable summary of why a Gemini response contained no image."""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "")
            if block_reason:
                return f"Prompt was blocked: {block_reason}"
            return "No candidates returned."
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts") or []
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        if text_parts:
            snippet = " | ".join(text_parts)[:300]
            return f"Model returned text only (finishReason={finish_reason}): {snippet}"
        return f"No image part in response (finishReason={finish_reason})."
    except Exception:
        return ""


def _extract_image_from_gemini_response(data, log_name):
    """Extract the first non-thought inline image from a Gemini generateContent response."""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            logging.warning("%s: response has no candidates. Full response: %s", log_name, data)
            return None
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts") or []
        logging.info("%s: finishReason=%s, part count=%d", log_name, finish_reason, len(parts))
        text_parts = []
        for part in parts:
            if part.get("thought"):
                continue
            if "inlineData" in part:
                b64 = part["inlineData"].get("data")
                if b64:
                    return base64.b64decode(b64)
            if "text" in part:
                text_parts.append(part["text"])
        if text_parts:
            logging.warning(
                "%s: API returned text instead of image (finishReason=%s). Text: %s",
                log_name, finish_reason, " | ".join(text_parts)[:500],
            )
        else:
            logging.warning(
                "%s: no image or text in response. finishReason=%s. Full response: %s",
                log_name, finish_reason, str(data)[:1000],
            )
    except (KeyError, TypeError, ValueError) as e:
        logging.warning("%s parse response: %s — raw: %s", log_name, e, str(data)[:500])
    return None


def _api_error_msg(service, e):
    msg = f"{service} API HTTP error: {e.response.status_code}"
    try:
        body = e.response.json()
        if "error" in body:
            msg += f" — {body.get('error', body)}"
        elif "message" in body:
            msg += f" — {body.get('message', body)}"
    except Exception:
        text = e.response.text
        if text:
            msg += f" — {text[:200]}"
    return msg


def _image_bytes_to_tensor(raw, log_name):
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logging.error("%s: failed to decode image: %s", log_name, e)
        raise RuntimeError(f"Failed to decode image from API: {e}") from e
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]
    return (tensor,)


# ---------- ComfyUI registration ----------

NODE_CLASS_MAPPINGS = {
    "NanoBananaProEditAPINode": NanoBananaProEditAPINode,
    "NanoBanana2EditAPINode": NanoBanana2EditAPINode,
    "SeedreamEditAPINode": SeedreamEditAPINode,
    "SeedreamEditFalAPINode": SeedreamEditFalAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProEditAPINode": "Onyx Nano Banana Pro Edit",
    "NanoBanana2EditAPINode": "Onyx Nano Banana 2 Edit",
    "SeedreamEditAPINode": "Onyx Seedream 4.5 Edit(Wavespeed)",
    "SeedreamEditFalAPINode": "Onyx Seedream 4.5 Edit(fal.ai)",
}
