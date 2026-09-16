"""
Onyx Social Face Swap
Downloads posts from Instagram, Threads or Pinterest, filters images,
and runs the face swap automation on each one with model fallback chain.
"""

import os
import io
import sys
import json
import base64
import math
import time
import torch
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image, ImageFilter

from ..utils.image_utils import tensor_to_pil, pil_to_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Instaloader auto-install
# ─────────────────────────────────────────────────────────────────────────────

from .onyx_render_profile import ensure_profile_ready
_GALLERY_DL_UPGRADE_MARKER = os.path.join(
    os.path.dirname(__file__), ".gallery_dl_last_upgrade")
_GALLERY_DL_UPGRADE_COOLDOWN = 24 * 3600  # 1 fois par jour max


def _pip_install_gallery_dl(upgrade: bool) -> bool:
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "gallery-dl",
           "--quiet", "--break-system-packages"]
    if upgrade:
        cmd.insert(4, "--upgrade")
    subprocess.check_call(cmd, timeout=120)
    try:
        with open(_GALLERY_DL_UPGRADE_MARKER, "w") as fh:
            fh.write(str(time.time()))
    except OSError:
        pass
    return True


def _ensure_gallery_dl() -> bool:
    """Installe gallery-dl, et le met a jour tout seul.

    Instagram change son site frequemment, et gallery-dl sort de nouvelles
    versions rien que pour reparer ses extracteurs Instagram/Threads/Pinterest.
    L'ancien code n'installait qu'une fois ("if not already present"), donc la
    version se figeait pour toujours des sa premiere installation : le
    telechargement finissait par se casser silencieusement chez tout le monde,
    des le jour ou Instagram changeait quelque chose - sans jamais se reparer.

    On verifie donc, au plus une fois par jour (fichier marqueur horodate),
    s'il existe une mise a jour, et on l'installe.
    """
    try:
        import gallery_dl  # noqa: F401
        installed = True
    except ImportError:
        installed = False

    if not installed:
        print("[gallery-dl] gallery-dl not found — installing...")
        try:
            _pip_install_gallery_dl(upgrade=False)
            print("[gallery-dl] gallery-dl installed successfully.")
            return True
        except Exception as e:
            print(f"[gallery-dl] gallery-dl install failed: {e}")
            return False

    # Deja installe : on ne retente une mise a jour qu'une fois par jour, pour
    # ne pas ralentir chaque run avec un appel reseau a pip/PyPI.
    last = 0.0
    try:
        with open(_GALLERY_DL_UPGRADE_MARKER) as fh:
            last = float(fh.read().strip())
    except (OSError, ValueError):
        pass
    if time.time() - last > _GALLERY_DL_UPGRADE_COOLDOWN:
        try:
            print("[gallery-dl] Checking for a newer version (Instagram/Threads/"
                  "Pinterest extractors break often, this keeps them current)...")
            _pip_install_gallery_dl(upgrade=True)
        except Exception as e:
            print(f"[gallery-dl] Upgrade check failed (using current version): {e}")
    return True


def _force_upgrade_gallery_dl() -> bool:
    """Appelee quand un telechargement a echoue : force une mise a jour hors
    cooldown, au cas ou l'echec vienne justement d'une version perimee."""
    try:
        print("[gallery-dl] Download failed/empty — forcing an upgrade and retrying once...")
        return _pip_install_gallery_dl(upgrade=True)
    except Exception as e:
        print(f"[gallery-dl] Forced upgrade failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Image quality detection  (numpy + PIL only, no extra deps)
# Thresholds are intentionally low — adjust as needed.
# ─────────────────────────────────────────────────────────────────────────────

# Brightness threshold: below this value the image is considered low-light.
# 0.20 = 20 % of maximum brightness — conservative, only very dark images trigger.
_BRIGHTNESS_THRESHOLD = 0.20

# Laplacian variance threshold: below this value the image is considered blurry/noisy.
# 20.0 is conservative — only very soft / heavily noisy images trigger.
_LAPLACIAN_THRESHOLD = 20.0


def _laplacian_variance(gray_2d: np.ndarray) -> float:
    """Approximate Laplacian variance using PIL FIND_EDGES on a 256×256 thumbnail."""
    try:
        h, w = gray_2d.shape
        pil_g = Image.fromarray(np.clip(gray_2d, 0, 255).astype(np.uint8))
        pil_g = pil_g.resize((256, 256), Image.LANCZOS)
        lap   = pil_g.filter(ImageFilter.FIND_EDGES)
        return float(np.var(np.array(lap, dtype=np.float32)))
    except Exception:
        return 999.0  # assume sharp on error


def _detect_image_quality(pil_img: Image.Image):
    """
    Returns (is_low_light: bool, is_degraded: bool).
    is_degraded = low_light AND low_quality (blurry / heavy noise).
    Only is_degraded triggers GPT Image 2 priority routing.
    """
    try:
        arr  = np.array(pil_img.convert("RGB"), dtype=np.float32)
        gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        mean_brightness = float(gray.mean()) / 255.0
        lap_var         = _laplacian_variance(gray)

        is_low_light  = mean_brightness < _BRIGHTNESS_THRESHOLD
        is_low_qual   = lap_var          < _LAPLACIAN_THRESHOLD
        is_degraded   = is_low_light and is_low_qual
        return is_low_light, is_degraded
    except Exception:
        return False, False


# ─────────────────────────────────────────────────────────────────────────────
# Model fallback chain builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_order(
    use_nb_pro:       bool,
    use_nb2:          bool,
    use_seedream5pro: bool,
    use_gpt2:         bool,
    is_degraded:      bool,
    provider:         str,
) -> list:
    """
    Build the ordered list of models to try.
    Normal order  : NB Pro → NB2 → Seedream 5 Pro → GPT2
    Degraded order: GPT2 → NB Pro → NB2 → Seedream 5 Pro
    GPT2 and Seedream 5 Pro are only available on KIE, FAL, WAVESPEED.
    """
    gpt2_ok        = use_gpt2         and provider in ("KIE", "FAL", "WAVESPEED")
    seedream5pro_ok = use_seedream5pro and provider in ("KIE", "FAL", "WAVESPEED")

    if is_degraded and gpt2_ok:
        order = []
        if gpt2_ok:          order.append("GPT Image 2.0")
        if use_nb_pro:       order.append("Nano Banana Pro")
        if use_nb2:          order.append("Nano Banana 2")
        if seedream5pro_ok:  order.append("Seedream 5 Pro")
    else:
        order = []
        if use_nb_pro:       order.append("Nano Banana Pro")
        if use_nb2:          order.append("Nano Banana 2")
        if seedream5pro_ok:  order.append("Seedream 5 Pro")
        if gpt2_ok:          order.append("GPT Image 2.0")

    return order


# ─────────────────────────────────────────────────────────────────────────────
# Skip log (processed.json per account folder)
# ─────────────────────────────────────────────────────────────────────────────

def _load_skip_log(folder: str) -> dict:
    path = os.path.join(folder, "processed.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_skip_log(folder: str, log: dict):
    path = os.path.join(folder, "processed.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Social] Warning: could not save skip log: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Face-swap dispatcher  (extends _run_face_swap_automation to GPT2 / Seedream)
# ─────────────────────────────────────────────────────────────────────────────

def _has_face(pil_img) -> bool:
    """Quick face presence check using YuNet (same model as _mask_face_in_tensor).
    Returns True if at least one face is detected, False otherwise."""
    try:
        import cv2
        import numpy as np
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
            print("⚠️  [FaceCheck] No usable detection model — 0 face will be reported.")
            return False
        detector = cv2.FaceDetectorYN.create(
            model_path, "", (w, h),
            score_threshold=0.6, nms_threshold=0.3, top_k=5000,
        )
        min_face = min(w, h) * 0.04
        _, detections = detector.detect(img_bgr)
        if detections is not None:
            for det in detections:
                fw, fh = int(det[2]), int(det[3])
                if fw >= min_face and fh >= min_face:
                    return True
        return False
    except Exception as e:
        print(f"⚠️  [FaceCheck] Detection failed ({e}) — assuming face present")
        return True  # fail-safe: don't skip if detection errors


_EXPRESSION_MAP = {
    "Auto":         "",
    "Neutral":      "neutral facial expression",
    "Sensual":      "sensual facial expression, intense eyes look",
    "Playful":      "playful facial expression",
    "Subtle smile": "subtle smile",
    "Smile":        "warm smile",
    "Laugh":        "she's laughing",
}


def _do_face_swap(
    aio,
    model_name:             str,
    face_ref_tensor,        # IMAGE tensor  (the reference face)
    target_tensor,          # IMAGE tensor  (the Instagram image)
    provider:               str,
    image_size:             str,
    disable_safety:         bool,
    face_expression:        str,
    gemini_api_key:         str,
    wavespeed_api_key:      str,
    kie_api_key:            str,
    fal_api_key:            str,
    vertex_json_folder:     str,
    vertex_location:        str,
    vertex_json_file_override: str = "",   # single JSON file — bypasses folder scan
    custom_prompt:          str = "",      # appended to face swap prompt
    yunet_score_threshold:  float = 0.7,
    temperature:            float = 1.0,
):
    """
    Route a single face-swap to the right provider method.
    NB Pro / NB2  → _run_face_swap_automation (existing, well-tested).
    GPT2 / Seedream → manual routing using the same masked_image + fs_prompt.
    """
    # ── NB Pro / NB2 use the existing automation (unless Auto expression) ────────
    if model_name in ("Nano Banana Pro", "Nano Banana 2") and face_expression != "Auto":
        return aio._run_face_swap_automation(
            yunet_score_threshold  = yunet_score_threshold,
            provider               = provider,
            model                  = model_name,
            image_size             = image_size,
            aspect_ratio           = "1:1",   # auto-detected inside _run_face_swap_automation
            batch_size             = 1,
            disable_safety         = disable_safety,
            face_expression        = face_expression,
            image_1                = face_ref_tensor,
            image_2                = target_tensor,
            gemini_api_key         = gemini_api_key,
            wavespeed_api_key      = wavespeed_api_key,
            kie_api_key            = kie_api_key,
            fal_api_key            = fal_api_key,
            vertex_json_folder     = vertex_json_folder,
            breast_refiner_enabled = False,
            low_neck_enabled       = False,
            face_swap_custom_prompt = custom_prompt,
        )

    # ── Auto expression: build prompt + masked image manually (no expression) ────
    from .nano_banana_aio import (
        _FACESWAP_PROMPT_B64,
        _SEEDREAM5PRO_FACESWAP_PROMPT_B64,
        _MODEL_MAP,
    )

    # Mask face in target image (same logic as the existing automation)
    masked_target = aio._mask_face_in_tensor(target_tensor)

    # Decode face-swap prompt — Seedream 5 Pro always uses its dedicated prompt
    if model_name == "Seedream 5 Pro":
        fs_prompt = base64.b64decode(_SEEDREAM5PRO_FACESWAP_PROMPT_B64).decode("utf-8")
    else:
        fs_prompt = base64.b64decode(_FACESWAP_PROMPT_B64).decode("utf-8")
    _expr = _EXPRESSION_MAP.get(face_expression, "neutral facial expression")
    if _expr:
        fs_prompt += f"\n\n{_expr}"
    if custom_prompt:
        fs_prompt += f"\n\n{custom_prompt}"

    image_tensors = [face_ref_tensor, masked_target]

    # Auto-detect aspect ratio from target image
    auto_ratio = aio._detect_aspect_ratio(target_tensor)

    # ── NB Pro / NB2 (Auto expression only) ──────────────────────────────────
    if model_name in ("Nano Banana Pro", "Nano Banana 2"):
        is_nb2 = (model_name == "Nano Banana 2")
        if provider == "WAVESPEED":
            return aio._generate_wavespeed(
                fs_prompt, image_tensors, image_size, "jpeg",
                auto_ratio, batch_size=1,
                ws_api_key=wavespeed_api_key, is_nb2=is_nb2,
            )
        elif provider == "KIE":
            return aio._generate_kie(
                fs_prompt, image_tensors, image_size, auto_ratio,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, is_nb2=is_nb2,
            )
        elif provider == "FAL":
            return aio._generate_fal(
                prompt            = fs_prompt,
                image_tensors     = image_tensors,
                image_size        = image_size,
                aspect_ratio      = auto_ratio,
                batch_size        = 1,
                fal_api_key       = fal_api_key,
                is_nb2            = is_nb2,
                safety_tolerance  = "4",
                enable_web_search = False,
            )
        elif provider == "GOOGLE":
            if not gemini_api_key:
                return aio._handle_error("❌ gemini_api_key missing for GOOGLE provider.")
            if is_nb2:
                return aio._generate_nb2_google(
                    prompt=fs_prompt, image_tensors=image_tensors,
                    image_size=image_size, aspect_ratio=auto_ratio,
                    temperature=temperature, g_key=gemini_api_key,
                    batch_size=1, disable_safety=disable_safety,
                )
            return aio._generate_google(
                prompt=fs_prompt, image_tensors=image_tensors,
                image_size=image_size, aspect_ratio=auto_ratio,
                temperature=temperature, top_p=0.95, use_search=False,
                system_instructions=None, batch_size=1,
                g_key=gemini_api_key,
                safety_threshold="OFF" if disable_safety else None,
            )
        elif provider == "VERTEX":
            if not vertex_json_folder and not vertex_json_file_override:
                return aio._handle_error("❌ vertex_json_folder missing for VERTEX provider.")
            from .nano_banana_aio import _load_vertex_json_folder
            vj_files = ([vertex_json_file_override] if vertex_json_file_override
                        else _load_vertex_json_folder(vertex_json_folder))
            if is_nb2:
                return aio._generate_nb2_vertex(
                    prompt=fs_prompt, image_tensors=image_tensors,
                    image_size=image_size, aspect_ratio=auto_ratio,
                    temperature=temperature, vertex_json_files=vj_files,
                    vertex_location=vertex_location, batch_size=1,
                    disable_safety=disable_safety,
                )
            return aio._generate_vertex(
                prompt=fs_prompt, image_tensors=image_tensors,
                image_size=image_size, aspect_ratio=auto_ratio,
                temperature=temperature, top_p=0.95, use_search=False,
                system_instructions=None, batch_size=1,
                vertex_json_files=vj_files, vertex_location=vertex_location,
                model_name=_MODEL_MAP.get(model_name, "gemini-3-pro-image"),
                safety_threshold="BLOCK_NONE" if disable_safety else None,
            )
        else:
            raise ValueError(f"NB Pro/NB2 is not available on provider '{provider}'.")

    # ── GPT Image 2.0 ─────────────────────────────────────────────────────────
    if model_name == "GPT Image 2.0":
        if provider == "KIE":
            return aio._generate_gpt2_kie(
                fs_prompt, image_tensors, auto_ratio, image_size,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, disable_safety=disable_safety,
            )
        elif provider == "FAL":
            return aio._generate_gpt2_fal(
                fs_prompt, image_tensors, auto_ratio, image_size,
                gpt2_quality="high", batch_size=1, fal_api_key=fal_api_key,
            )
        elif provider == "WAVESPEED":
            return aio._generate_gpt2_wavespeed(
                fs_prompt, image_tensors, auto_ratio, image_size,
                batch_size=1, ws_api_key=wavespeed_api_key,
            )
        else:
            raise ValueError(f"GPT Image 2.0 is not available on provider '{provider}'.")

    # ── Seedream 4.5 ──────────────────────────────────────────────────────────
    if model_name == "Seedream 4.5":
        if provider == "KIE":
            return aio._generate_seedream_kie(
                fs_prompt, image_tensors, image_size, auto_ratio,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, disable_safety=disable_safety,
            )
        elif provider == "FAL":
            return aio._generate_seedream_fal(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, fal_api_key=fal_api_key,
                disable_safety=disable_safety,
            )
        elif provider == "WAVESPEED":
            return aio._generate_seedream_wavespeed(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, ws_api_key=wavespeed_api_key,
                disable_safety=disable_safety,
            )
        else:
            raise ValueError(f"Seedream 4.5 is not available on provider '{provider}'.")

    # ── Seedream 5 Pro ────────────────────────────────────────────────────────
    if model_name == "Seedream 5 Pro":
        if provider == "KIE":
            return aio._generate_seedream5pro_kie(
                fs_prompt, image_tensors, image_size, auto_ratio,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=1, disable_safety=disable_safety,
            )
        elif provider == "FAL":
            return aio._generate_seedream5pro_fal(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, fal_api_key=fal_api_key,
                disable_safety=disable_safety,
            )
        elif provider == "WAVESPEED":
            return aio._generate_seedream5pro_wavespeed(
                fs_prompt, image_tensors, image_size, auto_ratio,
                batch_size=1, ws_api_key=wavespeed_api_key,
                disable_safety=disable_safety,
            )
        else:
            raise ValueError(f"Seedream 5 Pro is not available on provider '{provider}'.")

    raise ValueError(f"Unknown model: '{model_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class OnyxInstagramFaceSwapNode:
    """
    Onyx Content Remaker
    Downloads posts from Instagram, Threads or Pinterest, filters images (skips videos),
    and runs the face swap automation on each image.
    Supports a model fallback chain: NB Pro → NB2 → GPT Image 2.
    Low-light + blurry images can be automatically routed to GPT Image 2 first.
    Already processed images are skipped on re-run (resume support).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference": ("IMAGE",),

                "source_url": ("STRING", {
                    "default": "https://www.instagram.com/username/",
                    "tooltip": "Instagram, Threads or Pinterest URL.\n"
                               "• Instagram : https://www.instagram.com/username/\n"
                               "• Threads   : https://www.threads.net/@username/\n"
                               "• Pinterest : https://www.pinterest.com/username/ or /username/board/",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder where swapped images will be saved. "
                               "A sub-folder named after the account will be created.",
                }),
                "max_posts": ("INT", {
                    "default": 50, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Maximum number of image posts to process. 0 = all.",
                }),

                # ── Provider ──────────────────────────────────────────────────
                "provider": (["WAVESPEED", "GOOGLE", "FAL", "KIE", "VERTEX"], {
                    "default": "WAVESPEED",
                    "tooltip": "API provider. GPT Image 2 only works on KIE / FAL / WAVESPEED.",
                }),

                # ── Model toggles ─────────────────────────────────────────────
                "use_nano_banana_pro": ("BOOLEAN", {
                    "default": True,
                    "label_on":  "Nano Banana Pro ✅",
                    "label_off": "Nano Banana Pro ❌",
                }),
                "use_nano_banana_2": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "Nano Banana 2 ✅",
                    "label_off": "Nano Banana 2 ❌",
                }),
                "use_seedream_5_pro": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "Seedream 5 Pro ✅",
                    "label_off": "Seedream 5 Pro ❌",
                    "tooltip": "Available on KIE, FAL, WAVESPEED only.",
                }),
                "use_gpt_image_2": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "GPT Image 2 ✅",
                    "label_off": "GPT Image 2 ❌",
                    "tooltip": "Available on KIE, FAL, WAVESPEED only.",
                }),

                # ── Quality routing ───────────────────────────────────────────
                "auto_quality_routing": ("BOOLEAN", {
                    "default": True,
                    "label_on":  "Quality Routing ON",
                    "label_off": "Quality Routing OFF",
                    "tooltip": "When ON: low-light AND blurry images will use GPT Image 2 first "
                               "(if selected and provider supports it).",
                }),

                # ── Generation settings ───────────────────────────────────────
                "retry_count": ("INT", {
                    "default": 3, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Number of attempts per model before moving to the next one in the fallback chain.",
                }),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
                "face_expression": (
                    ["Auto", "Neutral", "Sensual", "Playful", "Subtle smile", "Smile", "Laugh"],
                    {"default": "Auto"},
                ),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional extra instructions appended to the face swap prompt. "
                        "Examples: 'Preserve heterochromia', 'Keep the freckles', "
                        "'The character has a beauty mark under her left eye'."
                    ),
                }),
                "disable_safety": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "Safety OFF",
                    "label_off": "Safety ON",
                }),

                # ── Instagram auth (optional, fixes 403 blocks) ──────────────────
                "cookies_file": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to a cookies.txt file exported from your browser (Netscape format). "
                        "How to get it: install the 'Get cookies.txt LOCALLY' extension in Chrome, "
                        "go to instagram.com while logged in, click the extension and export. "
                        "This is the safest method and supports 2FA accounts. "
                        "Leave empty to browse anonymously (may cause 403 blocks)."
                    ),
                }),

                # ── API keys ──────────────────────────────────────────────────
                "gemini_api_key":      ("STRING", {"default": ""}),
                "wavespeed_api_key":   ("STRING", {"default": ""}),
                "fal_api_key":         ("STRING", {"default": ""}),
                "kie_api_key":         ("STRING", {"default": ""}),
                "vertex_json_folder":  ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("last_image", "summary")
    OUTPUT_NODE   = True
    FUNCTION      = "run"
    CATEGORY      = "Onyx/NanoBanana"
    DESCRIPTION   = (
        "Onyx Content Remaker\n"
        "Downloads posts from Instagram, Threads or Pinterest and runs the face swap automation "
        "on every image.\n"
        "Fallback chain: Nano Banana Pro → Nano Banana 2 → Seedream 5 Pro → GPT Image 2.\n"
        "Low-light + blurry images can be routed to GPT Image 2 first.\n"
        "Already processed images are skipped on re-run (resume support)."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        face_reference,
        source_url,
        output_folder,
        max_posts,
        provider,
        use_nano_banana_pro,
        use_nano_banana_2,
        use_seedream_5_pro,
        use_gpt_image_2,
        auto_quality_routing,
        retry_count,
        image_size,
        face_expression,
        disable_safety,
        gemini_api_key,
        wavespeed_api_key,
        kie_api_key,
        fal_api_key,
        vertex_json_folder,
        cookies_file,
        custom_prompt,
    ):
        ensure_profile_ready()
        from .nano_banana_aio import OnyxNanoBananaAIO

        # ── Guard: instaloader ────────────────────────────────────────────────
        if not _ensure_gallery_dl():
            msg = "❌ gallery-dl could not be installed. Run: pip install gallery-dl"
            print(f"[{platform}] {msg}")
            return (torch.zeros(1, 64, 64, 3), msg)

        import subprocess

        # ── Guard: at least one model selected ───────────────────────────────
        if not any([use_nano_banana_pro, use_nano_banana_2,
                    use_seedream_5_pro, use_gpt_image_2]):
            msg = "❌ No model selected. Enable at least one model."
            return (torch.zeros(1, 64, 64, 3), msg)

        # ── Detect platform & parse username ─────────────────────────────────
        _url = source_url.strip().rstrip("/")
        _url_lower = _url.lower()
        if "threads.net" in _url_lower:
            platform = "Threads"
            platform_icon = "🧵"
        elif "pinterest.com" in _url_lower:
            platform = "Pinterest"
            platform_icon = "📌"
        else:
            platform = "Instagram"
            platform_icon = "📸"

        username = _url.split("/")[-1].replace("@", "")
        if not username:
            return (torch.zeros(1, 64, 64, 3), f"❌ Invalid {platform} URL / username.")

        print(f"\n{'═'*60}")
        print(f"{platform_icon}  ONYX {platform.upper()} FACE SWAP")
        print(f"    Account   : @{username}")
        print(f"    Provider  : {provider}")
        print(f"    Max posts : {max_posts if max_posts > 0 else 'all'}")
        models_on = [
            m for m, on in [
                ("NB Pro",  use_nano_banana_pro),
                ("NB2",     use_nano_banana_2),
                ("SD5Pro",  use_seedream_5_pro),
                ("GPT2",    use_gpt_image_2),
                ] if on
        ]
        print(f"    Models    : {' → '.join(models_on)}")
        print(f"    Quality routing: {'ON' if auto_quality_routing else 'OFF'}")
        print(f"{'═'*60}\n")

        # ── Prepare output folder ─────────────────────────────────────────────
        if not output_folder or not output_folder.strip():
            output_folder = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "instagram_output",
            )
        account_folder = os.path.join(output_folder, username)
        os.makedirs(account_folder, exist_ok=True)

        # ── Load skip log ─────────────────────────────────────────────────────
        skip_log = _load_skip_log(account_folder)
        print(f"[{platform}] Output folder : {account_folder}")
        print(f"[{platform}] Already processed: {len(skip_log)} post(s)")

        # ── Download images via gallery-dl ───────────────────────────────────
        raw_folder = os.path.join(account_folder, "_raw")
        os.makedirs(raw_folder, exist_ok=True)

        cmd = [sys.executable, "-m", "gallery_dl", "--dest", raw_folder, "--no-mtime"]

        _cfile = cookies_file.strip()
        if _cfile and os.path.isfile(_cfile):
            cmd += ["--cookies", _cfile]
            print(f"[{platform}] ✅ Using cookies from {_cfile!r}")
        else:
            print(f"[{platform}] ℹ️  No cookies — browsing anonymously (may hit 403)")

        if max_posts > 0:
            cmd += ["--range", f"1-{max_posts}"]

        # Build the source URL for gallery-dl
        if platform == "Threads":
            dl_url = f"https://www.threads.net/@{username}/"
        elif platform == "Pinterest":
            # Use URL as-is for Pinterest (supports /user/ and /user/board/)
            dl_url = _url if _url.startswith("http") else f"https://www.pinterest.com/{username}/"
        else:
            dl_url = f"https://www.instagram.com/{username}/"
        cmd.append(dl_url)

        print(f"[{platform}] Downloading posts from @{username} via gallery-dl...")
        try:
            dl_result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if dl_result.stdout:
                for line in dl_result.stdout.strip().splitlines()[-5:]:
                    print(f"   {line}")
            if dl_result.returncode not in (0, 1):
                print(f"[{platform}] ⚠️  gallery-dl exited with code {dl_result.returncode}")
                if dl_result.stderr:
                    print(f"   {dl_result.stderr.strip()[:300]}")
        except subprocess.TimeoutExpired:
            print(f"[{platform}] ⚠️  gallery-dl timed out after 10 min — processing partial results")
        except Exception as e:
            msg = f"❌ gallery-dl failed: {e}"
            print(f"[{platform}] {msg}")
            return (torch.zeros(1, 64, 64, 3), msg)

        # ── Collect downloaded image files ────────────────────────────────────
        _img_exts = {".jpg", ".jpeg", ".png", ".webp"}

        def _scan():
            return sorted([
                os.path.join(root, f)
                for root, _, files in os.walk(raw_folder)
                for f in files
                if os.path.splitext(f)[1].lower() in _img_exts
            ])

        image_files = _scan()
        # 0 fichier alors que gallery-dl n'a pas signale d'erreur (code 0/1) est
        # le symptome typique d'un extracteur perime par un changement du site :
        # on force une mise a jour et on retente une fois avant d'abandonner.
        if not image_files:
            print(f"[{platform}] ⚠️  0 image downloaded — retrying once after a forced "
                  f"gallery-dl upgrade (a stale extractor is the usual cause)...")
            if _force_upgrade_gallery_dl():
                try:
                    dl_result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if dl_result.stdout:
                        for line in dl_result.stdout.strip().splitlines()[-5:]:
                            print(f"   {line}")
                    image_files = _scan()
                except Exception as e:
                    print(f"[{platform}] Retry after upgrade failed: {e}")

        if max_posts > 0 and len(image_files) > max_posts:
            image_files = image_files[:max_posts]
            print(f"[{platform}] {len(image_files)} image file(s) kept (max_posts={max_posts} cap applied).")
        else:
            print(f"[{platform}] {len(image_files)} image file(s) found.")
        total = len(image_files)

        if total == 0:
            msg = f"❌ No images downloaded for @{username}. Check your cookies file or try again later."
            return (torch.zeros(1, 64, 64, 3), msg)

        # ── Instantiate OnyxNanoBananaAIO (reuse all face-swap methods) ───────────
        aio = OnyxNanoBananaAIO()

        # Shared kwargs for _do_face_swap
        swap_kwargs = dict(
            provider               = provider,
            image_size             = image_size,
            disable_safety         = disable_safety,
            face_expression        = face_expression,
            custom_prompt          = custom_prompt.strip(),
            gemini_api_key         = gemini_api_key,
            wavespeed_api_key      = wavespeed_api_key,
            kie_api_key            = kie_api_key,
            fal_api_key            = fal_api_key,
            vertex_json_folder     = vertex_json_folder,
            vertex_location        = "us-central1",
        )

        # ── Main loop ─────────────────────────────────────────────────────────
        _AUTO_BATCH_SIZE = 10   # parallel workers for non-Vertex providers

        processed_count = 0
        skipped_count   = 0
        no_face_count   = 0
        failed_count    = 0
        last_tensor     = torch.zeros(1, 64, 64, 3)
        summary_lines   = []
        _lock           = threading.Lock()

        # ── Per-image worker (closure captures all needed variables) ──────────
        def _process_one(idx, img_path, vj_file_override=""):
            shortcode = os.path.splitext(os.path.basename(img_path))[0]
            print(f"\n[{idx:>4}/{total}] @{username}/{shortcode}")

            # Load
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"   ❌ [{idx}/{total}] Load failed: {e}")
                return (shortcode, None, None, "load_failed", str(e))

            target_tensor = pil_to_tensor(pil_img)

            # ── Face presence check — skip non-person images ──────────────────
            if not _has_face(pil_img):
                print(f"   🚫 [{idx}/{total}] No face detected — skipping (landscape/object photo).")
                return (shortcode, None, None, "no_face", None)

            # Quality detection
            is_low_light = False
            is_degraded  = False
            if auto_quality_routing:
                is_low_light, is_degraded = _detect_image_quality(pil_img)
                if is_degraded:
                    print(f"   \U0001f311 [{idx}/{total}] Low-light + blurry → GPT Image 2 priority")
                elif is_low_light:
                    print(f"   \U0001f319 [{idx}/{total}] Low-light but sharp → normal chain")

            # Model order
            model_order = _build_model_order(
                use_nb_pro       = use_nano_banana_pro,
                use_nb2          = use_nano_banana_2,
                use_seedream5pro = use_seedream_5_pro,
                use_gpt2         = use_gpt_image_2,
                is_degraded      = is_degraded,
                provider         = provider,
            )
            print(f"   [{idx}/{total}] Chain: {' → '.join(model_order)}")

            # Try each model with retries
            result_tensor = None
            used_model    = None
            for model_name in model_order:
                for attempt in range(1, retry_count + 1):
                    print(f"   \U0001f680 [{idx}/{total}] {model_name} (attempt {attempt}/{retry_count})...")
                    try:
                        result  = _do_face_swap(
                            aio, model_name,
                            face_reference, target_tensor,
                            **swap_kwargs,
                            vertex_json_file_override=vj_file_override,
                        )
                        img_out = result[0]
                        if img_out.shape[1] > 64 and img_out.shape[2] > 64:
                            result_tensor = img_out
                            used_model    = model_name
                            print(f"   ✅ [{idx}/{total}] {model_name} — success! (attempt {attempt})")
                            break
                        else:
                            print(f"   ⚠️  [{idx}/{total}] {model_name} returned error placeholder "
                                  f"(attempt {attempt}/{retry_count}).")
                    except Exception as e:
                        print(f"   ❌ [{idx}/{total}] {model_name} attempt {attempt}/{retry_count} failed: {e}")
                    # Délai entre retries uniquement pour Vertex (évite le 429)
                    if result_tensor is None and attempt < retry_count and provider == "VERTEX":
                        print(f"   ⏳ [{idx}/{total}] Vertex retry delay 10s...")
                        time.sleep(10)
                if result_tensor is not None:
                    break
                print(f"   ↩️  [{idx}/{total}] {model_name} exhausted — next model...")

            return (shortcode, result_tensor, used_model,
                    "done" if result_tensor is not None else "all_failed", None)

        # ── Result handler (called after each completed image) ────────────────
        def _commit_result(idx, img_path, shortcode, result_tensor, used_model, status, error):
            nonlocal processed_count, failed_count, no_face_count, skipped_count, last_tensor

            if status == "no_face":
                no_face_count += 1
                with _lock:
                    skip_log[shortcode] = {"status": "no_face"}
                    _save_skip_log(account_folder, skip_log)
                return

            if status == "load_failed":
                failed_count += 1
                with _lock:
                    skip_log[shortcode] = {"status": "load_failed", "error": error}
                    _save_skip_log(account_folder, skip_log)
                return

            if result_tensor is None:
                print(f"   ❌ All models failed for {shortcode}.")
                failed_count += 1
                with _lock:
                    skip_log[shortcode] = {"status": "all_failed"}
                    _save_skip_log(account_folder, skip_log)
                return

            # Save to disk
            save_name = f"{idx:04d}_{shortcode}.png"
            save_path = os.path.join(account_folder, save_name)
            try:
                result_pil = Image.fromarray(
                    (result_tensor[0].cpu().numpy() * 255)
                    .clip(0, 255).astype(np.uint8)
                )
                result_pil.save(save_path, "PNG", optimize=True)
                print(f"   \U0001f4be Saved → {save_path}")
            except Exception as e:
                print(f"   ⚠️  Save failed: {e}")

            last_tensor     = result_tensor
            processed_count += 1
            with _lock:
                skip_log[shortcode] = {"status": "done", "model": used_model, "file": save_name}
                _save_skip_log(account_folder, skip_log)
            summary_lines.append(f"  ✅ {shortcode} → {used_model}")

        # ── Separate already-done from pending ────────────────────────────────
        pending_imgs = []
        for idx, img_path in enumerate(image_files, 1):
            shortcode = os.path.splitext(os.path.basename(img_path))[0]
            if shortcode in skip_log and skip_log[shortcode].get("status") == "done":
                skipped_count += 1
                print(f"⏭️   [{idx:>4}/{total}] {shortcode} — already processed, skipping.")
            else:
                pending_imgs.append((idx, img_path))

        # ── Pre-load Vertex JSON files for round-robin distribution ──────────
        _vj_files = []
        if provider == "VERTEX" and vertex_json_folder:
            from .nano_banana_aio import _load_vertex_json_folder
            _vj_files = _load_vertex_json_folder(vertex_json_folder)
            print(f"[{platform}] 🔑 {len(_vj_files)} Vertex project(s) found.")

        # ── Decide batch size ─────────────────────────────────────────────────
        if provider == "VERTEX":
            _batch_workers = len(_vj_files) if _vj_files else 1
        else:
            _batch_workers = _AUTO_BATCH_SIZE

        use_parallel = len(pending_imgs) > 1 and _batch_workers > 1
        if use_parallel:
            print(f"\n[{platform}] ⚡ Pool mode: {len(pending_imgs)} images — "
                  f"{_batch_workers} workers in parallel (provider={provider})")
        else:
            print(f"\n[{platform}] 🔁 Sequential mode: {len(pending_imgs)} image(s) to process.")

        if use_parallel:
            # Submit ALL images at once — executor keeps _batch_workers busy at all times.
            # As soon as a worker finishes, it immediately picks up the next image.
            # Vertex: round-robin across JSON files so each project stays in use.
            with ThreadPoolExecutor(max_workers=_batch_workers) as executor:
                future_map = {}
                for i, (idx, ip) in enumerate(pending_imgs):
                    vj_override = _vj_files[i % len(_vj_files)] if _vj_files else ""
                    fut = executor.submit(_process_one, idx, ip, vj_override)
                    future_map[fut] = (idx, ip)
                done_count = 0
                for future in as_completed(future_map):
                    idx, img_path = future_map[future]
                    done_count += 1
                    remaining = len(pending_imgs) - done_count
                    try:
                        sc, rt, um, status, err = future.result()
                        _commit_result(idx, img_path, sc, rt, um, status, err)
                    except Exception as e:
                        sc = os.path.splitext(os.path.basename(img_path))[0]
                        print(f"   ❌ Unexpected thread error [{idx}/{total}] {sc}: {e}")
                        failed_count += 1
                    print(f"[{platform}] ⏳ {remaining} image(s) remaining in queue...")
        else:
            # Sequential (1 image or 1 worker)
            for idx, img_path in pending_imgs:
                vj_override = _vj_files[0] if _vj_files else ""
                sc, rt, um, status, err = _process_one(idx, img_path, vj_override)
                _commit_result(idx, img_path, sc, rt, um, status, err)

        # ── Final summary ─────────────────────────────────────────────────────────────────────────
        summary = (
            f"{platform_icon}  {platform} Face Swap — @{username}\n"
            f"   Processed : {processed_count}\n"
            f"   Skipped   : {skipped_count} (already done)\n"
            f"   No face   : {no_face_count} (landscape/object)\n"
            f"   Failed    : {failed_count}\n"
            f"   Output    : {account_folder}\n"
        )
        if summary_lines:
            summary += "\nLast processed:\n" + "\n".join(summary_lines[-20:])

        print(f"\n{'=' * 60}")
        print(summary)
        print(f"{'=' * 60}\n")

        return (last_tensor, summary)


# ─────────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "OnyxInstagramFaceSwapNode": OnyxInstagramFaceSwapNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxInstagramFaceSwapNode": "Onyx Content Remaker",
}
