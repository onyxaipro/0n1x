"""
Onyx Re-pose for Carousel
Generates N pose variations of a single input image, with presets controlling
the degree of freedom -- from a barely-changed re-pose to a whole new location.
"""

import time
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Preset definitions
# ─────────────────────────────────────────────────────────────────────────────
from .onyx_render_profile import ensure_profile_ready
REPOSE_PRESETS = [
    "Subtle",
    "Normal",
    "Heavy",
]

_PRESET_DIRECTIVES = {
    "Subtle": (
        "Picture 1 defines the identity, facial anatomy, facial proportions, skin texture and hair characteristics of the subject.\n"
        "Picture 2 defines the environment, outfit, camera framing, pose, perspective and overall scene composition.\n"
        "Generate one photorealistic image of the exact same person from picture 1 naturally existing inside the scene of picture 2.\n"
        "The subject must preserve the exact facial likeness from picture 1:\n"
        "same eye shape, eye spacing, eyelids, nose shape, lips, jawline, cheek structure, facial proportions, skin tone and hair characteristics.\n"
        "Maintain extremely high identity consistency with picture 1.\n"
        "Zero identity drift.\n"
        "Keep the exact outfit, hairstyle, body proportions, environment and general aesthetic from picture 2.\n"
        "The final image must look like a brand new moment captured in the same place as picture 2, but with a completely different pose and body language.\n"
        "Create a subtle pose variation:\n"
        "the subject must have a slightly different pose, it can be a new arm placement, a new posture and a slightly different camera framing while remaining naturally inside the same environment.\n"
        "The overall body pose do not differs from image 2.\n"
        "Do not recreate or imitate the exact composition of picture 2.\n"
        "The output must feel like another spontaneous photo taken a few moments later in the same location, with potentially a different facial expression.\n"
        "The face must remain fully visible, perfectly recognizable and tack-sharp.\n"
        "Preserve natural facial asymmetry and realistic facial geometry from picture 1.\n"
        "Do not beautify, stylize or alter the identity.\n"
        "Lighting and rendering:\n"
        "The lighting on the face, skin and hair must perfectly match the lighting conditions of picture 2.\n"
        "The face must inherit the same RAW smartphone photo rendering as the rest of the body and scene.\n"
        "No pasted-face effect.\n"
        "No floating face effect.\n"
        "No mismatched skin tones.\n"
        "No incorrect head scale or head placement.\n"
        "Camera/look:\n"
        "Amateur smartphone raw crispness.\n"
        "Shot with a wide-angle lens (24mm equivalent).\n"
        "No portrait mode.\n"
        "Deep focus, entire scene sharp and detailed.\n"
        "Background fully in focus.\n"
        "No bokeh, no shallow depth of field, no motion blur.\n"
        "No haze, no bloom, no glow, no beauty filter, no skin smoothing.\n"
        "Extremely detailed realistic skin texture:\n"
        "visible pores, subtle natural dermal texture, slight natural skin irregularities, realistic sensor grain.\n"
        "Clean skin but not artificially perfect.\n"
        "8K native resolution.\n"
        "Ultra detailed.\n"
        "Photorealistic.\n"
        "Natural RAW photo aesthetic.\n"
        "If anything is uncertain, always prioritize picture 1 for facial identity and picture 2 for environment, outfit and composition logic."
    ),
    "Normal": (
        "Picture 1 defines the identity, facial anatomy, facial proportions, skin texture and hair characteristics of the subject.\n"
        "Picture 2 defines the environment, outfit, camera framing, pose, perspective and overall scene composition.\n"
        "Generate one photorealistic image of the exact same person from picture 1 naturally existing inside the scene of picture 2.\n"
        "The subject must preserve the exact facial likeness from picture 1:\n"
        "same eye shape, eye spacing, eyelids, nose shape, lips, jawline, cheek structure, facial proportions, skin tone and hair characteristics.\n"
        "Maintain extremely high identity consistency with picture 1.\n"
        "Zero identity drift.\n"
        "Keep the exact outfit, hairstyle, body proportions, environment and general aesthetic from picture 2.\n"
        "The final image must look like a brand new moment captured in the same place as picture 2, but with a different pose.\n"
        "Create a pose variation:\n"
        "the subject must have a different pose, it can be a new body positioning, a new arm placement, a new posture and a slightly different camera angle/framing while remaining naturally inside the same environment.\n"
        "Always generate a natural and realistic pose, accorded to instagram vibes.\n"
        "Do not recreate or imitate the exact composition of picture 2.\n"
        "The output must feel like another spontaneous photo taken a few moments later in the same location, with potentially a different facial expression.\n"
        "The face must remain fully visible, perfectly recognizable and tack-sharp.\n"
        "Preserve natural facial asymmetry and realistic facial geometry from picture 1.\n"
        "Do not beautify, stylize or alter the identity.\n"
        "Lighting and rendering:\n"
        "The lighting on the face, skin and hair must perfectly match the lighting conditions of picture 2.\n"
        "The face must inherit the same RAW smartphone photo rendering as the rest of the body and scene.\n"
        "No pasted-face effect.\n"
        "No floating face effect.\n"
        "No mismatched skin tones.\n"
        "No incorrect head scale or head placement.\n"
        "Camera/look:\n"
        "Amateur smartphone raw crispness.\n"
        "Shot with a wide-angle lens (24mm equivalent).\n"
        "No portrait mode.\n"
        "Deep focus, entire scene sharp and detailed.\n"
        "Background fully in focus.\n"
        "No bokeh, no shallow depth of field, no motion blur.\n"
        "No haze, no bloom, no glow, no beauty filter, no skin smoothing.\n"
        "Extremely detailed realistic skin texture:\n"
        "visible pores, subtle natural dermal texture, slight natural skin irregularities, realistic sensor grain.\n"
        "Clean skin but not artificially perfect.\n"
        "8K native resolution.\n"
        "Ultra detailed.\n"
        "Photorealistic.\n"
        "Natural RAW photo aesthetic.\n"
        "If anything is uncertain, always prioritize picture 1 for facial identity and picture 2 for environment, outfit and composition logic."
    ),
    "Heavy": (
        "Picture 1 defines the identity, facial anatomy, facial proportions, skin texture and hair characteristics of the subject.\n"
        "Picture 2 defines the environment, outfit, camera framing, pose, perspective and overall scene composition.\n"
        "Generate one photorealistic image of the exact same person from picture 1 naturally existing inside the scene of picture 2.\n"
        "The subject must preserve the exact facial likeness from picture 1:\n"
        "same eye shape, eye spacing, eyelids, nose shape, lips, jawline, cheek structure, facial proportions, skin tone and hair characteristics.\n"
        "Maintain extremely high identity consistency with picture 1.\n"
        "Zero identity drift.\n"
        "Keep the exact outfit, hairstyle, body proportions, environment and general aesthetic from picture 2.\n"
        "The final image must look like a brand new moment captured in the same place as picture 2, but with a completely different pose and body language.\n"
        "Create a strong pose variation:\n"
        "the subject must have a fully new pose, new body positioning, new arm placement, new posture and different camera angle/framing while remaining naturally inside the same environment.\n"
        "Always generate a natural and realistic pose, accorded to instagram vibes.\n"
        "Do not recreate or imitate the exact composition of picture 2. SHE MUST BE LOCATED SOMEWHERE ELSE IN THE SCENE, with a strong pose and composition variation.\n"
        "The output must feel like another spontaneous photo taken a few moments later in the same location, but somewhere else in this location.\n"
        "The face must remain fully visible, perfectly recognizable and tack-sharp.\n"
        "Preserve natural facial asymmetry and realistic facial geometry from picture 1.\n"
        "Do not beautify, stylize or alter the identity.\n"
        "Lighting and rendering:\n"
        "The lighting on the face, skin and hair must perfectly match the lighting conditions of picture 2.\n"
        "The face must inherit the same RAW smartphone photo rendering as the rest of the body and scene.\n"
        "No pasted-face effect.\n"
        "No floating face effect.\n"
        "No mismatched skin tones.\n"
        "No incorrect head scale or head placement.\n"
        "Camera/look:\n"
        "Amateur smartphone raw crispness.\n"
        "Shot with a wide-angle lens (24mm equivalent).\n"
        "No portrait mode.\n"
        "Deep focus, entire scene sharp and detailed.\n"
        "Background fully in focus.\n"
        "No bokeh, no shallow depth of field, no motion blur.\n"
        "No haze, no bloom, no glow, no beauty filter, no skin smoothing.\n"
        "Extremely detailed realistic skin texture:\n"
        "visible pores, subtle natural dermal texture, slight natural skin irregularities, realistic sensor grain.\n"
        "Clean skin but not artificially perfect.\n"
        "8K native resolution.\n"
        "Ultra detailed.\n"
        "Photorealistic.\n"
        "Natural RAW photo aesthetic.\n"
        "If anything is uncertain, always prioritize picture 1 for facial identity and picture 2 for environment, outfit and composition logic."
    ),
}

# Image models available for this node (image-only, no video)
_IMAGE_MODELS = ["Nano Banana Pro", "Nano Banana 2", "Seedream 5 Pro", "GPT Image 2.0"]

# GPT Image 2.0 constraints, mirrored from nano_banana_aio.py so this node can
# refuse an impossible combination *before* spending API calls on it. Left as
# plain constants rather than imported: importing them would pull the whole AIO
# module in at INPUT_TYPES time, and this node deliberately imports it lazily
# inside run().
_GPT2_PROVIDERS   = ("WAVESPEED", "KIE", "FAL")
_GPT2_MAX_SIZE    = "4K"          # no 8K tier exists for this model
_GPT2_SIZES       = ("1K", "2K", "4K")


# ─────────────────────────────────────────────────────────────────────────────
# Node class
# ─────────────────────────────────────────────────────────────────────────────
class OnyxReposeCarouselNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1":  ("IMAGE", {"tooltip": "Identity reference (single image -- face/person to preserve)."}),
                "provider": (["GOOGLE", "WAVESPEED", "KIE", "FAL", "VERTEX"], {"default": "GOOGLE"}),
                "model":    (_IMAGE_MODELS, {"default": "Nano Banana Pro"}),
                "preset":   (REPOSE_PRESETS, {
                    "default": "Subtle",
                    "tooltip": "Controls how much the model is allowed to deviate from the original.",
                }),
                "prompt":   ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Describe the subject, style, or any specific detail.",
                }),
                "count":    ("INT", {
                    "default": 4, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Number of re-posed images to generate per scene image (image_2 frame).",
                }),
                "retry_count": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of retries per image on failure.",
                }),
                "image_size": (["1K", "2K", "4K", "8K"], {
                    "default": "2K",
                    "tooltip": "Output resolution. 8K available with Nano Banana Pro via WaveSpeed only.\n"
                               "GPT Image 2.0 tops out at 4K — 8K is snapped down automatically.",
                }),
            },
            "optional": {
                "image_2":  ("IMAGE", {"tooltip": "Scene/environment reference. Can be a batch -- count variations will be generated per frame."}),
                "is_selfie": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adds selfie framing instructions to the prompt. Ignored when image_2 is a batch.",
                }),
                "is_mirror_selfie": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adds mirror selfie instructions to the prompt. Ignored when image_2 is a batch.",
                }),
                "gemini_api_key":    ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[GOOGLE] Google AI Studio API key."}),
                "wavespeed_api_key": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[WAVESPEED] WaveSpeed API key."}),
                "kie_api_key":       ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[KIE] Kie.ai API key."}),
                "fal_api_key":       ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "[FAL] Fal.ai API key."}),
                "vertex_json_folder": ("STRING", {"default": "", "multiline": False,
                                       "tooltip": "[VERTEX] Path to service account JSON folder."}),
                "disable_safety_threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable the safety filter.",
                }),
                "fal_safety_tolerance": (["1", "2", "3", "4", "5", "6"], {
                    "default": "4",
                    "tooltip": "[FAL] 1 = strict | 6 = permissive",
                }),
                "gpt2_image_quality": (["high", "medium", "low"], {
                    "default": "high",
                    "tooltip": "[GPT Image 2.0] Generation quality. Ignored by every other model.\n"
                               "Lower tiers are markedly cheaper — worth using while you dial in a "
                               "preset, since this node fires count x scenes requests per run.",
                }),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                     "9:16", "16:9", "21:9"],
                    {"default": "auto",
                     "tooltip": "auto = detected from image_2 (or image_1 if no image_2)."},
                ),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "[GOOGLE / VERTEX] Model creativity.",
                }),
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
        image_1,
        image_2                  = None,
        provider                 = "GOOGLE",
        model                    = "Nano Banana Pro",
        preset                   = "Subtle",
        prompt                   = "",
        count                    = 4,
        retry_count              = 3,
        image_size               = "2K",
        is_selfie                = False,
        is_mirror_selfie         = False,
        gemini_api_key           = "",
        wavespeed_api_key        = "",
        kie_api_key              = "",
        fal_api_key              = "",
        vertex_json_folder       = "",
        disable_safety_threshold = False,
        fal_safety_tolerance     = "4",
        aspect_ratio             = "auto",
        temperature              = 1.0,
        gpt2_image_quality       = "high",
    ):
        ensure_profile_ready()
        from .nano_banana_aio import OnyxNanoBananaAIO

        aio = OnyxNanoBananaAIO()

        # ── GPT Image 2.0 guards ──────────────────────────────────────────────
        # Checked here, up front, rather than left to the AIO node. This node
        # loops scenes x retries, so an unsupported combination would otherwise
        # fail once per attempt — the same error printed a dozen times, after a
        # dozen round trips, with the real cause buried in the middle.
        is_gpt2 = (model == "GPT Image 2.0")
        if is_gpt2:
            if provider not in _GPT2_PROVIDERS:
                raise RuntimeError(
                    f"[Re-pose] GPT Image 2.0 is only available via "
                    f"{', '.join(_GPT2_PROVIDERS)} — provider is set to {provider!r}.\n"
                    f"→ Switch the provider, or pick another model."
                )
            if image_size not in _GPT2_SIZES:
                print(
                    f"⚠️  [Re-pose] GPT Image 2.0 has no {image_size} tier "
                    f"→ using {_GPT2_MAX_SIZE}."
                )
                image_size = _GPT2_MAX_SIZE

        # ── Clamp count ───────────────────────────────────────────────────────
        count = max(1, min(count, 10))

        # ── Scene count ───────────────────────────────────────────────────────
        num_scenes = image_2.shape[0] if image_2 is not None else 1

        # ── Build final prompt ────────────────────────────────────────────────
        directive    = _PRESET_DIRECTIVES.get(preset, "")
        base         = prompt.strip()
        final_prompt = (base + "\n\n" + directive) if base else directive

        if is_selfie:
            final_prompt += (
                "\n\nIt's a selfie, smartphone perspective view, smartphone is not visible, "
                "imperfect framing, slightly angled framing."
            )

        if is_mirror_selfie:
            final_prompt += (
                "\n\nIt's a mirror selfie, the smartphone must be hold next to her face, "
                "the smartphone does not hide her face. We can see the back of the phone, "
                "and we do not see the screen of the phone."
            )

        if model == "Seedream 5 Pro":
            final_prompt += " Realistic anatomy, No extra limbs, avoid shiny skins, matte skin"

        is_vertex   = (provider == "VERTEX")
        retry_delay = 10.0 if is_vertex else 2.0

        print(f"[Re-pose] preset={preset!r} | model={model} | provider={provider} | count={count} | retries={retry_count} | scenes={num_scenes}")
        if is_gpt2:
            print(f"[Re-pose] GPT Image 2.0 quality={gpt2_image_quality} | size={image_size}")
        if is_selfie:
            print("[Re-pose] selfie mode ON")
        if is_mirror_selfie:
            print("[Re-pose] mirror selfie mode ON")

        # ── Helper: extract valid image tensors from a generate_unified result ─
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

        # ── Helper: apply 1% center crop to force pixel mismatch ────────────
        def _tiny_crop(tensor, factor=0.985):
            _, H, W, _ = tensor.shape
            new_H = int(H * factor)
            new_W = int(W * factor)
            oh = (H - new_H) // 2
            ow = (W - new_W) // 2
            return tensor[:, oh:oh + new_H, ow:ow + new_W, :]

        # ── Helper: generate `count` variations for one scene frame ───────────
        def _generate_for_scene(scene_frame):
            """scene_frame: shape [1, H, W, 3] or None"""
            ar = aspect_ratio
            if ar == "auto":
                ref = scene_frame if scene_frame is not None else image_1
                ar = OnyxNanoBananaAIO._detect_aspect_ratio(ref)
                print(f"[Re-pose] Auto AR -> {ar}")

            collected = []
            remaining = count

            for attempt in range(1, retry_count + 1):
                if remaining <= 0:
                    break

                print(f"\n[Re-pose] Attempt {attempt}/{retry_count} -- requesting {remaining} image(s)...")
                try:
                    result = aio.generate_unified(
                        provider                 = provider,
                        prompt                   = final_prompt,
                        negative_prompt          = "",
                        image_size               = image_size,
                        gemini_api_key           = gemini_api_key,
                        wavespeed_api_key        = wavespeed_api_key,
                        kie_api_key              = kie_api_key,
                        fal_api_key              = fal_api_key,
                        vertex_json_folder       = vertex_json_folder,
                        disable_safety_threshold = disable_safety_threshold,
                        model                    = model,
                        batch_size               = remaining,
                        use_search               = False,
                        system_instructions      = None,
                        aspect_ratio             = ar,
                        temperature              = temperature,
                        top_p                    = 0.95,
                        fal_safety_tolerance     = fal_safety_tolerance,
                        fal_enable_web_search    = False,
                        image_1                  = image_1,
                        image_2                  = _tiny_crop(scene_frame) if scene_frame is not None else None,
                        video_mode_enabled       = False,
                        face_swap_enabled        = False,
                        breast_refiner_enabled   = False,
                        low_neck_enabled         = False,
                        face_expression          = "Neutral",
                        gpt2_image_quality       = gpt2_image_quality,
                    )
                    got = _extract_tensors(result)
                    collected.extend(got)
                    remaining -= len(got)
                    print(f"[Re-pose] Got {len(got)} image(s) -- total {len(collected)}/{count} -- remaining {remaining}")
                except Exception as e:
                    print(f"[Re-pose] Attempt {attempt} failed: {e}")

                if remaining > 0 and attempt < retry_count:
                    print(f"[Re-pose] Waiting {retry_delay:.0f}s before retry...")
                    time.sleep(retry_delay)

            return collected

        # ── Iterate over each frame in image_2 batch (image_1 is always ref) ─
        print(f"[Re-pose] image_2 frames: {num_scenes} -- generating {count} variation(s) per scene")

        all_collected = []
        for i in range(num_scenes):
            print(f"\n[Re-pose] === Processing scene {i + 1}/{num_scenes} ===")
            scene = image_2[i:i+1] if image_2 is not None else None
            results = _generate_for_scene(scene)
            all_collected.extend(results)
            print(f"[Re-pose] Scene {i + 1} done -- {len(results)} image(s)")

        # ── Assemble batch ────────────────────────────────────────────────────
        if not all_collected:
            print("[Re-pose] No images generated.")
            return (torch.zeros(1, 64, 64, 3),)

        batch = torch.stack(all_collected, dim=0)
        print(f"\n[Re-pose] Done -- {len(all_collected)}/{num_scenes * count} image(s) generated ({num_scenes} scene(s) x {count} variation(s)).")
        return (batch,)
