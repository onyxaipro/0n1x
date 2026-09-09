"""
ComfyUI node — Onyx Lora Caption Generator.
Captions every image in a folder through Gemini (API key), Vertex (service
account) or Grok, one request per image.
By default the images are left untouched and each caption is written to
<image name>.txt beside it. Switch keep_original_names off to renumber the
folder to <trigger>_0001.ext with a matching <trigger>_0001.txt.
"""

import io
import os
import json
import base64
import logging

import requests
from PIL import Image
import folder_paths


_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _list_input_subdirs():
    """List subdirectories inside ComfyUI's input folder."""
    input_dir = folder_paths.get_input_directory()
    dirs = []
    for f in os.listdir(input_dir):
        full = os.path.join(input_dir, f)
        if os.path.isdir(full):
            dirs.append(f)
    dirs.sort(key=str.lower)
    if not dirs:
        dirs = ["(no folders found)"]
    return dirs


_PROVIDERS = ["gemini", "vertex", "grok"]

# Meme liste que gemini_prompt.py, pour que les deux nodes du pack proposent les
# memes modeles. L'ancienne s'arretait a la 2.0, qui ne repond plus.
_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# temperature et top_p sont ignores a partir de ces modeles : les envoyer ne
# provoque pas d'erreur aujourd'hui mais ne fait rien. Meme constat que dans
# gemini_prompt.py.
_GEMINI_NO_SAMPLING = {"gemini-3.6-flash", "gemini-3.5-flash-lite"}

# "global", pas une region. Les modeles Gemini recents sont publies sur le point
# d'entree global de Vertex et sont absents des endpoints regionaux : demander
# us-central1 renvoie un 404 qui se lit comme un probleme de droits alors que
# c'est un probleme de routage.
_VERTEX_LOCATION = "global"

_GROK_MODELS = [
    "grok-4-1-fast-non-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4.20-0309-reasoning",
]

_DEFAULT_CAPTION_INSTRUCTION = (
    "Describe this image in detail for AI image training. "
    "Include the subject's appearance, clothing, pose, expression, hair, "
    "background environment, lighting, colors, camera angle, and overall mood. "
    "Output ONLY the caption as a single paragraph. No bullet points, no headers, no labels. "
    "Do not start with 'This image shows' or 'The image depicts'. "
    "Just describe what is in the image directly."
)

_SDXL_CAPTION_INSTRUCTION = """You are generating a single caption for an SDXL LoRA training image of a recurring subject. Another mechanism prepends the subject's trigger word to your output, so DO NOT include any name, "she", "the woman", "the person", or any subject reference at the start.

WHAT TO EXCLUDE (never describe these):
- Face, facial features, expression as identity (a smile or frown is OK to mention; eye color, face shape, nose, lips are NOT)
- Hair (color, length, style, texture)
- Skin tone, ethnicity, age
- Body proportions, build, weight
- Subjective quality words ("beautiful", "stunning", "gorgeous", "perfect", "masterpiece", "amazing"). Stick to objective neutral description.

WHAT TO INCLUDE (always, when visible):
- Clothing: every garment, with material, fit, color, pattern, neckline, straps, fabric finish, waistband, closures, prints. Be technical and precise.
- Pose, body position, what the visible limbs are doing
- Setting: location, surfaces, notable furniture or objects, foreground and background elements
- Lighting: source, direction, quality (hard/soft/diffused), shadow behavior, color temperature
- Camera framing: angle (eye-level / low-angle / high-angle / overhead) and shot type (close-up, medium shot, full-body, three-quarter, side-profile, selfie)

FORMAT RULES:
- Output starts directly with "wearing ..." — or with "headless, ..." if the face/head is cropped, occluded, or not visible (then describe what IS visible).
- 2 to 4 short sentences. End with a single shot-type phrase like "Eye-level medium shot." or "High-angle selfie."
- No quotes, no labels, no preamble, no markdown, no explanation.
- Output ONLY the caption text.

EXAMPLES OF VALID OUTPUTS:
wearing a black microfiber spaghetti-strap sports bra with a scoop neckline and a wide underbust band, paired with high-waisted black athletic leggings featuring a seamless waistband. The scene is a sun-drenched public park with a grey paved walking path and dense green deciduous trees in the background. Bright, direct natural light from the front-right creates soft, well-defined shadows. Eye-level medium shot.

wearing a hot pink ribbed knit sports bra with thin double-cord straps and light-wash distressed denim shorts. Seated on a weathered brown wooden deck with horizontal slats. Intense direct side-angle sunlight casts sharp diagonal shadows across the deck. High-angle overhead selfie."""

_CAPTION_MODES = ["non sdxl", "sdxl", "sdxl-caption-json"]


def rename_folder(directory, trigger):
    """Renumber every image in `directory` to <trigger>_01.ext, _02, _03...

    Two rules make this safe to run on a folder that has already been captioned:

    - The .txt that sits beside an image follows it. Splitting an image from its
      caption is silent — nothing errors, the dataset is just wrong at training
      time — so the pair moves as one unit or not at all.
    - Everything goes through temporary names first. A folder that already
      contains <trigger>_01.jpg would otherwise see that file overwritten as the
      target of another rename, and the loss is unrecoverable.
    """
    trigger = (trigger or "").strip()
    if not trigger:
        raise ValueError("trigger_word is empty — nothing to rename the files to.")

    images = sorted((f for f in os.listdir(directory)
                     if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTENSIONS),
                    key=str.lower)
    if not images:
        raise FileNotFoundError(f"no image found in {directory}")

    # 01..99 comme demande, mais un dossier de 100 images et plus repasserait a
    # _100 apres _99 et se trierait dans le desordre : la largeur suit le lot.
    width = max(2, len(str(len(images))))

    staged = []
    for i, fname in enumerate(images, start=1):
        base, ext = os.path.splitext(fname)
        src_txt = os.path.join(directory, base + ".txt")
        has_txt = os.path.isfile(src_txt)

        tmp_img = os.path.join(directory, f"__ofm_rn_{i:06d}{ext.lower()}")
        os.rename(os.path.join(directory, fname), tmp_img)
        tmp_txt = None
        if has_txt:
            tmp_txt = os.path.join(directory, f"__ofm_rn_{i:06d}.txt")
            os.rename(src_txt, tmp_txt)
        staged.append((tmp_img, tmp_txt, ext.lower()))

    renamed = 0
    captions = 0
    for i, (tmp_img, tmp_txt, ext) in enumerate(staged, start=1):
        final_base = f"{trigger}_{i:0{width}d}"
        os.rename(tmp_img, os.path.join(directory, final_base + ext))
        renamed += 1
        if tmp_txt:
            os.rename(tmp_txt, os.path.join(directory, final_base + ".txt"))
            captions += 1

    logging.info("Lora Caption Generator: renamed %d image(s) and %d caption(s) in %s",
                 renamed, captions, directory)
    return renamed, captions


# ---------- Serve prompts to frontend (so the JS can populate the textbox) ----------

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/lora_caption/get_prompts")
    async def _lora_caption_get_prompts(request):
        return web.json_response({
            "non sdxl":            _DEFAULT_CAPTION_INSTRUCTION,
            "sdxl":                _SDXL_CAPTION_INSTRUCTION,
            "sdxl-caption-json": "",
        })

    @PromptServer.instance.routes.post("/lora_caption/rename_files")
    async def _lora_caption_rename_files(request):
        """Renumber a folder on demand, from the node's button.

        Not part of process_folder: renaming is destructive and must happen when
        the user asks for it, not as a side effect of running the graph.
        """
        try:
            data = await request.json()
            folder = (data.get("folder") or "").strip()
            trigger = (data.get("trigger") or "").strip()

            if not folder or folder == "(no folders found)":
                return web.json_response({"error": "no input folder selected"}, status=400)
            if not trigger:
                return web.json_response({"error": "trigger_word is empty"}, status=400)

            root = os.path.realpath(folder_paths.get_input_directory())
            directory = os.path.realpath(os.path.join(root, folder))
            # La route est joignable par n'importe qui sur le reseau : sans ce
            # test, un "../.." renommerait des fichiers hors de ComfyUI/input.
            if directory != root and not directory.startswith(root + os.sep):
                return web.json_response({"error": "folder outside ComfyUI/input"}, status=400)
            if not os.path.isdir(directory):
                return web.json_response({"error": f"not a directory: {folder}"}, status=400)

            renamed, captions = rename_folder(directory, trigger)
            return web.json_response({"renamed": renamed, "captions": captions,
                                      "folder": folder})
        except Exception as e:
            logging.exception("Lora Caption Generator: rename failed")
            return web.json_response({"error": str(e)}, status=500)
except Exception:
    logging.warning("Lora Caption Generator: could not register prompt route (server not available)")


def _image_file_to_b64(filepath):
    """Read an image file from disk and return base64 PNG string."""
    pil = Image.open(filepath).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _list_vertex_json(folder_path):
    """Sorted list of service-account .json files in a folder."""
    folder_path = (folder_path or "").strip()
    if not folder_path:
        raise ValueError("Lora Caption Generator: vertex_json_folder is empty.")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Lora Caption Generator: folder not found: {folder_path}")
    files = sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(".json"))
    if not files:
        raise FileNotFoundError(f"Lora Caption Generator: no .json in {folder_path}")
    return files


def _call_vertex(json_path, model, image_b64, instruction, temperature, max_tokens):
    """Same request as _call_gemini, through Vertex with a service account."""
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account

    with open(json_path, "r", encoding="utf-8") as fh:
        project_id = json.load(fh).get("project_id", "")
    credentials = service_account.Credentials.from_service_account_file(
        json_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    client = genai.Client(vertexai=True, project=project_id,
                          location=_VERTEX_LOCATION, credentials=credentials)

    cfg = {"max_output_tokens": max_tokens}
    if model not in _GEMINI_NO_SAMPLING:
        cfg["temperature"] = temperature
    try:
        # OFF plutot que BLOCK_NONE : c'est le seuil le plus permissif accepte
        # par les modeles recents, et il correspond a ce qu'envoie deja le
        # chemin Gemini direct de ce meme node.
        cfg["safety_settings"] = [
            types.SafetySetting(category=c, threshold="OFF")
            for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ]
    except Exception as e:
        logging.warning("Lora Caption Generator (Vertex): safety settings not applied (%s)", e)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/png"),
            instruction,
        ],
        config=types.GenerateContentConfig(**cfg),
    )
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Vertex: empty response.")
    return text


def _call_gemini(api_key, model, image_b64, instruction, temperature, max_tokens):
    """Call Gemini API with image and return text response."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    gen_cfg = {"maxOutputTokens": max_tokens}
    if model not in _GEMINI_NO_SAMPLING:
        gen_cfg["temperature"] = temperature

    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                {"text": instruction},
            ]
        }],
        "generationConfig": gen_cfg,
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        ],
    }

    resp = requests.post(url, json=payload,
                         headers={"Content-Type": "application/json"},
                         timeout=120)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        block = data.get("promptFeedback", {}).get("blockReason", "")
        if block:
            raise RuntimeError(f"Gemini blocked prompt: {block}")
        raise RuntimeError("Gemini: no candidates in response.")

    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
    if not texts:
        raise RuntimeError("Gemini: no text in response.")
    return "\n".join(texts).strip()


def _call_grok(api_key, model, image_b64, instruction, temperature, max_tokens):
    """Call Grok vision API (xAI, OpenAI-compatible) with image and return text response."""
    url = "https://api.x.ai/v1/chat/completions"

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Grok: no choices in response.")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Grok: empty response content.")
    return content.strip()


class OnyxLoraCaptionGeneratorNode:
    """Generate LoRA training captions for all images in a folder.

    Images are sent one at a time. With keep_original_names on (default) nothing
    is renamed and each caption lands in <image name>.txt beside its image; with
    it off, images are renumbered to <trigger>_0001.ext with a matching .txt.
    """

    _cached_api_key = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": (_list_input_subdirs(), {"default": _list_input_subdirs()[0]}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "provider": (_PROVIDERS, {"default": "gemini"}),
                "trigger_word": ("STRING", {"default": "", "multiline": False}),
                "mode": (_CAPTION_MODES, {
                    "default": "non sdxl",
                    "tooltip": (
                        "sdxl: caption style for SDXL person LoRA training "
                        "(no face/hair/identity, focus on outfit/scene/lighting/framing). "
                        "non sdxl: generic detailed caption (default)."
                    ),
                }),
            },
            "optional": {
                "gemini_model": (_GEMINI_MODELS, {"default": "gemini-3.6-flash"}),
                "vertex_json_folder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Folder of service-account .json files, for provider=vertex. "
                               "The first one is used; api_key is ignored in that mode."}),
                "keep_original_names": ("BOOLEAN", {
                    "default": True,
                    "label_on": "keep original filenames",
                    "label_off": "rename to <trigger>_0001",
                    "tooltip": "On: images are left untouched and each caption is written as "
                               "<image name>.txt beside it.\n"
                               "Off: the historical behaviour — images are renamed to "
                               "<trigger>_0001.ext with a matching .txt."}),
                "grok_model": (_GROK_MODELS, {"default": "grok-4-1-fast-non-reasoning"}),
                "caption_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional override. If empty, mode determines the prompt.",
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "process_folder"
    OUTPUT_NODE = True
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def process_folder(self, input_folder, api_key, provider, trigger_word,
                       mode="non sdxl",
                       gemini_model="gemini-3.6-flash",
                       vertex_json_folder="", keep_original_names=True,
                       grok_model="grok-4-1-fast-non-reasoning",
                       caption_instruction="", temperature=0.7,
                       max_tokens=8192):
        if not input_folder or input_folder == "(no folders found)":
            raise RuntimeError("Lora Caption Generator: no input folder selected. Create a subfolder in ComfyUI/input/.")

        directory = os.path.join(folder_paths.get_input_directory(), input_folder)
        if not os.path.isdir(directory):
            raise RuntimeError(f"Lora Caption Generator: '{directory}' is not a valid directory.")

        # sdxl-caption-json: skip all API calls and renames, just pack captions.json
        # from existing <image>/<image_basename>.txt pairs already in the folder.
        if mode == "sdxl-caption-json":
            logging.info("Lora Caption Generator: mode=sdxl-caption-json — packing captions.json from existing pairs in %s", directory)
            self._write_captions_json(directory)
            return {}

        vertex_files = []
        if provider == "vertex":
            # Vertex s'authentifie par compte de service : la cle API n'a pas
            # cours ici, et l'exiger bloquerait un mode qui n'en a pas besoin.
            vertex_files = _list_vertex_json(vertex_json_folder)
            logging.info("Lora Caption Generator: Vertex, using %s",
                         os.path.basename(vertex_files[0]))
        else:
            if api_key.strip():
                OnyxLoraCaptionGeneratorNode._cached_api_key = api_key.strip()
            api_key = api_key.strip() or OnyxLoraCaptionGeneratorNode._cached_api_key
            if not api_key:
                raise RuntimeError("Lora Caption Generator: API key is required.")

        trigger = trigger_word.strip()
        if not trigger and not keep_original_names:
            raise RuntimeError(
                "Lora Caption Generator: trigger_word is required when renaming files.\n"
                "-> Fill it in, or switch keep_original_names on."
            )

        # Collect all image files
        image_files = []
        for f in os.listdir(directory):
            ext = os.path.splitext(f)[1].lower()
            if ext in _SUPPORTED_EXTENSIONS:
                image_files.append(f)

        if not image_files:
            raise RuntimeError(f"Lora Caption Generator: no images found in '{directory}'.")

        image_files.sort(key=str.lower)
        total = len(image_files)
        logging.info("Lora Caption Generator: processing %d images in %s", total, directory)

        if caption_instruction.strip():
            instruction = caption_instruction.strip()
        elif mode == "sdxl":
            instruction = _SDXL_CAPTION_INSTRUCTION
        else:
            instruction = _DEFAULT_CAPTION_INSTRUCTION
        logging.info("Lora Caption Generator: mode=%s instruction_len=%d", mode, len(instruction))

        # Phase 1. Le passage par des noms temporaires n'existe que pour le
        # renommage : sans lui, renommer en <trigger>_0003 pourrait ecraser un
        # fichier source qui porte deja ce nom. Quand on garde les noms
        # d'origine, il n'y a aucune collision possible et toucher aux fichiers
        # serait un risque gratuit — un plantage en cours de route laisserait le
        # dossier plein de __ofm_temp_XXXX.
        temp_files = []
        if keep_original_names:
            for fname in image_files:
                temp_files.append((os.path.join(directory, fname),
                                   os.path.splitext(fname)[1].lower()))
        else:
            for i, fname in enumerate(image_files, start=1):
                ext = os.path.splitext(fname)[1].lower()
                src = os.path.join(directory, fname)
                temp_path = os.path.join(directory, f"__ofm_temp_{i:04d}{ext}")
                os.rename(src, temp_path)
                temp_files.append((temp_path, ext))

        # Phase 2: caption each temp file, then rename to final name and write .txt
        success_count = 0
        for i, (temp_path, ext) in enumerate(temp_files, start=1):
            if keep_original_names:
                # Le .txt prend exactement le nom de l'image, extension mise a
                # part : c'est la convention que lisent les outils
                # d'entrainement, et elle survit a un tri ou a un ajout de
                # fichiers dans le dossier.
                final_base = os.path.splitext(os.path.basename(temp_path))[0]
                final_image_path = temp_path
            else:
                final_base = f"{trigger}_{i:04d}"
                final_image_path = os.path.join(directory, f"{final_base}{ext}")
            final_txt_path = os.path.join(directory, f"{final_base}.txt")

            try:
                image_b64 = _image_file_to_b64(temp_path)

                if provider == "vertex":
                    caption = _call_vertex(vertex_files[0], gemini_model, image_b64,
                                           instruction, temperature, max_tokens)
                elif provider == "gemini":
                    caption = _call_gemini(api_key, gemini_model, image_b64,
                                           instruction, temperature, max_tokens)
                else:
                    caption = _call_grok(api_key, grok_model, image_b64,
                                         instruction, temperature, max_tokens)

                if trigger:
                    caption = f"{trigger}, {caption}"

                if not keep_original_names:
                    os.rename(temp_path, final_image_path)

                # Write caption file alongside
                with open(final_txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                success_count += 1
                logging.info("Lora Caption Generator: [%d/%d] %s (%d chars)",
                             i, total, final_base, len(caption))
            except Exception as e:
                logging.error("Lora Caption Generator: [%d/%d] failed for %s: %s",
                              i, total, os.path.basename(temp_path), e)
                # On failure, still rename temp file to final image name so user
                # can see which one failed (no caption written)
                if not keep_original_names:
                    try:
                        os.rename(temp_path, final_image_path)
                    except Exception:
                        pass

        logging.info("Lora Caption Generator: done — %d/%d captioned in %s",
                     success_count, total, directory)

        # SDXL mode: also emit captions.json mapping {image_filename: caption}
        if mode == "sdxl":
            self._write_captions_json(directory)

        return {}

    @staticmethod
    def _write_captions_json(directory):
        """Build {image_filename: caption_text} from .txt+image pairs in `directory`
        and write captions.json. Mirrors the user's script.python.py logic.
        """
        import json

        all_files   = os.listdir(directory)
        image_files = {os.path.splitext(f)[0]: f for f in all_files
                       if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTENSIONS}
        txt_files   = {os.path.splitext(f)[0]: f for f in all_files
                       if f.endswith(".txt")}

        dataset      = {}
        missing_txt  = []
        missing_img  = []

        for name, imgfile in sorted(image_files.items()):
            if name in txt_files:
                txt_path = os.path.join(directory, name + ".txt")
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        dataset[imgfile] = f.read().strip()
                except Exception as e:
                    logging.warning("captions.json: failed to read %s: %s", txt_path, e)
            else:
                missing_txt.append(imgfile)

        for name, txtfile in sorted(txt_files.items()):
            if name not in image_files and txtfile != "captions.json":
                missing_img.append(txtfile)

        output_path = os.path.join(directory, "captions.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error("captions.json: write failed: %s", e)
            return

        logging.info("captions.json: %d entries written to %s", len(dataset), output_path)
        if missing_txt:
            logging.warning("captions.json: %d image(s) without caption: %s",
                            len(missing_txt), ", ".join(missing_txt[:10]))
        if missing_img:
            logging.warning("captions.json: %d txt(s) without image: %s",
                            len(missing_img), ", ".join(missing_img[:10]))
