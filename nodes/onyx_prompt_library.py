# -*- coding: utf-8 -*-
"""
ComfyUI nodes - Onyx Prompt Saver / Prompt Gallery.

A small prompt library with thumbnails, stored next to the pack.

The awkward part of the design, and why it is built this way: a Save button is
clicked whenever you like, but the image that would illustrate the prompt only
exists AFTER a run. A button alone can therefore never have a thumbnail, and
saving automatically on every run would flood the library with near-duplicates.

So the Saver does both halves separately. On every execution it keeps the
prompt and a thumbnail of the image in a pending slot, in memory, keyed by node
id. The button then commits whatever is pending. You run once, look at the
result, and decide - which is the moment you actually know whether the prompt
was any good.

Layout on disk:

    _prompt_library/
        library.json          one list of entries, human-readable
        thumbs/<id>.jpg       320 px, quality 80

library.json is plain JSON on purpose: if this pack ever disappears, the prompts
are still readable in a text editor.
"""

import base64
import io
import json
import logging
import os
import time
import uuid

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_HERE, "_prompt_library")
_THUMB_DIR = os.path.join(_LIB_DIR, "thumbs")
_INDEX = os.path.join(_LIB_DIR, "library.json")

_THUMB_MAX = 320

# Reserve en memoire : {node_id: {"prompt": str, "thumb": bytes|None}}
_PENDING = {}


def _load_index() -> list:
    try:
        with open(_INDEX, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as e:
        logging.warning("Prompt Library: unreadable index (%s)", e)
        return []


def _save_index(entries: list) -> None:
    os.makedirs(_LIB_DIR, exist_ok=True)
    tmp = _INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    # Ecriture atomique : une coupure pendant json.dump laisserait sinon un
    # index tronque, et toute la bibliotheque avec lui.
    os.replace(tmp, _INDEX)


def _make_thumb(image) -> bytes | None:
    """First frame of an IMAGE batch -> JPEG bytes, longest side 320 px."""
    if image is None or Image is None:
        return None
    try:
        frame = image[0]
        arr = (255.0 * frame.cpu().numpy()).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
        pil.thumbnail((_THUMB_MAX, _THUMB_MAX), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=80, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logging.warning("Prompt Library: thumbnail failed (%s)", e)
        return None


def commit(prompt: str, thumb: bytes | None, name: str = "", tags: str = "") -> dict:
    """Write one entry to the library and return it."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("nothing to save: the prompt is empty.")

    entry_id = uuid.uuid4().hex[:12]
    os.makedirs(_THUMB_DIR, exist_ok=True)
    has_thumb = False
    if thumb:
        with open(os.path.join(_THUMB_DIR, f"{entry_id}.jpg"), "wb") as fh:
            fh.write(thumb)
        has_thumb = True

    entry = {
        "id": entry_id,
        "name": (name or "").strip() or prompt[:48].replace("\n", " "),
        "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],
        "prompt": prompt,
        "thumb": has_thumb,
        "created": time.time(),
        "chars": len(prompt),
    }
    entries = _load_index()
    entries.insert(0, entry)          # le plus recent en tete
    _save_index(entries)
    return entry


def delete(entry_id: str) -> bool:
    entries = _load_index()
    kept = [e for e in entries if e.get("id") != entry_id]
    if len(kept) == len(entries):
        return False
    _save_index(kept)
    path = os.path.join(_THUMB_DIR, f"{entry_id}.jpg")
    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception:
            pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

class OnyxPromptSaver:
    """Hold the current prompt and a thumbnail, ready for the Save button."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The prompt to file away. Usually linked from whatever produced it."}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "The result this prompt produced. Its first frame becomes the "
                               "thumbnail — that is what makes the library browsable."}),
                "name": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional. Left empty, the first 48 characters of the prompt are used."}),
                "tags": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Comma-separated, searchable in the gallery."}),
                "autosave": ("BOOLEAN", {
                    "default": False,
                    "label_on": "save on every run",
                    "label_off": "save on button only",
                    "tooltip": "Off: a run only ARMS the button, nothing is written. Turn it on for "
                               "a batch you want kept wholesale — it will file one entry per run."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Onyx/Prompt"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Le node doit re-tourner a chaque queue, sinon la reserve garde la
        # paire d'un run precedent et le bouton enregistre la mauvaise image.
        return float("nan")

    def run(self, prompt, image=None, name="", tags="", autosave=False, unique_id=None):
        thumb = _make_thumb(image)
        _PENDING[str(unique_id)] = {"prompt": prompt or "", "thumb": thumb}

        if autosave:
            try:
                entry = commit(prompt, thumb, name, tags)
                print(f"💾 [Prompt Saver] autosaved '{entry['name']}' "
                      f"({entry['chars']} chars, thumbnail: {'yes' if entry['thumb'] else 'no'})")
            except Exception as e:
                print(f"⚠️  [Prompt Saver] autosave skipped: {e}")
        else:
            print(f"📌 [Prompt Saver] ready — {len((prompt or '').strip())} chars, "
                  f"thumbnail: {'yes' if thumb else 'no'}. Click Save to file it.")

        return (prompt,)


class OnyxPromptGallery:
    """Pick a saved prompt and feed it into the graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Filled by the Browse button. It stays a normal text field, so the "
                               "value is saved with the workflow and still works if the library "
                               "is ever moved or lost."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "Onyx/Prompt"

    def run(self, prompt):
        return (prompt,)


# ─────────────────────────────────────────────────────────────────────────────
# REST
# ─────────────────────────────────────────────────────────────────────────────

try:
    from server import PromptServer
    from aiohttp import web

    if not getattr(PromptServer.instance, "_onyx_prompt_lib_routes", False):
        PromptServer.instance._onyx_prompt_lib_routes = True

        @PromptServer.instance.routes.get("/onyx/prompt_library/list")
        async def _pl_list(request):
            entries = _load_index()
            # Le prompt complet part avec la liste : la galerie doit pouvoir
            # l'inserer sans un second aller-retour par entree.
            return web.json_response({"success": True, "entries": entries})

        @PromptServer.instance.routes.get("/onyx/prompt_library/thumb")
        async def _pl_thumb(request):
            entry_id = (request.query.get("id") or "").strip()
            # Le nom de fichier vient du client : sans ce filtre, un ".." irait
            # lire n'importe quel fichier de la machine.
            if not entry_id.isalnum():
                return web.Response(status=400, text="bad id")
            path = os.path.join(_THUMB_DIR, f"{entry_id}.jpg")
            if not os.path.isfile(path):
                return web.Response(status=404, text="no thumbnail")
            return web.FileResponse(path, headers={"Cache-Control": "max-age=31536000"})

        @PromptServer.instance.routes.post("/onyx/prompt_library/save")
        async def _pl_save(request):
            try:
                body = await request.json()
                node_id = str(body.get("node_id", ""))
                pending = _PENDING.get(node_id)
                prompt = (body.get("prompt") or "").strip()
                thumb = pending.get("thumb") if pending else None

                # Le texte du widget prime sur la reserve : il reflete ce qui est
                # a l'ecran maintenant, la reserve date du dernier rendu.
                if not prompt and pending:
                    prompt = pending.get("prompt", "")
                if not prompt:
                    return web.json_response(
                        {"success": False, "error": "the prompt is empty"}, status=400)

                entry = commit(prompt, thumb, body.get("name", ""), body.get("tags", ""))
                print(f"💾 [Prompt Saver] saved '{entry['name']}' "
                      f"(thumbnail: {'yes' if entry['thumb'] else 'no'})")
                return web.json_response({"success": True, "entry": entry})
            except Exception as e:
                logging.exception("Prompt Library: save failed")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/onyx/prompt_library/delete")
        async def _pl_delete(request):
            try:
                body = await request.json()
                ok = delete(str(body.get("id", "")))
                return web.json_response({"success": ok})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=500)

except Exception:
    logging.warning("Onyx Prompt Library: routes not registered (server unavailable)")
