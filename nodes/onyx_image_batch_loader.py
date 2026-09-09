"""
Onyx Image and Video Batch Loader
Sequential batch loader with drag-and-drop UI.
Loads images and videos one at a time in order, cycling through the uploaded list.
"""

import os
import json
import uuid
import hashlib
import torch
import numpy as np
from PIL import Image
from aiohttp import web
import folder_paths
from server import PromptServer

# ─────────────────────────────────────────────────────────────────────────────
_POOL_SUBDIR = "Onyx_ImagePool"
_THUMB_PREFIX = "thumb_"

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}

_EMPTY_IMAGE_SIDE = 64


def _pool_dir() -> str:
    d = os.path.join(folder_paths.get_input_directory(), _POOL_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _is_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


class _AnyType(str):
    """A type string that ComfyUI's link check accepts against any socket.

    Needed for `path`. LoadVideoUI declares its `video` input as a list, which the
    frontend presents as a COMBO socket, and a STRING output simply cannot be
    linked to it — the check is a plain type comparison. Returning False from
    __ne__ makes that comparison pass while the value stays an ordinary string,
    which is exactly what LoadVideoUI wants: it resolves `video` by testing the
    raw value as a filesystem path before anything else.

    The trade is that this output can be dropped onto sockets where a path makes
    no sense; ComfyUI will accept the link and the receiving node will fail on the
    value instead of at connection time.
    """

    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


# ─────────────────────────────────────────────────────────────────────────────
# Video decoding
#
# Trois backends essayes dans l'ordre plutot qu'un seul suppose : ce pack tourne
# sur des installations tres differentes, et une dependance absente doit donner
# un message lisible, pas un ImportError au milieu d'un run. PyAV en premier
# parce que c'est ce que ComfyUI utilise nativement pour la video, donc celui qui
# a le plus de chances d'etre deja la et de decoder les memes fichiers.
# ─────────────────────────────────────────────────────────────────────────────

def _decode_with_av(path, max_side):
    import av
    frames, fps = [], 0.0
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(stream):
            pil = frame.to_image()
            frames.append(_fit(pil, max_side))
    return frames, fps


def _decode_with_cv2(path, max_side):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cv2 could not open the file")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(_fit(Image.fromarray(rgb), max_side))
    cap.release()
    return frames, fps


def _decode_with_imageio(path, max_side):
    import imageio.v3 as iio
    meta = iio.immeta(path)
    fps = float(meta.get("fps") or 0.0)
    frames = [_fit(Image.fromarray(f), max_side) for f in iio.imiter(path)]
    return frames, fps


def _fit(pil: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the long edge is at most max_side. 0 keeps native size.

    A 15 s 1080p clip is 360 frames; held as float32 RGB that is over 2 GB of
    RAM before anything else runs. The default cap exists so dropping a normal
    phone video into the node does not take the machine down.
    """
    if not max_side:
        return pil.convert("RGB")
    w, h = pil.size
    longest = max(w, h)
    if longest <= max_side:
        return pil.convert("RGB")
    scale = max_side / float(longest)
    return pil.convert("RGB").resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS
    )


def _decode_video(path: str, max_side: int = 1024):
    """Return (frames_tensor (N,H,W,3), fps). Raises with the list of failures."""
    errors = []
    for name, fn in (("av", _decode_with_av), ("cv2", _decode_with_cv2),
                     ("imageio", _decode_with_imageio)):
        try:
            frames, fps = fn(path, max_side)
            if not frames:
                errors.append(f"{name}: decoded 0 frames")
                continue
            arr = np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in frames])
            print(f"[Onyx Batch] 🎞️  decoded with {name}: {len(frames)} frames "
                  f"@ {fps:.2f} fps, {frames[0].size[0]}x{frames[0].size[1]}")
            return torch.from_numpy(arr), (fps or 24.0)
        except ImportError:
            errors.append(f"{name}: not installed")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(
        "[Onyx Batch] Could not decode the video. Tried:\n  "
        + "\n  ".join(errors)
        + "\n-> Install one of: av (recommended, same decoder as ComfyUI), "
          "opencv-python, imageio[ffmpeg]."
    )


def _first_frame(path: str) -> Image.Image:
    """First frame of a video, for the gallery thumbnail.

    Decodes one frame and stops, rather than reusing _decode_video: pulling a
    whole 15 s clip into memory just to make a 300 px preview would stall the
    upload for every file dropped on the node.
    """
    errors = []
    try:
        import av
        with av.open(path) as container:
            for frame in container.decode(container.streams.video[0]):
                return frame.to_image().convert("RGB")
        errors.append("av: no video frame")
    except ImportError:
        errors.append("av: not installed")
    except Exception as e:
        errors.append(f"av: {e}")

    try:
        import cv2
        cap = cv2.VideoCapture(path)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        errors.append("cv2: could not read a frame")
    except ImportError:
        errors.append("cv2: not installed")
    except Exception as e:
        errors.append(f"cv2: {e}")

    raise RuntimeError("no video backend could read a first frame — " + "; ".join(errors))


def _resize_frames(frames, method: str, target_w: int, target_h: int):
    """Resize a (N,H,W,C) batch to target_w x target_h.

    Runs on the whole batch at once through torch rather than frame by frame with
    PIL: 360 frames is 360 round trips otherwise, and the tensor is already in
    memory in the right layout apart from the channel axis.
    """
    import torch.nn.functional as F

    if method == "none" or (target_w <= 0 and target_h <= 0):
        return frames

    n, h, w, c = frames.shape

    # Une seule dimension fournie : l'autre suit le rapport d'origine, ce qui
    # evite d'avoir a saisir les deux quand on veut juste "toutes en 480 de haut".
    if target_w <= 0:
        target_w = max(1, round(w * target_h / h))
    if target_h <= 0:
        target_h = max(1, round(h * target_w / w))

    chw = frames.permute(0, 3, 1, 2)  # (N,C,H,W) pour interpolate

    if method == "stretch to fit":
        out = F.interpolate(chw, size=(target_h, target_w), mode="bilinear", align_corners=False)

    elif method == "crop":
        # Couvre la cible puis recadre au centre : aucune deformation, on perd
        # les bords du plan le plus long.
        scale = max(target_w / w, target_h / h)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        res = F.interpolate(chw, size=(rh, rw), mode="bilinear", align_corners=False)
        top, left = (rh - target_h) // 2, (rw - target_w) // 2
        out = res[:, :, top:top + target_h, left:left + target_w]

    elif method == "pad":
        # Tient dans la cible puis complete en noir : rien n'est perdu ni
        # deforme, on ajoute des bandes.
        scale = min(target_w / w, target_h / h)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        res = F.interpolate(chw, size=(rh, rw), mode="bilinear", align_corners=False)
        pad_t, pad_l = (target_h - rh) // 2, (target_w - rw) // 2
        out = F.pad(res, (pad_l, target_w - rw - pad_l, pad_t, target_h - rh - pad_t))

    else:  # "maintain aspect ratio" — tient dans la boite, sans completer
        scale = min(target_w / w, target_h / h)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        out = F.interpolate(chw, size=(rh, rw), mode="bilinear", align_corners=False)

    out = out.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
    print(f"[Onyx Batch] 📐 resized {w}x{h} -> {out.shape[2]}x{out.shape[1]} ({method})")
    return out


def _extract_audio(path: str):
    """Return a ComfyUI AUDIO dict, or None when the file carries no audio.

    Missing audio is a normal case, not an error: plenty of clips are silent,
    and refusing to load them would be worse than returning nothing.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            sr = int(stream.rate)
            chunks = [f.to_ndarray() for f in container.decode(stream)]
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=-1) if chunks[0].ndim > 1 else np.concatenate(chunks)
        if data.ndim == 1:
            data = data[None, :]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        wav = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0)
        print(f"[Onyx Batch] 🔊 audio: {wav.shape[1]} channel(s) @ {sr} Hz")
        return {"waveform": wav, "sample_rate": sr}
    except Exception as e:
        print(f"[Onyx Batch] audio not extracted: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
class OnyxImageBatchLoader:
    """
    Onyx Image and Video Batch Loader
    Upload images and videos via the node UI and process them sequentially.
    Each execution loads the next item in the list.
    Click «Queue All» to process every image.
    """

    _states: dict = {}  # state_key → {"current_index": int}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Managed entirely by the JS widget — hidden from the user via JS.
                "batch_data": ("STRING", {"default": "{}"}),
                "video_max_side": ("INT", {
                    "default": 1024, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Longest edge of the decoded video frames. 0 keeps native size.\n"
                               "A 15 s 1080p clip is 360 frames, over 2 GB held as float32 RGB. "
                               "The cap keeps a normal phone video from taking the machine down.\n"
                               "Irrelevant for images, which are never resized.",
                }),
                # Le decoupage est une REGLE, pas une valeur : la meme pour tous
                # les clips de la file, alors que leur duree differe. D'ou la
                # convention 0 = fin de la video, qui rend la regle applicable
                # a des longueurs quelconques sans rien recalculer a la main.
                "trim_mode": (["seconds", "frames"], {
                    "default": "seconds",
                    "tooltip": "Whether the trim below is expressed in seconds or in frames.",
                }),
                "trim_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "Where each clip starts. 0 keeps the beginning.",
                }),
                "trim_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "Where each clip ends. **0 means the end of the video**, so the "
                               "same setting works across clips of different lengths — which is "
                               "the whole point of trimming in a batch.\n"
                               "Set 8 with trim_mode=seconds to keep the first 8 seconds of every "
                               "clip, whatever its duration.",
                }),
                "resize_method": (
                    ["none", "maintain aspect ratio", "stretch to fit", "pad", "crop"], {
                        "default": "none",
                        "tooltip": "How each clip is fitted to custom_width x custom_height.\n"
                                   "maintain aspect ratio: fits inside the box, output size varies "
                                   "with each source's ratio.\n"
                                   "stretch to fit: exact size, distorts.\n"
                                   "pad: exact size, black bars, nothing lost.\n"
                                   "crop: exact size, no distortion, edges lost.\n\n"
                                   "Use pad or crop when every clip must come out the same size — "
                                   "'maintain aspect ratio' does not guarantee that across a batch "
                                   "of mixed sources.",
                    }),
                "custom_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "0 derives the width from the height and the source ratio.",
                }),
                "custom_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "0 derives the height from the width and the source ratio.",
                }),
                "force_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Resample every clip to this frame rate. 0 keeps each clip's native "
                               "rate.\nSet 24 to normalise a batch of mixed 30 and 60 fps sources — "
                               "MiniMax H3 works at 24, and a clip left at 30 plays back at the "
                               "wrong speed.",
                }),
                # Les deux widgets ci-dessous sont ajoutes EN DERNIER : ComfyUI
                # serialise les valeurs par POSITION, donc les inserer plus haut
                # decalerait tous les reglages des workflows deja enregistres.
                "consume_on_load": ("BOOLEAN", {
                    "default": False,
                    "label_on": "remove each item once loaded",
                    "label_off": "keep the whole list",
                    "tooltip": "ON: an item disappears from the list as soon as it has been "
                               "loaded. A crash three quarters of the way through a batch then "
                               "costs you only what had not run yet — re-queue and it carries on "
                               "instead of starting over.\n"
                               "The file itself is NOT deleted from disk, only removed from this "
                               "node's list.\n"
                               "Pair it with queue_batch_size = 1: an item is consumed when it is "
                               "LOADED, not when the graph finishes, so anything still queued "
                               "behind a failure is lost from the list.",
                }),
                "queue_batch_size": ("INT", {
                    "default": 50, "min": 1, "max": 50, "step": 1,
                    "tooltip": "How many runs 'Queue All' submits at a time.\n"
                               "50 is fastest. 1 is the safe setting: a failure then costs one "
                               "item instead of everything already sitting in the queue.\n"
                               "With consume_on_load ON, the button waits for the queue to empty "
                               "between batches — it has to, because each run rewrites the list "
                               "the next submission is built from.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # `video` is a separate output on purpose. It is an IMAGE batch like `image`,
    # so ComfyUI would happily let them be swapped, and a frame sequence wired
    # into a single-image input fails much later with a confusing error.
    # `path` is the absolute path of the current item in the pool. LoadVideoUI
    # resolves its `video` argument with `video_path = video` before trying any
    # ComfyUI lookup, and its VALIDATE_INPUTS returns True unconditionally, so an
    # absolute path fed there loads the file directly — which is what turns a
    # batch of videos into a dynamic source for a workflow built around it.
    # duration / frame_count / video_fps are OUTPUTS, not refreshed widgets.
    # LoadVideoUI needs its browser to probe the file and fill its widgets because
    # they are editable trim controls; that refresh only fires on a manual click,
    # which is precisely what a queue cannot do. Reading the real values in Python
    # at execution time gives the correct figures for every item in the batch,
    # with no round trip through the frontend.
    RETURN_TYPES  = ("IMAGE", "IMAGE", "FLOAT", "FLOAT", "INT", "AUDIO", _ANY)
    RETURN_NAMES  = ("image", "video", "video_fps", "duration", "frame_count", "audio", "path")
    OUTPUT_NODE   = False
    FUNCTION      = "load_next"
    CATEGORY      = "Onyx"
    DESCRIPTION   = (
        "Sequential batch loader for images and videos.\n"
        "Upload via the node UI, then click Queue All.\n"
        "Each queue run outputs the next item: images leave through `image`, "
        "videos through `video` + `video_fps` + `duration` + `frame_count` + `audio`.\n"
        "Trim and frame rate are applied per clip, using each clip's own real values."
    )

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_trim(frames, fps, trim_mode, trim_start, trim_end, force_fps):
        """Trim then optionally resample. Returns (frames, fps).

        Resampling is nearest-neighbour on the time axis: it drops or repeats
        whole frames rather than blending them. Blending would invent motion that
        never happened, which is the opposite of what a motion reference is for.
        """
        total = int(frames.shape[0])
        if total == 0:
            return frames, fps

        if trim_mode == "frames":
            start = int(trim_start)
            end = int(trim_end) if trim_end > 0 else total
        else:
            start = int(round(trim_start * fps))
            end = int(round(trim_end * fps)) if trim_end > 0 else total

        start = max(0, min(start, total - 1))
        end = max(start + 1, min(end, total))

        if (start, end) != (0, total):
            frames = frames[start:end]
            print(f"[Onyx Batch] ✂️  trimmed to frames {start}-{end} "
                  f"({(end - start) / fps:.3f}s of {total / fps:.3f}s)")

        if force_fps > 0 and abs(force_fps - fps) > 1e-6:
            n_in = int(frames.shape[0])
            n_out = max(1, int(round(n_in * force_fps / fps)))
            idx = torch.linspace(0, n_in - 1, n_out).round().long().clamp(0, n_in - 1)
            frames = frames[idx]
            print(f"[Onyx Batch] ⏱️  resampled {fps:.2f} -> {force_fps:.2f} fps "
                  f"({n_in} -> {n_out} frames)")
            fps = float(force_fps)

        return frames, fps

    # ─────────────────────────────────────────────────────────────────────────
    def load_next(self, batch_data: str = "{}", video_max_side: int = 1024,
                  trim_mode: str = "seconds", trim_start: float = 0.0,
                  trim_end: float = 0.0, resize_method: str = "none",
                  custom_width: int = 0, custom_height: int = 0,
                  force_fps: float = 0.0, consume_on_load: bool = False,
                  queue_batch_size: int = 50, unique_id=None):
        # ── Parse data ────────────────────────────────────────────────────────
        try:
            data = json.loads(batch_data) if batch_data else {}
        except (json.JSONDecodeError, TypeError):
            data = {}

        images_meta = [
            x for x in data.get("images", [])
            if isinstance(x, dict) and x.get("id")
        ]
        order = [x for x in data.get("order", []) if isinstance(x, str)]

        empty = torch.zeros((1, _EMPTY_IMAGE_SIDE, _EMPTY_IMAGE_SIDE, 3), dtype=torch.float32)
        # Le placeholder image reste 64x64 noir : grok_prompt.py le detecte par
        # cette taille exacte pour ne pas facturer une analyse vision sur du vide.
        # Ne pas changer sans mettre a jour _is_placeholder_frame la-bas.
        blank = (empty, empty, 0.0, 0.0, 0, None, "")

        if not images_meta or not order:
            return blank

        # Build ordered flat list
        flat = []
        for img_id in order:
            meta = next((m for m in images_meta if m["id"] == img_id), None)
            if meta:
                flat.append(meta)

        total = len(flat)
        if total == 0:
            return blank

        # ── Sequential state ──────────────────────────────────────────────────
        state_key = f"{unique_id}_{hashlib.md5(batch_data.encode()).hexdigest()[:8]}"
        if state_key not in self._states:
            self._states[state_key] = {"current_index": 0}

        idx   = self._states[state_key]["current_index"] % total
        meta  = flat[idx]
        _next = (idx + 1) % total
        self._states[state_key]["current_index"] = _next

        # ── Load ──────────────────────────────────────────────────────────────
        pool = _pool_dir()
        path = os.path.join(pool, meta["filename"])
        name = meta.get("original_name", meta["filename"])

        if not os.path.exists(path):
            print(f"[Onyx Batch] File not found: {path}")
            self._notify(unique_id, idx, total)
            return blank

        if _is_video(meta["filename"]):
            try:
                # Le plafond memoire ne doit pas descendre sous la cible de
                # redimensionnement, sinon on decoderait en dessous puis on
                # reagrandirait — une perte de detail invisible dans les reglages.
                cap = int(video_max_side)
                if resize_method != "none":
                    cap = max(cap, int(custom_width), int(custom_height))
                frames, fps = _decode_video(path, cap)
                frames = _resize_frames(
                    frames, resize_method, int(custom_width), int(custom_height)
                )
                frames, fps = self._apply_trim(
                    frames, fps, trim_mode, float(trim_start), float(trim_end), float(force_fps)
                )
                audio = _extract_audio(path)
                n = int(frames.shape[0])
                duration = n / fps if fps else 0.0
                print(f"[Onyx Batch] ✅ [{idx + 1}/{total}] 🎬 {name} — "
                      f"{n} frames, {duration:.3f}s @ {fps:.2f} fps")
                self._notify(unique_id, idx, total)
                if consume_on_load:
                    self._consume(unique_id, meta["id"], name, total - 1)
                # `image` reste le placeholder : cet item est une video, et
                # renvoyer sa premiere frame sur la sortie image ferait passer
                # une video pour une photo dans un graphe branche sur les deux.
                return (empty, frames, float(fps), float(duration), n, audio, path)
            except Exception as e:
                print(f"[Onyx Batch] ❌ {name}: {e}")
                self._notify(unique_id, idx, total)
                return blank

        try:
            pil_img = Image.open(path).convert("RGB")
            arr     = np.array(pil_img).astype(np.float32) / 255.0
            tensor  = torch.from_numpy(arr).unsqueeze(0)
            print(f"[Onyx Batch] ✅ [{idx + 1}/{total}] 🖼️  {name}")
        except Exception as e:
            print(f"[Onyx Batch] ❌ Error loading {meta['filename']}: {e}")
            self._notify(unique_id, idx, total)
            return blank

        self._notify(unique_id, idx, total)
        # Consomme UNIQUEMENT apres un chargement reussi : un fichier illisible
        # doit rester dans la liste, sinon il disparait sans avoir rien produit
        # et sans qu'on sache lequel c'etait.
        if consume_on_load:
            self._consume(unique_id, meta["id"], name, total - 1)
        return (tensor, empty, 0.0, 0.0, 0, None, path)

    @staticmethod
    def _consume(node_id, img_id: str, name: str, left: int):
        """Tell the frontend to drop this item from the list.

        It has to go through the frontend: the list lives in the batch_data
        widget, which only the browser owns. Python cannot edit a widget, but it
        can say what happened and let the JS rewrite it.

        The already-queued prompts each carry their OWN snapshot of batch_data,
        taken when they were submitted, so removing an entry here never disturbs
        a run that is already in flight. It only shortens the list the NEXT
        submission is built from — which is exactly what makes a batch
        resumable.
        """
        try:
            PromptServer.instance.send_sync(
                "onyx_batch_loader_consume",
                {"node_id": str(node_id), "image_id": str(img_id),
                 "name": name, "left": int(left)},
            )
            print(f"[Onyx Batch] 🗑️  consumed '{name}' — {left} left in the list.")
        except Exception as e:
            # Le rendu est deja fait : perdre la notification coute une entree
            # en trop dans la liste, pas la generation.
            print(f"[Onyx Batch] ⚠️  could not notify the frontend ({e}) — "
                  f"'{name}' stays in the list.")

    @staticmethod
    def _notify(node_id, current_index: int, total: int):
        try:
            PromptServer.instance.send_sync(
                "onyx_batch_loader_update",
                {"node_id": str(node_id), "current_index": current_index, "total": total},
            )
        except Exception:
            pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
#
# Guard : evite le crash "Cannot register a resource into frozen router"
# quand ComfyUI re-importe le module au restart. Sans ce guard, les
# decorateurs ci-dessous sont re-executes -> aiohttp leve une exception ->
# shutdown bloque -> port 8190 + DB lock restent detenus -> nouveau ComfyUI
# ne peut pas demarrer. Pattern identique a celui utilise dans nano_banana_aio.
# ─────────────────────────────────────────────────────────────────────────────

if not getattr(PromptServer.instance, "_onyx_batch_routes_registered", False):
    PromptServer.instance._onyx_batch_routes_registered = True

    @PromptServer.instance.routes.post("/onyx/batch_upload")
    async def _onyx_batch_upload(request):
        """Upload one or more images to the shared pool."""
        try:
            reader  = await request.multipart()
            pool    = _pool_dir()
            results = []

            async for field in reader:
                if field.name != "files":
                    continue
                raw_name = field.filename or "image.jpg"
                data     = await field.read()

                file_id  = str(uuid.uuid4())
                ext      = os.path.splitext(raw_name)[1].lower() or ".jpg"
                safe     = f"{file_id}{ext}"
                path     = os.path.join(pool, safe)

                with open(path, "wb") as fh:
                    fh.write(data)

                try:
                    if _is_video(safe):
                        # Vignette = premiere frame. Le thumbnail est toujours un
                        # PNG, quelle que soit l'extension source, pour que la
                        # galerie du widget n'ait pas a savoir lire de la video.
                        img = _first_frame(path)
                        is_video = True
                    else:
                        img = Image.open(path)
                        is_video = False

                    w, h = img.size

                    thumb      = img.convert("RGB").copy()
                    thumb.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    thumb_name = f"{_THUMB_PREFIX}{os.path.splitext(safe)[0]}.png"
                    thumb.save(os.path.join(pool, thumb_name))

                    results.append({
                        "id":            file_id,
                        "filename":      safe,
                        "original_name": raw_name,
                        "thumbnail":     thumb_name,
                        "width":         w,
                        "height":        h,
                        "is_video":      is_video,
                    })
                except Exception as e:
                    print(f"[Onyx Batch] Error processing {raw_name}: {e}")
                    if os.path.exists(path):
                        os.remove(path)

            return web.json_response({"success": True, "images": results})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)


    @PromptServer.instance.routes.delete("/onyx/batch_delete/{image_id}")
    async def _onyx_batch_delete(request):
        """Delete an image (and its thumbnail) from the pool."""
        image_id = request.match_info["image_id"]
        try:
            pool    = _pool_dir()
            deleted = []
            for fn in os.listdir(pool):
                if image_id in fn:
                    os.remove(os.path.join(pool, fn))
                    deleted.append(fn)
            return web.json_response({"success": True, "deleted": deleted})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)


    @PromptServer.instance.routes.get("/onyx/view/{filename}")
    async def _onyx_view(request):
        """Serve an image from the pool via ComfyUI's standard /view endpoint."""
        filename = request.match_info["filename"]
        pool     = _pool_dir()
        if os.path.exists(os.path.join(pool, filename)):
            return web.HTTPFound(
                f"/view?filename={filename}&type=input&subfolder={_POOL_SUBDIR}"
            )
        return web.Response(status=404, text=f"Image not found: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "OnyxImageBatchLoader": OnyxImageBatchLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # La cle reste OnyxImageBatchLoader : elle est inscrite dans chaque
    # workflow sauvegarde, seul le libelle affiche change.
    "OnyxImageBatchLoader": "Onyx Image and Video Batch Loader",
}
