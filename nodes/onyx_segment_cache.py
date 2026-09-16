# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - Onyx Segment Cache.

Keeps a finished segment on disk so a later failure in the chain does not throw
it away.

The problem it solves: a four-segment render is forty minutes of sampling in one
queue. If segment 3 dies - a driver fault, a bad allocation, a mistyped widget -
segments 1 and 2 die with it, and the next attempt re-renders them from nothing.
Nothing was wrong with them.

How the sampler is actually skipped, and why it is not obvious: ComfyUI is a
DAG, so a node placed AFTER the sampler cannot stop it from running. What it can
do is refuse to ask for it. `images` and `audio` are LAZY inputs: ComfyUI
evaluates them only when this node requests them in check_lazy_status. On a
cache hit it requests nothing, and the whole upstream branch - sampler, model
load, VAE - is never executed. That is the same mechanism the Join uses to skip
unneeded segments.

What the key is built from, and why it is built from those things:

    session + segment_index          which segment of which job
    fingerprint_image (optional)     the reference slice; changes if the source
                                     video, the crop, the mask, the window or
                                     the resolution changed
    cache_version (widget)           bumped BY HAND when something the image
                                     cannot see has changed: a seed, a sampler
                                     setting, the intent text

The fingerprint is what stops a stale hit. Keyed on the index alone, this node
would happily hand back yesterday's segment 2 for today's completely different
video, and the mistake would only surface in the finished file.

DO NOT feed the Context-IR's compiled prompt in here, tempting as it looks. That
prompt is written by an LLM, so it comes back different on every run even when
nothing upstream changed - 4937, then 5605, then 5983 characters for the same
segment and the same inputs. A key built on it can never match twice, and the
cache silently never hits. `fingerprint_text` exists for text you know to be
deterministic, which the compiled prompt is not.

Frames are stored as uint8 by default. The segment ends up in an 8-bit video, so
the rounding costs nothing that survives to the file; float16 is there for when
the batch still has grading or compositing ahead of it.
"""

import hashlib
import json
import logging
import os
import shutil
import time

import numpy as np
import torch


_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_ROOT = os.path.join(_HERE, "_cache", "segments")

_PRECISION = {
    "uint8 (smallest, 8-bit output)": "uint8",
    "float16 (keeps grading headroom)": "float16",
}

_MODES = {
    "read + write": "rw",
    "write only (force re-render)": "w",
    "off (passthrough)": "off",
}


def _safe(name: str, fallback: str) -> str:
    out = "".join(c for c in str(name or "") if c.isalnum() or c in "-_")
    return out or fallback


def _digest_image(tensor) -> str:
    """Short digest of an IMAGE batch, from a strided subsample.

    Strided on purpose: two different renders differ everywhere, not in one
    pixel, so reading every value to tell them apart would cost seconds per
    segment for no extra certainty. Shape is folded in, which is what catches a
    resolution or length change even if the sampled pixels happen to match.
    """
    if tensor is None:
        return ""
    h = hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode())
    sub = tensor[::4, ::16, ::16, :].detach().cpu().contiguous()
    h.update(sub.numpy().astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def _digest_text(text) -> str:
    if not text:
        return ""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


class OnyxSegmentCache:
    """Store a rendered segment, and skip its sampler entirely on a re-run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_index": ("INT", {
                    "forceInput": True,
                    "tooltip": "Connect the Segment node's `next_index` MINUS nothing — use the "
                               "same index that drove this segment. Segment 1's node emits 1."}),
                "session": ("STRING", {
                    "default": "chain", "multiline": False,
                    "tooltip": "One name per job. Change it and you start a fresh cache; reuse it "
                               "and a re-queue picks up where the last one died."}),
                "mode": (list(_MODES), {
                    "default": "read + write",
                    "tooltip": "read + write: reuse a matching segment, otherwise render and keep it.\n"
                               "write only: always re-render, and overwrite what is stored — use "
                               "this when you changed something the fingerprints cannot see, like a seed.\n"
                               "off: this node does nothing at all."}),
                "precision": (list(_PRECISION), {
                    "default": "uint8 (smallest, 8-bit output)",
                    "tooltip": "uint8: ~0.75 GB per 243-frame segment at 768x1344.\n"
                               "float16: twice that, and keeps values a later grade could use."}),
                "cache_version": ("INT", {
                    "default": 1, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "Bump this by one whenever you change something the fingerprint "
                               "image cannot see — a seed, steps, cfg, the intent text.\n"
                               "It is the manual half of the key, and it is manual on purpose: "
                               "the alternative is hashing the compiled prompt, which an LLM "
                               "rewrites on every run, so the cache would never hit at all."}),
            },
            "optional": {
                "images": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "The sampler's output. LAZY: on a cache hit this is never "
                               "evaluated, so the sampler does not run."}),
                "audio": ("AUDIO", {"lazy": True}),
                "fingerprint_image": ("IMAGE", {
                    "tooltip": "Connect the Segment node's `reference_slice`. This is what makes a "
                               "hit trustworthy: change the source video and the key changes with it.\n"
                               "Leave it empty and the cache trusts the index alone."}),
                "fingerprint_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "For DETERMINISTIC text only.\n"
                               "Do NOT connect the Context-IR's compiled `prompt`: an LLM rewrites "
                               "it on every run, so the key would change every time and the cache "
                               "would never hit. The node refuses it and warns if you do."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN", "STRING")
    RETURN_NAMES = ("images", "audio", "was_cached", "info")
    FUNCTION = "run"
    CATEGORY = "Onyx"
    DESCRIPTION = ("Cache a rendered segment on disk. On a re-run, a matching segment is read "
                   "back and its sampler is never executed.")

    # ── clef et chemins ──────────────────────────────────────────────────────

    # Longueur au-dela de laquelle un texte est considere comme une sortie de
    # LLM et non comme un identifiant. Les prompts Context-IR compiles font
    # 4000 a 6000 caracteres ; une etiquette deterministe, quelques dizaines.
    _LLM_TEXT_CHARS = 1500

    @classmethod
    def _key(cls, session, segment_index, fingerprint_image=None, fingerprint_text=None,
             cache_version=1):
        # Le filtre est ICI et nulle part ailleurs. Le mettre dans run() laissait
        # check_lazy_status calculer une clef differente de celle qui avait ete
        # ecrite - donc un miss permanent, exactement le bug qu'il corrige.
        if fingerprint_text and len(str(fingerprint_text)) > cls._LLM_TEXT_CHARS:
            fingerprint_text = None
        parts = [
            _safe(session, "chain"),
            f"seg{int(segment_index or 0):02d}",
            f"v{int(cache_version or 1)}",
            _digest_image(fingerprint_image),
            _digest_text(fingerprint_text),
        ]
        return "_".join(p for p in parts if p)

    @classmethod
    def _dir(cls, session):
        return os.path.join(_CACHE_ROOT, _safe(session, "chain"))

    @classmethod
    def _paths(cls, session, key):
        base = os.path.join(cls._dir(session), key)
        return base + ".meta.json", base + ".images.npy", base + ".audio.npy"

    @classmethod
    def _hit(cls, session, key):
        meta_p, img_p, _ = cls._paths(session, key)
        # Le .meta.json est ecrit EN DERNIER, apres les tableaux : sa presence
        # est donc la preuve que l'ecriture est allee jusqu'au bout. Une coupure
        # de courant en plein milieu laisse un .npy orphelin, jamais un faux hit.
        return os.path.isfile(meta_p) and os.path.isfile(img_p)

    # ── evaluation paresseuse ────────────────────────────────────────────────

    @classmethod
    def check_lazy_status(cls, segment_index=None, session="chain", mode="read + write",
                          precision=None, cache_version=1,
                          fingerprint_image=None, fingerprint_text=None, **kwargs):
        """Ask for the sampler only when the cache cannot answer.

        Returning an empty list here is what skips the whole upstream branch.
        """
        wanted = [k for k in ("images", "audio") if k in kwargs and kwargs[k] is None]
        if _MODES.get(mode, "rw") != "rw":
            return wanted

        key = cls._key(session, segment_index, fingerprint_image, fingerprint_text, cache_version)
        if cls._hit(session, key):
            print(f"⚡ [Segment Cache] hit for segment {int(segment_index or 0)} ({key}) — "
                  f"the sampler for this segment will not run.")
            return []

        # Un miss silencieux est le pire des cas : douze minutes de rendu
        # repartent sans que rien n'indique pourquoi. On dit donc QUELLE partie
        # de la clef a bouge, en comparant aux entrees deja presentes.
        cls._explain_miss(session, segment_index, key)
        return wanted

    @classmethod
    def _explain_miss(cls, session, segment_index, key):
        idx = int(segment_index or 0)
        print(f"🔍 [Segment Cache] miss for segment {idx} — this sampler WILL run.")
        print(f"     looked for : {key}")
        d = cls._dir(session)
        if not os.path.isdir(d):
            print(f"     nothing stored yet for session '{_safe(session, 'chain')}'.")
            return
        stored = sorted(f[:-len(".meta.json")] for f in os.listdir(d) if f.endswith(".meta.json"))
        same_seg = [k for k in stored if f"_seg{idx:02d}_" in k or k.endswith(f"_seg{idx:02d}")]
        if not same_seg:
            print(f"     no entry for segment {idx} ({len(stored)} stored for other segments).")
            return
        want = key.split("_")
        for k in same_seg[:3]:
            have = k.split("_")
            diff = []
            for a, b in zip(want, have):
                if a != b:
                    what = ("version" if a.startswith("v") and b.startswith("v")
                            else "fingerprint")
                    diff.append(f"{what}: {b} -> {a}")
            print(f"     stored     : {k}"
                  + (f"\n                  differs by {', '.join(diff)}" if diff else ""))

    # ── execution ────────────────────────────────────────────────────────────

    def run(self, segment_index, session, mode, precision, cache_version=1,
            images=None, audio=None, fingerprint_image=None, fingerprint_text=None):
        ensure_profile_ready()
        m = _MODES.get(mode, "rw")
        idx = int(segment_index or 0)
        if fingerprint_text and len(str(fingerprint_text)) > self._LLM_TEXT_CHARS:
            print(f"⚠️  [Segment Cache] fingerprint_text is {len(str(fingerprint_text))} chars — "
                  f"that looks like a compiled Context-IR prompt.\n"
                  f"     LLM output differs on every run, so it is IGNORED for the key "
                  f"(_key drops it). Use cache_version instead, and unplug it to save the call.")
        key = self._key(session, segment_index, fingerprint_image, fingerprint_text, cache_version)
        meta_p, img_p, aud_p = self._paths(session, key)

        if m == "off":
            if images is None:
                raise ValueError("[Segment Cache] mode is 'off' but no images arrived. "
                                 "Connect the sampler to `images`.")
            return (images, audio, False, "cache off — passthrough")

        # ── lecture ──────────────────────────────────────────────────────────
        if m == "rw" and self._hit(session, key) and images is None:
            with open(meta_p, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            arr = np.load(img_p, mmap_mode="r")
            out = torch.from_numpy(np.ascontiguousarray(arr))
            if out.dtype == torch.uint8:
                out = out.float() / 255.0
            else:
                out = out.float()

            aud = None
            if os.path.isfile(aud_p) and meta.get("audio_rate"):
                wav = torch.from_numpy(np.ascontiguousarray(np.load(aud_p))).float()
                aud = {"waveform": wav.unsqueeze(0), "sample_rate": int(meta["audio_rate"])}

            age = (time.time() - float(meta.get("saved", 0))) / 3600.0
            info = (f"segment {idx} restored from cache — {int(out.shape[0])} frame(s) "
                    f"{int(out.shape[2])}x{int(out.shape[1])}, stored {age:.1f} h ago")
            print(f"⚡ [Segment Cache] {info}\n     key: {key}")
            return (out, aud, True, info)

        # ── ecriture ─────────────────────────────────────────────────────────
        if images is None:
            raise ValueError(
                f"[Segment Cache] segment {idx}: nothing cached and no images arrived.\n"
                f"-> Connect the sampler's IMAGE output to `images`."
            )

        os.makedirs(self._dir(session), exist_ok=True)
        want = _PRECISION.get(precision, "uint8")
        src = images.detach().cpu()
        if want == "uint8":
            arr = (src.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
        else:
            arr = src.to(torch.float16).numpy()

        try:
            np.save(img_p, arr)
            rate = None
            if audio is not None and audio.get("waveform") is not None:
                wav = audio["waveform"]
                if wav.ndim == 3:
                    wav = wav[0]
                np.save(aud_p, wav.detach().cpu().float().numpy())
                rate = int(audio.get("sample_rate", 44100))

            meta = {
                "session": _safe(session, "chain"), "segment_index": idx, "key": key,
                "frames": int(images.shape[0]),
                "height": int(images.shape[1]), "width": int(images.shape[2]),
                "precision": want, "audio_rate": rate, "saved": time.time(),
                "cache_version": int(cache_version or 1),
            }
            # Ecrit en dernier : voir _hit(). Tant que ce fichier n'existe pas,
            # l'entree n'est pas consideree comme valide.
            tmp = meta_p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            os.replace(tmp, meta_p)
        except Exception as e:
            # Un cache qui echoue ne doit jamais faire perdre le rendu qu'on
            # vient de payer : on signale et on laisse passer les images.
            for p in (img_p, aud_p, meta_p):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            print(f"⚠️  [Segment Cache] could not store segment {idx} ({e}) — "
                  f"the render itself is fine and passes through.")
            return (images, audio, False, f"not cached: {e}")

        size_mb = os.path.getsize(img_p) / 1024 ** 2
        warn = ""
        if not (fingerprint_image is not None or fingerprint_text):
            warn = ("\n     ⚠️  no fingerprint connected: this entry is keyed on the segment "
                    "index alone.\n"
                    "        Change the source video without changing the session name and it "
                    "will be served anyway.")
        info = (f"segment {idx} cached — {int(images.shape[0])} frame(s) as {want}, "
                f"{size_mb:.0f} MB")
        print(f"💾 [Segment Cache] {info}\n     key: {key}{warn}")
        return (images, audio, False, info)


# ─────────────────────────────────────────────────────────────────────────────
# Entretien
# ─────────────────────────────────────────────────────────────────────────────

def clear_session(session: str) -> tuple:
    """Delete one session's cache. Returns (files removed, MB freed)."""
    path = OnyxSegmentCache._dir(session)
    if not os.path.isdir(path):
        return 0, 0.0
    total = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
    count = len(os.listdir(path))
    shutil.rmtree(path, ignore_errors=True)
    return count, total / 1024 ** 2


try:
    from server import PromptServer
    from aiohttp import web

    if not getattr(PromptServer.instance, "_onyx_segcache_routes", False):
        PromptServer.instance._onyx_segcache_routes = True

        @PromptServer.instance.routes.post("/onyx/segment_cache/clear")
        async def _sc_clear(request):
            try:
                body = await request.json()
                n, mb = clear_session(str(body.get("session", "chain")))
                return web.json_response({"success": True, "files": n, "mb": round(mb, 1)})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.get("/onyx/segment_cache/list")
        async def _sc_list(request):
            out = []
            if os.path.isdir(_CACHE_ROOT):
                for sess in sorted(os.listdir(_CACHE_ROOT)):
                    d = os.path.join(_CACHE_ROOT, sess)
                    if not os.path.isdir(d):
                        continue
                    files = os.listdir(d)
                    mb = sum(os.path.getsize(os.path.join(d, f)) for f in files) / 1024 ** 2
                    out.append({"session": sess,
                                "segments": len([f for f in files if f.endswith(".meta.json")]),
                                "mb": round(mb, 1)})
            return web.json_response({"success": True, "sessions": out})
except Exception:
    logging.warning("Onyx Segment Cache: routes not registered (server unavailable)")


NODE_CLASS_MAPPINGS = {"OnyxSegmentCache": OnyxSegmentCache}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxSegmentCache": "Onyx Segment Cache"}
