# -*- coding: utf-8 -*-
"""
ComfyUI node - Onyx Save Video (no metadata).

Saves a video that already exists as a FILE to ComfyUI's output folder, stripped
of every container tag, without touching a single frame.

Why it takes a path and not an IMAGE batch: the AIO node's `video` output is the
mp4 the API returned. Re-encoding it from frames would throw away the only clean
copy - the API's own bitstream - and add a generation of compression that no
setting can recover. So this node copies the bitstream and rewrites only the
container.

Two levels, and the difference matters:

- remux (default): PyAV opens the source, creates a NEW container, and copies
  the encoded packets across untouched. The output is built from nothing but the
  streams, so no tag can survive by being in a box nobody thought to clear. This
  is the only method that is exhaustive by construction rather than by a list.
- byte copy: the file is copied verbatim, then `udta` and the `hdlr` vendor
  fields are cleared in place. Keeps the original container byte-for-byte
  elsewhere. Use it if a downstream tool is picky about the muxer.

Neither re-encodes. The frames in the output are bit-identical to the input's.
"""

import os
import shutil
import time
import logging

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Une seule implementation du nettoyage binaire pour tout le pack : celle de
# l'AIO. La dupliquer ici garantirait qu'un jour l'une des deux nettoie une
# boite que l'autre laisse passer.
try:
    from .nano_banana_aio import _mp4_strip_metadata
except Exception:
    _mp4_strip_metadata = None


_METHODS = {
    "remux (rebuild container, most thorough)": "remux",
    "byte copy + strip tags": "copy",
}


def _resolve(video, video_path):
    """The single source path, from whichever input is connected."""
    for candidate in (video, video_path):
        if candidate is None:
            continue
        text = str(candidate).strip().strip('"')
        if not text or text == "(not saved)":
            continue
        return text
    return None


def _strip_in_place(path, tag="[Save Video]"):
    """Clear udta / hdlr vendor on an mp4 already on disk. Returns True if changed."""
    if _mp4_strip_metadata is None:
        print(f"⚠️  {tag} binary strip unavailable (nano_banana_aio not importable) — "
              f"container tags left as the muxer wrote them.")
        return False
    try:
        with open(path, "rb") as fh:
            data = bytearray(fh.read())
        if _mp4_strip_metadata(data, 0, len(data)):
            with open(path, "wb") as fh:
                fh.write(bytes(data))
            return True
        return False
    except Exception as e:
        print(f"⚠️  {tag} binary strip failed: {e}")
        return False


def _remux(src, dst, tag="[Save Video]"):
    """Stream-copy src into a fresh container at dst. No re-encode.

    Raises on failure so the caller can fall back to a byte copy rather than
    silently leaving a truncated file behind.
    """
    import av

    inp = av.open(src)
    out = av.open(dst, mode="w")
    try:
        mapping = {}
        # Tous les flux sont declares avant le premier mux : le conteneur ecrit
        # son en-tete au premier paquet, et un flux ajoute ensuite n'y figure pas.
        for stream in inp.streams:
            if stream.type not in ("video", "audio"):
                # Les pistes de donnees et de sous-titres sont precisement la ou
                # se logent les tags d'appareil et de logiciel. Ne pas les
                # recopier fait partie du travail.
                continue
            # add_stream_from_template n'existe qu'a partir de PyAV 12 ; avant,
            # le meme service passait par le mot-cle template. Une seule des
            # deux formes existe selon la version installee.
            if hasattr(out, "add_stream_from_template"):
                new = out.add_stream_from_template(stream)
            else:
                new = out.add_stream(template=stream)
            # add_stream_from_template recopie aussi le dictionnaire de
            # metadonnees du flux ; on le vide.
            try:
                new.metadata.clear()
            except Exception:
                pass
            mapping[stream.index] = new

        if not mapping:
            raise ValueError("the source has no video or audio stream to copy.")

        try:
            out.metadata.clear()
        except Exception:
            pass

        copied = 0
        for packet in inp.demux(list(mapping)):
            # Un paquet de vidage (dts nul) marque la fin d'un flux et n'a rien
            # a muxer.
            if packet.dts is None:
                continue
            packet.stream = mapping[packet.stream.index]
            out.mux(packet)
            copied += 1
        return copied, len(mapping)
    finally:
        out.close()
        inp.close()


class OnyxSaveVideoNoMetadata:
    """Copy a video file to output/ with every container tag removed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "onyx", "multiline": False,
                    "tooltip": "Subfolders are allowed: 'client/janvier' writes to "
                               "output/client/janvier/."}),
                "method": (list(_METHODS), {
                    "default": "remux (rebuild container, most thorough)",
                    "tooltip": "remux: rebuilds the container from the streams alone, so no tag "
                               "survives in a box that was not on a list.\n"
                               "byte copy: keeps the original container and clears udta + hdlr "
                               "vendor in place.\n"
                               "Neither one re-encodes — the frames are bit-identical either way."}),
            },
            "optional": {
                "video": ("AB_VIDEO", {
                    "tooltip": "From Video Loader or Images to Video."}),
                "video_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "From the AIO node's `video` output, which is a path string.\n"
                               "Connect this one, not `images` — going through the frames would "
                               "re-encode and add a generation of compression."}),
                "add_timestamp": ("BOOLEAN", {
                    "default": True,
                    "label_on": "prefix_00001_1699999999.mp4",
                    "label_off": "prefix_00001.mp4",
                    "tooltip": "On by default: two runs a second apart cannot collide."}),
            },
        }

    RETURN_TYPES = ("AB_VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Onyx"
    DESCRIPTION = ("Save a video FILE to output/ with all container metadata removed. "
                   "Copies the bitstream — never re-encodes.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Le node doit re-tourner a chaque queue : sinon un second rendu, dont
        # le fichier temporaire porte un autre nom, ne serait jamais enregistre
        # parce que les widgets, eux, n'ont pas change.
        return float("nan")

    def save(self, filename_prefix, method, video=None, video_path=None,
             add_timestamp=True):
        tag = "[Save Video]"
        src = _resolve(video, video_path)

        if src is None:
            raise ValueError(
                f"{tag} no video connected.\n"
                f"-> Connect the AIO node's `video` output to `video_path`, or an "
                f"AB_VIDEO source to `video`.\n"
                f"   `images` is the wrong output here: it would have to be re-encoded.")
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"{tag} the file no longer exists:\n   {src}\n"
                f"-> The AIO node writes to ComfyUI's temp folder, which is emptied on "
                f"restart. Re-run the generation, then save in the same session.")

        out_root = (folder_paths.get_output_directory() if folder_paths
                    else os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "_output"))

        prefix = (filename_prefix or "onyx").strip().replace("\\", "/")
        subdir, _, stem = prefix.rpartition("/")
        stem = "".join(c for c in stem if c.isalnum() or c in "-_") or "onyx"
        # Les segments de dossier sont filtres un par un : sans cela un ".." dans
        # le prefixe ecrirait hors de output/.
        parts = [p for p in subdir.split("/") if p and p not in (".", "..")]
        dest_dir = os.path.join(out_root, *parts) if parts else out_root
        os.makedirs(dest_dir, exist_ok=True)

        existing = [f for f in os.listdir(dest_dir)
                    if f.startswith(stem + "_") and f.lower().endswith(".mp4")]
        counter = len(existing) + 1
        ext = os.path.splitext(src)[1].lower() or ".mp4"
        name = f"{stem}_{counter:05d}"
        if add_timestamp:
            name += f"_{int(time.time())}"
        dest = os.path.join(dest_dir, name + ext)

        mode = _METHODS[method]
        used = mode
        if mode == "remux":
            try:
                packets, streams = _remux(src, dest, tag)
                print(f"🎬 {tag} remuxed {packets} packet(s) across {streams} stream(s) — "
                      f"container rebuilt, no metadata carried over.")
            except Exception as e:
                # Un remux rate laisse un fichier partiel : il doit disparaitre
                # avant la reprise, sinon on garderait une video tronquee sous
                # un nom qui annonce un succes.
                if os.path.isfile(dest):
                    try:
                        os.remove(dest)
                    except Exception:
                        pass
                print(f"⚠️  {tag} remux failed ({e}) — falling back to a byte copy.")
                used = "copy"

        if used == "copy":
            shutil.copy2(src, dest)
            changed = _strip_in_place(dest, tag)
            print(f"🎬 {tag} byte copy — "
                  + ("udta + hdlr vendor cleared." if changed else "no tag found to clear."))
        else:
            # Le muxeur mp4 de PyAV ecrit son propre identifiant d'encodeur dans
            # hdlr : le conteneur est neuf, mais pas anonyme pour autant.
            _strip_in_place(dest, tag)

        size_mb = os.path.getsize(dest) / 1024 ** 2
        src_mb = os.path.getsize(src) / 1024 ** 2
        print(f"     → {dest}\n"
              f"     {size_mb:.1f} MB (source {src_mb:.1f} MB) — no frame re-encoded.")
        logging.info("Onyx Save Video (no metadata): %s", dest)

        return (dest, dest)


NODE_CLASS_MAPPINGS = {"OnyxSaveVideoNoMetadata": OnyxSaveVideoNoMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxSaveVideoNoMetadata": "Onyx Save Video (no metadata)"}
