# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - Onyx Images to Video (AB_VIDEO).

Turns an IMAGE batch into an .mp4 on disk and returns its path as AB_VIDEO, the
type `Onyx Image & Video Edit AIO` expects on `video_reference`.

Why it is needed: AB_VIDEO is a file path, not a tensor. Everything the graph
produces - a SAM3 composite, a masked clip, a chained render - lives as an IMAGE
batch, so there was no way to send it as a reference video without saving it by
hand, then picking it back up with the Video Loader on the next run. This node
closes that loop inside a single execution.

Encoding notes that are not optional:
- H.264 with yuv420p refuses odd dimensions. The frame is padded by one pixel of
  edge replication rather than cropped, so nothing at the border is lost.
- The batch is written frame by frame. Building the whole file in memory would
  reproduce the very allocation failures that batching was meant to avoid.
"""

import os
import time
import logging
from fractions import Fraction

import numpy as np
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None


# Chaque aller-retour vers une API re-compresse : encoder -> uploader ->
# generer -> telecharger -> decoder -> re-encoder. Les pertes s'additionnent et
# ne se retirent jamais. Sur une video qui sert d'ENTREE a un modele, la taille
# du fichier ne coute qu'un peu d'upload, alors que la degradation, elle, reste
# dans le resultat.
_QUALITY = {
    "lossless (crf 0, 4:4:4)": 0,
    "near-lossless (crf 10)": 10,
    "high (crf 17)": 17,
    "good (crf 20)": 20,
    "medium (crf 23)": 23,
    "small file (crf 28)": 28,
}

# Au-dela de cette qualite, le plafond de debit devient le facteur limitant et
# annule le gain : on le retire pour laisser crf decider seul.
_UNCAPPED_BELOW_CRF = 17


def _even(tensor):
    """Pad to even width/height by replicating the last row/column.

    yuv420p subsamples chroma by two, so an odd side cannot be encoded at all.
    Replicating the edge keeps the framing intact; cropping would shift it by a
    pixel, which matters when the clip is used as a motion reference against
    another clip that was not cropped.
    """
    h, w = int(tensor.shape[1]), int(tensor.shape[2])
    if h % 2:
        tensor = torch.cat([tensor, tensor[:, -1:, :, :]], dim=1)
    if w % 2:
        tensor = torch.cat([tensor, tensor[:, :, -1:, :]], dim=2)
    return tensor


def _audio_array(audio, duration_s):
    """ComfyUI AUDIO -> (float32 (C, N), sample_rate, layout), or None."""
    if audio is None:
        return None
    wav = audio.get("waveform")
    sr = int(audio.get("sample_rate", 44100))
    if wav is None or int(wav.numel()) <= 1:
        return None

    if wav.ndim == 3:      # (B, C, N) -> premier element
        wav = wav[0]
    if wav.ndim == 1:      # (N,) -> (1, N)
        wav = wav.unsqueeze(0)
    wav = wav.detach().cpu().float()

    if int(wav.shape[0]) > 2:
        wav = wav[:2]
    layout = "stereo" if int(wav.shape[0]) == 2 else "mono"

    # L'audio plus long que l'image ferait un fichier dont la duree ne
    # correspond pas au nombre de frames, ce qui deroute les APIs qui lisent la
    # duree du conteneur pour facturer ou valider.
    max_samples = int(round(duration_s * sr))
    if max_samples > 0 and int(wav.shape[1]) > max_samples:
        wav = wav[:, :max_samples]

    return wav.clamp(-1.0, 1.0).numpy().astype(np.float32), sr, layout


def _write_audio(container, stream, arr, sr, layout):
    import av

    chunk = int(getattr(stream, "frame_size", 0) or 1024)
    pts = 0
    for start in range(0, arr.shape[1], chunk):
        block = np.ascontiguousarray(arr[:, start:start + chunk])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
        frame.sample_rate = sr
        frame.time_base = Fraction(1, sr)
        frame.pts = pts
        pts += block.shape[1]
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def _duplicate_report(images, fps):
    """Count consecutive frames that are identical, and say what that implies.

    Judder in a file whose container rate is correct almost always means the
    batch itself repeats frames: a clip read at a rate higher than its own is
    padded by duplication, so 30 containers-frames may carry only 24 distinct
    images per second. That is invisible in the metadata and obvious here.

    Compared on a strided subsample - a duplicate is exact, so there is no need
    to look at every pixel to find one.
    """
    total = int(images.shape[0])
    if total < 3:
        return ""

    small = images[:, ::8, ::8, :]
    arr = small.cpu().numpy().reshape(total, -1)
    diffs = np.abs(np.diff(arr, axis=0)).mean(axis=1)
    dup = np.flatnonzero(diffs < 1e-6)
    if dup.size == 0:
        return ""

    ratio = 1.0 - dup.size / float(total - 1)
    effective = fps * ratio
    listed = ", ".join(str(int(i) + 1) for i in dup[:12]) + (" ..." if dup.size > 12 else "")
    return (f"\n⚠️  {dup.size} duplicate frame(s) out of {total - 1} transitions "
            f"(after frames {listed})\n"
            f"     → only ~{effective:.2f} distinct image(s) per second at {fps:g} fps. "
            f"The batch was rate-converted upstream;\n"
            f"       re-encoding cannot undo it. Load the clip at its native rate instead "
            f"of forcing one.")


class OnyxImagesToVideo:
    """Encode an IMAGE batch to .mp4 and return its path as AB_VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "quality": (list(_QUALITY), {"default": "high (crf 17)"}),
                "filename_prefix": ("STRING", {"default": "ab_video", "multiline": False}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Muxed as an AAC track. Trimmed to the video's "
                                               "own duration so the container length matches "
                                               "the frame count."}),
                "save_to_output": ("BOOLEAN", {
                    "default": False,
                    "label_on": "keep it in output/",
                    "label_off": "temporary file",
                    "tooltip": "Off: written to ComfyUI's temp folder — it is only a hand-off to "
                               "the API node.\nOn: written to output/ so you can keep or inspect it."}),
            },
        }

    RETURN_TYPES = ("AB_VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "encode"
    CATEGORY = "Onyx"

    def encode(self, images, fps, quality, filename_prefix,
               audio=None, save_to_output=False):
        ensure_profile_ready()
        import av

        total = int(images.shape[0])
        if total == 0:
            raise ValueError("[Images to Video] the image batch is empty.")

        images = _even(images)
        height, width = int(images.shape[1]), int(images.shape[2])
        fps = fps if fps > 0 else 24.0

        if folder_paths is None:
            base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
        else:
            base = (folder_paths.get_output_directory() if save_to_output
                    else folder_paths.get_temp_directory())
        os.makedirs(base, exist_ok=True)

        safe = "".join(c for c in (filename_prefix or "ab_video") if c.isalnum() or c in "-_") or "ab_video"
        path = os.path.join(base, f"{safe}_{int(time.time() * 1000)}.mp4")

        # PyAV veut un entier ou une Fraction ici, jamais un float : un float
        # leve "'float' object has no attribute 'numerator'" au moment d'ouvrir
        # le flux, bien avant la premiere frame.
        rate = Fraction(fps).limit_denominator(10000)
        container = av.open(path, mode="w")
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        # yuv420p sous-echantillonne la chrominance d'un facteur 2 : meme a crf 0
        # la couleur perd la moitie de sa resolution. Un encodage reellement sans
        # perte impose donc 4:4:4, pas seulement un crf bas.
        stream.pix_fmt = "yuv444p" if _QUALITY[quality] == 0 else "yuv420p"
        # Un pts explicite par frame, exprime dans une time_base de 1/fps posee
        # sur la FRAME. La poser sur le flux fait rejeter le muxeur mp4, qui
        # impose la sienne ; le conteneur reechelonne ensuite tout seul.
        frame_tb = Fraction(1, 1) / rate
        # Une image-cle par seconde. Le defaut de x264 en pose une toutes les 250
        # frames, soit une toutes les 8 s a 30 fps : un lecteur qui se repositionne
        # doit alors redecoder jusqu'a 8 s pour afficher une frame, ce qui se voit
        # comme une lecture qui accroche alors que les timestamps sont justes.
        stream.gop_size = max(1, int(round(fps)))
        _crf = _QUALITY[quality]
        stream.options = {
            "crf": str(_crf),
            "preset": "medium",
            "profile": "high",
        }
        if _crf == 0:
            # High 4:4:4 Predictive. Tous les lecteurs ne le prennent pas, et
            # beaucoup d'APIs le refusent ou le retranscodent - ce qui annulerait
            # tout le benefice. A reserver aux fichiers qui restent en local.
            stream.options["profile"] = "high444"
            stream.options["qp"] = "0"
        if _crf >= _UNCAPPED_BELOW_CRF:
            # Repartit le debit au lieu de laisser des pics sur les plans charges,
            # ce qui est ce qui met un lecteur en difficulte sur de la video
            # verticale en 1080. Inutile en quasi-sans-perte, ou il ne ferait que
            # brider la qualite qu'on cherche justement a preserver.
            stream.options["x264-params"] = "vbv-maxrate=20000:vbv-bufsize=40000"

        # Les DEUX flux sont declares avant la premiere frame. Le conteneur
        # ecrit son en-tete au premier mux(), et un flux ajoute apres coup n'y
        # figure pas : le muxage audio echoue alors sur "Cannot rebase to zero
        # time", une erreur qui ne dit rien de sa cause.
        prepared = _audio_array(audio, total / fps)
        astream = None
        if prepared is not None:
            arr_a, sr_a, layout_a = prepared
            astream = container.add_stream("aac", rate=sr_a)
            astream.layout = layout_a
            astream.time_base = Fraction(1, sr_a)

        for i in range(total):
            arr = (images[i].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = i
            frame.time_base = frame_tb
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

        if astream is not None:
            _write_audio(container, astream, arr_a, sr_a, layout_a)
            audio_note = f"{arr_a.shape[1] / sr_a:.2f}s of {layout_a} audio"
        else:
            audio_note = "no audio"
        container.close()

        size_mb = os.path.getsize(path) / 1024 ** 2
        if _crf == 0:
            print("     ⚠️  lossless 4:4:4 — keep this one local. Most upload APIs either "
                  "refuse it or transcode it, which throws the benefit away.")
        print(f"🎬 [Images to Video] {total} frame(s) {width}x{height} @ {fps:g} fps "
              f"= {total / fps:.3f}s, {audio_note}, {size_mb:.1f} MB\n"
              f"     → {path}"
              + _duplicate_report(images, fps))
        logging.info("Onyx Images to Video: %s", path)
        return (path, path)
