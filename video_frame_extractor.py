"""
ComfyUI node - Onyx Video Frame Extractor.

Outputs one frame from a video: either a file picked in ComfyUI's input folder,
or a frame sequence arriving on `video_frames` — which is what lets a batch of
videos drive it without choosing a file by hand for each one.

Deliberately plain. An earlier version carried a JS extension with an on-node
gallery, an Extract button and an animated theme; it wrote preview PNGs to temp
on every run, forced its own size, and grew a widget row at each restart. None of
that helped pick a frame index, so it is gone. What remains is a node that reads
a video and returns a frame.
"""

import os
import logging
import random

import av
import numpy as np
import torch
from PIL import Image
import folder_paths

from .nodes.onyx_render_profile import ensure_profile_ready
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _list_video_files():
    """List video files in ComfyUI's input folder."""
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in _VIDEO_EXTENSIONS and os.path.isfile(os.path.join(input_dir, f)):
            files.append(f)
    files.sort(key=str.lower)
    if not files:
        files = ["(no videos found)"]
    return files


def _decode_selection(video_path, start_second, frame_count, frame_interval):
    """Decode the sampled frames as a list of (H,W,3) float tensors.

    Stops as soon as `frame_count` frames are held: on a long clip there is no
    reason to walk to the end when only the first sampled frames can be selected.
    """
    container = av.open(video_path, mode="r")
    stream = container.streams.video[0]

    start_pts = 0
    if start_second > 0:
        start_pts = int(start_second / stream.time_base)
        container.seek(start_pts, stream=stream)

    tensors = []
    idx = 0
    for frame in container.decode(stream):
        if start_second > 0 and frame.pts is not None and frame.pts < start_pts:
            continue
        if idx % frame_interval == 0:
            arr = frame.to_ndarray(format="rgb24")
            tensors.append(torch.from_numpy(arr.copy()).float() / 255.0)
        idx += 1
        if len(tensors) >= frame_count:
            break

    container.close()
    return tensors


class OnyxVideoFrameExtractorNode:
    """Extract one frame from a video file or from a connected frame sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_list_video_files(), {"video_upload": True}),
                "start_second": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1}),
                "frame_count": ("INT", {"default": 10, "min": 1, "max": 9999, "step": 1,
                                        "tooltip": "How many frames are sampled, and therefore how "
                                                   "far selected_frame can reach."}),
                "frame_interval": ("INT", {"default": 1, "min": 1, "max": 300, "step": 1,
                                           "tooltip": "Take one frame every N. 1 samples every frame."}),
                "selected_frame": ("INT", {"default": 0, "min": 0, "max": 9998, "step": 1,
                                           "tooltip": "Which sampled frame to output. 0 with "
                                                      "start_second=0 gives the very first frame."}),
            },
            "optional": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Connect the `video` output of Onyx Image and Video Batch Loader.\n"
                               "When connected it REPLACES the file dropdown entirely — nothing is "
                               "read from disk."}),
                "video_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate of the connected frames, so start_second can be turned "
                               "into a frame index. Left at 0, start_second is ignored."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract"
    CATEGORY = "image"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        # Le placeholder n'est pas une erreur : le node peut etre pilote par
        # video_frames, et VALIDATE_INPUTS ne sait pas si une entree est branchee.
        # Le controle reel est dans extract(), ou l'information existe.
        if video == "(no videos found)":
            return True
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True

    def extract(self, video, start_second=0.0, frame_count=10, frame_interval=1,
                selected_frame=0, video_frames=None, video_fps=0.0):

        # ── Frames deja decodees ──────────────────────────────────────────────
        # Prioritaire sur le menu fichier : quand un loader alimente ce node, le
        # menu pointe encore sur un fichier arbitraire, et lire celui-la donnerait
        # un resultat silencieusement faux.
        ensure_profile_ready()
        if video_frames is not None and int(video_frames.shape[0]) > 0:
            total = int(video_frames.shape[0])

            start_idx = 0
            if start_second > 0 and video_fps > 0:
                start_idx = min(int(round(start_second * video_fps)), total - 1)
            elif start_second > 0:
                print("[Video Frame Extractor] start_second ignored — connect video_fps "
                      "so seconds can be converted to a frame index.")

            picked = list(range(start_idx, total, max(1, frame_interval)))[:frame_count]
            if not picked:
                raise RuntimeError(
                    f"[Video Frame Extractor] start_second={start_second} lands past the end "
                    f"of a {total}-frame clip."
                )

            idx = min(selected_frame, len(picked) - 1)
            print(f"[Video Frame Extractor] connected frames: {total} in, "
                  f"{len(picked)} sampled, outputting source frame #{picked[idx]}")
            return (video_frames[picked[idx]].unsqueeze(0),)

        # ── Fichier ───────────────────────────────────────────────────────────
        if video == "(no videos found)":
            raise RuntimeError(
                "[Video Frame Extractor] No video selected and nothing connected to "
                "video_frames.\n-> Upload a video, or connect the `video` output of "
                "Onyx Image and Video Batch Loader."
            )

        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.isfile(video_path):
            raise RuntimeError(f"[Video Frame Extractor] File not found: {video_path}")

        tensors = _decode_selection(video_path, start_second, frame_count, frame_interval)
        if not tensors:
            raise RuntimeError(
                f"[Video Frame Extractor] No frame decoded from {os.path.basename(video_path)} "
                f"at start_second={start_second}."
            )

        idx = min(selected_frame, len(tensors) - 1)
        logging.info("Video Frame Extractor: frame %d/%d from %s",
                     idx + 1, len(tensors), os.path.basename(video_path))
        return (tensors[idx].unsqueeze(0),)
