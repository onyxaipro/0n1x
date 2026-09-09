"""
ComfyUI node — onyx Directory Image Loader.
Loads images from a directory one at a time, auto-incrementing through all files.
Designed for batch workflows like Seedream 4.5 Edit.
"""

import os
import logging

import numpy as np
import torch
from PIL import Image, ImageOps
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


try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/directory_loader/list_folders")
    async def _list_folders(request):
        return web.json_response(_list_input_subdirs())

    @PromptServer.instance.routes.get("/directory_loader/status")
    async def _get_status(request):
        """Return current counter state for the frontend to poll."""
        return web.json_response(OnyxDirectoryImageLoaderNode._counters)

    @PromptServer.instance.routes.post("/directory_loader/reset_batch")
    async def _reset_batch(request):
        """Reset batch counters so a failed batch doesn't leave stale state."""
        OnyxDirectoryImageLoaderNode._batch_runs.clear()
        return web.json_response({"ok": True})
except Exception:
    pass


class OnyxDirectoryImageLoaderNode:
    """Load images from a directory one at a time.

    Set run_count to how many images to process.
    Queue once — it auto-queues the rest via frontend JS.
    """

    _counters = {}  # {directory: index} — which image to load next
    _batch_runs = {}  # {directory: count} — how many runs done in current batch

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_absolute_path": ("BOOLEAN", {"default": False}),
                "run_count": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
            },
            "optional": {
                "input_folder": (_list_input_subdirs(), {"default": _list_input_subdirs()[0]}),
                "absolute_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "filename", "filepath", "current_index", "total_count")
    FUNCTION = "load_image"
    OUTPUT_NODE = True
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load_image(self, use_absolute_path, run_count, input_folder=None, absolute_path=""):
        if use_absolute_path:
            directory = absolute_path.strip()
            if not directory:
                raise RuntimeError("Directory Image Loader: absolute path is required.")
        else:
            if not input_folder or input_folder == "(no folders found)":
                raise RuntimeError("Directory Image Loader: no input folder selected. Create a subfolder in ComfyUI/input/.")
            directory = os.path.join(folder_paths.get_input_directory(), input_folder)

        if not os.path.isdir(directory):
            raise RuntimeError(f"Directory Image Loader: '{directory}' is not a valid directory.")

        files = []
        for f in os.listdir(directory):
            ext = os.path.splitext(f)[1].lower()
            if ext in _SUPPORTED_EXTENSIONS:
                files.append(f)

        if not files:
            raise RuntimeError(f"Directory Image Loader: no images found in '{directory}'.")

        files.sort(key=str.lower)
        total = len(files)
        actual_run_count = min(run_count, total)

        key = directory

        # Track which image index we're on
        if key not in OnyxDirectoryImageLoaderNode._counters:
            OnyxDirectoryImageLoaderNode._counters[key] = 0
        current = OnyxDirectoryImageLoaderNode._counters[key] % total
        OnyxDirectoryImageLoaderNode._counters[key] = current + 1

        # Track how many runs done in this batch
        if key not in OnyxDirectoryImageLoaderNode._batch_runs:
            OnyxDirectoryImageLoaderNode._batch_runs[key] = 0
        OnyxDirectoryImageLoaderNode._batch_runs[key] += 1
        batch_done = OnyxDirectoryImageLoaderNode._batch_runs[key]
        remaining = actual_run_count - batch_done

        filename = files[current]
        filepath = os.path.join(directory, filename)

        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]

        logging.info("Directory Image Loader: [%d/%d] %s (batch %d/%d, %d remaining)",
                     current + 1, total, filename, batch_done, actual_run_count, max(0, remaining))

        # Reset batch counter when done
        if remaining <= 0:
            OnyxDirectoryImageLoaderNode._batch_runs[key] = 0

        return {"ui": {"current_index": [current], "total_count": [total],
                       "run_count": [actual_run_count], "remaining": [max(0, remaining)],
                       "filename": [filename]},
                "result": (tensor, filename, filepath, current, total)}
