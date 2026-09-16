import os
from datetime import datetime

import folder_paths
from PIL import Image
import numpy as np


from .onyx_render_profile import ensure_profile_ready
class SaveImageWithoutMetadata:
    """Base class — used internally by OnyxPreviewImageWithoutMetadata."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "IMG"}),
            },
            "optional": {
                "video": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Onyx/image"

    def save_images(self, images, filename_prefix="IMG", video="", prompt=None, extra_pnginfo=None):
        # ── Mode vidéo ───────────────────────────────────────────
        ensure_profile_ready()
        if video and video != "(non sauvegardé)" and os.path.isfile(video):
            ext = os.path.splitext(video)[1].lower()
            if ext in (".mp4", ".webm", ".gif"):
                return self._preview_video(video)

        # ── Mode image (comportement original) ───────────────────
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None  # No metadata saved

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            if len(images) > 1:
                file = f"{filename_prefix}_{timestamp}_{batch_number:03d}.png"
            else:
                file = f"{filename_prefix}_{timestamp}.png"

            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })

        return {"ui": {"images": results}}

    def _preview_video(self, video_path: str) -> dict:
        """Retourne un dict UI ComfyUI pour afficher un player vidéo natif."""
        try:
            tmp_dir = folder_paths.get_temp_directory()
            rel_sub = os.path.relpath(os.path.dirname(video_path), tmp_dir)
            if rel_sub == ".":
                rel_sub = ""
        except ValueError:
            rel_sub = ""

        # Détection fps via cv2 si disponible
        fps = 24.0
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            detected = cap.get(cv2.CAP_PROP_FPS)
            if detected and detected > 0:
                fps = detected
            cap.release()
        except Exception:
            pass

        filename = os.path.basename(video_path)
        return {
            "ui": {
                "gifs": [{
                    "filename":   filename,
                    "subfolder":  rel_sub,
                    "type":       "temp",
                    "format":     "video/h264-mp4",
                    "frame_rate": fps,
                }]
            }
        }


class OnyxPreviewImageWithoutMetadata(SaveImageWithoutMetadata):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "video": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Onyx/image"
