# -*- coding: utf-8 -*-
"""
Onyx Save Image No Metadata
Saves images exactly like ComfyUI's native SaveImage node but without
embedding the workflow JSON or prompt metadata into the file.
Supports PNG and JPEG output.
"""

import os
import logging
import numpy as np
from PIL import Image
import folder_paths


class OnyxSaveImageNoMetadataNode:

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for saved filenames. Supports ComfyUI date tokens like %date:yyyy-MM-dd%.",
                }),
                "format": (["PNG", "JPEG"], {
                    "default": "PNG",
                    "tooltip": "Output format. PNG is lossless. JPEG is smaller but lossy.",
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPEG quality (1-100). Ignored for PNG.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Onyx/Image"

    def save_images(self, images, filename_prefix="ComfyUI", format="PNG", quality=95):
        # Resolve output path and counter (same logic as native SaveImage)
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                images[0].shape[1],
                images[0].shape[0],
            )

        results = []
        for batch_idx in range(images.shape[0]):
            # Tensor [H, W, 3] float32 0-1 -> uint8 PIL
            np_img = (images[batch_idx].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img, mode="RGB")

            if format == "JPEG":
                ext = "jpg"
                file = f"{filename}_{counter:05}_.{ext}"
                pil_img.save(
                    os.path.join(full_output_folder, file),
                    "JPEG",
                    quality=quality,
                    optimize=True,
                )
            else:
                ext = "png"
                file = f"{filename}_{counter:05}_.{ext}"
                # No pnginfo= argument -> zero metadata embedded
                pil_img.save(
                    os.path.join(full_output_folder, file),
                    compress_level=self.compress_level,
                )

            logging.info("[Onyx SaveNoMeta] Saved %s/%s", subfolder or "output", file)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1

        return {"ui": {"images": results}}
