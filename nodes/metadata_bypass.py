"""
Onyx Metadata Bypass
Strips all metadata (EXIF, XMP, PNG chunks, ICC, C2PA) from one or more images
and returns clean IMAGE tensors. No settings -- just plug it in before Save Image.
"""

import io
import numpy as np
import torch
from PIL import Image


class OnyxMetadataBypassNode:
    """Pass-through node that strips all metadata from images and returns clean tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
    OUTPUT_NODE   = False
    FUNCTION      = "run"
    CATEGORY      = "Onyx/Automation"

    def run(self, images):
        cleaned = []
        for i in range(images.shape[0]):
            # Tensor [H, W, 3] float32 0-1 → uint8 PIL
            np_img = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img, mode="RGB")

            # Round-trip through PNG in memory — strips all metadata chunks
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            clean = Image.open(buf).convert("RGB")

            # Back to float32 tensor [H, W, 3]
            clean_np = np.array(clean).astype(np.float32) / 255.0
            cleaned.append(torch.from_numpy(clean_np))

        batch = torch.stack(cleaned, dim=0)
        print(f"[Metadata Bypass] Cleaned {len(cleaned)} image(s).")
        return (batch,)
