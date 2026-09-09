"""
Utilitaires de conversion image — Ru4ls / NanoBanana pack.
"""
import torch
import numpy as np
from PIL import Image
 
 
def tensor_to_pil(image_tensor) -> Image.Image | None:
    """Convertit un tensor PyTorch [B, H, W, C] en image PIL (premier frame)."""
    if image_tensor is None:
        return None
    return Image.fromarray(
        (image_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    )
 
 
def pil_to_tensor(pil_image) -> torch.Tensor | None:
    """Convertit une image PIL en tensor PyTorch [1, H, W, C]."""
    if pil_image is None:
        return None
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np)[None,]