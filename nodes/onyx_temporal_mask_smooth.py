"""
Aiorbust Temporal Mask Smooth - ComfyUI custom node.

Stabilizes a MASK batch coming out of SAM3 video propagation before it is used
to composite or condition a video edit. Four stages, in order:

  1. temporal median  - kills single-frame masks that pop in from nowhere
  2. closing          - fills pinholes and reconnects broken edges
  3. dilate           - pushes the boundary outside the subject
  4. gaussian blur    - soft edge, so no hard seam survives the re-encode
"""

import torch
import torch.nn.functional as F

# Meme budget que le reste du pack (aiorbust_image_blur_batched.py) : F.pad /
# F.conv2d / F.max_pool2d indexent en 32 bits (2**31 elements max), et gardent
# plusieurs copies (entree, tenseur padde, sortie) en memoire a la fois. Sans
# ca, dilate/erode/blur traitaient TOUT le batch de masques en un seul appel -
# correct sur un clip court, mais "you tried to allocate 46581350400 bytes"
# des qu'un batch un peu long rencontre un rayon de flou/dilate un peu large.
from .onyx_render_profile import ensure_profile_ready
_SAFE_ELEMENTS = int((2 ** 31) * 0.90)
_TARGET_BYTES_PER_SLICE = 1024 ** 3  # 1 GiB par tranche


def _auto_slice_size(total: int, h: int, w: int, radius: int, frames_per_slice: int = 0) -> int:
    if frames_per_slice > 0:
        return max(1, min(frames_per_slice, total))
    padded = (h + 2 * radius) * (w + 2 * radius)
    by_elements = _SAFE_ELEMENTS // max(1, padded)
    by_bytes = _TARGET_BYTES_PER_SLICE // max(1, padded * 4)
    return max(1, min(total, by_elements, by_bytes))


def _temporal_median(masks: torch.Tensor, radius: int, chunk_size: int = 32) -> torch.Tensor:
    """
    Median filter along the frame axis. masks: [B, H, W].
    Edges are handled by clamping indices, so the first and last frames
    replicate rather than fade out.
    Processed in chunks - a full [B, window, H, W] stack blows up VRAM on long clips.
    """
    if radius < 1:
        return masks

    frame_count = masks.shape[0]
    window = torch.arange(-radius, radius + 1, device=masks.device)
    output = torch.empty_like(masks)

    for start in range(0, frame_count, chunk_size):
        end = min(start + chunk_size, frame_count)
        centers = torch.arange(start, end, device=masks.device).unsqueeze(1)
        indices = (centers + window.unsqueeze(0)).clamp(0, frame_count - 1)
        stack = masks[indices]                      # [chunk, window, H, W]
        output[start:end] = stack.median(dim=1).values

    return output


def _dilate(masks: torch.Tensor, radius: int, frames_per_slice: int = 0) -> torch.Tensor:
    """masks: [B, 1, H, W]. Chunked along B — one max_pool2d call over the
    whole batch is what produced the 46 GB allocation on a long clip."""
    if radius < 1:
        return masks
    size = 2 * radius + 1
    total, _, h, w = masks.shape
    slice_size = _auto_slice_size(total, h, w, radius, frames_per_slice)
    if slice_size >= total:
        return F.max_pool2d(masks, kernel_size=size, stride=1, padding=radius)
    out = []
    for start in range(0, total, slice_size):
        part = masks[start:start + slice_size]
        out.append(F.max_pool2d(part, kernel_size=size, stride=1, padding=radius))
    return torch.cat(out, dim=0)


def _erode(masks: torch.Tensor, radius: int, frames_per_slice: int = 0) -> torch.Tensor:
    if radius < 1:
        return masks
    return -_dilate(-masks, radius, frames_per_slice)


def _gaussian_blur(masks: torch.Tensor, radius: int, frames_per_slice: int = 0) -> torch.Tensor:
    """Separable gaussian. sigma derived from radius so one knob controls both.
    Chunked along B for the same reason as _dilate."""
    if radius < 1:
        return masks

    size = 2 * radius + 1
    sigma = radius / 2.0
    coords = torch.arange(size, dtype=masks.dtype, device=masks.device) - radius
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kh = kernel.view(1, 1, 1, size)
    kv = kernel.view(1, 1, size, 1)

    def _pass(chunk):
        blurred = F.conv2d(F.pad(chunk, (radius, radius, 0, 0), mode="replicate"), kh)
        blurred = F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode="replicate"), kv)
        return blurred

    total, _, h, w = masks.shape
    slice_size = _auto_slice_size(total, h, w, radius, frames_per_slice)
    if slice_size >= total:
        return _pass(masks)
    out = []
    for start in range(0, total, slice_size):
        out.append(_pass(masks[start:start + slice_size]))
    return torch.cat(out, dim=0)


class OnyxTemporalMaskSmooth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "temporal_radius": ("INT", {"default": 2, "min": 0, "max": 15}),
                "closing_radius": ("INT", {"default": 3, "min": 0, "max": 64}),
                "dilate_radius": ("INT", {"default": 10, "min": 0, "max": 128}),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 128}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Widgets ajoutes EN DERNIER, dans l'ordre ou ils ont ete ajoutes :
                # deplacer leur position casserait la deserialisation positionnelle
                # des workflows deja sauvegardes.
                "frames_per_slice": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "0 = auto : la plus grande tranche qui reste sous le budget "
                               "memoire (limite d'indexation 32 bits + budget de 1 GiB), "
                               "calculee depuis la resolution et les rayons de ce batch.\n"
                               "A baisser a la main seulement si le pic memoire est encore trop haut."}),
                "opening_radius": ("INT", {
                    "default": 2, "min": 0, "max": 64,
                    "tooltip": "Erode puis dilate (dans cet ordre), AVANT le closing. Supprime "
                               "les petits blobs isoles que la median temporelle n'a pas filtres "
                               "(rares faux positifs de detection SAM3) : un blob plus petit que "
                               "ce rayon disparait a l'erosion et ne revient pas au dilate — "
                               "contrairement au closing (dilate puis erode), qui reconnecte/comble "
                               "mais ne supprime rien. Augmente si des artefacts isoles persistent, "
                               "0 pour desactiver."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "Onyx"
    DESCRIPTION = ("Stabilizes a MASK batch from SAM3 video propagation: temporal median, "
                   "closing, dilate, gaussian blur — in that order. Processed in memory-bounded "
                   "chunks, so long clips don't blow up RAM/VRAM.")

    def process(self, masks, temporal_radius, closing_radius, dilate_radius, blur_radius,
                threshold, frames_per_slice=0, opening_radius=2):
        ensure_profile_ready()
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)

        total, h, w = int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2])
        widest_radius = max(closing_radius, dilate_radius, blur_radius, opening_radius, 1)
        print(f"[Aiorbust Temporal Mask Smooth] {total} frame(s) at {w}x{h} — "
              f"processing dilate/erode/blur in memory-bounded chunks "
              f"({'auto' if frames_per_slice <= 0 else 'manual=' + str(frames_per_slice)}, "
              f"widest radius {widest_radius}).")
        if temporal_radius >= 3:
            print(f"   ↳ temporal_radius={temporal_radius} melange {2 * temporal_radius + 1} "
                  f"frames par position : sur un sujet qui bouge vite, ca peut donner un effet "
                  f"de mouvement lisse/interpole. Baisse-le (1, voire 0) si le masque semble "
                  f"'glisser' entre les positions plutot que suivre le mouvement reel.")

        working = masks.float()

        # Binarize first. SAM3 confidence gradients make the median unstable,
        # and morphology on soft values smears the boundary.
        working = (working > threshold).float()

        working = _temporal_median(working, temporal_radius)

        # Opening = erode puis dilate, AVANT le closing. Un blob isole plus
        # petit que opening_radius disparait a l'erosion et ne revient pas au
        # dilate : c'est ce qui manquait pour nettoyer les rares faux positifs
        # de detection SAM3 que la mediane temporelle seule ne filtre pas.
        # Le closing (dilate puis erode, ci-dessous) fait l'inverse : il
        # reconnecte/comble, il ne supprime jamais un blob isole.
        working = _dilate(_erode(working.unsqueeze(1), opening_radius, frames_per_slice),
                           opening_radius, frames_per_slice).squeeze(1)

        # Closing = dilate then erode. Fills holes without growing the shape.
        working = _erode(_dilate(working.unsqueeze(1), closing_radius, frames_per_slice),
                          closing_radius, frames_per_slice).squeeze(1)

        working = _dilate(working.unsqueeze(1), dilate_radius, frames_per_slice).squeeze(1)
        working = _gaussian_blur(working.unsqueeze(1), blur_radius, frames_per_slice).squeeze(1)

        return (working.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {"OnyxTemporalMaskSmooth": OnyxTemporalMaskSmooth}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxTemporalMaskSmooth": "Aiorbust Temporal Mask Smooth"}
