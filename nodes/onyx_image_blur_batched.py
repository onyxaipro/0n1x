# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - Onyx Image Blur (batched).

Drop-in replacement for ComfyUI's ImageBlur on long video batches.

Why it exists: ImageBlur pads with F.pad(..., 'reflect'), and PyTorch's padding
kernels index in 32 bits. A tensor of more than 2**31 elements is refused
outright, whatever the amount of VRAM available:

    RuntimeError: input tensor must fit into 32-bit index math

At 768x1344 RGB that ceiling lands at 693 frames - a 29-second clip at 24 fps.
Above it, ImageBlur cannot run at all.

This node slices the batch, blurs each slice and concatenates. The blur itself
is NOT reimplemented here: the node looks up ComfyUI's own ImageBlur class and
calls it per slice, so the result is bit-identical and `sigma` keeps exactly the
meaning it has in the stock node. Reimplementing a "Gaussian" here would have
silently changed what sigma means, and every saved workflow with it.
"""

import logging

import torch

# La limite est sur le nombre d'elements, pas sur les octets. On garde 10 % de
# marge : le tenseur PADDE est plus grand que l'entree, et c'est lui qui doit
# tenir aussi.
_INDEX_LIMIT = 2 ** 31
_SAFE_ELEMENTS = int(_INDEX_LIMIT * 0.90)

# Une tranche vise ~1 GiB par tenseur. Blur et composite en manipulent plusieurs
# a la fois - entree, tenseur intermediaire, sortie - donc le pic reel vaut
# plusieurs fois cette valeur, et viser plus haut relance l'OOM qu'on evite.
_TARGET_BYTES_PER_SLICE = 1024 ** 3


def _stock_callable(node_key="ImageBlur"):
    """A stock ComfyUI node class + its method, resolved from the node registry.

    Going through NODE_CLASS_MAPPINGS rather than importing comfy_extras
    directly: the module path and the method name have both moved between
    ComfyUI versions (blur() in the v1 schema, execute() in v3), while the
    registry key "ImageBlur" has not.
    """
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get(node_key)
    if cls is None:
        raise RuntimeError(
            f"[Onyx batched] ComfyUI's {node_key} node was not found in "
            f"NODE_CLASS_MAPPINGS. This node borrows its maths, so it cannot run "
            f"without it."
        )

    for name in ("execute", getattr(cls, "FUNCTION", None)):
        if name and hasattr(cls, name):
            return cls, getattr(cls, name)

    raise RuntimeError(
        f"[Onyx batched] Found {node_key} ({cls.__name__}) but none of "
        f"execute/FUNCTION is callable on it."
    )


def _first_image(result):
    """Pull the IMAGE tensor out of whatever ImageBlur returned.

    v1 nodes return a plain tuple; v3 nodes return an io.NodeOutput. Both are
    indexable in current ComfyUI, but NodeOutput has not always been, so the
    fallbacks stay.
    """
    if torch.is_tensor(result):
        return result
    for attempt in (lambda r: r[0],
                    lambda r: r.result[0],
                    lambda r: r.outputs[0]):
        try:
            value = attempt(result)
            if torch.is_tensor(value):
                return value
        except Exception:
            continue
    raise RuntimeError(
        f"[Onyx Image Blur] Could not read an IMAGE out of ImageBlur's "
        f"return value ({type(result).__name__})."
    )


class OnyxImageBlurBatched:
    """Blur a long image batch in slices, so the 32-bit index limit never applies."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 31, "step": 1}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "frames_per_slice": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "0 = auto: the largest slice that stays under PyTorch's "
                               "2^31-element indexing limit, computed from this batch's "
                               "own resolution.\n"
                               "Set it lower by hand only to cut VRAM peaks further."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blur"
    CATEGORY = "image/postprocessing"

    def blur(self, image, blur_radius, sigma, frames_per_slice=0):
        ensure_profile_ready()
        if blur_radius <= 0:
            return (image,)

        total, height, width = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        channels = int(image.shape[3]) if image.ndim == 4 else 1

        # Deux plafonds INDEPENDANTS, et il faut respecter les deux.
        #
        # 1. Le nombre d'elements : F.pad indexe en 32 bits, donc 2**31 elements
        #    au maximum. C'est une limite de kernel, elle ne depend pas de la RAM.
        # 2. Le nombre d'OCTETS : en float32 chaque element pese 4 octets, donc
        #    une tranche sous la limite d'elements peut quand meme demander 7 Go.
        #    Et le blur en manipule plusieurs a la fois - entree, tenseur paddé,
        #    sortie - donc le pic vaut plusieurs fois cette valeur.
        #
        # Ne regarder que le premier plafond produit exactement l'erreur que ce
        # node etait cense eviter, deplacee du kernel vers l'allocateur.
        padded_per_frame = (height + 2 * blur_radius) * (width + 2 * blur_radius) * channels
        by_elements = _SAFE_ELEMENTS // max(1, padded_per_frame)
        by_bytes = _TARGET_BYTES_PER_SLICE // max(1, padded_per_frame * 4)
        auto = max(1, min(by_elements, by_bytes))
        slice_size = int(frames_per_slice) if frames_per_slice > 0 else auto
        slice_size = max(1, min(slice_size, total))

        cls, fn = _stock_callable("ImageBlur")

        if slice_size >= total:
            print(f"[Onyx Image Blur] {total} frame(s) at {width}x{height} — "
                  f"one pass, under the limit.")
            return (_first_image(_call(cls, fn, image=image, blur_radius=blur_radius, sigma=sigma)),)

        n_slices = (total + slice_size - 1) // slice_size
        print(f"[Onyx Image Blur] {total} frame(s) at {width}x{height}x{channels} — "
              f"{n_slices} slice(s) of up to {slice_size} "
              f"({'auto' if frames_per_slice <= 0 else 'manual'}); "
              f"limits here: {by_elements} frame(s) by 32-bit indexing, "
              f"{by_bytes} by the {_TARGET_BYTES_PER_SLICE / 1024 ** 3:.0f} GiB budget.")

        out = []
        for start in range(0, total, slice_size):
            part = image[start:start + slice_size]
            blurred = _first_image(_call(cls, fn, image=part, blur_radius=blur_radius, sigma=sigma))
            out.append(blurred.cpu())
            del blurred

        result = torch.cat(out, dim=0)
        logging.info("Onyx Image Blur: %d frames blurred in %d slices", total, n_slices)
        return (result,)


def _call(cls, fn, **kwargs):
    """Call the stock method whether it is a classmethod or an instance method."""
    try:
        return fn(**kwargs)
    except TypeError:
        # Methode d'instance non liee : ComfyUI l'appelle sur une instance, pas
        # sur la classe. On en fabrique une, ces nodes n'ont pas d'etat.
        return fn(cls(), **kwargs)


class OnyxImageCompositeMaskedBatched:
    """Composite a long image batch in slices, so peak RAM stays bounded.

    ImageCompositeMasked builds `source_portion + destination_portion` for the
    whole batch at once. At 1080x1920 that is 24.9 MB per frame per tensor, so
    435 frames need 10.8 GB per copy and roughly three copies live at the same
    moment. It fails on RAM, not on VRAM, and long before the resolution itself
    is a problem:

        DefaultCPUAllocator: not enough memory: you tried to allocate 10824192000 bytes

    Slicing bounds the peak to one slice's worth. As with the blur, the compositing
    is ComfyUI's own - this node only decides how much of it happens at a time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
                "frames_per_slice": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "0 = auto: sized so one slice stays near 1 GiB, from this batch's "
                               "own resolution.\nLower it by hand if you still run out of RAM."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "image"

    def composite(self, destination, source, x, y, resize_source,
                  mask=None, frames_per_slice=0):
        ensure_profile_ready()
        total = int(destination.shape[0])
        height, width = int(destination.shape[1]), int(destination.shape[2])
        channels = int(destination.shape[3]) if destination.ndim == 4 else 1

        per_frame = height * width * channels * 4  # float32
        auto = max(1, _TARGET_BYTES_PER_SLICE // max(1, per_frame))
        slice_size = int(frames_per_slice) if frames_per_slice > 0 else auto
        slice_size = max(1, min(slice_size, total))

        cls, fn = _stock_callable("ImageCompositeMasked")

        def pick(tensor, start, end):
            """Slice along the batch, or reuse a single frame for the whole batch.

            A mask or a source given as one frame is meant to apply to every
            frame; slicing it would hand an empty tensor to every pass but the
            first.
            """
            if tensor is None:
                return None
            return tensor[start:end] if int(tensor.shape[0]) > 1 else tensor

        if slice_size >= total:
            out = _call(cls, fn, destination=destination, source=source, x=x, y=y,
                        resize_source=resize_source, mask=mask)
            return (_first_image(out),)

        n_slices = (total + slice_size - 1) // slice_size
        print(f"[Onyx Composite] {total} frame(s) at {width}x{height}x{channels} — "
              f"{n_slices} slice(s) of up to {slice_size} "
              f"({'auto' if frames_per_slice <= 0 else 'manual'}); "
              f"one pass would need {total * per_frame / 1024 ** 3:.1f} GiB per tensor.")

        parts = []
        for start in range(0, total, slice_size):
            end = start + slice_size
            out = _call(cls, fn,
                        destination=pick(destination, start, end),
                        source=pick(source, start, end),
                        x=x, y=y, resize_source=resize_source,
                        mask=pick(mask, start, end))
            parts.append(_first_image(out).cpu())

        result = torch.cat(parts, dim=0)
        logging.info("Onyx Composite: %d frames composited in %d slices", total, n_slices)
        return (result,)
