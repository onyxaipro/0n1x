# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - Onyx RIFE VFI (batched).

RIFE frame interpolation that does not need twice the RAM of its own output.

What goes wrong in the original, and it is not the model: RIFE_VFI collects every
frame into a Python list and finishes with

    out_tensor = torch.cat(output_frames, dim=0).to(torch.float32)

`cat` allocates a fresh contiguous block while the list still holds all the
pieces, so the peak is TWICE the output. On 1505 frames of 768x1344 that is
17.4 GiB needed twice - 34.7 GiB - and the run dies on the very last line, after
the interpolation has already succeeded:

    DefaultCPUAllocator: not enough memory: you tried to allocate 18641387520 bytes

Note `clear_cache_after_n_frames` is no help here at all. It empties the CUDA
cache during the loop; this allocation is host RAM, after the loop.

What this node does instead: it computes the output length first, allocates the
destination ONCE, and writes each frame straight into its slot. No list, no cat,
no second copy. Peak host memory is the output itself plus one batch.

`storage` decides where that single allocation lives:

  - RAM: one float32 tensor, 17.4 GiB for the case above instead of 34.7.
  - memory-mapped file: the same array backed by a .npy in ComfyUI's temp
    folder. Resident memory stays near zero and the pages are read back on
    demand, which is exactly right when the next node walks the batch in order
    to encode a video. It is the only option that does not need the whole thing
    in RAM at any point.

The interpolation itself is not reimplemented. IFNet and the weights are the
ones from ComfyUI-Frame-Interpolation, imported from it, so the frames this node
produces are the frames RIFE_VFI produces.
"""

import gc
import os
import sys
import typing

import numpy as np
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None


_FI_DIRNAMES = ("comfyui-frame-interpolation", "ComfyUI-Frame-Interpolation")


def _frame_interpolation_root():
    """Locate ComfyUI-Frame-Interpolation, whatever the folder is called.

    Installed from git it is lower-case, from a zip it keeps the repository's
    capitalisation. Both are looked for rather than assuming one.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(here)
    for name in _FI_DIRNAMES:
        cand = os.path.join(parent, name)
        if os.path.isdir(os.path.join(cand, "vfi_models", "rife")):
            return cand
    for name in os.listdir(parent):
        cand = os.path.join(parent, name)
        if os.path.isdir(os.path.join(cand, "vfi_models", "rife")):
            return cand
    return None


def _load_upstream():
    """Import IFNet and the helpers from the third-party pack.

    Its own package does sys.path.insert on its root, but this node may be
    imported before it, so the path is ensured here too.
    """
    root = _frame_interpolation_root()
    if root is None:
        raise RuntimeError(
            "[RIFE batched] ComfyUI-Frame-Interpolation was not found next to this pack.\n"
            "-> Install it (Manager: 'ComfyUI Frame Interpolation'). This node borrows its "
            "model and weights rather than shipping a second copy."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    from vfi_utils import load_file_from_github_release, InterpolationStateList  # noqa
    sys.path.insert(0, os.path.join(root, "vfi_models", "rife"))
    from rife_arch import IFNet  # noqa
    return IFNet, load_file_from_github_release, InterpolationStateList


_CKPT_VER = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
}

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

_STORAGE = {
    "RAM (float32, one allocation)": "ram",
    "memory-mapped file (lowest RAM)": "mmap",
}

# Cache de modele partage, sur le meme principe que celui d'origine : recharger
# les poids a chaque execution coute plusieurs secondes pour rien.
_model_cache: typing.Dict[tuple, torch.nn.Module] = {}


class OnyxRifeVfiBatched:
    """RIFE interpolation writing into a preallocated output. Half the peak RAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "ckpt_name": (sorted(_CKPT_VER), {"default": "rife49.pth"}),
                "multiplier": ("INT", {
                    "default": 2, "min": 1, "max": 16,
                    "tooltip": "2 doubles the frame count: 24 fps in, 48 fps out."}),
                "storage": (list(_STORAGE), {
                    "default": "RAM (float32, one allocation)",
                    "tooltip": "RAM: one float32 tensor. Already half the original node's peak, "
                               "because nothing is concatenated.\n"
                               "memory-mapped: the array lives in a temp .npy instead. Resident "
                               "memory stays near zero — use it when the output alone does not "
                               "fit, and when the next node reads the batch in order (a video "
                               "encoder does)."}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "dtype": (list(_DTYPES), {
                    "default": "float32",
                    "tooltip": "Inference precision on the GPU. The output is always float32, "
                               "which is what ComfyUI's IMAGE type is."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Interpolated frames per GPU call. Costs VRAM, not host RAM."}),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 1000,
                    "tooltip": "Empties the CUDA cache every N pairs. This only ever concerned "
                               "VRAM — the failure this node fixes was host RAM, after the loop."}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "vfi"
    CATEGORY = "Onyx"
    DESCRIPTION = ("RIFE interpolation that writes into a preallocated output instead of "
                   "concatenating a list. Halves peak RAM, or removes it entirely with "
                   "memory-mapped storage.")

    def vfi(self, frames, ckpt_name, multiplier, storage, fast_mode, ensemble,
            scale_factor, dtype, batch_size, clear_cache_after_n_frames,
            optional_interpolation_states=None):
        ensure_profile_ready()
        from comfy.model_management import get_torch_device, soft_empty_cache

        IFNet, fetch_weights, _ = _load_upstream()

        n_in = int(frames.shape[0])
        if n_in < 2:
            raise ValueError(f"[RIFE batched] needs at least 2 frames, got {n_in}.")

        height, width = int(frames.shape[1]), int(frames.shape[2])
        arch_ver = _CKPT_VER[ckpt_name]
        torch_dtype = _DTYPES[dtype]
        device = get_torch_device()
        mult = max(1, int(multiplier))
        if arch_ver == "4.26":
            ensemble = False          # non supporte par cette architecture

        # ── plan de sortie ───────────────────────────────────────────────────
        # Calcule AVANT toute inference : c'est ce qui permet d'allouer une
        # seule fois. Chaque paire fournit sa frame d'origine puis ses
        # intermediaires ; la derniere frame ferme la serie.
        n_pairs = n_in - 1
        steps = []
        for pair in range(n_pairs):
            skipped = (optional_interpolation_states is not None
                       and optional_interpolation_states.is_frame_skipped(pair))
            steps.append(0 if skipped else max(mult - 1, 0))
        n_out = n_pairs + sum(steps) + 1

        need_gb = n_out * height * width * 3 * 4 / 1024 ** 3
        mode = _STORAGE[storage]

        # ── l'unique allocation ──────────────────────────────────────────────
        tmp_path = None
        if mode == "mmap":
            base = (folder_paths.get_temp_directory() if folder_paths
                    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp"))
            os.makedirs(base, exist_ok=True)
            tmp_path = os.path.join(base, f"ab_rife_{os.getpid()}_{id(self)}.npy")
            # open_memmap ecrit un vrai .npy : le tenseur rendu est adosse au
            # fichier, donc la memoire residente reste proche de zero et les
            # pages sont relues a la demande.
            arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32,
                                            shape=(n_out, height, width, 3))
            out = torch.from_numpy(arr)
        else:
            out = torch.empty((n_out, height, width, 3), dtype=torch.float32)

        print(f"🎞️  [RIFE batched] {n_in} frame(s) x{mult} -> {n_out} frame(s) "
              f"{width}x{height}\n"
              f"     output = {need_gb:.2f} GiB, allocated once "
              f"({'memory-mapped' if mode == 'mmap' else 'in RAM'}). "
              f"The original node would peak at {need_gb * 2:.2f} GiB.")
        if tmp_path:
            print(f"     backing file: {tmp_path}")

        # ── modele ───────────────────────────────────────────────────────────
        key = (ckpt_name, dtype)
        if key not in _model_cache:
            model_path = fetch_weights("rife", ckpt_name)
            model = IFNet(arch_ver=arch_ver)
            model.load_state_dict(torch.load(model_path, weights_only=False))
            if torch_dtype != torch.float32:
                model = model.to(torch_dtype)
            model.eval().to(device)
            _model_cache[key] = model
            print(f"     loaded {ckpt_name} ({dtype})")
        else:
            model = _model_cache[key]

        if arch_ver == "4.26":
            scale_list = [16 / scale_factor, 8 / scale_factor, 4 / scale_factor,
                          2 / scale_factor, 1 / scale_factor]
        else:
            scale_list = [8 / scale_factor, 4 / scale_factor,
                          2 / scale_factor, 1 / scale_factor]

        # ── taches, dans l'ordre de la SORTIE ────────────────────────────────
        # Chaque tache porte l'index exact de sa place. C'est ce qui remplace le
        # dictionnaire `results` de l'original, qui gardait tous les
        # intermediaires en memoire jusqu'a l'assemblage final.
        tasks = []
        slot = 0
        for pair in range(n_pairs):
            out[slot] = frames[pair, :, :, :3]      # frame d'origine, copiee en place
            slot += 1
            for step in range(1, steps[pair] + 1):
                tasks.append((pair, step / mult, slot))
                slot += 1
        out[slot] = frames[n_in - 1, :, :, :3]
        slot += 1
        assert slot == n_out, f"plan mismatch: {slot} != {n_out}"

        # ── inference ────────────────────────────────────────────────────────
        done_pairs = 0
        last_pair = -1
        pos = 0
        try:
            with torch.inference_mode():
                while pos < len(tasks):
                    chunk = tasks[pos:pos + max(1, int(batch_size))]
                    f0 = torch.cat([frames[p:p + 1, :, :, :3] for p, _, _ in chunk], dim=0)
                    f1 = torch.cat([frames[p + 1:p + 2, :, :, :3] for p, _, _ in chunk], dim=0)
                    # IFNet travaille en NCHW ; permute est une vue, la copie ne
                    # se fait qu'au transfert vers le GPU.
                    f0 = f0.permute(0, 3, 1, 2).to(device, dtype=torch_dtype)
                    f1 = f1.permute(0, 3, 1, 2).to(device, dtype=torch_dtype)
                    ts = torch.tensor([t for _, t, _ in chunk],
                                      dtype=torch_dtype, device=device).view(-1, 1, 1, 1)

                    mid = model(f0, f1, ts, scale_list, fast_mode, ensemble).clamp(0, 1)
                    mid = mid.permute(0, 2, 3, 1).to(torch.float32).cpu()

                    for i, (pair, _, dest) in enumerate(chunk):
                        # Ecriture directe dans la destination. Rien n'est
                        # conserve : la frame produite est libre au tour suivant.
                        out[dest] = mid[i]
                        if pair != last_pair:
                            last_pair, done_pairs = pair, done_pairs + 1
                            if done_pairs >= clear_cache_after_n_frames:
                                soft_empty_cache()
                                gc.collect()
                                done_pairs = 0
                    del mid, f0, f1
                    pos += len(chunk)
        except BaseException:
            # Un plantage ou une interruption laisserait sinon un fichier de
            # plusieurs gigaoctets dans temp/, invisible et jamais reclame.
            if tmp_path and os.path.isfile(tmp_path):
                del out
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

        soft_empty_cache()
        print(f"✅ [RIFE batched] {n_out} frame(s) written in place — no concatenation.")
        return (out,)


NODE_CLASS_MAPPINGS = {"OnyxRifeVfiBatched": OnyxRifeVfiBatched}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxRifeVfiBatched": "Onyx RIFE VFI (batched)"}
