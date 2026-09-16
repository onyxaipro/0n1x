"""Onyx Speed HD Sampler.

Ported into the Onyx pack from the user's modified copy of ComfyUI-SPEED-SwarmNeo
(MIT license, (c) 2026 A. Izzuddin Al Faruq — see LICENSE_SPEED_HD.txt), itself based on
https://github.com/howardhx/speed — "Spectral Progressive Diffusion for Efficient Image
and Video Generation" (Xiao, Chao, Yariv & Wetzstein, 2026). Progressively expands the
latent resolution during denoising, reducing computation while preserving quality.

The original node (speed_sampler.py in the source repo) used ComfyUI's newer
io.ComfyNode/Schema API. This is a straight port to the classic INPUT_TYPES /
RETURN_TYPES / FUNCTION style used everywhere else in this pack, so it registers the
same way as every other Onyx node. The framework-agnostic math itself
(speed_hd_core.py / speed_hd_spectral_utils.py) is otherwise unchanged.
"""
from __future__ import annotations

from typing import List

import torch

import comfy.samplers
import comfy.k_diffusion.sampling as kds

from .speed_hd_core import (
    _PRESETS,
    _parse_scales,
    _parse_sigmas,
    sample_speed_core,
)


from .onyx_render_profile import ensure_profile_ready
@torch.no_grad()
def sample_speed_hd(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    *,
    transform: str = "dct",
    base_sampler: str = "euler",
    mode: str = "delta_optimal",
    scales: List[float] = None,
    delta: float = 0.01,
    spectrum_A: float = 203.615097,
    spectrum_beta: float = 1.915461,
    manual_sigmas: List[float] = None,
    seed: int = 0,
):
    """Comfy-compatible ``sample_*`` function — resolves the base solver from
    comfy.k_diffusion.sampling, then delegates the segmented spectral-expansion
    sampling to sample_speed_core."""
    sampler_fn = getattr(kds, f"sample_{base_sampler}", None)
    if sampler_fn is None:
        raise ValueError(f"[Onyx Speed HD] Unknown base sampler {base_sampler!r}.")

    return sample_speed_core(
        sampler_fn, model, x, sigmas,
        extra_args=extra_args, callback=callback, disable=disable,
        transform=transform, mode=mode, scales=scales, delta=delta,
        spectrum_A=spectrum_A, spectrum_beta=spectrum_beta,
        manual_sigmas=manual_sigmas, seed=seed,
    )


def _list_samplers() -> List[str]:
    """Return supported k-diffusion sampler names."""
    excluded = {"dpm_fast", "dpm_adaptive", "lcm"}
    try:
        names = [a[len("sample_"):] for a in dir(kds) if a.startswith("sample_")]
    except Exception:
        names = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "uni_pc"]
    return sorted(n for n in names if n not in excluded)


class OnyxSpeedHDSampler:
    """Spectral Progressive Diffusion sampler node — Onyx pack integration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sampler": (_list_samplers(), {
                    "default": "euler",
                    "tooltip": "Underlying ODE solver. Any comfy k_diffusion sampler is "
                               "supported. Multistep solvers restart at each transition "
                               "because the schedule is segmented.",
                }),
                "transform": (["dct", "dwt", "fft"], {
                    "default": "dct",
                    "tooltip": "Spectral basis used at each transition. DCT (default) "
                               "supports any scale ratio; DWT requires consecutive "
                               "scales to differ by exactly 2x (needs PyWavelets); FFT "
                               "accepts any ratio.",
                }),
                "mode": (["delta_optimal", "manual"], {
                    "default": "delta_optimal",
                    "tooltip": "'delta_optimal' computes transitions from 'scales', "
                               "'delta', and the selected power-spectrum preset. "
                               "'manual' uses user-specified sigma thresholds.",
                }),
                "model_preset": (list(_PRESETS.keys()), {
                    "default": "flux",
                    "tooltip": "Power-spectrum preset for delta-optimal mode. 'flux' "
                               "and 'wan21' use measured (A, beta). 'custom' uses the "
                               "manual spectrum_A / spectrum_beta inputs below.",
                }),
                "scales": ("STRING", {
                    "default": "0.5,1.0",
                    "tooltip": "Comma-separated resolution fractions ending at 1.0. "
                               "Used in delta_optimal mode. Example: 0.5,1.0 or "
                               "0.25,0.5,1.0.",
                }),
                "delta": ("FLOAT", {
                    "default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001,
                    "tooltip": "Noise-dominated tolerance. Smaller values transition later.",
                }),
                "manual_sigmas": ("STRING", {
                    "default": "0.85",
                    "tooltip": "Comma-separated sigma thresholds, one per transition "
                               "(length = number of scales minus 1). Used in manual "
                               "mode. Example for scales=0.25,0.5,1.0: 0.95,0.85.",
                }),
                "spectrum_A": ("FLOAT", {
                    "default": 203.615097, "min": 0.0, "max": 1e6, "step": 0.001,
                    "tooltip": "Power-spectrum amplitude A (used when model_preset=custom).",
                }),
                "spectrum_beta": ("FLOAT", {
                    "default": 1.915461, "min": 0.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Power-spectrum decay exponent beta (used when model_preset=custom).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31 - 1, "step": 1,
                    "tooltip": "Seed for the spectral-noise padding at each transition.",
                }),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION     = "get_sampler"
    CATEGORY     = "Onyx/Sampling"
    DESCRIPTION  = (
        "Onyx Speed HD Sampler\n"
        "Spectral Progressive Diffusion — progressively expands the latent resolution "
        "during denoising for faster sampling with preserved quality.\n"
        "Connect the output to SamplerCustomAdvanced."
    )

    def get_sampler(
        self,
        base_sampler, transform, mode, model_preset, scales, delta,
        manual_sigmas, spectrum_A, spectrum_beta, seed,
    ):
        ensure_profile_ready()
        preset = _PRESETS.get(model_preset)
        if preset is not None:
            A, beta = preset["A"], preset["beta"]
        else:
            A, beta = float(spectrum_A), float(spectrum_beta)

        parsed_scales = _parse_scales(scales)
        parsed_sigmas = _parse_sigmas(manual_sigmas) if mode == "manual" else []

        sampler = comfy.samplers.KSAMPLER(
            sample_speed_hd,
            extra_options={
                "transform": transform,
                "base_sampler": base_sampler,
                "mode": mode,
                "scales": parsed_scales,
                "delta": float(delta),
                "spectrum_A": A,
                "spectrum_beta": beta,
                "manual_sigmas": parsed_sigmas,
                "seed": int(seed),
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "OnyxSpeedHDSampler": OnyxSpeedHDSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxSpeedHDSampler": "Onyx Speed HD Sampler",
}
