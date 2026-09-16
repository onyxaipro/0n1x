"""Wan Animate 2 — arbitrary-length generation in a single node.
Features auto-segmentation, Replacement Mode, and an optional Low VRAM mode.
"""

import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage

import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from comfy_extras.nodes_wan import WanAnimate2ToVideo

from .onyx_render_profile import ensure_profile_ready
_GB = 1024 ** 3


def _snap_length(n):
    """Animate 2 wants 4k+1 frames, at least 5."""
    n = max(5, int(n))
    return ((n - 1) // 4) * 4 + 1


def _peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(mm.get_torch_device()) / _GB


def process_mask_tensor(mask, dilation=4, feather=8):
    """Dilate and blur/feather the mask to create seamless boundaries."""
    if mask.ndim == 4:
        mask = mask.mean(dim=-1)  # Convert [F, H, W, C] to [F, H, W]
    
    mask_np = mask.cpu().numpy()
    processed = []
    
    for m in mask_np:
        # Dilation / Erosion
        if dilation > 0:
            m = scipy.ndimage.binary_dilation(m > 0.5, iterations=dilation).astype(np.float32)
        elif dilation < 0:
            m = scipy.ndimage.binary_erosion(m > 0.5, iterations=-dilation).astype(np.float32)
        else:
            m = (m > 0.5).astype(np.float32)
        
        # Feather (Gaussian Blur)
        if feather > 0:
            m = scipy.ndimage.gaussian_filter(m, sigma=feather)
            
        processed.append(m)
        
    return torch.from_numpy(np.array(processed)).to(mask.device)


class OnyxAnimate2Infinity:
    """Generate a Wan Animate 2 video of any length with flat VRAM use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Animate 2 UNet."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "segment_length": ("INT", {
                    "default": 81, "min": 5, "max": 321, "step": 4,
                    "tooltip": "Frames generated per segment, snapped to 4k+1."}),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Hard cap on total output frames. 0 = follow pose_video to end."}),
                "vary_seed_per_segment": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF = same seed every segment. ON = adds segment index to seed."}),
                "decode_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Decode each segment in tiles to save VRAM."}),
                # --- LOW VRAM OPTIMIZATION SWITCH ---
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ON = Purges CLIP / TextEncoder from VRAM before sampling. Saves 3-6 GB VRAM to prevent OOM."}),
                # --- REPLACEMENT MODE SWITCH ---
                "replacement_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True = Keeps original background. Composites the generated character back onto original background."}),
                "mask_dilation": ("INT", {
                    "default": 4, "min": -100, "max": 100, "step": 1,
                    "tooltip": "Expand (+) or contract (-) mask boundary."}),
                "mask_feather": ("INT", {
                    "default": 8, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Blur mask edges for a seamless blend."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "The character to animate."}),
                "pose_video": ("IMAGE", {"tooltip": "Frames carrying the motion."}),
                "pose_video_mask": ("IMAGE", {"tooltip": "Black & white mask of the character."}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "positive_pose": ("CONDITIONING",),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT",),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "info")
    FUNCTION = "generate"
    CATEGORY = "Onyx/Video"
    DESCRIPTION = ("Wan Animate 2 at any length. Runs segment chain internally so peak VRAM stays flat.")

    def _decode(self, vae, latent, decode_tiled):
        if not decode_tiled:
            return vae.decode(latent)
        try:
            return vae.decode_tiled(latent)
        except TypeError:
            return vae.decode(latent)

    def generate(self, model, positive, negative, vae, width, height, seed, steps, cfg,
                 sampler_name, scheduler, segment_length, max_frames,
                 vary_seed_per_segment, decode_tiled, low_vram, replacement_mode,
                 mask_dilation, mask_feather,
                 reference_image=None, pose_video=None, pose_video_mask=None,
                 clip_vision_output=None, positive_pose=None, clip_vision_output_pose=None,
                 pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0,
                 reference_image_strength=1.0):

        ensure_profile_ready()
        if pose_video is None:
            raise ValueError("pose_video is required: its length is what drives the chain.")

        width = (int(width) // 16) * 16
        height = (int(height) // 16) * 16
        segment_length = _snap_length(segment_length)

        available = int(pose_video.shape[0])
        target = available if max_frames <= 0 else min(available, int(max_frames))

        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps).cpu()

        chunks = []
        produced = 0
        offset = 0
        continue_motion = None
        segment = 0
        started = time.time()
        lines = []
        guard = 0

        while produced < target and guard < 10000:
            guard += 1
            remaining = target - produced
            wanted = remaining + (1 if continue_motion is not None else 0)
            seg_len = _snap_length(min(segment_length, max(5, wanted)))

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(mm.get_torch_device())

            left = max(1, -(-(target - produced) // max(1, seg_len - 1)))
            print(f"[Onyx Animate 2] --- segment {segment + 1} --- {seg_len} frames, {steps} steps | {produced}/{target} frames done", flush=True)

            pos, neg, latent, trim_latent, trim_image, next_offset = WanAnimate2ToVideo.execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=seg_len,
                batch_size=1,
                video_frame_offset=offset,
                reference_image=reference_image,
                pose_video=pose_video,
                clip_vision_output=clip_vision_output,
                positive_pose=positive_pose,
                clip_vision_output_pose=clip_vision_output_pose,
                continue_motion=continue_motion,
                pose_strength=pose_strength,
                pose_start_percent=pose_start_percent,
                pose_end_percent=pose_end_percent,
                reference_image_strength=reference_image_strength,
            ).result

            seg_seed = seed + segment if vary_seed_per_segment else seed
            samples = latent["samples"]
            samples = comfy.sample.fix_empty_latent_channels(model, samples)
            noise = Noise_RandomNoise(seg_seed).generate_noise({"samples": samples})

            # === PURGE OPTIONNELLE POUR LE MODE LOW VRAM ===
            if low_vram:
                print("[Onyx Animate 2] Low VRAM active: Purge des modèles CLIP/TextEncoder de la VRAM...", flush=True)
                mm.unload_all_models()
                mm.soft_empty_cache()
                gc.collect()

            print("[Onyx Animate 2] sampling...", flush=True)
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
            out = comfy.sample.sample_custom(
                model, noise, cfg, sampler, sigmas, pos, neg, samples,
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seg_seed)

            out = out[:, :, trim_latent:]

            if low_vram:
                mm.soft_empty_cache()

            print(f"[Onyx Animate 2] decoding{' (tiled)' if decode_tiled else ''}...", flush=True)
            images = self._decode(vae, out, decode_tiled)
            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])
            if trim_image > 0:
                images = images[trim_image:]

            continue_motion = images[-1:].clone().cpu()

            keep = min(images.shape[0], remaining)
            chunks.append(images[:keep].cpu())
            produced += keep
            offset = next_offset
            segment += 1

            peak = _peak_gb()
            lines.append(f"segment {segment:>2}  {keep:>4} frames  offset -> {offset:<6} peak {peak:.2f} GB")
            print("[Onyx Animate 2] " + lines[-1], flush=True)

            del out, samples, noise, images, latent, pos, neg
            gc.collect()
            mm.soft_empty_cache()

            if offset >= available:
                break

        if not chunks:
            raise RuntimeError("No segment was produced — check pose_video and max_frames.")

        result = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

        # --- LOGIQUE DE REPLACEMENT MODE ---
        if replacement_mode:
            if pose_video_mask is None:
                print("[Onyx Animate 2] WARNING: replacement_mode is enabled, but pose_video_mask is missing.", flush=True)
            else:
                print("[Onyx Animate 2] Applying Replacement Mode (keeping original background)...", flush=True)
                num_frames = result.shape[0]
                
                bg_video = pose_video[:num_frames].clone().to(result.device)
                bg_mask = pose_video_mask[:num_frames].clone().to(result.device)
                
                # Resize background
                bg_video_resized = F.interpolate(
                    bg_video.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False
                ).permute(0, 2, 3, 1)
                
                # Convert mask to grayscale mono channel FIRST
                if bg_mask.ndim == 4:
                    bg_mask_mono = bg_mask.mean(dim=-1)
                else:
                    bg_mask_mono = bg_mask
                    
                # Resize mask
                bg_mask_resized = F.interpolate(
                    bg_mask_mono.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
                
                # Process mask (dilation & feathering)
                processed_mask = process_mask_tensor(bg_mask_resized, dilation=mask_dilation, feather=mask_feather).unsqueeze(-1)
                
                # Compositing
                result = (result * processed_mask) + (bg_video_resized * (1.0 - processed_mask))
                result = torch.clamp(result, 0.0, 1.0)

        elapsed = time.time() - started

        info = "\n".join([
            f"Animate 2 — {result.shape[0]} frames in {segment} segment(s) of {segment_length} at {width}x{height}",
            f"elapsed: {elapsed:.1f}s ({elapsed / max(1, result.shape[0]):.2f}s per frame)",
            "",
        ] + lines)
        print("[Onyx Animate 2]\n" + info, flush=True)

        return (result, int(result.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "OnyxAnimate2Infinity": OnyxAnimate2Infinity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxAnimate2Infinity": "Onyx Animate 2 Infinity",
}