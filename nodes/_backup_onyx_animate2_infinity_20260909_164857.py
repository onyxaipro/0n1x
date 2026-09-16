"""Wan Animate 2 — arbitrary-length generation in a single node.

Why this exists
---------------
`WanAnimate2ToVideo` encodes the whole clip through the VAE in one shot: once for
a grey placeholder tensor of `length` frames, once for `pose_video`. `VAEDecode`
then decodes the whole latent in one shot. Those three passes are the only part
of an Animate 2 graph that scales with clip duration, and none of them can be
split along time by ComfyUI's automatic paths — for a video VAE the tensor is
reshaped to (1, C, T, H, W), so the batching loop in `VAE.encode` runs exactly
once, and the OOM fallback `encode_tiled_3d` defaults to `tile_t=9999`.

Context windows do not help: they patch the sampler, not the VAE. The only way
to bound the memory is to bound the number of frames that reach the VAE, which
means generating in segments.

This node runs that segment loop internally — build conditioning, sample, decode,
keep the last frame as the temporal anchor, advance into the pose video, repeat —
so peak VRAM is set by `segment_length` and stays flat no matter how long the
input video is. Decoded frames are moved to CPU between segments so the output
does not accumulate on the card.

Segments overlap by one frame by design: the model re-renders the anchor frame to
keep the motion continuous, and `trim_image` says how many head frames to drop.
"""

import gc
import time

import torch

import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from comfy_extras.nodes_wan import WanAnimate2ToVideo

_GB = 1024 ** 3


def _snap_length(n):
    """Animate 2 wants 4k+1 frames, at least 5."""
    n = max(5, int(n))
    return ((n - 1) // 4) * 4 + 1


def _peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(mm.get_torch_device()) / _GB


class OnyxAnimate2Infinity:
    """Generate a Wan Animate 2 video of any length with flat VRAM use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Animate 2 UNet, after your LoRA / attention / cache patches. "
                               "Do NOT put Context Windows in this chain — this node already "
                               "bounds the work, and the two mechanisms fight each other."}),
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
                    "tooltip": "Frames generated per segment, snapped to 4k+1. This is the ONLY "
                               "knob that sets peak VRAM. 81 is what the model was trained on; "
                               "drop to 49 or 33 if you still OOM, raise it only if you have "
                               "headroom to spare."}),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Hard cap on total output frames. 0 = follow the pose video to "
                               "its end. Set it to `segment_length` to render exactly one "
                               "segment while you dial in settings."}),
                "vary_seed_per_segment": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF keeps one seed for every segment, which holds the look "
                               "steady across joints. ON adds the segment index to the seed; "
                               "use it only if repeated segments look mechanically identical."}),
                "decode_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Decode each segment in tiles. Slower and very slightly softer, "
                               "but it is the last thing to try before lowering resolution."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "The character to animate."}),
                "pose_video": ("IMAGE", {"tooltip": "Frames carrying the motion. Its length "
                                                    "decides how many segments run."}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "positive_pose": ("CONDITIONING", {"tooltip": "Prompt for the pose branch."}),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT",),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_image_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Above 1.0 tightens identity against drift, which is what long "
                               "chains suffer from. 1.05-1.15 is a reasonable place to start."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "info")
    FUNCTION = "generate"
    CATEGORY = "Onyx/Video"
    DESCRIPTION = ("Wan Animate 2 at any length. Runs the segment chain internally so the "
                   "VAE never sees more than `segment_length` frames and peak VRAM stays "
                   "flat regardless of input duration.")

    def _decode(self, vae, latent, decode_tiled):
        if not decode_tiled:
            return vae.decode(latent)
        try:
            return vae.decode_tiled(latent)
        except TypeError:
            # older/newer signatures differ on the tile kwargs; plain decode is the safe path
            return vae.decode(latent)

    def generate(self, model, positive, negative, vae, width, height, seed, steps, cfg,
                 sampler_name, scheduler, segment_length, max_frames,
                 vary_seed_per_segment, decode_tiled,
                 reference_image=None, pose_video=None, clip_vision_output=None,
                 positive_pose=None, clip_vision_output_pose=None,
                 pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0,
                 reference_image_strength=1.0):

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

        pbar = comfy.utils.ProgressBar(max(1, target))

        while produced < target and guard < 10000:
            guard += 1
            remaining = target - produced
            # after the first segment one frame is the re-rendered anchor, so ask for one more
            wanted = remaining + (1 if continue_motion is not None else 0)
            seg_len = _snap_length(min(segment_length, max(5, wanted)))

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(mm.get_torch_device())

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

            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
            out = comfy.sample.sample_custom(
                model, noise, cfg, sampler, sigmas, pos, neg, samples,
                callback=callback, disable_pbar=True, seed=seg_seed)

            # drop the reference frame(s) that live at the head of the latent
            out = out[:, :, trim_latent:]

            images = self._decode(vae, out, decode_tiled)
            if images.ndim == 5:  # (B, T, H, W, C) -> (T, H, W, C)
                images = images.reshape(-1, *images.shape[-3:])
            if trim_image > 0:
                images = images[trim_image:]

            # the anchor for the next segment, kept before we ship the chunk off the GPU
            continue_motion = images[-1:].clone().cpu()

            keep = min(images.shape[0], remaining)
            chunks.append(images[:keep].cpu())
            produced += keep
            offset = next_offset
            segment += 1

            peak = _peak_gb()
            lines.append("segment {:>2}  {:>4} frames  offset -> {:<6} peak {:.2f} GB".format(
                segment, keep, offset, peak))
            print("[Animate2Infinity] " + lines[-1], flush=True)
            pbar.update_absolute(min(produced, target), target)

            del out, samples, noise, images, latent, pos, neg
            gc.collect()
            mm.soft_empty_cache()

            if offset >= available:
                break

        if not chunks:
            raise RuntimeError("No segment was produced — check pose_video and max_frames.")

        result = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        elapsed = time.time() - started

        info = "\n".join([
            "Animate 2 — {} frames in {} segment(s) of {} at {}x{}".format(
                result.shape[0], segment, segment_length, width, height),
            "pose video: {} frames available, {} requested".format(available, target),
            "elapsed: {:.1f}s  ({:.2f}s per frame)".format(
                elapsed, elapsed / max(1, result.shape[0])),
            "",
        ] + lines)
        print("[Animate2Infinity]\n" + info, flush=True)

        return (result, int(result.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "OnyxAnimate2Infinity": OnyxAnimate2Infinity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxAnimate2Infinity": "Onyx Animate 2 Infinity",
}
