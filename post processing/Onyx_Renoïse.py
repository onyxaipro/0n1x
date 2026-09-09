import torch
import torch.nn.functional as F
import math

try:
    import kornia.filters as kfilters
    import kornia.color as kcolor
    KORNIA_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    KORNIA_AVAILABLE = False
    print("⚠️ Onyx Renoïse: Kornia not found. Install with 'pip install kornia'.")

# ---------------------------------------------------------------------------
# ISO preset configurations: (sensor_grain, color_scatter, grain_softness, color_softness, shadow_emphasis)
# ---------------------------------------------------------------------------
ISO_PRESETS = {
    "ISO 0 (Off)":     (0.000, 0.000, 0.00, 0.0, 0.00),
    "ISO 50":          (0.004, 0.002, 0.35, 1.2, 0.98),
    "ISO 100 (Clean)": (0.008, 0.005, 0.40, 1.5, 0.95),
    "ISO 200":         (0.012, 0.008, 0.45, 1.8, 0.90),
    "ISO 400":         (0.018, 0.012, 0.50, 2.0, 0.85),
    "ISO 800":         (0.025, 0.016, 0.55, 2.5, 0.78),
    "ISO 1600":        (0.035, 0.025, 0.65, 3.0, 0.70),
    "Night Mode":      (0.025, 0.035, 1.20, 4.0, 0.50),
}


class Onyx_Renoise:
    """
    Onyx Renoïse — Injects realistic digital sensor noise.

    Separates luminance grain (fine structure) from chroma scatter (color blotches),
    applies independent Gaussian softening to each, and biases the effect toward
    darker regions where real sensor noise is most visible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        if not KORNIA_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {
                        "default": "Kornia not installed. Run: pip install kornia",
                        "multiline": True,
                    })
                }
            }

        return {
            "required": {
                "image":      ("IMAGE",),
                "seed":       ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": "randomize",
                }),
                "iso_preset": (list(ISO_PRESETS.keys()), {
                    "default": "ISO 400",
                    "tooltip": "ISO noise level. ISO 0 = off | ISO 50 = barely visible | ISO 100 = clean | ISO 1600 = heavy grain | Night Mode = chroma-heavy.",
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Taille des grains de bruit (PAS l'intensite, qui vient de l'ISO preset). 1.0 = pixel par pixel (comportement d'origine). Au-dessus de 1 = grains plus gros/visibles (bruit genere a resolution reduite puis agrandi en nearest-neighbor). En dessous de 1 (jusqu'a 0.1) = grain plus fin/serre (bruit genere a resolution superieure puis moyenne vers le bas).",
                }),
            },
        }

    CATEGORY    = "Onyx/Post-Processing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION    = "renoise"

    DESCRIPTION = (
        "Onyx Renoïse\n"
        "Injects authentic digital camera sensor noise.\n"
        "Select an ISO preset — from ISO 0 (off) to ISO 1600 (strong grain).\n"
        "Night Mode adds chroma-heavy noise for a low-light look."
    )

    # ------------------------------------------------------------------
    def _blur(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur if sigma > 0."""
        if sigma <= 0:
            return tensor
        ks = 2 * math.ceil(3.0 * sigma) + 1
        return kfilters.gaussian_blur2d(tensor, (ks, ks), (sigma, sigma))

    # ------------------------------------------------------------------
    def _gen_grain_noise(self, B: int, C: int, H: int, W: int, grain_size: float, device) -> torch.Tensor:
        """Random noise at (B,C,H,W). grain_size == 1 -> per-pixel noise
        (original behavior). grain_size > 1 -> noise generated at a reduced
        resolution then upscaled with nearest-neighbor, producing visibly
        bigger noise blobs. grain_size < 1 -> noise generated at a higher
        resolution then averaged down (area), producing a finer/tighter
        texture than plain per-pixel noise. Neither direction changes
        intensity (see ISO preset)."""
        if grain_size == 1.0:
            return torch.randn(B, C, H, W, device=device)
        rh = max(1, round(H / grain_size))
        rw = max(1, round(W / grain_size))
        small = torch.randn(B, C, rh, rw, device=device)
        mode = "nearest" if grain_size > 1.0 else "area"
        return F.interpolate(small, size=(H, W), mode=mode)

    # ------------------------------------------------------------------
    def renoise(
        self,
        image:      torch.Tensor,
        seed:       int,
        iso_preset: str,
        grain_size: float = 1.0,
    ):
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia is required. Install with: pip install kornia")

        preset_values = ISO_PRESETS.get(iso_preset)
        if preset_values is None:
            return (image,)
        sensor_grain, color_scatter, grain_softness, color_softness, shadow_emphasis = preset_values

        # Short-circuit if nothing to do
        if sensor_grain == 0.0 and color_scatter == 0.0:
            return (image,)

        torch.manual_seed(seed)

        B, H, W, _ = image.shape
        img = image.permute(0, 3, 1, 2).to(DEVICE)   # BCHW

        noise = torch.zeros_like(img)

        # --- Luminance grain (same pattern on all 3 channels) ---
        if sensor_grain > 0:
            luma_noise = self._gen_grain_noise(B, 1, H, W, grain_size, DEVICE).expand(B, 3, H, W).clone()
            luma_noise = self._blur(luma_noise, grain_softness)
            noise = noise + luma_noise * sensor_grain

        # --- Chroma scatter (independent per channel) ---
        if color_scatter > 0:
            chroma_noise = self._gen_grain_noise(B, 3, H, W, grain_size, DEVICE)
            chroma_noise = self._blur(chroma_noise, color_softness)
            noise = noise + chroma_noise * color_scatter

        # --- Shadow emphasis mask (noise strongest in darks, fades in highlights) ---
        if shadow_emphasis > 0:
            luminance = kcolor.rgb_to_grayscale(img)   # (B, 1, H, W)
            # mask = 1 in pure black, 0 in pure white, transition at shadow_emphasis threshold
            mask = 1.0 - torch.clamp(
                (luminance - shadow_emphasis) / (1.0 - shadow_emphasis + 1e-6),
                0.0, 1.0
            )
            noise = noise * mask

        out = torch.clamp(img + noise, 0.0, 1.0)
        return (out.permute(0, 2, 3, 1).to(image.device),)


NODE_CLASS_MAPPINGS = {
    "Onyx_Renoise": Onyx_Renoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Onyx_Renoise": "🎞️ Onyx Renoïse",
}