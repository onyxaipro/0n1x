import os
import numpy as np
import torch

# LUTS_DIR : dossier "luts/" à la racine du pack (un niveau au-dessus de "post processing/")
LUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts")


def _list_luts():
    """Liste les fichiers .cube disponibles dans LUTS_DIR."""
    if not os.path.isdir(LUTS_DIR):
        return ["<aucun fichier .cube trouvé>"]
    files = sorted([f for f in os.listdir(LUTS_DIR) if f.lower().endswith(".cube")])
    return files if files else ["<aucun fichier .cube trouvé>"]


class Onyx_Apply_LUT:
    """
    Onyx Apply LUT
    Applique un fichier .cube (LUT 3D ou 1D) sur une image.
    Placez vos fichiers .cube dans le dossier luts/ à la racine du pack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":            ("IMAGE",),
                "lut_file":         (_list_luts(),),
                "gamma_correction": ("BOOLEAN", {"default": True}),
                "clip_values":      ("BOOLEAN", {"default": True}),
                "strength":         ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "execute"
    CATEGORY      = "Onyx/Post-Processing"

    DESCRIPTION = (
        "Onyx Apply LUT\n"
        "Applies a .cube LUT file to your image.\n"
        "Place your .cube files in the luts/ folder at the pack root.\n"
        "gamma_correction: applies a 2.2 gamma correction before/after.\n"
        "clip_values: clips LUT values within the declared domain.\n"
        "strength: 1.0 = full effect, 0.0 = original image."
    )

    def execute(self, image, lut_file, gamma_correction, clip_values, strength):
        from colour.io.luts.iridas_cube import read_LUT_IridasCube

        lut_path = os.path.join(LUTS_DIR, lut_file)
        if not os.path.isfile(lut_path):
            raise FileNotFoundError(f"[Onyx Apply LUT] LUT file not found: {lut_path}")

        device = image.device
        lut = read_LUT_IridasCube(lut_path)
        lut.name = lut_file

        # --- Clip des valeurs dans le domaine ---
        if clip_values:
            if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
            else:
                if len(lut.table.shape) == 2:  # LUT 3×1D
                    for dim in range(3):
                        lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                else:  # LUT 3D
                    for dim in range(3):
                        lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

        out = []
        for img in image:
            lut_img = img.cpu().numpy().copy()

            is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
            dom_scale = None
            if is_non_default_domain:
                dom_scale = lut.domain[1] - lut.domain[0]
                lut_img = lut_img * dom_scale + lut.domain[0]

            if gamma_correction:
                lut_img = lut_img ** (1 / 2.2)

            lut_img = lut.apply(lut_img)

            if gamma_correction:
                lut_img = lut_img ** 2.2

            if is_non_default_domain:
                lut_img = (lut_img - lut.domain[0]) / dom_scale

            lut_img = torch.from_numpy(lut_img).to(device)

            if strength < 1.0:
                lut_img = strength * lut_img + (1.0 - strength) * img

            out.append(lut_img)

        out = torch.stack(out)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Onyx_Apply_LUT": Onyx_Apply_LUT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Onyx_Apply_LUT": "🎨 Onyx Apply LUT",
}
