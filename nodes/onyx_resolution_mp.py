# -*- coding: utf-8 -*-
"""
ComfyUI node - Onyx Resolution (MP).

Picks a width/height from a megapixel budget and an aspect ratio, snapped to a
multiple the model actually accepts.

Why this is not a one-liner: video models want dimensions that are multiples of
32 (MiniMax H3 states this explicitly, and its reference 1344x768 follows it), so
you cannot simply scale a ratio to hit a pixel count. Snapping afterwards moves
BOTH the megapixel total and the aspect ratio, and naive rounding can land
several percent away from the target. This node searches the valid grid instead
of rounding once, and reports what it actually produced — because a resolution
widget showing 0.98 while the model receives 1.04 is worse than no widget.
"""

import math


_ASPECT_PRESETS = {
    "from image (input)": None,
    "1:1":   (1, 1),
    "4:3":   (4, 3),
    "3:4":   (3, 4),
    "3:2":   (3, 2),
    "2:3":   (2, 3),
    "16:9":  (16, 9),
    "9:16":  (9, 16),
    "21:9":  (21, 9),
    "5:4":   (5, 4),
    "4:5":   (4, 5),
}


def _nearest_ratio_label(ar: float) -> str:
    """Closest named ratio to `ar`, compared in log space.

    Log space, not a plain difference: 21:9 and 16:9 sit 0.55 apart in raw value
    while 3:4 and 2:3 sit 0.08 apart, so a linear metric would treat the second
    pair as nearly identical and always favour the wide end of the list.
    """
    best = None
    for label, wh in _ASPECT_PRESETS.items():
        if wh is None:
            continue
        cand = wh[0] / wh[1]
        err = abs(math.log(cand / ar))
        if best is None or err < best[0]:
            best = (err, label, cand)
    return best[1]


def _snap_to_budget(megapixels: float, ar: float, multiple: int,
                    min_side: int = 64, max_side: int = 8192) -> tuple:
    """Best (width, height) on the `multiple` grid for a MP budget and ratio.

    Searches candidate heights around the ideal rather than rounding once. For
    each candidate height the matching width is derived from the ratio and
    snapped, then the pair is scored on how far it lands from the requested
    megapixels, with the aspect-ratio error as a tie-breaker.

    Megapixel accuracy is weighted far above ratio accuracy on purpose: the MP
    budget is what governs VRAM and how close you stay to the model's training
    regime, whereas a fraction of a percent of ratio drift is invisible.
    """
    target_px = max(1.0, megapixels * 1_000_000.0)
    ideal_h = math.sqrt(target_px / ar)

    def snap(v):
        return max(multiple, int(round(v / multiple)) * multiple)

    best = None
    base_h = snap(ideal_h)
    # Both sides are searched, not just the height. Deriving a single width per
    # height (w = snap(h * ar)) looks natural but leaves the best cells of the
    # grid unvisited: at 0.20 MP in 16:9 it can only reach 576x320, which is 7.8%
    # short of budget, while 608x320 sits within 2.7% for a ratio error nobody
    # sees. On a coarse grid the exact ratio is a luxury; the pixel budget is not.
    for kh in range(-3, 4):
        h = base_h + kh * multiple
        if h < min_side or h > max_side:
            continue
        base_w = snap(h * ar)
        for kw in range(-3, 4):
            w = base_w + kw * multiple
            if w < min_side or w > max_side:
                continue
            px = w * h
            mp_err = abs(px - target_px) / target_px
            ar_err = abs((w / h) - ar) / ar
            score = mp_err + 0.25 * ar_err
            if best is None or score < best[0]:
                best = (score, w, h)

    if best is None:  # ratio so extreme that nothing fits the bounds
        h = max(min_side, min(max_side, snap(ideal_h)))
        w = max(min_side, min(max_side, snap(h * ar)))
        return w, h
    return best[1], best[2]


class OnyxResolutionMP:
    """Megapixel-budget resolution picker with 0.01 MP granularity."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixels": ("FLOAT", {
                    "default": 0.98, "min": 0.05, "max": 8.0, "step": 0.01, "round": 0.01,
                    "tooltip": "Pixel budget in megapixels, to two decimals.\n"
                               "Models behave best near the resolution they were trained on — "
                               "0.98 and 1.05 are not interchangeable even though both 'look like 1MP'.\n"
                               "MiniMax H3 reference: 1344x768 = 1.03 MP.",
                }),
                "aspect_ratio": (list(_ASPECT_PRESETS.keys()), {
                    "default": "16:9",
                    "tooltip": "Target ratio. 'from image (input)' takes it from the connected image "
                               "instead, which is what you want when matching a source video.",
                }),
                "multiple_of": ("INT", {
                    "default": 32, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Both sides are snapped to a multiple of this.\n"
                               "32 for MiniMax H3 and most video models — sending a non-multiple "
                               "is either refused or silently resized by the model.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional. Its aspect ratio is used when 'from image (input)' "
                               "is selected. The image itself is not resized or passed through.",
                }),
            },
        }

    # aspect_ratio is emitted as a label, not as the raw width:height. Downstream
    # consumers such as H3 Context-IR take a ratio from a fixed list, and a value
    # outside it fails combo validation. With "from image (input)" the exact ratio
    # is rarely one of those labels — 768x1344 is 4:7, not 9:16 — so the closest
    # listed ratio is emitted and both values are printed, rather than sending
    # something that will be rejected.
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_ratio")
    FUNCTION = "compute"
    CATEGORY = "Onyx/Utils"
    DESCRIPTION = (
        "Convert a megapixel budget + aspect ratio into a width/height snapped to a "
        "valid multiple. The resolution actually produced is printed to the console."
    )

    def compute(self, megapixels, aspect_ratio, multiple_of, image=None):
        preset = _ASPECT_PRESETS.get(aspect_ratio)

        if preset is None:
            if image is None:
                raise RuntimeError(
                    "[Onyx Resolution] 'from image (input)' is selected but no image is "
                    "connected.\n-> Connect an image, or pick a fixed ratio."
                )
            # ComfyUI IMAGE is [B, H, W, C]
            src_h, src_w = int(image.shape[1]), int(image.shape[2])
            if src_h <= 0 or src_w <= 0:
                raise RuntimeError(f"[Onyx Resolution] Invalid image dimensions: {src_w}x{src_h}")
            ar = src_w / src_h
            ar_label = f"{src_w}x{src_h}"
            ratio_out = _nearest_ratio_label(ar)
        else:
            ar = preset[0] / preset[1]
            ar_label = aspect_ratio
            ratio_out = aspect_ratio

        width, height = _snap_to_budget(float(megapixels), ar, int(multiple_of))

        # Toujours imprimé, même sans sortie dédiée : snapper sur la grille déplace
        # le budget réel, et une valeur de 0.98 affichée dans le widget pendant que
        # le modèle reçoit 1.03 se remarque tard, sur la qualité, sans rien à lire.
        actual_mp = (width * height) / 1_000_000.0
        drift = (actual_mp - megapixels) / megapixels * 100.0
        print(
            f"📐 [Onyx Resolution] {ar_label} -> {width}x{height} | "
            f"{actual_mp:.3f} MP ({drift:+.1f}% vs {megapixels:.2f} requested) | "
            f"ratio {width / height:.4f} (target {ar:.4f}) | /{multiple_of} | "
            f"aspect_ratio out: {ratio_out}"
        )

        return (width, height, ratio_out)


# ─────────────────────────────────────────────────────────────────────────────
# Live preview
#
# The widget preview goes through this route rather than re-deriving the maths
# in JavaScript. _snap_to_budget searches the grid and weighs two competing
# errors; a second implementation of that in another language would drift from
# this one the first time either is touched, and the node would then display a
# resolution it does not produce - which is precisely the failure this node was
# written to prevent.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from server import PromptServer
    from aiohttp import web

    if not getattr(PromptServer.instance, "_onyx_res_mp_route", False):
        PromptServer.instance._onyx_res_mp_route = True

        @PromptServer.instance.routes.get("/onyx/resolution_mp/preview")
        async def _res_mp_preview(request):
            try:
                q = request.query
                mp = float(q.get("mp", 1.0))
                label = q.get("ar", "16:9")
                mult = int(q.get("mult", 32))

                preset = _ASPECT_PRESETS.get(label)
                if preset is None:
                    # "from image (input)" : le ratio vient d'un tenseur que le
                    # frontend n'a pas. Mieux vaut le dire que d'afficher un
                    # chiffre invente.
                    return web.json_response({
                        "success": True,
                        "text": "from image — computed at run time",
                    })

                ar = preset[0] / preset[1]
                w, h = _snap_to_budget(mp, ar, mult)
                actual = (w * h) / 1_000_000.0
                drift = (actual - mp) / mp * 100.0 if mp else 0.0
                return web.json_response({
                    "success": True,
                    "width": w, "height": h,
                    "mp": round(actual, 3), "drift": round(drift, 1),
                    "text": f"{w} x {h}  ·  {actual:.3f} MP ({drift:+.1f}%)  ·  {w / h:.4f}",
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=500)
except Exception:
    pass


NODE_CLASS_MAPPINGS = {"OnyxResolutionMP": OnyxResolutionMP}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxResolutionMP": "Onyx Resolution (MP)"}
