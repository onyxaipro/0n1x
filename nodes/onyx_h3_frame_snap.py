# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - Onyx H3 Frame Snap.

Snaps a frame count down to a length MiniMax H3 can actually generate, so the
target video and the reference video land on the same number of frames.

Why this node exists: H3's video VAE runs on a 17k+5 grid (5, 22, 39, ..., 192,
209, ...). Feeding it an off-grid length is not refused - it is silently
realigned, and the target and the reference are realigned in OPPOSITE
directions:

    target    length -> rounded UP   to the next 17k+5
    reference length -> rounded DOWN to the previous 17k+5

At 198 frames that means a 209-frame video generated against 192 frames of
reference: 17 frames - 0.71 s at 24 fps - with no reference conditioning at all.
The model has nothing left to follow there, so it invents an ending.

On top of that, H3 clocks audio at 40 Hz against 24 fps video. A frame boundary
only lands exactly on the audio grid when (frames * 40) % 24 == 0. Miss it and
the audio stream drifts against the video stream over the tail, which is what
breaks lip sync even when the picture still looks fine.

Only lengths that satisfy BOTH grids are safe: 39, 90, 141, 192, 243, 294, ...
- a run of 51 frames. That second constraint cannot be written as a single
expression in a generic math node, which is the whole reason this file exists.

Rounding is always DOWN. Rounding up would push the target past the reference
and recreate the exact gap this node removes.
"""

import math


# Grille du VAE video H3 : 5, 22, 39, 56, ... (17k + 5).
_VIDEO_OFFSET = 5
_VIDEO_STEP = 17

# Sous-ensemble de la grille video dont la fin retombe pile sur l'horloge audio
# 40 Hz : 39, 90, 141, 192, ... (51j + 39). Demonstration : n = 17k+5 tombe sur
# l'horloge audio ssi n*40 % 24 == 0, c'est-a-dire n divisible par 3, ce qui
# impose k = 2 mod 3 — donc un pas de 3*17 = 51 a partir de k=2 (n=39).
_AV_OFFSET = 39
_AV_STEP = 51

# Entree native maximale de H3, reprise de la meme limite que le code natif.
_MAX_FRAMES = 3600

_FPS = 24.0

MODE_AV = "Video + Audio (17k+5 and 40Hz)"
MODE_VIDEO = "Video only (17k+5)"
_MODES = [MODE_AV, MODE_VIDEO]

ROUND_NEAREST = "nearest (may exceed source)"
ROUND_DOWN = "down (never exceed source)"
_ROUNDINGS = [ROUND_NEAREST, ROUND_DOWN]


def reference_floor(frames: int) -> int:
    """Length H3 truncates a reference video to: the 17k+5 run at or below `frames`.

    This is the real ceiling for the target, not the source frame count. H3 aligns
    the target UP to the grid and the reference DOWN to it, so an off-grid source
    silently loses frames on the reference side before the target is even chosen.
    A 188-frame source becomes a 175-frame reference; asking for 192 then leaves
    17 frames with nothing to follow, which is where the picture jumps.
    """
    n = int(frames)
    if n < _VIDEO_OFFSET:
        return 0
    return n - ((n - _VIDEO_OFFSET) % _VIDEO_STEP)


def _snap_down(frames: int, offset: int, step: int) -> int:
    """Largest value of the form offset + step*j that is <= frames."""
    if frames < offset:
        return 0
    return offset + ((frames - offset) // step) * step


def _snap_nearest(frames: int, offset: int, step: int) -> int:
    """Closest value of the form offset + step*j, above or below `frames`.

    Overshoot is bounded by step/2 by construction - 25 frames (~1.0 s at 24 fps)
    on the audio-aligned grid, 8 frames (~0.35 s) on the video grid. That upper
    bound is what makes rounding up acceptable here: the tail beyond the source
    is short enough to trim away, whereas rounding down can cost a full grid
    step - 51 frames, over two seconds - for the sake of four missing frames.
    """
    if frames <= offset:
        return offset
    j = int(round((frames - offset) / float(step)))
    value = offset + max(0, j) * step
    return min(value, _MAX_FRAMES)


def snap_h3_frames(frames: int, audio_aligned: bool = True,
                   allow_overshoot: bool = True) -> tuple:
    """Return (snapped_frames, note).

    With allow_overshoot the result may exceed `frames`. That is deliberate: the
    audio-aligned grid steps by 51 frames, so landing four frames short of a run
    costs two seconds of video when rounding down. Generating slightly past the
    source and trimming the result is the cheaper trade - the frames past the
    reference are unconditioned, but there are at most 25 of them and they are
    discarded anyway.
    """
    n = max(0, int(frames))
    if n > _MAX_FRAMES:
        n = _MAX_FRAMES

    if n < _VIDEO_OFFSET:
        raise RuntimeError(
            f"[Onyx H3 Frame Snap] {frames} frame(s) is below H3's minimum of "
            f"{_VIDEO_OFFSET}.\n-> Feed a longer video."
        )

    snap = _snap_nearest if allow_overshoot else _snap_down

    if audio_aligned:
        snapped = snap(n, _AV_OFFSET, _AV_STEP)
        if snapped >= _AV_OFFSET:
            return snapped, ""
        # Sous 39 frames aucune longueur ne satisfait les deux grilles. Plutot
        # que d'echouer, on retombe sur la grille video seule : le rendu reste
        # valide, seul le calage audio de fin est approximatif.
        fallback = snap(n, _VIDEO_OFFSET, _VIDEO_STEP)
        return fallback, (
            f"under {_AV_OFFSET} frames no length satisfies both grids -> "
            f"video grid only, audio boundary is approximate"
        )

    return snap(n, _VIDEO_OFFSET, _VIDEO_STEP), ""


class OnyxH3FrameSnap:
    """Snap a frame count down to a valid MiniMax H3 run length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {
                    "default": 198, "min": 1, "max": _MAX_FRAMES, "step": 1,
                    "tooltip": "Plug the frame count coming out of your video loader.",
                }),
                "alignment": (_MODES, {
                    "default": MODE_AV,
                    "tooltip": "Video + Audio: only lengths that sit on H3's 17k+5 video grid "
                               "AND land exactly on its 40 Hz audio clock (39, 90, 141, 192, 243...). "
                               "Use this whenever the clip has speech - it is what keeps lip sync "
                               "locked to the last frame.\n"
                               "Video only: the full 17k+5 grid (5, 22, 39, ... 192, 209...). "
                               "Finer granularity, but the audio stream can drift over the tail.",
                }),
                "rounding": (_ROUNDINGS, {
                    "default": ROUND_NEAREST,
                    "tooltip": "nearest: pick the closest valid run, above or below. Overshoot is "
                               "capped at half a grid step (25 frames on the audio-aligned grid, "
                               "8 on the video grid) and those extra frames are simply trimmed "
                               "from the result.\n"
                               "down: never exceed the source. Safer on paper, but on the "
                               "audio-aligned grid missing a run by a few frames costs a full "
                               "51-frame step - over two seconds of video for the sake of four "
                               "missing frames.",
                }),
            },
        }

    # `seconds` exists because the snapped length, not the source length, is what
    # actually gets rendered: a 7.822 s source snapped to 175 frames renders 7.292 s.
    # Anything downstream that needs a duration - H3 Context-IR plans its shot
    # timing from one - must read it from here, or it plans for a clip that will
    # never exist.
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("frames", "seconds")
    FUNCTION = "snap"
    CATEGORY = "Onyx/Utils"
    DESCRIPTION = (
        "Snap a frame count to a length MiniMax H3 can generate without realigning it. "
        "Prevents the tail H3 invents when it rounds the target up and the reference down."
    )

    def snap(self, frame_count, alignment, rounding=ROUND_NEAREST):
        ensure_profile_ready()
        requested = int(frame_count)
        snapped, note = snap_h3_frames(
            requested,
            audio_aligned=(alignment == MODE_AV),
            allow_overshoot=(rounding == ROUND_NEAREST),
        )

        # Imprime systematiquement : la valeur corrigee ne s'affiche nulle part
        # dans l'interface, et l'ecart entre ce que la video contient et ce qui
        # part reellement dans H3 est exactement l'information qui manquait
        # quand la fin des rendus partait en vrille.
        delta = snapped - requested
        # Le chiffre qui compte n'est pas snapped - requested, mais
        # snapped - reference_floor : H3 tronque la reference vers le bas sur la
        # grille avant meme de comparer. Une source de 188 frames ne conditionne
        # que 175 frames, donc viser 192 laisse 17 frames a l'aveugle et non 4.
        floor = reference_floor(requested)
        gap = snapped - floor
        msg = (
            f"🎬 [Onyx H3 Frame Snap] {requested} -> {snapped} frames "
            f"({snapped / _FPS:.3f}s @ {_FPS:.0f}fps) | {delta:+d} frame(s) vs source "
            f"| {alignment} | {rounding}"
        )
        if gap > 0:
            msg += (
                f"\n⚠️  [Onyx H3 Frame Snap] H3 truncates the {requested}-frame source "
                f"to a {floor}-frame reference, so the last {gap} frame(s) "
                f"({gap / _FPS:.3f}s) generate with no reference to follow — expect the "
                f"picture to drift there, and trim them.\n"
                f"   Zero-gap alternatives: {floor} frames ({floor / _FPS:.3f}s) on the "
                f"video grid"
            )
            av_safe = _snap_down(floor, _AV_OFFSET, _AV_STEP)
            if av_safe >= _AV_OFFSET:
                msg += f", {av_safe} frames ({av_safe / _FPS:.3f}s) audio-aligned"
            msg += "."
        elif delta == 0:
            msg += "\n   Already on the grid — nothing to trim."
        if note:
            msg += f"\n⚠️  [Onyx H3 Frame Snap] {note}"

        # Le mode 'down' peut couter un pas de grille entier. Quand la perte
        # depasse un pas de la grille video, c'est presque toujours un reglage
        # subi plutot que choisi — d'ou l'avertissement chiffre.
        if delta < 0 and -delta > _VIDEO_STEP:
            alt, _ = snap_h3_frames(
                requested, audio_aligned=(alignment == MODE_AV), allow_overshoot=True
            )
            if alt != snapped:
                msg += (
                    f"\n⚠️  [Onyx H3 Frame Snap] dropping {-delta} frames "
                    f"({-delta / _FPS:.3f}s). 'nearest' would give {alt} frames "
                    f"({alt / _FPS:.3f}s)."
                )
        print(msg)

        return (snapped, snapped / _FPS)


NODE_CLASS_MAPPINGS = {"OnyxH3FrameSnap": OnyxH3FrameSnap}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3FrameSnap": "Onyx H3 Frame Snap"}
