# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - Onyx Audio Switch.

Turns an audio connection on and off from a widget instead of unplugging the
cable.

Why this exists: MiniMax H3 is a joint audio-video model. Feeding it a reference
audio track makes it drive mouth movement from that audio - which is what you
want on a talking clip, and exactly what ruins a clip where the subject is silent
over background music. There the audio conditioning and the video conditioning
pull in opposite directions, and the motion that loses is the one from the
reference video.

Switching that off is therefore a per-clip decision, not a per-workflow one.
Bypassing a node with Ctrl+B does not help: ComfyUI's bypass forwards the input
straight to the matching output, so the audio still arrives. Passing None is what
actually severs it, and that is all this node does.
"""


class OnyxAudioSwitch:
    """Pass an AUDIO through, or output nothing, from a boolean widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "audio connected",
                    "label_off": "audio cut",
                    "tooltip": "On: the audio passes through unchanged.\n"
                               "Off: nothing is sent, exactly as if the cable were unplugged.\n\n"
                               "Turn it off when the subject is NOT speaking or singing in the "
                               "source clip. H3 will otherwise build mouth movement from whatever "
                               "voice it hears, including song lyrics, and fight the reference "
                               "video for control of the motion.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional. With nothing connected the node simply outputs nothing, "
                               "whatever the toggle says.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "switch"
    CATEGORY = "Onyx/Utils"
    DESCRIPTION = (
        "Enable or disable an audio connection from a widget. Off is equivalent to "
        "unplugging the cable - useful for MiniMax H3, where reference audio drives "
        "mouth movement and competes with the reference video."
    )

    def switch(self, enabled, audio=None):
        ensure_profile_ready()
        if not enabled:
            # Imprime les deux etats, pas seulement celui qui coupe : un toggle
            # silencieux se retrouve dans le mauvais sens deux workflows plus
            # tard, et l'effet ne se voit qu'apres plusieurs minutes de rendu.
            print("🔇 [Onyx Audio Switch] audio cut — nothing sent downstream.")
            return (None,)

        if audio is None:
            print("🔇 [Onyx Audio Switch] enabled, but no audio connected — nothing sent.")
            return (None,)

        print("🔊 [Onyx Audio Switch] audio connected — passing through.")
        return (audio,)


NODE_CLASS_MAPPINGS = {"OnyxAudioSwitch": OnyxAudioSwitch}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxAudioSwitch": "Onyx Audio Switch"}
