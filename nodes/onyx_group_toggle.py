# -*- coding: utf-8 -*-
"""
ComfyUI node - Onyx Group Toggle.

One switch for ONE group, instead of a panel listing every group in the graph.

Why a node rather than a menu entry: a node lives in the workflow, so the group
it drives is saved with the graph and sits next to what it controls. Muters that
enumerate every group are fine on a small canvas and unusable on a large one,
where the list is long and the entry you want is never the one under the cursor.

Everything happens client-side. Muting is `node.mode = 2` on each member of the
group, which is a canvas state, not a computation — so this node declares no
output and is never executed by the backend. The Python side exists only to
register the widgets that the JS extension then drives.
"""


class OnyxGroupToggle:
    """Mute or bypass a single named group from a toggle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Declared as a COMBO with a placeholder, not as a STRING. The JS
                # swaps the value provider so the list shows the groups of the
                # open workflow — but the widget must already BE a combo: turning
                # a text widget into one after ComfyUI has built its DOM does not
                # produce a working dropdown.
                "group": (["(no group)"], {
                    "tooltip": "Group this switch drives. The list is filled from the groups "
                               "present in the open workflow.",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "group active",
                    "label_off": "group off",
                    "tooltip": "Off applies the mode below to every node in the group.",
                }),
                "off_mode": (["mute", "bypass"], {
                    "default": "mute",
                    "tooltip": "mute: the group does not run, and nothing downstream of it runs "
                               "either.\n"
                               "bypass: the group is skipped but its inputs are passed through to "
                               "its outputs, so the rest of the graph keeps working around it.\n\n"
                               "Use bypass on a group sitting in the middle of a chain, mute on a "
                               "branch that ends in its own output.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Onyx/Utils"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Mute or bypass one named group from a single toggle. Acts on the canvas, "
        "not during execution."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, group=None, **kwargs):
        # Le combo est declare avec un placeholder ; sa vraie liste est produite
        # cote navigateur. Toute valeur reelle serait donc rejetee par la
        # validation standard, qui compare a la liste declaree a l'import.
        return True

    def noop(self, group="", enabled=True, off_mode="mute"):
        return ()


NODE_CLASS_MAPPINGS = {"OnyxGroupToggle": OnyxGroupToggle}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxGroupToggle": "Onyx Group Toggle"}
