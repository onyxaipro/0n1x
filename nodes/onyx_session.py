"""
Onyx Session — the one place a user pastes their key.

OUTPUT_NODE = True on purpose: ComfyUI always executes an OUTPUT_NODE on every
queue, even with nothing wired to its output (exactly how Save Image / Preview
Image run without being connected to anything downstream). So this node just
needs to exist ANYWHERE in the workflow — no wiring to the other nodes needed —
and it refreshes the local sync cache every single run. Every other gated node
then reads that cache on its own; see onyx_render_profile.py.
"""

from .onyx_render_profile import sync_with_server


class OnyxSessionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Your Onyx key. Paste it once — this node re-checks it "
                               "automatically on every queue, from anywhere in the workflow.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Onyx"
    DESCRIPTION = "Paste your Onyx key here once. Runs automatically on every queue."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Doit re-tourner a chaque queue pour rafraichir le cache, meme si le
        # widget key n'a pas change.
        return float("nan")

    def run(self, key):
        result = sync_with_server(key)
        if result["valid"]:
            status = "✅ Onyx session active"
        else:
            status = f"🔴 Onyx session inactive ({result['reason']})"
        print(f"[Onyx Session] {status}")
        return {"ui": {"text": [status]}, "result": (status,)}


NODE_CLASS_MAPPINGS = {"OnyxSessionNode": OnyxSessionNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxSessionNode": "Onyx Session"}
