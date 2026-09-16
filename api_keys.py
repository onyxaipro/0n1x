"""
ComfyUI node — API Keys Loader (Onyx pack).
Entrez vos clés API une seule fois ici et câblez les sorties vers les autres nodes.
"""


from .nodes.onyx_render_profile import ensure_profile_ready
class ApiKeysLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "gemini_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Gemini API Key — aistudio.google.com",
                }),
                "wavespeed_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "WaveSpeed API Key — wavespeed.ai",
                }),
                "fal_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "fal.ai API Key — fal.ai/dashboard/keys",
                }),
                "kie_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Kie.ai API Key — à venir",
                }),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("gemini_api_key", "wavespeed_api_key", "fal_api_key", "kie_api_key")
    FUNCTION      = "load"
    CATEGORY      = "Onyx/NanoBanana"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, gemini_api_key="", wavespeed_api_key="", fal_api_key="", kie_api_key=""):
        ensure_profile_ready()
        return (
            gemini_api_key.strip(),
            wavespeed_api_key.strip(),
            fal_api_key.strip(),
            kie_api_key.strip(),
        )


# Alias pour compatibilité avec l'ancien __init__.py
OnyxLoadAPIKeysNode = ApiKeysLoaderNode