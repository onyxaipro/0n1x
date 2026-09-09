from .nano_banana_aio import OnyxNanoBananaAIO
from ..api_keys import ApiKeysLoaderNode  # au lieu de ..core.api_keys

NODE_CLASS_MAPPINGS = {
    "OnyxNanoBananaAIO":       OnyxNanoBananaAIO,
    "Onyx_Api_Loader": ApiKeysLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnyxNanoBananaAIO":       "Onyx Image and Video Edit AIO",
    "Onyx_Api_Loader": "Onyx Api Loader",
}