"""ComfyUI custom nodes for Nano Banana Pro and Nano Banana 2 APIs (Monthly)."""
from .Node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .gemini_prompt import OnyxGeminiPromptNode
from .prompt_selector import PromptSelectorNode
from .directory_loader import OnyxDirectoryImageLoaderNode
from .api_keys import OnyxLoadAPIKeysNode
from .video_frame_extractor import OnyxVideoFrameExtractorNode
from .image_processing import MetadataRemoveNode, SaveAsPhonePhotoNode
from .lora_caption import OnyxLoraCaptionGeneratorNode
from .nodes.nano_banana_aio import OnyxNanoBananaAIO
from .nodes.imageUtils import OnyxPreviewImageWithoutMetadata
from .nodes.onyx_video_loader import OnyxVideoLoader
from .nodes.instagram_faceswap import OnyxInstagramFaceSwapNode
from .nodes.dataset_creator          import OnyxDatasetCreatorNode
from .nodes.onyx_image_batch_loader import OnyxImageBatchLoader
from .nodes.repose_carousel import OnyxReposeCarouselNode
from .nodes.metadata_bypass import OnyxMetadataBypassNode
from .nodes.image_enhancement import OnyxImageEnhancementNode
from .nodes.image_black_check import OnyxImageBlackCheckNode
from .nodes.grok_prompt import OnyxGrokPromptNode
from .nodes.onyx_resolution_mp import OnyxResolutionMP
from .nodes.onyx_h3_frame_snap import OnyxH3FrameSnap
from .nodes.onyx_audio_switch import OnyxAudioSwitch
from .nodes.h3_context_ir import OnyxH3ContextIR
from .nodes.onyx_group_toggle import OnyxGroupToggle
from .nodes.onyx_image_blur_batched import OnyxImageBlurBatched, OnyxImageCompositeMaskedBatched
from .nodes.onyx_images_to_video import OnyxImagesToVideo
from .nodes.onyx_save_video_no_metadata import OnyxSaveVideoNoMetadata
from .nodes.onyx_segment_cache import OnyxSegmentCache
from .nodes.onyx_rife_batched import OnyxRifeVfiBatched
from .nodes.onyx_prompt_library import OnyxPromptSaver, OnyxPromptGallery
from .nodes.onyx_video_chain import (OnyxVideoChainSegment, OnyxVideoChainJoin,
                                         OnyxVideoChainPrepare, OnyxVideoChainCommit)
from .nodes.save_image_no_metadata import OnyxSaveImageNoMetadataNode
from .nodes.onyx_speed_hd_sampler import OnyxSpeedHDSampler
from .nodes.onyx_eye_detailer import OnyxEyeBBoxDetectorProvider, OnyxDetailer

# Post-processing nodes (dossier avec espace + noms unicode -> importlib)
import importlib.util as _ilu, os as _os

_pack_root = _os.path.dirname(_os.path.abspath(__file__))

def _load_pp(filename, modname):
    _path = _os.path.join(_pack_root, "post processing", filename)
    _spec = _ilu.spec_from_file_location(modname, _path)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod

try:
    _renoise_mod = _load_pp("Onyx_Renoïse.py", "Onyx_Renoise_mod")
    Onyx_Renoise = _renoise_mod.Onyx_Renoise
    print("[Onyx] ✅ Renoïse chargé")
except Exception as _e:
    Onyx_Renoise = None
    print(f"[Onyx] ❌ Renoïse failed: {_e}")

try:
    _camera_look_mod = _load_pp("Onyx_camera_look.py", "Onyx_Camera_Look_mod")
    Onyx_Camera_Look = _camera_look_mod.Onyx_Camera_Look
    print("[Onyx] ✅ Camera Look chargé")
except Exception as _e:
    Onyx_Camera_Look = None
    print(f"[Onyx] ❌ Camera Look failed: {_e}")

try:
    _apply_lut_mod = _load_pp("Onyx_Apply_LUT.py", "Onyx_Apply_LUT_mod")
    Onyx_Apply_LUT = _apply_lut_mod.Onyx_Apply_LUT
    print("[Onyx] ✅ Apply LUT chargé")
except Exception as _e:
    Onyx_Apply_LUT = None
    print(f"[Onyx] ❌ Apply LUT failed: {_e}")

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS["OnyxNanoBananaAIO"] = OnyxNanoBananaAIO
NODE_CLASS_MAPPINGS["OnyxGeminiPromptNode"] = OnyxGeminiPromptNode
NODE_CLASS_MAPPINGS["OnyxPromptSelectorNodeMonthly"] = PromptSelectorNode
NODE_CLASS_MAPPINGS["OnyxDirectoryImageLoaderNode"] = OnyxDirectoryImageLoaderNode
NODE_CLASS_MAPPINGS["OnyxLoadAPIKeysNode"] = OnyxLoadAPIKeysNode
NODE_CLASS_MAPPINGS["OnyxVideoFrameExtractorNode"] = OnyxVideoFrameExtractorNode
NODE_CLASS_MAPPINGS["OnyxMetadataRemoveNodeMonthly"] = MetadataRemoveNode
NODE_CLASS_MAPPINGS["OnyxSaveAsPhonePhotoNodeMonthly"] = SaveAsPhonePhotoNode
NODE_CLASS_MAPPINGS["OnyxLoraCaptionGeneratorNode"] = OnyxLoraCaptionGeneratorNode
NODE_CLASS_MAPPINGS["OnyxPreviewImageWithoutMetadata"] = OnyxPreviewImageWithoutMetadata
NODE_CLASS_MAPPINGS["OnyxVideoLoader"] = OnyxVideoLoader
NODE_CLASS_MAPPINGS["OnyxInstagramFaceSwapNode"] = OnyxInstagramFaceSwapNode
NODE_CLASS_MAPPINGS["OnyxDatasetCreatorNode"]        = OnyxDatasetCreatorNode
NODE_CLASS_MAPPINGS["OnyxImageBatchLoader"] = OnyxImageBatchLoader
NODE_CLASS_MAPPINGS["OnyxReposeCarouselNode"] = OnyxReposeCarouselNode
NODE_CLASS_MAPPINGS["OnyxMetadataBypassNode"] = OnyxMetadataBypassNode
NODE_CLASS_MAPPINGS["OnyxImageEnhancementNode"] = OnyxImageEnhancementNode
NODE_CLASS_MAPPINGS["OnyxImageBlackCheckNode"] = OnyxImageBlackCheckNode
NODE_CLASS_MAPPINGS["OnyxGrokPromptNode"] = OnyxGrokPromptNode
NODE_CLASS_MAPPINGS["OnyxResolutionMP"] = OnyxResolutionMP
NODE_CLASS_MAPPINGS["OnyxH3FrameSnap"] = OnyxH3FrameSnap
NODE_CLASS_MAPPINGS["OnyxImageBlurBatched"] = OnyxImageBlurBatched
NODE_CLASS_MAPPINGS["OnyxImageCompositeMaskedBatched"] = OnyxImageCompositeMaskedBatched
NODE_CLASS_MAPPINGS["OnyxImagesToVideo"] = OnyxImagesToVideo
NODE_CLASS_MAPPINGS["OnyxSaveVideoNoMetadata"] = OnyxSaveVideoNoMetadata
NODE_CLASS_MAPPINGS["OnyxSegmentCache"] = OnyxSegmentCache
NODE_CLASS_MAPPINGS["OnyxRifeVfiBatched"] = OnyxRifeVfiBatched
NODE_CLASS_MAPPINGS["OnyxPromptSaver"] = OnyxPromptSaver
NODE_CLASS_MAPPINGS["OnyxPromptGallery"] = OnyxPromptGallery
NODE_CLASS_MAPPINGS["OnyxVideoChainSegment"] = OnyxVideoChainSegment
NODE_CLASS_MAPPINGS["OnyxVideoChainJoin"] = OnyxVideoChainJoin
NODE_CLASS_MAPPINGS["OnyxVideoChainPrepare"] = OnyxVideoChainPrepare
NODE_CLASS_MAPPINGS["OnyxVideoChainCommit"] = OnyxVideoChainCommit
NODE_CLASS_MAPPINGS["OnyxAudioSwitch"] = OnyxAudioSwitch
NODE_CLASS_MAPPINGS["OnyxH3ContextIR"] = OnyxH3ContextIR
NODE_CLASS_MAPPINGS["OnyxGroupToggle"] = OnyxGroupToggle
NODE_CLASS_MAPPINGS["OnyxSaveImageNoMetadataNode"] = OnyxSaveImageNoMetadataNode
NODE_CLASS_MAPPINGS["OnyxSpeedHDSampler"] = OnyxSpeedHDSampler
NODE_CLASS_MAPPINGS["OnyxEyeBBoxDetectorProvider"] = OnyxEyeBBoxDetectorProvider
NODE_CLASS_MAPPINGS["OnyxDetailer"] = OnyxDetailer
if Onyx_Renoise:     NODE_CLASS_MAPPINGS["Onyx_Renoise"]     = Onyx_Renoise
if Onyx_Camera_Look: NODE_CLASS_MAPPINGS["Onyx_Camera_Look"] = Onyx_Camera_Look
if Onyx_Apply_LUT:   NODE_CLASS_MAPPINGS["Onyx_Apply_LUT"]   = Onyx_Apply_LUT

NODE_DISPLAY_NAME_MAPPINGS["OnyxNanoBananaAIO"] = "Onyx Image and Video Edit AIO"
NODE_DISPLAY_NAME_MAPPINGS["OnyxGeminiPromptNode"] = "Onyx Prompt Generator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPromptSelectorNodeMonthly"] = "Onyx Prompt Selector"
NODE_DISPLAY_NAME_MAPPINGS["OnyxDirectoryImageLoaderNode"] = "Onyx Directory Image Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxLoadAPIKeysNode"] = "Onyx Load API Keys"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoFrameExtractorNode"] = "Onyx Video Frame Extractor"
NODE_DISPLAY_NAME_MAPPINGS["OnyxMetadataRemoveNodeMonthly"] = "Onyx Remove Image Metadata"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSaveAsPhonePhotoNodeMonthly"] = "Onyx Save as Phone Photo"
NODE_DISPLAY_NAME_MAPPINGS["OnyxLoraCaptionGeneratorNode"] = "Onyx Lora Caption Generator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPreviewImageWithoutMetadata"] = "Onyx Preview No Metadata"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoLoader"]      = "Onyx Video Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxInstagramFaceSwapNode"]   = "Onyx Content Remaker"
NODE_DISPLAY_NAME_MAPPINGS["OnyxDatasetCreatorNode"]        = "Onyx Dataset Creator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageBatchLoader"] = "Onyx Image and Video Batch Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxReposeCarouselNode"] = "Onyx Re-pose for Carousel"
NODE_DISPLAY_NAME_MAPPINGS["OnyxMetadataBypassNode"] = "Onyx Metadata Bypass"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageEnhancementNode"] = "Onyx Image Enhancement"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageBlackCheckNode"] = "Onyx Image Black Check"
NODE_DISPLAY_NAME_MAPPINGS["OnyxGrokPromptNode"] = "Onyx Grok Prompt Generator"
NODE_DISPLAY_NAME_MAPPINGS["OnyxResolutionMP"] = "Onyx Resolution (MP)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxH3FrameSnap"] = "Onyx H3 Frame Snap"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageBlurBatched"] = "Onyx Image Blur (batched)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImageCompositeMaskedBatched"] = "Onyx Image Composite Masked (batched)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxImagesToVideo"] = "Onyx Images to Video (AB_VIDEO)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSaveVideoNoMetadata"] = "Onyx Save Video (no metadata)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSegmentCache"] = "Onyx Segment Cache"
NODE_DISPLAY_NAME_MAPPINGS["OnyxRifeVfiBatched"] = "Onyx RIFE VFI (batched)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPromptSaver"] = "Onyx Prompt Saver"
NODE_DISPLAY_NAME_MAPPINGS["OnyxPromptGallery"] = "Onyx Prompt Gallery"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainSegment"] = "Onyx Video Chain — Segment"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainJoin"] = "Onyx Video Chain — Join"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainPrepare"] = "Onyx Video Chain — Prepare"
NODE_DISPLAY_NAME_MAPPINGS["OnyxVideoChainCommit"] = "Onyx Video Chain — Commit"
NODE_DISPLAY_NAME_MAPPINGS["OnyxAudioSwitch"] = "Onyx Audio Switch"
NODE_DISPLAY_NAME_MAPPINGS["OnyxH3ContextIR"] = "H3 Context-IR (Gemini)"
NODE_DISPLAY_NAME_MAPPINGS["OnyxGroupToggle"] = "Onyx Group Toggle"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSaveImageNoMetadataNode"] = "Onyx Save Image No Metadata"
NODE_DISPLAY_NAME_MAPPINGS["OnyxSpeedHDSampler"] = "Onyx Speed HD Sampler"
NODE_DISPLAY_NAME_MAPPINGS["OnyxEyeBBoxDetectorProvider"] = "Onyx HD Ultralytic BBox Loader"
NODE_DISPLAY_NAME_MAPPINGS["OnyxDetailer"] = "Onyx Detailer"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Renoise"]     = "Onyx Renoise"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Camera_Look"] = "📷 Onyx Camera Look"
NODE_DISPLAY_NAME_MAPPINGS["Onyx_Apply_LUT"]   = "🎨 Onyx Apply LUT"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
