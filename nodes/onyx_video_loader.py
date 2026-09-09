"""
Onyx Video Loader — Charge un fichier vidéo depuis input/ de ComfyUI.
Sortie unique : AB_VIDEO (chemin absolu) → à connecter à video_reference du node AIO.
"""
import os

try:
    import folder_paths as _fp
except ImportError:
    _fp = None

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _list_videos():
    if not _fp:
        return ["[folder_paths non disponible]"]
    input_dir = _fp.get_input_directory()
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in _VIDEO_EXTENSIONS
        and os.path.isfile(os.path.join(input_dir, f))
    )
    return files or ["[aucune vidéo dans input/]"]


class OnyxVideoLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_list_videos(), {
                    "video_upload": True,
                }),
            }
        }

    RETURN_TYPES = ("AB_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION     = "load_video"
    CATEGORY     = "Onyx"

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        if _fp:
            path = os.path.join(_fp.get_input_directory(), video)
            if os.path.isfile(path):
                return os.path.getmtime(path)
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not _fp:
            return "folder_paths non disponible."
        if video.startswith("["):
            return "Aucune vidéo dans ComfyUI/input/ — uploadez-en une d'abord."
        if not _fp.exists_annotated_filepath(video):
            return f"Fichier introuvable : {video}"
        return True

    def load_video(self, video):
        if not _fp:
            raise RuntimeError("[Onyx Video Loader] folder_paths non disponible.")
        video_path = _fp.get_annotated_filepath(video)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"[Onyx Video Loader] Fichier introuvable : {video_path}")
        print(f"🎞️  [Onyx Video Loader] {os.path.basename(video_path)}")
        return (video_path,)
