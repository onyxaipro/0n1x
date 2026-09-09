"""
ComfyUI node for selecting fixed prompts or example prompts from text files.
Prompts are stored in the prompts/ directory, separated by '---'.
"""

import logging
import os


_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

_FIXED_PROMPTS = {
    "Simple Pose and Style Transfer": "image 2 posing and dressing like image 1",
}


def _get_prompt_files():
    """List .txt files in the prompts directory, with Pose and Style Transfer Prompts first."""
    files = ["Pose and Style Transfer Prompts"]
    if os.path.isdir(_PROMPTS_DIR):
        txt_files = [f for f in os.listdir(_PROMPTS_DIR) if f.endswith(".txt")]
        files.extend(sorted(txt_files))
    return files if len(files) > 1 else ["Pose and Style Transfer Prompts"]


def _load_all_prompts():
    """Load prompts from all files.

    Returns:
        prompt_map: {label: full_text} for all prompts
        all_labels: flat list of all labels
        by_file: {file_stem: [label, ...]} grouped by source file
    """
    prompt_map = {}
    all_labels = []
    by_file = {}
    counter = 1
    for fname in _get_prompt_files():
        if fname.startswith("("):
            continue
        fpath = os.path.join(_PROMPTS_DIR, fname)
        file_stem = os.path.splitext(fname)[0]
        file_labels = []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw = f.read()
            for block in raw.split("---"):
                text = block.strip()
                if text:
                    preview = text[:60].replace("\n", " ")
                    if len(text) > 60:
                        preview += "..."
                    label = f"[{file_stem}] {counter}. {preview}"
                    prompt_map[label] = text
                    all_labels.append(label)
                    file_labels.append(label)
                    counter += 1
        except Exception as e:
            logging.warning("PromptSelector: failed to read %s: %s", fname, e)
        if file_labels:
            by_file[file_stem] = file_labels
    if not all_labels:
        all_labels = ["(no prompts found)"]
    return prompt_map, all_labels, by_file


_PROMPT_MAP, _PROMPT_LABELS, _PROMPTS_BY_FILE = _load_all_prompts()

# Build fixed prompt labels
_FIXED_LABELS = list(_FIXED_PROMPTS.keys())
_ALL_LABELS = _FIXED_LABELS + _PROMPT_LABELS

# Serve prompt data to frontend (unique routes for Monthly pack)
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/prompt_selector_monthly/get_prompts")
    async def _get_prompts_monthly_api(request):
        combined = dict(_FIXED_PROMPTS)
        combined.update(_PROMPT_MAP)
        return web.json_response(combined)

    @PromptServer.instance.routes.get("/prompt_selector_monthly/get_prompts_by_file")
    async def _get_prompts_by_file_monthly_api(request):
        result = {"Pose and Style Transfer Prompts": _FIXED_LABELS}
        result.update(_PROMPTS_BY_FILE)
        return web.json_response(result)
except Exception:
    logging.warning("PromptSelector: could not register API routes (server not available)")


class PromptSelectorNode:
    """Select from fixed prompts or pre-written example prompts stored in text files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_file": (_get_prompt_files(),),
                "example_prompt": (_ALL_LABELS,),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_prompt"
    CATEGORY = "utils"

    def select_prompt(self, prompt_file, example_prompt, prompt_text=""):
        if prompt_text.strip():
            return (prompt_text.strip(),)
        # Check fixed prompts first
        if example_prompt in _FIXED_PROMPTS:
            return (_FIXED_PROMPTS[example_prompt],)
        full_text = _PROMPT_MAP.get(example_prompt, "")
        return (full_text,)
