# onyx OFM ComfyUI Nodes (Monthly)

Custom ComfyUI nodes for AI-powered image editing and prompt management.

## Nodes

### Nano Banana Pro Edit

Edit and transform images using Google's Gemini 3 Pro model through a simple API call. Supports custom aspect ratios, resolution up to 4K, and adjustable safety filters.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| images | IMAGE | — | Input image(s) to edit |
| api_key | String | — | Your Google Gemini API key |
| prompt | Multiline String | — | Instructions for how to edit the image |
| aspect_ratio | Dropdown | `auto` | Output aspect ratio: auto, 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9 |
| image_size | Dropdown | `2K` | Output resolution: 1K, 2K, or 4K |
| temperature | Float (0–2) | 1.0 | Controls creativity — higher = more creative |
| safety_threshold | Dropdown | `BLOCK_ONLY_HIGH` | Content safety filter level |

**Output:** `IMAGE` tensor

---

### Nano Banana 2 Edit

A faster, lighter image editor powered by Gemini 3.1 Flash for quick image edits and modifications. Same flexible controls as Pro but optimized for speed.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| images | IMAGE | — | Input image(s) to edit |
| api_key | String | — | Your Google Gemini API key |
| prompt | Multiline String | — | Instructions for how to edit the image |
| aspect_ratio | Dropdown | `auto` | Output aspect ratio |
| image_size | Dropdown | `1K` | Output resolution: 1K, 2K, or 4K |
| temperature | Float (0–2) | 1.0 | Controls creativity |
| safety_threshold | Dropdown | `BLOCK_ONLY_HIGH` | Content safety filter level |

**Output:** `IMAGE` tensor

---

### Example Prompt Selector

Browse and select from pre-written example prompts stored in text files. Outputs the selected prompt as a STRING to connect to other nodes.

**Inputs:**

| Input | Type | Description |
|---|---|---|
| prompt_file | Dropdown | Select which `.txt` file to browse prompts from |
| example_prompt | Dropdown | Browse prompts from the selected file |
| prompt_text | Multiline String | Editable text area — auto-populated when you select a prompt. Edit to customize. |

**Output:** `STRING` — the prompt text. Uses `prompt_text` if manually edited, otherwise the selected example prompt.

**Navigation:** Use the `Prev Prompt` / `Next Prompt` buttons to cycle through prompts, or click the dropdown to pick directly.

#### Adding Your Own Prompts

1. Create a `.txt` file in the `prompts/` folder (e.g., `prompts/my_styles.txt`)
2. Write one prompt per block, separated by `---` on its own line:

```
A beautiful sunset over the ocean, golden light, dramatic clouds
---
A cyberpunk street scene at night, neon reflections, rain-soaked pavement
---
A cozy cabin interior with fireplace, warm lighting, rustic furniture
```

3. Restart ComfyUI — the new file appears in the `prompt_file` dropdown

## File Structure

```
Onyx/
├── __init__.py              # Node registration
├── Node.py                  # NanoBananaProEditAPINode, NanoBanana2EditAPINode
├── prompt_selector.py       # PromptSelectorNode + API routes
├── js/
│   └── prompt_selector.js   # Frontend: prompt navigation + text population
└── prompts/
    └── *.txt                # Prompt files (--- separated)
```
