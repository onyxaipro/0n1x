# Onyx OFM Pack (Monthly) - Node Documentation

## Table of Contents

1. [Nano Banana Pro Edit](#nano-banana-pro-edit)
2. [Nano Banana 2 Edit](#nano-banana-2-edit)
3. [Seedream 4.5 Edit (Wavespeed)](#seedream-45-edit-wavespeed)
4. [Seedream 4.5 Edit (fal.ai)](#seedream-45-edit-falai)
5. [Prompt Generator](#prompt-generator)
6. [Prompt Selector](#prompt-selector)
7. [Directory Image Loader](#directory-image-loader)
8. [Video Frame Extractor](#video-frame-extractor)
9. [Lora Caption Generator](#lora-caption-generator)
10. [Remove Image Metadata](#remove-image-metadata)
11. [Save as Phone Photo](#save-as-phone-photo)
12. [Load API Keys](#load-api-keys)

---

## Nano Banana Pro Edit

**Display Name:** Onyx Nano Banana Pro Edit
**Category:** image

Edits or generates images using the Google Gemini 3 Pro image model. Accepts one or more input images along with a text prompt to guide the edit. Automatically detects aspect ratio from the input image when set to "auto".

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| images | IMAGE | Yes | — | Input image(s) to edit |
| api_key | STRING | Yes | — | Google Gemini API key |
| prompt | STRING | Yes | — | Text instruction describing the desired edit |
| aspect_ratio | Dropdown | No | auto | `auto`, `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| image_size | Dropdown | No | 2K | `1K`, `2K`, `4K` |
| temperature | FLOAT | No | 1.0 | Creativity control (0.0 - 2.0) |
| seed | INT | No | 0 | Random seed for reproducibility |
| safety_threshold | Dropdown | No | OFF | Content safety filter level |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | The edited/generated image |

---

## Nano Banana 2 Edit

**Display Name:** Onyx Nano Banana 2 Edit
**Category:** image

Edits images using the Google Gemini 3.1 Flash image model. Faster than Nano Banana Pro but with a smaller model. Defaults to 1K output resolution.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| images | IMAGE | Yes | — | Input image(s) to edit |
| api_key | STRING | Yes | — | Google Gemini API key |
| prompt | STRING | Yes | — | Text instruction describing the desired edit |
| aspect_ratio | Dropdown | No | auto | `auto`, `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| image_size | Dropdown | No | 1K | `1K`, `2K`, `4K` |
| temperature | FLOAT | No | 1.0 | Creativity control (0.0 - 2.0, step 0.05) |
| seed | INT | No | 0 | Random seed for reproducibility |
| safety_threshold | Dropdown | No | OFF | Content safety filter level |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | The edited/generated image |

---

## Seedream 4.5 Edit (Wavespeed)

**Display Name:** Onyx Seedream 4.5 Edit(Wavespeed)
**Category:** image

Edits images using ByteDance's Seedream 4.5 model hosted on WaveSpeed API. Supports 1-10 reference images for context-aware editing with facial feature, lighting, and color tone preservation. Can output up to 4 images per run. When size is set to "auto", it scales the input aspect ratio to approximately 2K resolution.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| images | IMAGE | Yes | — | 1-10 reference images |
| wavespeed_apikey | STRING | Yes | — | WaveSpeed API key |
| prompt | STRING | Yes | — | Text instruction for the edit |
| size | Dropdown | No | auto — match input (2K) | Preset sizes: auto, square 1:1, landscape 4:3, portrait 3:4, portrait 4:5, landscape 5:4, landscape 16:9, portrait 9:16 |
| num_images | INT | No | 1 | Number of output images (1-4) |
| use_custom_size | BOOLEAN | No | False | Enable custom width/height instead of presets |
| custom_width | INT | No | 2048 | Custom output width (512-8192, step 64) |
| custom_height | INT | No | 2048 | Custom output height (512-8192, step 64) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Edited image(s) as a batch tensor |

---

## Seedream 4.5 Edit (fal.ai)

**Display Name:** Onyx Seedream 4.5 Edit(fal.ai)
**Category:** image

Same Seedream 4.5 model as above but hosted on fal.ai. Functionally identical with a different API backend. Use this if you have a fal.ai API key instead of WaveSpeed.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| images | IMAGE | Yes | — | 1-10 reference images |
| fal_apikey | STRING | Yes | — | fal.ai API key |
| prompt | STRING | Yes | — | Text instruction for the edit |
| size | Dropdown | No | auto — match input (2K) | Preset sizes (same options as WaveSpeed version) |
| num_images | INT | No | 1 | Number of output images (1-4) |
| use_custom_size | BOOLEAN | No | False | Enable custom width/height instead of presets |
| custom_width | INT | No | 2048 | Custom output width (512-8192, step 64) |
| custom_height | INT | No | 2048 | Custom output height (512-8192, step 64) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Edited image(s) as a batch tensor |

---

## Prompt Generator

**Display Name:** Onyx Prompt Generator
**Category:** utils

Generates prompts using Gemini or Grok LLMs. Supports 6 modes: JSON image analysis (detailed visual breakdown), style transfer prompt extraction, 4 Seedream Edit configurations for multi-face/body reference workflows, and a custom prompt mode. Can optionally analyze input images.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| mode | Dropdown | Yes | JSON Image Analysis | `JSON Image Analysis`, `Style Transfer Prompt`, `Seedream Edit (2 face + 1 ref)`, `Seedream Edit (1 face + 1 ref)`, `Seedream Edit (2 face and body + 1 ref)`, `Seedream Edit (2 faces + 2 bodies + 1ref)`, `Custom Prompt` |
| provider | Dropdown | Yes | Gemini | `Gemini` or `Grok` |
| gemini_api_key | STRING | No | — | Google Gemini API key (required when provider is Gemini) |
| grok_api_key | STRING | No | — | xAI Grok API key (required when provider is Grok) |
| images | IMAGE | No | — | Input images for analysis |
| custom_prompt | STRING | No | — | Custom instruction text (used in Custom Prompt mode) |
| gemini_model | Dropdown | No | gemini-2.5-pro | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash` |
| grok_model | Dropdown | No | grok-4.20-0309-reasoning | Various Grok models (reasoning and non-reasoning variants) |
| temperature | FLOAT | No | 1.0 | Creativity control (0.0 - 2.0) |
| max_tokens | INT | No | 8192 | Maximum output length (64 - 65536) |
| safety_threshold | Dropdown | No | BLOCK_ONLY_HIGH | Content safety filter level |
| seed | INT | No | 0 | Random seed |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| prompt | STRING | The generated prompt text |

---

## Prompt Selector

**Display Name:** Onyx Prompt Selector
**Category:** utils

Select from pre-written example prompts stored in text files, or type your own. Prompts are loaded from `.txt` files in the `prompts/` directory (separated by `---` delimiters). If user-provided text is entered, it takes priority over the selected example.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| prompt_file | Dropdown | Yes | — | Select a prompt file or "Pose and Style Transfer Prompts" |
| example_prompt | Dropdown | Yes | — | Select a specific prompt from the loaded file |
| prompt_text | STRING | Yes | — | Custom prompt text (overrides selection if non-empty) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| prompt | STRING | The selected or typed prompt |

---

## Directory Image Loader

**Display Name:** Onyx Directory Image Loader
**Category:** image

Loads images one at a time from a folder, auto-incrementing through all files. Designed for batch workflows — automatically queues the next image after each run. Supports both relative paths (subfolders of ComfyUI/input/) and absolute paths.

Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.tif`

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| use_absolute_path | BOOLEAN | Yes | False | Toggle between input subfolder or absolute path |
| run_count | INT | Yes | 1 | Controls which image index to load (auto-increments) |
| input_folder | Dropdown | No | — | Subfolder within ComfyUI/input/ |
| absolute_path | STRING | No | — | Full path to an image directory |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | The loaded image tensor |
| filename | STRING | Name of the current file |
| filepath | STRING | Full path to the current file |
| current_index | INT | Current image index (1-based) |
| total_count | INT | Total number of images in the folder |

---

## Video Frame Extractor

**Display Name:** Onyx Video Frame Extractor
**Category:** image

Extracts frames from video files using PyAV. Upload a video to ComfyUI's input folder, choose a start time and frame count, and select which extracted frame to output. Includes a preview button in the UI to browse frames visually.

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v`

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| video | Dropdown | Yes | — | Video file from ComfyUI/input/ (supports upload) |
| start_second | FLOAT | Yes | 0.0 | Start time in seconds (0 - 86400) |
| frame_count | INT | Yes | 10 | Number of frames to extract (1 - 9999) |
| frame_interval | INT | Yes | 1 | Sample every Nth frame (1 - 300) |
| selected_frame | INT | Yes | 0 | Which extracted frame to use as output (0 - 9998) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | The selected frame as an image tensor |

---

## Lora Caption Generator

**Display Name:** Onyx Lora Caption Generator
**Category:** image

Generates LoRA training captions for images using Gemini or Grok vision APIs. Creates a `.txt` caption file per image with the same filename. Designed to work with the Directory Image Loader for batch captioning. Optionally prepends a trigger word to each caption.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| image | IMAGE | Yes | — | Input image to caption |
| filename | STRING | Yes | — | Filename (must be connected from another node) |
| api_key | STRING | Yes | — | Gemini or Grok API key |
| provider | Dropdown | Yes | gemini | `gemini` or `grok` |
| trigger_word | STRING | Yes | — | Word prepended to each caption (e.g. "ohwx") |
| output_folder_name | STRING | Yes | lora_captions | Subfolder name in ComfyUI/output/ |
| gemini_model | Dropdown | No | gemini-2.5-flash | Gemini model to use |
| grok_model | Dropdown | No | grok-2-vision-latest | Grok model to use |
| caption_instruction | STRING | No | (detailed default) | Custom instruction for the vision model |
| temperature | FLOAT | No | 0.7 | Creativity control (0.0 - 2.0) |
| max_tokens | INT | No | 512 | Maximum caption length (64 - 4096) |
| overwrite_existing | BOOLEAN | No | False | Whether to overwrite existing caption files |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| caption | STRING | The generated caption text |
| caption_filepath | STRING | Path to the saved .txt file |

---

## Remove Image Metadata

**Display Name:** Onyx Remove Image Metadata
**Category:** image

Batch-strips all metadata (EXIF, XMP, PNG chunks, ICC profiles, C2PA) from images in a folder and re-saves them as clean JPEGs. Optionally adds subtle noise to further break any steganographic fingerprints.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| folder | Dropdown | Yes | — | Subfolder within ComfyUI/input/ |
| filename_prefix | STRING | Yes | ComfyUI | Prefix for output filenames |
| output_folder | STRING | Yes | cleaned | Subfolder name in ComfyUI/output/ |
| quality | INT | Yes | 93 | JPEG quality (1-100) |
| add_noise | BOOLEAN | No | True | Add subtle noise to break fingerprints |

### Outputs

None (output node only). Saves cleaned images to `ComfyUI/output/<output_folder>/`.

---

## Save as Phone Photo

**Display Name:** Onyx Save as Phone Photo
**Category:** image

Batch-saves images with realistic phone camera EXIF metadata. Supports iPhone 15/16 Pro Max and Samsung Galaxy S24/S25 Ultra with authentic device-specific EXIF data (make, model, software version, focal length, f-number, ISO, exposure). Optionally adds GPS coordinates and subtle noise.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| folder | Dropdown | Yes | — | Subfolder within ComfyUI/input/ |
| filename_prefix | STRING | Yes | ComfyUI | Prefix for output filenames |
| phone_model | Dropdown | Yes | — | `iPhone 15 Pro Max`, `iPhone 16 Pro Max`, `Samsung Galaxy S24 Ultra`, `Samsung Galaxy S25 Ultra` |
| resolution | Dropdown | Yes | 12MP | `12MP`, `48MP/50MP`, `Original` |
| gps_location | Dropdown | Yes | Random US City | `None`, `New York`, `Los Angeles`, `London`, `Tokyo`, `Paris`, `Sydney`, `Dubai`, `Random US City` |
| output_folder | STRING | Yes | phone_photos | Subfolder name in ComfyUI/output/ |
| quality | INT | Yes | 95 | JPEG quality (1-100) |
| add_noise | BOOLEAN | No | True | Add subtle noise for realism |

### Outputs

None (output node only). Saves images to `ComfyUI/output/<output_folder>/`.

---

## Load API Keys

**Display Name:** Onyx Load API Keys
**Category:** utils

Simple utility node to enter API keys once and wire them to multiple nodes that need authentication. Avoids duplicating keys across your workflow.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| gemini_api_key | STRING | Yes | — | Google Gemini API key |
| wavespeed_api_key | STRING | Yes | — | WaveSpeed API key |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| gemini_api_key | STRING | Pass-through Gemini key |
| wavespeed_api_key | STRING | Pass-through WaveSpeed key |
