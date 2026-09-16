# -*- coding: utf-8 -*-
from .onyx_render_profile import ensure_profile_ready
"""
ComfyUI node - H3 Context-IR (Gemini).

Rebuilds, locally, the preprocessing stage the MiniMax H3 cloud API runs before
H3-Base: read the actual media, reason about it, and hand the video model a
structured, enriched instruction instead of a bare prompt.

Two passes, deliberately separate:

  Pass A - grounding. The real media go to a vision model, which returns plain
  facts as JSON. No H3 formatting at this stage, only "what is in these files".

  Pass B - compilation. That JSON plus the user's intent become an H3 prompt.
  This pass never sees a pixel.

The split is the whole point. Asked to look and to format in one call, the model
skims the media and falls back on generic phrasing, because most of its output
budget goes to satisfying a long format specification. Splitting forces it to
commit to observations first, then write from them.

Pass B's system prompt is the official MiniMax guide read from disk, verbatim.
The rules are never paraphrased here: a paraphrase drifts from the source the
moment the source is updated.
"""

import base64
import hashlib
import io
import json
import logging
import os
import struct
import time
import wave

import numpy as np
import requests
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Official guides
# ─────────────────────────────────────────────────────────────────────────────

# Both files are needed, not one. guides/README.md in ComfyUI-MiniMaxH3-Prompt-
# Writer states they are vendored verbatim from MiniMaxAI/MiniMax-H3 at revision
# bfc8ed0353f5a9733be73e6b2c98ec0948195b86, and the reference guide opens by
# saying its shot, camera, speaker and dialogue formats are "shared with" the
# base guide. Loading only the reference guide therefore leaves the compiler
# without the formats it is told to reuse.
_GUIDE_BASE = "VIDEO_PROMPT_WRITING_GUIDE_base_en.md"
_GUIDE_REF = "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"

_DEFAULT_GUIDE_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ComfyUI-MiniMaxH3-Prompt-Writer-main", "guides",
)

_guide_cache = {}


def _load_guides(folder: str) -> str:
    """Concatenate both official guides, verbatim, base first."""
    folder = (folder or "").strip() or _DEFAULT_GUIDE_FOLDER
    key = os.path.abspath(folder)
    if key in _guide_cache:
        return _guide_cache[key]

    if not os.path.isdir(folder):
        raise RuntimeError(
            f"[H3 Context-IR] Guide folder not found: {folder}\n"
            f"-> Point 'guide_folder' at the 'guides' directory of "
            f"ComfyUI-MiniMaxH3-Prompt-Writer."
        )

    chunks = []
    for name in (_GUIDE_BASE, _GUIDE_REF):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"[H3 Context-IR] Missing guide: {path}\n"
                f"-> Both {_GUIDE_BASE} and {_GUIDE_REF} are required; the reference "
                f"guide reuses the formats defined in the base guide."
            )
        with open(path, "r", encoding="utf-8") as fh:
            chunks.append(fh.read())

    text = "\n\n".join(chunks)
    _guide_cache[key] = text
    print(f"📖 [H3 Context-IR] Guides loaded from {folder} ({len(text)} chars)")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_PASS_A_SYSTEM = """You are a media analyst. You look at the supplied assets and report what is actually there.

Return ONLY a JSON object, no prose, no markdown fence. Schema:

{
  "assets": [
    {
      "label": "<Picture 1>|<Video 1>|<Audio 1>",
      "kind": "image|video|audio",
      "subjects": [
        {
          "id": "subject_1",
          "type": "person|animal|object",
          "face": "face shape; eye shape, colour and spacing; eyebrow shape and thickness; nose bridge and tip; lip shape and fullness; cheekbones; chin and jawline; every distinguishing mark with where it sits (freckles, moles, scars, dimples); visible age range; makeup actually visible",
          "hair": "colour; length; texture; parting; and above all HOW IT IS STRUCTURED RIGHT NOW - loose, braided (say which kind and where), in a bun, ponytail, half-up, tied back, pinned, with a fringe. Name the structure explicitly: a braid described as 'long hair' cannot be reproduced. Hair colour is ONE colour unless a second one is unmistakably there. Darkening at the parting, where the hair meets the scalp, or under any overhanging mass is SHADOW - it is on every head of hair ever photographed. Report a second tone only when its boundary shows a change of HUE, not merely of brightness. Dyed hair carries a strong expectation of grown-out roots; do not satisfy that expectation from a shadow. When the length is a single colour, say so in those words - 'uniform <colour> from the roots to the ends' - because saying nothing on the point lets whoever reads the report assume the usual two-tone pattern. When a second tone IS unmistakable, say for each one WHERE IT BEGINS AND WHERE IT ENDS and which occupies most of the length: 'red with dark roots' leaves the second tone free to appear anywhere, 'vivid red from two centimetres below the parting to the ends, darker roots in a narrow band at the parting only' does not",
          "skin": "tone, texture, visible marks",
          "wardrobe": "every visible garment: type, cut, colour, pattern, fabric",
          "accessories": "jewellery, glasses, headwear, or an empty string",
          "body": "build, proportions, visible posture"
        }
      ],
      "framing": "shot size, camera height, angle, what is inside and outside the frame",
      "environment": "location, surfaces, notable objects and their colours",
      "lighting": "direction, hardness, colour temperature, where the shadows fall",
      "timeline": [
        {"t": "0.000s", "event": "what is visibly happening at this timecode"}
      ],
      "audio_events": [
        {"t": "0.000s", "event": "what is audible, with its source"}
      ]
    }
  ],
  "relations": "how the assets relate to one another: same person, same scene, same moment, or unrelated"
}

Rules:

- Report only what is visible or audible. Never infer, complete, or assume.
- wardrobe MUST be null when the framing shows no garment at all. A close-up that stops above the shoulders shows no garment. Do not deduce an outfit from a neckline, a strap, a shadow, or the setting.
- accessories, timeline and audio_events are empty when there is nothing to report.
- Every timecode you write must come from a frame label given to you. Never invent one.
- Colours are named concretely: "dark olive cotton", not "dark top".
- If an asset contains no person, subjects is an empty list.
- A single image may be a contact sheet: several panels showing the SAME subject from different angles, often front, profile, three-quarter and back. Report it as ONE subject, and use every panel to describe that subject more completely — a profile panel tells you the nose and jaw line a frontal panel cannot. Never report the panels as separate people.
- When several assets show the same subject, say so explicitly in "relations" and describe the subject once, merging what each asset reveals.
- Describe a face by its measurable landmarks, not by an impression. "Almond eyes set wide apart, straight nose with a slightly upturned tip, full lower lip, defined cupid's bow, sharp jawline" can be reproduced; "pretty, delicate features" cannot. Whoever reads your report never sees the image.
- A flat region of uniform colour covering part of a body — a head, a face, a torso — is a MASK: an editing artefact standing where something was removed. It is not clothing, not a prop, not paint, not an object in the scene. Treat it as an absence. Set the field it covers to null and write nothing about it: not its colour, not its shape, not its edges, not that anything covers anything. Whoever reads your report must come away believing that region was simply never shown, because the only reason to mention a mask is to have it reproduced.
"""

# Chaque regle correspond a un echec de generation observe, pas couvert par le
# guide officiel. Elles sont concatenees APRES le guide, jamais a la place.
_PASS_B_ADDENDUM = """
---

# Output addendum

These rules are additional to the guide above. They never override it.

1. No negation in the output. Every exclusion is written as a positive, observable final state. Not "she does not move her hands" but "her hands stay flat on the table for the whole shot".

2. Spell out every inherited attribute. An attribute taken from a reference must be written in full words, not merely pointed at by its label. The model drifts on everything left unverbalised, wardrobe first.

3. Keep retention and replacement as two disjoint lists of named attributes. Never write a blanket clause such as "everything else stays the same": it drowns the replacement instruction and the model then replaces nothing.

4. Observable behaviour only. Emotion is written as what a camera records: where the eyes go, what the hands do, what stays still.

5. Coloured lighting is a physical off-frame source, with falloff and a defined shadow side. Never a wash saturating the whole image, which produces a whitish veil.

6. Resolve every label you use. Never point at an asset that was not supplied.

7. When a reference video supplies the motion, give that motion its own `<Subject N>` in `subject_definitions` — "whose motion comes from `<Video N>`", following the guide's own multi-asset subject pattern — and mark that subject `fully_preserved` in `retention_analysis`. The relationship marker applies to the role you defined, so a narrow role marked `fully_preserved` is a stronger instruction than the whole video marked `partially_preserved`, which states that its contribution is only partly retained.

8. When a reference video supplies the motion, keep the written account of that motion at the level of what is actually observable in the supplied frames. The video carries the exact movement already; a confident sentence that is coarser than the footage competes with it instead of reinforcing it.

Return ONLY the finished prompt. No preamble, no commentary, no markdown fence.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Seedance
#
# Different target, different grammar. ByteDance publishes a six-slot formula and
# an @-tag binding syntax, and its guidance runs opposite to H3's on one point
# that matters: length. H3's guide asks for an exhaustive detailed_description;
# Seedance's says a long paragraph "fights itself" and that two or three
# sentences is the sweet spot. Compiling a Seedance prompt against the H3 guide
# would therefore produce exactly the failure mode ByteDance warns about.
#
# Source: ByteDance's published grammar as documented at
# blog.segmind.com/the-official-seedance-2-5-prompt-guide-bytedances-six-part-
# formula-explained-with-examples. Secondary source, not ByteDance's own page.
# ─────────────────────────────────────────────────────────────────────────────

TARGET_H3 = "MiniMax H3"
TARGET_SEEDANCE = "Seedance"
_TARGETS = [TARGET_H3, TARGET_SEEDANCE]

_SEEDANCE_SYSTEM = """You write prompts for ByteDance Seedance from grounded observations. Follow its published grammar exactly.

# Slot order

Subject + Action or Event + Scene and Environment + Visual Style + Camera Movement or Cut + Audio.

Only Subject and Action are required. Leave a slot empty rather than filling it with something you were not asked for: competing style and camera instructions fight each other and the output turns to mush.

# Length

Two or three sentences per block is the target. A long paragraph that repeats the same instruction in six different wordings performs WORSE than a short one, because the model has to reconcile phrasings instead of following an instruction. Clearer beats longer. This rule overrides any instinct to be thorough.

# Reference binding

Files are tagged `@Image 1`, `@Video 1`, `@Audio 1` — with a space before the number.

Every reference gets a job AND an exclusion. The exclusion is the half most people skip, and it is where reference-driven generation fails: a reference image carries its subject, its background, its lighting and its framing, so a reference introduced without exclusions donates all of it.

    @Image 1 defines the woman's face and hair only. Do not use its background, its clothing or its framing.

An exclusion may only name something the observations actually recorded in that file. If the report says nothing about a face in a video, that video has no face, and "do not use its face" asserts one — the model reads the noun, not the "do not", and supplies what you named. Silence is the strongest exclusion there is: leave the attribute unwritten and nothing can be donated from it. So exclude the background, the wardrobe or the framing when the report describes them, and say nothing at all about what the report left empty.

One owner per attribute, and only the owner is named. Once you have assigned the face to `@Image 1`, no other file's face is mentioned at all — not to exclude it, not to contrast with it. The positive assignment already settles the question; adding "do not use the face or hair of @Video 1" puts a competing face and a competing head of hair into the prompt, and that is the whole of what the model receives. Write the exclusions a file needs for attributes NOBODY was assigned, and stay silent about every attribute that already has an owner.

Tie every character, prop and product to exactly one file. When two files could plausibly define the same element, the model is choosing at random.

# When an image is tagged "first frame of the target video"

That image is not a person to copy from. It IS frame one of the clip you are describing, already composited — the identity question is settled in its pixels before the model reads a word.

This changes what you write, in one direction: LESS.

Do not spell out the face in landmarks. Landmarks exist to rebuild an identity the model has to transfer; here it has nothing to transfer, and a written description competes with the pixels it already has. Name the person plainly — "the woman in @Image 1" — and move on to what she DOES.

Say that the clip opens on @Image 1 exactly, and put "her face and hair stay exactly as established in the opening frame" in `[Maintain Consistency]`. Those two lines carry the whole identity job.

The reference video then donates motion, timing, framing and scene. Its own subject is not the target and is never described — not their face, not their build, not their hair.

# Structure

    [Generation Goal] Video type and the central event.
    [Stage 1] Initial state, one primary event, end state.
    [Stage 2] Continue from that end state, new event, new end state.
    [Maintain Consistency] What must not change across the clip.

One main change and one clear end state per stage. Two events in a stage split the motion budget and soften the transition.

Use stages only when the clip is long enough to need them. A short single-action clip takes the goal, the bindings and the consistency block, and nothing else.

`[Maintain Consistency]` is where character count, clothing, prop ownership and spacing are pinned — those are exactly what drifts across a generation.

It is also where you cover the views the reference does NOT show. A portrait shows a face and hair from one angle; the clip turns the head to profile, moves the hair, shows the back of it. Those frames have no reference behind them, so the model falls back on the only other source it has — the video — and that is where a replaced attribute reappears in its original form, in patches rather than throughout. Name the donated attributes once more here and state that they hold through the turns and the movement, including the parts of the hair that fall behind the shoulders.

# Audio brackets

    ( )  music and ambient beds
    < >  sound effects
    { }  spoken dialogue
    【 】  on-screen subtitles

Dialogue and subtitles are separate channels: a line in `{ }` is heard, not seen. Declare the language and delivery before the braces, and put only the spoken words inside. Keep effects short and physical — two or three concrete sounds beat a paragraph of sound design.

Write nothing in these brackets unless the intent or the observations call for it.

# Output rules

1. Spell out every inherited attribute in full words. "Her face from @Image 1" drifts; "her oval face, light green eyes and bright red center-parted hair from @Image 1" holds. The model drifts on everything left unverbalised.

2. State exclusions positively where you can, and only name what the file must NOT donate when there is no positive way to say it. "Do not use its background" is fine; a paragraph of prohibitions is not.

3. Never point at a file that was not supplied.

Return ONLY the finished prompt. No preamble, no commentary, no markdown fence.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Media encoding
# ─────────────────────────────────────────────────────────────────────────────

_VIDEO_KEYFRAMES = 8
_AUDIO_SR = 16000  # mono 16-bit at 16 kHz: speech-grade, and small enough to inline


def _tensor_to_pil(frame) -> Image.Image:
    arr = (255.0 * frame.cpu().numpy()).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_png_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _audio_to_wav_bytes(audio) -> bytes:
    """ComfyUI AUDIO -> mono 16-bit PCM WAV at 16 kHz.

    Mono 16-bit is what actually populates overall_soundscape with dated events
    instead of a generic sentence: the model needs a waveform it can transcribe,
    not a high-fidelity stereo mix. 16 kHz keeps a 15 s clip near 480 KB, well
    inside an inline request.
    """
    import torch

    wav = audio["waveform"]
    sr = int(audio["sample_rate"])

    if wav.ndim == 3:      # (B, C, N) -> first item
        wav = wav[0]
    if wav.ndim == 2:      # (C, N) -> mono
        wav = wav.mean(dim=0)
    wav = wav.detach().cpu().float()

    if sr != _AUDIO_SR:
        n_out = max(1, int(round(wav.shape[0] * _AUDIO_SR / float(sr))))
        wav = torch.nn.functional.interpolate(
            wav.view(1, 1, -1), size=n_out, mode="linear", align_corners=False
        ).view(-1)

    peak = float(wav.abs().max()) if wav.numel() else 0.0
    if peak > 1.0:
        wav = wav / peak
    pcm = (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_AUDIO_SR)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# Le modele echantillonne la video de reference a 2 images par seconde, et pas
# autrement. Extrait de comfy_extras/nodes_minimax_h3.py :
#
#     sample_idx = list(range(0, frames.shape[0], FPS // 2))   # FPS = 24
#     qwen_frames = frames[sample_idx]
#     ref_items.append({"type": "video", "data": qwen_frames,
#                       "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
#
# C'est LA vue que le text encoder a du mouvement. Un timestamp d'action ecrit
# a 7.333s designe donc un instant que le modele ne possede pas : sa bande n'a
# que 7.0 et 7.5, et le repere tombe entre deux images.
# Regles ajoutees APRES le dernier rendu juge bon par l'utilisateur.
#
# Chacune corrigeait un defaut reel et observe. Ce qui n'a jamais ete
# mesure, c'est leur effet CUMULE : elles s'empilent dans le meme contexte
# que le guide officiel, et le compilateur en fait la moyenne. Regroupees
# ici et desactivees par defaut, pour que la forme du prompt reste celle du
# run de reference tant qu'on n'en redemande pas une explicitement.
_EXTRA_RULES = """
# Never write stillness

When a reference video supplies the motion, do not describe the subject as still. "Her torso stays still", "her eyes stay fixed forward", "she remains seated through 00:05.875" — every one of those is an instruction to stop moving, and the model obeys it. A subject told to hold still, in a prompt that also carries a reference photograph, converges on that photograph: a static portrait is the cheapest way to satisfy "no motion, facing forward". The clip then stops being the source video and becomes the still image, usually in the last seconds, where the described events have run out.

This is not the same as running out of things to say. If the observations record no notable event in a stretch, that stretch still has motion in it — breathing, chewing, small head and gaze shifts, the camera. So write, once: "her movement continues from @Video 1 through to the final frame." That assigns the whole interval to the file that actually holds it, and it costs one clause.

Concretely, for the tail of a clip:
- Never close a shot with a stasis clause. "through 00:05.875" attached to "stays still" is the single most reliable way to freeze an ending.
- Never write "remains", "stays", "holds steady", "unchanged" or "motionless" about a body part or a pose. Those verbs belong to IDENTITY - the face, the hair, the wardrobe - and never to movement.
- The last described beat should be an action or a continuation, not a resting state.
"""


_MODEL_REF_FPS = 2.0
_MODEL_FPS = 24


def _sample_keyframes(frames, fps: float, count: int = _VIDEO_KEYFRAMES,
                      model_aligned: bool = False) -> list:
    """Sample frames for pass A, each paired with its timecode in seconds.

    Two modes:

    even          `count` frames spread across the clip. Independent of what the
                  model sees - which is the point when the video is being read
                  for content rather than for timing.
    model_aligned exactly the frames H3 hands to Qwen: every 12th at 24 fps,
                  labelled 0.0, 0.5, 1.0 ... Pass A then describes the very
                  images the model will have, with the same labels, so a
                  timestamp in the prompt points at something rather than
                  between two things.

    `count` is ignored when model_aligned is on: the stride is the model's, not
    a budget. A 294-frame segment gives 25 frames either way, so this costs
    nothing in practice.
    """
    total = int(frames.shape[0])
    if total == 0:
        return []

    src_fps = fps if fps and fps > 0 else float(_MODEL_FPS)

    if model_aligned:
        # Le pas est celui du modele : une frame sur 12 a 24 fps. Sur une source
        # a un autre debit on garde 0.5s d'ecart reel, ce qui reproduit la meme
        # densite temporelle plutot que le meme entier.
        stride = max(1, int(round(src_fps / _MODEL_REF_FPS)))
        idxs = list(range(0, total, stride))
        return [(i / src_fps, _tensor_to_pil(frames[i])) for i in idxs]

    count = min(count, total)
    idxs = [int(round(i * (total - 1) / max(1, count - 1))) for i in range(count)] \
        if count > 1 else [0]
    return [(i / src_fps, _tensor_to_pil(frames[i])) for i in idxs]


# Roles mirror the official Context-IR payload, where every asset carries what it
# is FOR, not merely where it sits in the list. An image tagged first_frame is the
# opening frame of the render; a bare <Picture 1> only says "the first reference",
# and the compiler has to guess which of the two it is.
ROLE_REFERENCE = "reference"
ROLE_FIRST = "first_frame"
ROLE_LAST = "last_frame"
_ROLES = [ROLE_REFERENCE, ROLE_FIRST, ROLE_LAST]

_ROLE_LABEL = {
    ROLE_REFERENCE: "reference image",
    ROLE_FIRST: "first frame of the target video",
    ROLE_LAST: "last frame of the target video",
}


def _collect_pictures(slots, batch) -> list:
    """Ordered [(origin, frame, role)]: discrete slots first, then batch items.

    Labels are assigned by POSITION IN THIS LIST, not by socket number. Leaving
    image_2 empty while filling image_1 and image_3 yields <Picture 1> and
    <Picture 2>, never a gap: a prompt that resolves <Picture 3> when only two
    images were sent is exactly the dangling-label failure the addendum forbids.
    The mapping is printed so the numbering is never something to deduce.
    """
    frames = []
    for slot_no, (tensor, role) in enumerate(slots, 1):
        if tensor is None:
            continue
        for i in range(int(tensor.shape[0])):
            frames.append((f"image_{slot_no}", tensor[i], role))
    if batch is not None:
        for i in range(int(batch.shape[0])):
            frames.append((f"images_batch[{i}]", batch[i], ROLE_REFERENCE))
    return frames


def _build_grounding_parts(pictures, video, video_fps, video_audio, ref_audios=None,
                           keyframes=_VIDEO_KEYFRAMES, model_aligned=False) -> list:
    """Ordered [(kind, payload)] list for the vision pass.

    A text label precedes every asset. Without it the model receives an unordered
    bag of images and has to guess which is which, so its report attributes
    features to the wrong asset. The per-frame timecode exists for the same
    reason on the time axis: with no clock, shot boundaries get invented.
    """
    parts = []

    for i, (origin, frame, role) in enumerate(pictures, 1):
        parts.append(("text", f"<Picture {i}> — {_ROLE_LABEL[role]}:"))
        parts.append(("image", _pil_to_png_bytes(_tensor_to_pil(frame))))

    if video is not None and int(video.shape[0]) > 0:
        kfs = _sample_keyframes(video, video_fps, keyframes, model_aligned)
        dur = int(video.shape[0]) / (video_fps if video_fps > 0 else 24.0)
        parts.append((
            "text",
            f"<Video 1> — source video, {int(video.shape[0])} frames, "
            f"{dur:.3f}s at {video_fps:.2f} fps, sampled below as "
            f"{len(kfs)} keyframes in chronological order"
            + (". These are EXACTLY the frames the video model itself receives, at "
               "the same 2 frames per second and under the same timecodes, so a time "
               "you read here is a time it can act on."
               if model_aligned else "") + ":",
        ))
        for t, pil in kfs:
            parts.append(("text", f"<Video 1> frame at {t:07.3f}s:"))
            parts.append(("image", _pil_to_png_bytes(pil)))

    # Deux natures d'audio, deux etiquettes. La bande-son appartient a <Video 1>
    # et n'est pas un asset separe ; un audio fourni a part est <Audio N>, comme
    # dans la charge utile officielle ou reference_video et reference_audio sont
    # deux entrees distinctes. Les confondre ferait resoudre un <Audio 1> qui
    # n'existe pas, ou l'inverse.
    if video_audio is not None:
        parts.append(("text", "<Video 1> soundtrack — the audio that belongs to <Video 1>:"))
        parts.append(("audio", _audio_to_wav_bytes(video_audio)))

    for i, ref in enumerate(ref_audios or [], 1):
        if ref is None:
            continue
        parts.append(("text", f"<Audio {i}> — standalone reference audio:"))
        parts.append(("audio", _audio_to_wav_bytes(ref)))

    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Pass A0 - speech transcription with measured word timings
#
# Why this exists: pass A is a VISION pass. Asked when a line is spoken, it
# reads keyframes and estimates, and it estimates badly - a scream observed at
# 6.5s in the source came back written as 00:05.200, four timecoded keyframes
# away. H3 encodes time absolutely, so that error lands in the render as speech
# that starts a second early.
#
# gemini-3.5-transcribe does not estimate. It returns start and end offsets per
# word. Handing pass A a measurement instead of asking it for a guess is the
# whole idea.
#
# Two things about this endpoint that are unlike everything else in this file:
#
# Two routes, because the two providers expose this differently:
#
#   Vertex : model gemini-3.5-transcribe-preview, through the ordinary
#            generate_content, with audio_transcription_config in the config.
#            No extra key, no file upload - it reuses the service account this
#            node already authenticates with. `global` only, which is the
#            location this node already uses.
#   Gemini : model gemini-3.5-transcribe, through the Interactions API
#            (/v1beta/interactions) after a Files API upload. Different URL,
#            different request and response shapes entirely.
#
# Documented limits worth knowing before switching it on:
#   - Word-level timestamps are EXPERIMENTAL and "degrade transcription
#     accuracy". That trade is the right way round here: we want WHEN a line
#     starts, and a slightly worse spelling of it costs nothing.
#   - Diarization is experimental beyond 2 speakers.
#   - Audio is capped at 15 minutes with timestamps enabled.
#   - The model supports no system instructions, so this cannot go through
#     _dispatch: it needs its own call.
#
# Cost: ONE call per unique soundtrack, cached to disk beside the grounding. A
# four-segment chain transcribes once, not four times.
# ─────────────────────────────────────────────────────────────────────────────

# Deux identifiants distincts : Vertex publie la variante -preview, l'API Gemini
# la variante stable. Les confondre donne un 404 sans rapport apparent.
_TRANSCRIBE_MODEL_VERTEX = "gemini-3.5-transcribe-preview"
_TRANSCRIBE_MODEL_GEMINI = "gemini-3.5-transcribe"
_GENAI_ROOT = "https://generativelanguage.googleapis.com"


def _upload_audio_file(api_key: str, wav: bytes, timeout: int = 120) -> str:
    """Files API upload. Returns the file URI.

    The docs recommend it for anything longer than a few seconds, and a source
    video's soundtrack always is.
    """
    import requests
    start = requests.post(
        f"{_GENAI_ROOT}/upload/v1beta/files?key={api_key}",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(wav)),
            "X-Goog-Upload-Header-Content-Type": "audio/wav",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": "h3_context_ir_soundtrack"}},
        timeout=timeout,
    )
    start.raise_for_status()
    upload_url = start.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise RuntimeError("the Files API did not return an upload URL.")

    done = requests.post(
        upload_url,
        headers={
            "Content-Length": str(len(wav)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        data=wav,
        timeout=timeout,
    )
    done.raise_for_status()
    uri = (done.json().get("file") or {}).get("uri")
    if not uri:
        raise RuntimeError("the Files API returned no file URI.")
    return uri


def _parse_offset(value) -> float:
    """'1.625s' -> 1.625. Returns -1.0 when the field is missing or malformed."""
    if value is None:
        return -1.0
    try:
        return float(str(value).rstrip("s"))
    except (TypeError, ValueError):
        return -1.0


def _words_to_lines(words, gap: float = 0.6) -> list:
    """Group normalised words into utterances, splitting on silence or speaker.

    Takes the output of _normalise_words - {text, speaker, start, end} with float
    seconds - and nothing else. Parsing lives in one place; this function only
    groups, so there is no second reader of the API's offset format to drift out
    of step with the first.

    A per-word list is unusable in a prompt: hundreds of lines the compiler has
    to reassemble itself, badly. Grouping on a pause gives the unit that actually
    matters here - when a sentence STARTS, which is the number that goes into the
    render.
    """
    lines = []
    for w in words or []:
        text = (w.get("text") or "").strip()
        start = w.get("start")
        if not text or start is None or start < 0:
            continue
        end = w.get("end", start)
        spk = w.get("speaker") or ""
        if (lines and spk == lines[-1]["speaker"]
                and start - lines[-1]["end"] <= gap):
            lines[-1]["words"].append(text)
            lines[-1]["end"] = max(lines[-1]["end"], end)
        else:
            lines.append({"speaker": spk, "start": start,
                          "end": max(end, start), "words": [text]})
    return lines


def _pick_vertex_json(vertex_folder: str) -> str:
    """First .json in the folder, sorted. One implementation for every caller.

    _dispatch had this inline; pass A0 needs exactly the same choice, and two
    copies would eventually pick two different service accounts out of the same
    folder.
    """
    folder = (vertex_folder or "").strip()
    if not folder or not os.path.isdir(folder):
        raise RuntimeError(f"[H3 Context-IR] vertex_json_folder not found: {folder!r}")
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(".json"))
    if not files:
        raise RuntimeError(f"[H3 Context-IR] No .json in {folder}")
    return files[0]


def _transcribe_vertex(json_path: str, wav: bytes, language_codes=None) -> str:
    """Vertex route. Reuses the service account already configured on the node."""
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account

    with open(json_path, "r", encoding="utf-8") as fh:
        project_id = json.load(fh).get("project_id", "")
    credentials = service_account.Credentials.from_service_account_file(
        json_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    print(f"🌐 [H3 Context-IR] Vertex — project={project_id} | location=global | "
          f"model={_TRANSCRIBE_MODEL_VERTEX}")
    client = genai.Client(vertexai=True, project=project_id,
                          location="global", credentials=credentials)

    # google-genai valide AudioTranscriptionConfig avec pydantic en mode
    # "extra_forbidden" : un champ que la version installee ne connait pas ne
    # produit pas un avertissement, il fait echouer l'appel. Et le message brut
    # ("Extra inputs are not permitted") ne dit ni quel paquet ni quoi faire.
    # On regarde donc ce que la classe accepte reellement avant d'appeler.
    # pydantic v2 expose model_fields, v1 expose __fields__. None des deux =
    # introspection impossible, et dans ce cas seulement on tente a l'aveugle.
    _cls = types.AudioTranscriptionConfig
    _raw = getattr(_cls, "model_fields", None)
    if _raw is None:
        _raw = getattr(_cls, "__fields__", None)
    _fields = None if _raw is None else set(_raw)
    _wanted = {"word_timestamp": True, "diarization": True}
    if language_codes:
        _wanted["language_codes"] = list(language_codes)

    if _fields is not None:
        _missing = [k for k in _wanted if k not in _fields]
        if "word_timestamp" in _missing:
            # Sans horodatage par mot il ne reste qu'une transcription sans
            # timing, c'est-a-dire rien de ce pour quoi cette passe existe.
            raise RuntimeError(
                "the installed google-genai does not support word_timestamp on "
                "AudioTranscriptionConfig — it only accepts "
                f"{sorted(_fields) or 'nothing'}.\n"
                "     -> update it:  python_embeded\\python.exe -m pip install -U google-genai\n"
                "     Word timings are the whole point of this pass, so it is skipped "
                "rather than run blind."
            )
        for k in _missing:
            print(f"⚠️  [H3 Context-IR] '{k}' unsupported by the installed google-genai "
                  f"— dropped, the rest of the transcription still runs.")
            _wanted.pop(k, None)
    cfg = _wanted
    response = client.models.generate_content(
        model=_TRANSCRIBE_MODEL_VERTEX,
        contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav")],
        config=types.GenerateContentConfig(
            audio_transcription_config=types.AudioTranscriptionConfig(**cfg)
        ),
    )

    words, plain = [], []
    for part in (getattr(response, "parts", None) or []):
        tx = getattr(part, "audio_transcription", None)
        if getattr(part, "text", None):
            plain.append(part.text)
        if tx is None:
            continue
        speaker = getattr(tx, "speaker_label", "") or ""
        for w in (getattr(tx, "words", None) or []):
            words.append({
                "text": getattr(w, "word", "") or "",
                "speaker": getattr(w, "speaker", "") or speaker,
                "start_offset": getattr(w, "start_offset", None),
                "end_offset": getattr(w, "end_offset", None),
            })
    return _normalise_words(words), "".join(plain)


def _transcribe_gemini(api_key: str, wav: bytes, timeout: int = 180) -> str:
    """Gemini API route: Files API upload, then the Interactions endpoint."""
    import requests
    uri = _upload_audio_file(api_key, wav, timeout)
    body = {
        "model": _TRANSCRIBE_MODEL_GEMINI,
        "input": [{"type": "audio", "uri": uri, "mime_type": "audio/wav"}],
        "generation_config": {
            "transcription_config": {
                # verbatim, pas "smart" : smart est incompatible avec les
                # timestamps, et c'est exactement ce qu'on vient chercher.
                "mode": {
                    "type": "verbatim",
                    "timestamp_granularities": ["word"],
                    "diarization_mode": "speaker",
                }
            }
        },
    }
    resp = requests.post(
        f"{_GENAI_ROOT}/v1beta/interactions",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body, timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    words = []
    for step in data.get("steps") or []:
        for content in step.get("content") or []:
            for ann in content.get("annotations") or []:
                if ann.get("type") == "word_info":
                    words.append(ann)
    return _normalise_words(words), (data.get("output_text") or "").strip()


def _normalise_words(words) -> list:
    """Raw annotations -> [{text, speaker, start, end}] with float seconds.

    Stored in this shape rather than as finished text, because the block has to
    be cut to a different window for every segment. A formatted string can only
    be cached whole, and a whole-source transcript is exactly what must never
    reach a segment - see _render_transcript.
    """
    out = []
    for w in words or []:
        start = _parse_offset(w.get("start_offset"))
        if start < 0:
            continue
        text = (w.get("text") or "").strip()
        if not text:
            continue
        end = _parse_offset(w.get("end_offset"))
        out.append({"text": text, "speaker": w.get("speaker") or "",
                    "start": start, "end": end if end >= start else start})
    return out


def _render_transcript(words, w0: float, w1: float, plain_text: str = "") -> str:
    """The speech block for ONE render window, timed on that render's clock.

    Windowing is not a nicety. A segment handed the whole source's speech knows
    the whole story and plans it into the length it was given: with 32s of
    dialogue in front of it and 12.25s to fill, the compiler compresses - the
    scene plays faster than the source and stops tracking it. Cutting the
    transcript to the window removes the temptation at the root.

    Re-basing matters for the same reason as window_start_seconds: the times a
    prompt carries are the times H3 renders at, and those are relative to the
    first frame of THIS segment, not to the source.
    """
    lines = _words_to_lines(words)
    kept = []
    for ln in lines:
        # Une replique a cheval sur le bord est gardee : la couper en deux
        # inventerait un silence qui n'existe pas dans la bande-son.
        if ln["end"] < w0 or ln["start"] > w1:
            continue
        kept.append(ln)

    if not kept:
        if not lines and plain_text:
            return ("# Measured speech (NO word timings returned — content only)\n\n"
                    + plain_text + "\n\n")
        if lines:
            return ("# Measured speech\n\n"
                    "The soundtrack was transcribed and contains NO speech inside this "
                    "render's window. Write no dialogue and no <d> tag.\n\n")
        return ""

    out = ["# Measured speech (this render's own timeline)", "",
           "These times are MEASURED from the soundtrack, then re-based to the first frame "
           "of THIS render. They are not estimates from the keyframes: where the two "
           "disagree, these win.",
           "Only the speech that falls inside this render is listed. There is more of it "
           "before and after — it belongs to the other segments, and writing it here would "
           "compress this one.", ""]
    for ln in kept:
        who = f"[{ln['speaker']}] " if ln["speaker"] else ""
        a = max(0.0, ln["start"] - w0)
        b = max(0.0, ln["end"] - w0)
        flag = "  (starts before this render)" if ln["start"] < w0 else ""
        out.append(f"{a:07.3f}s -> {b:07.3f}s  {who}" + " ".join(ln["words"]) + flag)
    out.append("")
    return "\n".join(out)


def _transcript_cache_path(fingerprint: str) -> str:
    # .json et non .txt : on cache les mots horodates, pas un bloc deja mis en
    # forme, parce que chaque segment le veut coupe a SA fenetre.
    return os.path.join(_CACHE_DIR, f"{fingerprint}.transcript.json")


def _media_fingerprint(parts, model: str, media_res: str = "default") -> str:
    """Hash of everything that changes what pass A sees or how it sees it.

    media_res belongs in here, not just the bytes: at low the model receives a
    quarter of the tokens per image, so the same frames produce a different
    report. Leaving it out would serve a low-resolution grounding back when you
    switch to high, and the setting would look like it does nothing.
    """
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(media_res.encode("utf-8"))
    for kind, payload in parts:
        h.update(kind.encode("utf-8"))
        h.update(payload.encode("utf-8") if isinstance(payload, str) else payload)
    return h.hexdigest()


# Cache disque : le cache memoire seul disparait a chaque redemarrage de
# ComfyUI, et la passe A est justement l'appel cher. Un fichier par empreinte,
# supprimable a la main sans rien casser.
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "h3_context_ir",
)


def _cache_read(fingerprint: str):
    path = os.path.join(_CACHE_DIR, f"{fingerprint}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)["grounding"]
    except Exception:
        return None


def _cache_write(fingerprint: str, grounding: str, meta: dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, f"{fingerprint}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"grounding": grounding, **meta}, fh, ensure_ascii=False, indent=1)
    except Exception as e:
        # Un cache qui n'ecrit pas coute des tokens ; un cache qui fait planter
        # le run coute la generation entiere.
        print(f"⚠️  [H3 Context-IR] Could not write cache: {e}")


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Providers
# ─────────────────────────────────────────────────────────────────────────────

_MIME = {"image": "image/png", "audio": "audio/wav"}

# Verifie sur ai.google.dev/gemini-api/docs/models (page datee du 2026-09-02) :
# 3.8 et 3.7 flash sont tous deux en Stable. Les combos ComfyUI serialisent la
# CHAINE et non l'index, donc reordonner cette liste ne casse aucun workflow.
_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

_PROVIDERS = ["Gemini", "Vertex", "Grok"]

_XAI_URL = "https://api.x.ai/v1/chat/completions"

# Meme liste que _GROK_MODELS dans grok_prompt.py, modeles vision en tete.
_GROK_MODELS = [
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-2-vision-1212",
]

# Gemini 3 image token counts, from the media-resolution guide: 1120 at the
# default, 280 at low - a factor of four on every keyframe. The same guide
# recommends low for action recognition and description, which is exactly what
# pass A does with the video frames. Kept as a widget rather than forced to low,
# because the identity and wardrobe grounding does want the detail.
_MEDIA_RES = ["default", "low", "medium", "high"]
_MEDIA_RES_REST = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}
_MEDIA_RES_TOKENS = {"default": 1120, "low": 280, "medium": 560, "high": 1120}

# Union de la liste officielle de Context-IR et des presets que le node
# Onyx Resolution (MP) peut emettre.
_KNOWN_RATIOS = ("adaptive", "16:9", "9:16", "1:1", "4:3", "3:4",
                 "21:9", "3:2", "2:3", "5:4", "4:5")


def _call_gemini(api_key, model, system_prompt, parts, max_tokens, media_res="default"):
    """Google AI Studio REST. Media travel inline as base64."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")

    payload_parts = []
    for kind, payload in parts:
        if kind == "text":
            payload_parts.append({"text": payload})
        else:
            payload_parts.append({"inline_data": {
                "mime_type": _MIME[kind],
                "data": base64.b64encode(payload).decode("ascii"),
            }})

    gen_cfg = {"maxOutputTokens": max_tokens}
    if media_res in _MEDIA_RES_REST:
        gen_cfg["mediaResolution"] = _MEDIA_RES_REST[media_res]

    body = {
        "contents": [{"parts": payload_parts}],
        "generationConfig": gen_cfg,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    try:
        resp = requests.post(url, json=body,
                             headers={"Content-Type": "application/json"},
                             timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        msg = f"[H3 Context-IR] Gemini API error {e.response.status_code}"
        try:
            msg += f" — {e.response.json().get('error')}"
        except Exception:
            msg += f" — {e.response.text[:300]}"
        raise RuntimeError(msg)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[H3 Context-IR] Gemini request error: {e!s}")

    out = ""
    for cand in (data.get("candidates") or []):
        for part in (cand.get("content", {}).get("parts") or []):
            if "text" in part:
                out += part["text"]
    if not out.strip():
        raise RuntimeError(
            f"[H3 Context-IR] Gemini returned no text. "
            f"finishReason={(data.get('candidates') or [{}])[0].get('finishReason')}"
        )
    return out


def _call_vertex(json_path, model, system_prompt, parts, max_tokens, media_res="default"):
    """Vertex through google-genai. location='global': recent Gemini models are
    published there and are simply absent from regional endpoints, where the call
    returns 404 — same reason as gemini_prompt.py and nano_banana_aio.py."""
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account

    with open(json_path, "r", encoding="utf-8") as fh:
        project_id = json.load(fh).get("project_id", "")
    credentials = service_account.Credentials.from_service_account_file(
        json_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    print(f"🌐 [H3 Context-IR] Vertex — project={project_id} | location=global | model={model}")

    client = genai.Client(vertexai=True, project=project_id,
                          location="global", credentials=credentials)

    contents = []
    for kind, payload in parts:
        if kind == "text":
            contents.append(payload)
        else:
            contents.append(types.Part.from_bytes(data=payload, mime_type=_MIME[kind]))

    cfg = {"max_output_tokens": max_tokens, "system_instruction": system_prompt}
    if media_res in _MEDIA_RES_REST:
        # Enveloppe : media_resolution n'existe pas sur les anciennes versions de
        # google-genai, et rater une option de cout ne justifie pas de faire
        # echouer tout le run. On degrade vers le defaut en le disant.
        try:
            cfg["media_resolution"] = getattr(
                types.MediaResolution, _MEDIA_RES_REST[media_res]
            )
        except Exception as e:
            print(f"⚠️  [H3 Context-IR] media_resolution unsupported by the installed "
                  f"google-genai ({e}) — using the model default.")
    config = types.GenerateContentConfig(**cfg)
    response = client.models.generate_content(
        model=model, contents=contents, config=config
    )
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError("[H3 Context-IR] Vertex returned no text.")
    return text


def _call_grok(api_key, model, system_prompt, parts, max_tokens):
    """xAI, OpenAI-compatible chat/completions. Same shape as grok_prompt.py.

    The guide goes in as a `system` role message, which this API accepts natively
    — no reason it should be read any differently than by Gemini.

    Audio is dropped rather than sent: xAI's image parts are documented and used
    elsewhere in this pack, its audio input is not, and a silent 400 in the middle
    of a run is worse than a stated limitation.
    """
    content = []
    dropped_audio = 0
    for kind, payload in parts:
        if kind == "text":
            content.append({"type": "text", "text": payload})
        elif kind == "image":
            b64 = base64.b64encode(payload).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        else:
            dropped_audio += 1

    if dropped_audio:
        print(f"⚠️  [H3 Context-IR] Grok: {dropped_audio} audio asset(s) not sent — "
              f"audio input is not verified on this API. The soundscape sections will "
              f"be written without hearing anything. Use Gemini or Vertex when the audio "
              f"matters.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            _XAI_URL, json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        msg = f"[H3 Context-IR] Grok API error {e.response.status_code}"
        try:
            msg += f" — {e.response.json().get('error')}"
        except Exception:
            msg += f" — {e.response.text[:300]}"
        raise RuntimeError(msg)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[H3 Context-IR] Grok request error: {e!s}")

    choices = data.get("choices") or []
    text = (choices[0].get("message", {}).get("content", "") if choices else "") or ""
    if not text.strip():
        raise RuntimeError("[H3 Context-IR] Grok returned no text.")
    if data.get("usage"):
        u = data["usage"]
        print(f"📊 [H3 Context-IR] Grok usage — prompt={u.get('prompt_tokens','?')} "
              f"completion={u.get('completion_tokens','?')} total={u.get('total_tokens','?')}")
    return text


def _dispatch(provider, api_key, vertex_folder, model, system_prompt, parts, max_tokens,
              media_res="default", grok_api_key=""):
    if provider == "Grok":
        key = (grok_api_key or "").strip()
        if not key:
            raise RuntimeError("[H3 Context-IR] grok_api_key is required for provider=Grok.")
        return _call_grok(key, model, system_prompt, parts, max_tokens)

    if provider == "Vertex":
        return _call_vertex(_pick_vertex_json(vertex_folder), model, system_prompt,
                            parts, max_tokens, media_res)

    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("[H3 Context-IR] gemini_api_key is required for provider=Gemini.")
    return _call_gemini(key, model, system_prompt, parts, max_tokens, media_res)


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class OnyxH3ContextIR:
    """Two-pass local replacement for the H3 cloud preprocessing stage."""

    # Cle = empreinte des medias + modele. Iterer sur l'intent ne rappelle donc
    # que la passe texte : la passe vision, qui est la lente et la chere, n'est
    # rejouee que si les medias changent reellement.
    _grounding_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "intent": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Keep this SHORT, and write only what must CHANGE. The official "
                               "Context-IR calls send one sentence — 'Pull focus to the people in "
                               "the background and add more steam to the ramen bowl.' The media are "
                               "read for everything else; describing the scene here competes with "
                               "what pass A actually observed.",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 8.0, "min": 2.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Target length. The official API takes this as a Context-IR input, "
                               "not just a generation parameter: shot count, pacing and where "
                               "beats land all depend on it. A prompt written blind to duration "
                               "plans a timeline the render cannot hold.",
                }),
                # STRING and not a combo, on purpose: a combo input only accepts a
                # link whose output type is that same combo, so the STRING coming
                # out of Onyx Resolution (MP) could never be wired to it. The
                # value is checked in run() instead, which warns without blocking -
                # any ratio the compiler can read is legitimate here.
                "aspect_ratio": ("STRING", {
                    "default": "adaptive", "multiline": False,
                    "tooltip": "Target framing: adaptive, 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 3:2, "
                               "2:3, 5:4, 4:5.\n"
                               "'adaptive' is the official default and lets the compiler follow "
                               "the media. Connect the aspect_ratio output of Onyx Resolution "
                               "(MP) to keep it in sync with what you actually render.",
                }),
                "target_model": (_TARGETS, {
                    "default": TARGET_H3,
                    "tooltip": "Which video model the prompt is written for.\n"
                               "MiniMax H3: compiles against the two official MiniMax guides, "
                               "verbatim, with <Picture N> / <Video N> labels.\n"
                               "Seedance: compiles against ByteDance's six-slot grammar with "
                               "@Image 1 / @Video 1 tags.\n\n"
                               "The two are not interchangeable. H3's guide asks for an "
                               "exhaustive description; ByteDance's says a long paragraph fights "
                               "itself and that two or three sentences is the target. Pass A does "
                               "not change — it describes media, not a destination.",
                }),
                "provider": (_PROVIDERS, {"default": "Gemini"}),
                # Une seule liste pour les trois providers, comme le fait deja
                # gemini_prompt.py apres la fusion des widgets grok/gemini. Le
                # node verifie la coherence au run et le dit, plutot que de
                # laisser partir un modele Grok vers l'endpoint Google.
                "model": (_MODELS + _GROK_MODELS, {"default": "gemini-3.6-flash"}),
                "guide_folder": ("STRING", {
                    "default": _DEFAULT_GUIDE_FOLDER, "multiline": False,
                    "tooltip": "Folder holding the two official MiniMax guides. Both are "
                               "loaded verbatim: the reference guide reuses the shot, camera "
                               "and dialogue formats defined in the base guide.",
                }),
                "max_tokens": ("INT", {"default": 8192, "min": 512, "max": 65536, "step": 512}),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Reference image. Slots are labelled in order: the "
                                                 "first connected slot becomes <Picture 1>, the next "
                                                 "<Picture 2>, and so on. Skipping a slot never leaves "
                                                 "a gap in the numbering."}),
                "image_1_role": (_ROLES, {
                    "default": ROLE_REFERENCE,
                    "tooltip": "What image_1 is FOR, mirroring the official Context-IR payload.\n"
                               "reference: a picture of the subject.\n"
                               "first_frame: this image IS the opening frame of the render.\n"
                               "last_frame: this image IS the closing frame.\n\n"
                               "This is not a mode selector — you describe the asset, the compiler "
                               "still decides the mode on its own.",
                }),
                "image_2": ("IMAGE",),
                "image_2_role": (_ROLES, {
                    "default": ROLE_REFERENCE,
                    "tooltip": "Same as image_1_role. Set image_1 to first_frame and image_2 to "
                               "last_frame for a first-and-last-frame render.",
                }),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "images_batch": ("IMAGE", {"tooltip": "Optional extra images as a batch, appended "
                                                      "AFTER the numbered slots. Useful with the "
                                                      "Onyx Image Batch Loader."}),
                "video": ("IMAGE", {"tooltip": "Source video frames, labelled <Video 1>. "
                                               "Sampled to 8 timecoded keyframes."}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                                        "tooltip": "Used to write the timecode on each keyframe. "
                                                   "Wrong here means wrong shot boundaries."}),
                "media_resolution": (_MEDIA_RES, {
                    "default": "low",
                    "tooltip": "Token budget per image sent to pass A. Gemini 3: 1120 at default, "
                               "280 at low — a factor of four on every keyframe.\n"
                               "Google recommends low for action recognition and description, "
                               "which is what the video frames are for. Raise it only if the "
                               "grounding starts missing wardrobe or face detail, and consider "
                               "spending the saving on more keyframes instead.",
                }),
                "video_keyframes": ("INT", {
                    "default": 8, "min": 2, "max": 32, "step": 1,
                    "tooltip": "How many frames of the video pass A actually looks at.\n"
                               "The compiler writes the action sequence from these and nothing "
                               "else, so this sets the resolution of the timeline it can describe. "
                               "8 over a 15 s clip is one frame every 1.9 s — enough to say what "
                               "happens, not enough to say exactly how.\n"
                               "Roughly one per second is a good target. Each frame costs vision "
                               "tokens, so raise it for long or busy clips, not for everything."}),
                "video_audio": ("AUDIO", {
                    "tooltip": "The soundtrack that belongs to the video above. It is described as "
                               "part of <Video 1>, not as a separate asset.\n"
                               "This is what fills overall_soundscape with dated events instead of "
                               "a generic line."}),
                "ref_audio_1": ("AUDIO", {
                    "tooltip": "A standalone audio reference, labelled <Audio 1> — a voice timbre "
                               "to follow, a music bed, a sound to reproduce. Separate from the "
                               "video's own soundtrack, exactly as reference_audio is separate from "
                               "reference_video in the official payload."}),
                "ref_audio_2": ("AUDIO", {"tooltip": "Second standalone audio reference, <Audio 2>."}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
                "grok_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "xAI key (console.x.ai), for provider=Grok.\n"
                               "Audio assets are not sent to Grok: its image input is documented "
                               "and already used elsewhere in this pack, its audio input is not "
                               "verified. Use Gemini or Vertex when the soundtrack matters."}),
                "vertex_json_folder": ("STRING", {"default": "", "multiline": False}),
                "grounding_override": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Paste a corrected grounding JSON to skip pass A entirely and "
                               "recompile without another vision call.",
                }),
                "window_start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.001,
                    "tooltip": "Where this render starts inside the supplied source, in seconds.\n"
                               "Leave at 0 for a normal one-shot render.\n"
                               "For a chained render, give every segment the SAME full video and "
                               "only change this: the grounding is then computed once and served "
                               "from cache, and each segment gets its own window with timestamps "
                               "re-based to zero.\n"
                               "Added LAST on purpose: ComfyUI serialises widget values by "
                               "position, so inserting it higher would shift every widget in "
                               "already-saved workflows."}),
                "pinned_lead_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.001,
                    "tooltip": "How many seconds at the START of this render are already fixed "
                               "by a guide, and must therefore NOT be described.\n"
                               "For a chained render: connect the Segment node's `guide_seconds`. "
                               "Those frames are pinned by AddGuide — image AND audio — so any "
                               "line of dialogue the compiler places inside them is rendered a "
                               "second time and then thrown away by the Join. That is what makes "
                               "a character repeat a sentence across a seam.\n"
                               "The timeline origin does not move: 00:00.000 stays the first "
                               "frame of the render, because H3 encodes time absolutely and the "
                               "frames really are being generated. Only the DESCRIPTION starts "
                               "later."}),
                "transcribe_speech": ("BOOLEAN", {
                    "default": False,
                    "label_on": "measure speech timings",
                    "label_off": "off",
                    "tooltip": "Runs gemini-3.5-transcribe on the soundtrack first and gives both "
                               "passes the MEASURED start and end time of every spoken word.\n"
                               "Pass A is a vision pass: asked when a line is spoken it reads "
                               "keyframes and estimates, and it estimates badly. This replaces the "
                               "guess with a measurement.\n"
                               "REQUIRES a gemini_api_key: it uses the Interactions API, which is "
                               "documented on the Gemini API and which I could not confirm on "
                               "Vertex. Without a key the pass is skipped and the node behaves "
                               "exactly as before.\n"
                               "Costs one call per soundtrack for the whole render — it is cached "
                               "on disk beside the grounding, so a four-segment chain transcribes "
                               "once."}),
                "extra_prompt_rules": ("BOOLEAN", {
                    "default": False,
                    "label_on": "add the rules written after the reference run",
                    "label_off": "prompt as of the last known-good render",
                    "tooltip": "OFF restores the Pass B prompt to the shape it had for the render "
                               "you judged near-perfect, before a week of accumulated rules.\n\n"
                               "What it removes:\n"
                               "• 'Never write stillness' — 1464 characters that were always on\n"
                               "• the 'Render window' block on segment 1 (segments 2+ keep theirs, "
                               "they need the re-basing)\n\n"
                               "Each rule fixed something real. What was never measured is their "
                               "combined effect: they stack in the same context as the official "
                               "guide, and the compiler averages them. Turn them back on one at a "
                               "time if a defect they addressed comes back.",
                }),
                "keyframes_match_model": ("BOOLEAN", {
                    "default": False,
                    "label_on": "same frames as the model (2 fps)",
                    "label_off": "evenly spaced (video_keyframes)",
                    "tooltip": "Samples the reference video EXACTLY as H3 does — every 12th frame "
                               "at 24 fps, labelled 0.0, 0.5, 1.0 …\n\n"
                               "H3 hands its text encoder the video at 2 fps and nothing else "
                               "(nodes_minimax_h3.py: range(0, n, FPS // 2)). Evenly spaced "
                               "keyframes land on OTHER frames with OTHER labels, so a beat "
                               "written at 00:07.333 points at an instant the model does not "
                               "have — its strip only holds 7.0 and 7.5.\n\n"
                               "With this on, pass A describes the very images the model will "
                               "see, and pass B is told to put action beats on the same 0.5 s "
                               "grid.\n"
                               "video_keyframes is ignored: the stride belongs to the model. A "
                               "294-frame segment gives 25 frames either way, so it costs "
                               "nothing.",
                }),
                "motion_from_video_only": ("BOOLEAN", {
                    "default": False,
                    "label_on": "the video owns the timing",
                    "label_off": "the prompt may time actions",
                    "tooltip": "Stops the prompt from putting timestamps on physical actions.\n\n"
                               "In a reference-video render the footage already fixes WHEN every "
                               "movement happens, frame by frame. A timestamp in the prompt is a "
                               "SECOND answer to the same question — and it is an estimate: pass A "
                               "reads keyframes and misjudges by around a second. The model then "
                               "has two sources that disagree and splits the difference, which is "
                               "what 'the same scene, played differently' looks like.\n\n"
                               "Turn this on for motion control. Dialogue keeps its timestamps: "
                               "H3 generates the audio, so nothing else says when a line is "
                               "spoken.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "grounding_json")
    FUNCTION = "run"
    CATEGORY = "Onyx/Prompt"
    DESCRIPTION = (
        "Local rebuild of the H3 cloud preprocessing stage. Pass A reads the real media "
        "and reports facts as JSON; pass B compiles that JSON plus your intent into an H3 "
        "prompt, using the official MiniMax guides verbatim as its system prompt."
    )

    def run(self, intent, duration_seconds, aspect_ratio, target_model,
            provider, model, guide_folder, max_tokens,
            image_1=None, image_1_role=ROLE_REFERENCE,
            image_2=None, image_2_role=ROLE_REFERENCE,
            image_3=None, image_4=None,
            image_5=None, image_6=None, images_batch=None,
            video=None, video_fps=24.0, media_resolution="low",
            video_keyframes=_VIDEO_KEYFRAMES,
            video_audio=None, ref_audio_1=None, ref_audio_2=None,
            gemini_api_key="", grok_api_key="", vertex_json_folder="",
            grounding_override="", window_start_seconds=0.0,
            pinned_lead_seconds=0.0, transcribe_speech=False,
            motion_from_video_only=False, keyframes_match_model=False,
            extra_prompt_rules=False):

        ensure_profile_ready()
        if not (intent or "").strip():
            raise RuntimeError("[H3 Context-IR] intent is empty.")

        # Provider et modele viennent de deux widgets independants, donc rien
        # n'empeche d'envoyer un modele Grok a l'endpoint Google. L'erreur
        # remontee serait un 404 sur le nom du modele, qui se lit comme une faute
        # de frappe alors que c'est un mauvais aiguillage.
        is_grok_model = model in _GROK_MODELS
        if provider == "Grok" and not is_grok_model:
            raise RuntimeError(
                f"[H3 Context-IR] provider=Grok with model={model!r}.\n"
                f"-> Pick one of: {', '.join(_GROK_MODELS)}"
            )
        if provider != "Grok" and is_grok_model:
            raise RuntimeError(
                f"[H3 Context-IR] provider={provider} with the Grok model {model!r}.\n"
                f"-> Switch provider to Grok, or pick a Gemini model."
            )

        # Averti sans bloquer : la liste couvre l'API officielle et les presets du
        # node Resolution, mais un ratio hors liste reste lisible par le
        # compilateur. Refuser ici ne protegerait de rien et casserait un graphe
        # qui marche.
        aspect_ratio = (aspect_ratio or "adaptive").strip() or "adaptive"
        if aspect_ratio not in _KNOWN_RATIOS:
            print(f"⚠️  [H3 Context-IR] Unusual aspect_ratio {aspect_ratio!r} — "
                  f"passed through as-is. Known values: {', '.join(_KNOWN_RATIOS)}")

        # ── Pass A ────────────────────────────────────────────────────────────
        roles_note = ""
        override = (grounding_override or "").strip()
        if override:
            grounding = override
            print("📝 [H3 Context-IR] Pass A skipped — using grounding_override.")
        else:
            pictures = _collect_pictures(
                [
                    (image_1, image_1_role),
                    (image_2, image_2_role),
                    (image_3, ROLE_REFERENCE),
                    (image_4, ROLE_REFERENCE),
                    (image_5, ROLE_REFERENCE),
                    (image_6, ROLE_REFERENCE),
                ],
                images_batch,
            )
            if pictures:
                mapping = ", ".join(
                    f"<Picture {i}>={origin}({role})"
                    for i, (origin, _, role) in enumerate(pictures, 1)
                )
                print(f"🏷️  [H3 Context-IR] Label map — {mapping}")
                # Repete pour la passe B : elle ne voit pas les etiquettes de la
                # requete vision, seulement le JSON. Un role laisse implicite la
                # ferait retomber sur "reference", ce qui perd exactement
                # l'information que ce widget existe pour porter.
                roles_note = "\n".join(
                    f"<Picture {i}>: {_ROLE_LABEL[role]}"
                    for i, (_, _, role) in enumerate(pictures, 1)
                )
            parts = _build_grounding_parts(
                pictures, video, video_fps, video_audio,
                [ref_audio_1, ref_audio_2], int(video_keyframes),
                keyframes_match_model,
            )
            if not parts:
                raise RuntimeError(
                    "[H3 Context-IR] No media connected. Connect at least one of "
                    "image_1..image_6 / images_batch / video / video_audio / "
                    "ref_audio_1..2, or paste a grounding_override."
                )
            fp = _media_fingerprint(parts, model, media_resolution)

            # Passe A0 : la mesure avant l'estimation. Elle est faite AVANT la
            # passe A pour que le transcript entre dans le contexte de la vision
            # - qui n'a alors plus a dater ce qu'elle voit, seulement a le
            # decrire. Sa propre empreinte est celle de la bande-son seule, donc
            # changer media_resolution ou le modele de vision ne le refait pas.
            speech_words, speech_plain = [], ""
            if transcribe_speech and video_audio is not None:
                wav = _audio_to_wav_bytes(video_audio)
                a_fp = hashlib.sha256(wav).hexdigest()
                t_path = _transcript_cache_path(a_fp)
                if os.path.isfile(t_path):
                    try:
                        with open(t_path, "r", encoding="utf-8") as fh:
                            _blob = json.load(fh)
                        speech_words = _blob.get("words") or []
                        speech_plain = _blob.get("plain") or ""
                        print(f"♻️  [H3 Context-IR] Pass A0 transcript from cache "
                              f"({a_fp[:12]}) — {len(speech_words)} word(s)")
                    except Exception as e:
                        print(f"⚠️  [H3 Context-IR] unreadable transcript cache ({e}) — "
                              f"re-transcribing.")
                        speech_words = []
                if not speech_words and not speech_plain:
                    _route = ("vertex" if provider == "Vertex" and vertex_json_folder
                              else ("gemini" if (gemini_api_key or "").strip() else None))
                    if _route is None:
                        print("⚠️  [H3 Context-IR] transcribe_speech is ON but neither route is "
                              "available.\n"
                              "     Vertex: set provider=Vertex and fill vertex_json_folder "
                              "(uses gemini-3.5-transcribe-preview, no extra key).\n"
                              "     Gemini: fill gemini_api_key.\n"
                              "     Skipping: the node continues exactly as it did before.")
                    else:
                        _t0 = time.time()
                        print(f"🎙️  [H3 Context-IR] Pass A0 — transcribing "
                              f"{len(wav) / 1024:.0f} KB of audio via {_route} "
                              f"(word timings + speakers)...")
                        try:
                            if _route == "vertex":
                                speech_words, speech_plain = _transcribe_vertex(
                                    _pick_vertex_json(vertex_json_folder), wav)
                            else:
                                speech_words, speech_plain = _transcribe_gemini(
                                    gemini_api_key.strip(), wav)
                            os.makedirs(_CACHE_DIR, exist_ok=True)
                            with open(t_path, "w", encoding="utf-8") as fh:
                                json.dump({"words": speech_words, "plain": speech_plain},
                                          fh, ensure_ascii=False)
                            _last = speech_words[-1]["end"] if speech_words else 0.0
                            print(f"✅ [H3 Context-IR] Pass A0 done in {time.time() - _t0:.1f}s "
                                  f"— {len(speech_words)} word(s) up to {_last:.3f}s, cached.")
                        except Exception as e:
                            # Une transcription ratee ne doit jamais couter le
                            # rendu : on perd la precision, pas la generation.
                            print(f"⚠️  [H3 Context-IR] Pass A0 failed ({e}) — continuing "
                                  f"without measured timings.")
                            speech_words, speech_plain = [], ""

            # Passe A voit la bande-son entiere : sa sortie est mise en cache et
            # partagee par tous les segments, donc elle ne doit dependre d'aucune
            # fenetre. C'est la passe B, et elle seule, qui recoit la coupe.
            if speech_words or speech_plain:
                _full = _render_transcript(speech_words, 0.0, float("inf"), speech_plain)
                if _full:
                    parts = parts + [("text", _full)]

            cached = OnyxH3ContextIR._grounding_cache.get(fp)
            if cached is None:
                cached = _cache_read(fp)
                if cached is not None:
                    OnyxH3ContextIR._grounding_cache[fp] = cached
                    print(f"♻️  [H3 Context-IR] Pass A from disk cache ({fp[:12]})")
            elif cached is not None:
                print(f"♻️  [H3 Context-IR] Pass A from memory cache ({fp[:12]})")
            if cached is not None:
                grounding = cached
            else:
                n_media = sum(1 for k, _ in parts if k != "text")
                n_img = sum(1 for k, _ in parts if k == "image")
                est = n_img * _MEDIA_RES_TOKENS.get(media_resolution, 1120)
                print(f"🔍 [H3 Context-IR] Pass A — grounding {n_media} asset(s) "
                      f"via {provider}/{model} | media_resolution={media_resolution} "
                      f"(~{est} image tokens for {n_img} image(s))...")
                grounding = _strip_fence(_dispatch(
                    provider, gemini_api_key, vertex_json_folder, model,
                    _PASS_A_SYSTEM, parts, max_tokens, media_resolution,
                    grok_api_key,
                ))
                try:
                    json.loads(grounding)
                except Exception as e:
                    # On garde le texte brut : il part quand meme en passe B et
                    # sort sur grounding_json, ou tu peux le lire et le corriger
                    # pour le recoller dans grounding_override.
                    print(f"⚠️  [H3 Context-IR] Pass A output is not valid JSON ({e}). "
                          f"Passing it through anyway — inspect grounding_json.")
                OnyxH3ContextIR._grounding_cache[fp] = grounding
                _cache_write(fp, grounding, {
                    "model": model,
                    "media_resolution": media_resolution,
                    "images": n_img,
                    "assets": n_media,
                })
                print(f"💾 [H3 Context-IR] Grounding cached ({fp[:12]}) — "
                      f"identical media will not be re-analysed.")

        # ── Pass B ────────────────────────────────────────────────────────────
        if target_model == TARGET_SEEDANCE:
            # Le guide H3 n'est pas charge du tout ici. Ses 39 Ko demandent une
            # description exhaustive, ce que la grammaire Seedance qualifie
            # explicitement d'echec — concatener les deux reviendrait a demander
            # au compilateur d'obeir a deux consignes opposees sur la longueur.
            system_b = _SEEDANCE_SYSTEM
            label_note = ("Refer to the assets as @Image 1, @Video 1 and @Audio 1, "
                          "in the order they appear in the observations. The observations "
                          "label them <Picture N>, <Video N> and <Audio N>; <Picture 1> is "
                          "@Image 1, and so on.")
        else:
            system_b = _load_guides(guide_folder) + "\n" + _PASS_B_ADDENDUM
            label_note = ("Use only the assets listed in the observations, under the labels "
                          "they were given there.")
        # duration et ratio sont des entrees de Context-IR cote officiel, pas de
        # simples parametres de rendu : le decoupage en plans et le placement des
        # temps forts en dependent. Ecrire le prompt sans les connaitre revient a
        # planifier une timeline que le rendu ne pourra pas contenir.
        # Fenetre : le rendu ne couvre qu'une partie de la source.
        #
        # Le format H3 encode le temps en ABSOLU (`At MM:SS.mmm`). Un prompt
        # compile pour 20 s et donne a un segment de 12 s fait comprimer toute
        # l'action - un dialogue place a 12,9 s se retrouve joue vers 7,8 s.
        # D'ou le rebasage : ce que les observations situent a window_start
        # devient 00:00.000 dans le prompt.
        _w0 = float(window_start_seconds or 0.0)
        _w1 = _w0 + float(duration_seconds)

        # Le rendu est-il une FENETRE dans une source plus longue ?
        #
        # Il ne suffit pas de tester window_start > 0 : le segment 1 commence a
        # zero et n'en est pas moins une fenetre de 12.25s dans une source de
        # 32s. La condition d'origine le laissait sans aucune borne, et pass B
        # recevait alors la totalite des observations - et, depuis la passe A0,
        # la totalite des paroles horodatees - pour n'ecrire que 12.25s. Un
        # compilateur qui connait toute l'histoire la fait tenir dans la duree
        # qu'on lui donne : la scene se joue plus vite que la source et cesse de
        # la suivre.
        #
        # La longueur de la source est lue sur la video elle-meme : elle est deja
        # branchee, il n'y a rien de plus a cabler.
        _src_seconds = 0.0
        if video is not None and int(video.shape[0]) > 0:
            _src_seconds = int(video.shape[0]) / (video_fps if video_fps > 0 else 24.0)
        # Avant le 03/09 la condition etait _w0 > 0.0, donc le segment 1 (qui
        # commence a zero) ne recevait aucun bloc de fenetre. C'est la forme
        # qu'avait le prompt du rendu de reference.
        _is_window = _w0 > 0.0 or (extra_prompt_rules
                                   and _src_seconds > float(duration_seconds) + 0.05)

        window_note = ""
        if _is_window:
            window_note = (
                f"# Render window\n\n"
                f"This render covers ONLY {_w0:.3f}s to {_w1:.3f}s"
                + (f" of a {_src_seconds:.3f}s source." if _src_seconds > 0 else " of the source.")
                + " The observations describe the WHOLE source; this render is one window "
                  "inside it.\n"
                + (f"Everything before {_w0:.3f}s has already been rendered by an earlier "
                   f"segment. " if _w0 > 0.0 else "")
                + f"Everything after {_w1:.3f}s belongs to a later render — do not tell that "
                  f"part of the story here, do not summarise it, and do not hurry toward it. "
                  f"Fitting more of the source than this window contains is the one failure "
                  f"that makes the result stop following the source: the action then plays "
                  f"faster than the footage it is copying.\n"
                + (f"Re-base every timestamp to the START of this render: what the "
                   f"observations place at {_w0:.3f}s is 00:00.000 in your output. "
                   if _w0 > 0.0 else "")
                + f"The last moment you may write is {float(duration_seconds):.3f}s.\n\n"
            )

        # Tete de segment deja fixee par le guide.
        #
        # AddGuide epingle les premieres frames ET leur audio : elles ne sont pas
        # generees, elles sont imposees, puis le Join les jette au profit de la
        # sortie du segment precedent. Une replique que le compilateur place dans
        # cette fenetre est donc rendue deux fois - une fois ici, une fois dans le
        # segment d'avant - et le spectateur entend le personnage se repeter au
        # raccord.
        #
        # L'origine du temps NE bouge PAS. H3 encode le temps de facon absolue et
        # ces frames sont bel et bien produites, donc 00:00.000 reste la premiere
        # frame du rendu. Seule la DESCRIPTION commence plus tard. Deplacer
        # l'origine decalerait tout ce qui suit de la duree du guide.
        _lead = max(0.0, float(pinned_lead_seconds or 0.0))
        if _lead > 0.0 and _lead < float(duration_seconds):
            window_note += (
                f"# Already-rendered opening\n\n"
                f"The first {_lead:.3f}s of this render (00:00.000 to {_lead:.3f}) are ALREADY "
                f"FIXED: they are pinned frames carried over from the previous segment, with "
                f"their audio, and they will be discarded from this render when the segments "
                f"are joined.\n"
                f"Therefore:\n"
                f"- Place NO dialogue, NO <d> tag and NO new action before {_lead:.3f}s. A line "
                f"written there is spoken twice in the finished video, once at the end of the "
                f"previous segment and once here.\n"
                f"- Your first described beat starts at {_lead:.3f}s or later.\n"
                f"- You may still state, in one short clause, what the subject is already doing "
                f"at 00:00.000, so the continuation is unambiguous — but as a state, not as an "
                f"event with a timestamp.\n"
                f"- The last moment you may write remains {float(duration_seconds):.3f}s.\n\n"
            )

        # Coupe a CETTE fenetre, et re-basee sur elle. Voir _render_transcript.
        _win_speech = ""
        if speech_words or speech_plain:
            _win_speech = _render_transcript(speech_words, _w0, _w1, speech_plain)

        # Le timing des actions appartient a la video de reference, pas au texte.
        #
        # En Ref2VA la video fixe deja, frame par frame, l'instant de chaque
        # mouvement. Un timestamp d'action dans le prompt est une SECONDE reponse
        # a la meme question - et une reponse estimee : la passe A lit des
        # keyframes et se trompe d'environ une seconde. Le modele recoit alors
        # deux sources qui se contredisent et arbitre entre les deux.
        #
        # Le dialogue garde ses timestamps : H3 GENERE l'audio, donc rien
        # d'autre ne dit quand une replique est prononcee.
        # Grille de 0,5 s : celle sur laquelle le modele voit la video.
        grid_note = ""
        if keyframes_match_model:
            grid_note = (
                "# The model's own clock\n\n"
                "The video model receives @Video 1 at 2 frames per second, timecoded 0.0, 0.5, "
                "1.0, 1.5 and so on. The keyframes in the observations are those exact frames "
                "under those exact labels.\n"
                "So every timestamp you write for a physical action must land ON that grid: "
                "00:07.000 or 00:07.500, never 00:07.333. A time between two frames points at "
                "an instant the model does not hold, and it has to guess which side you meant.\n"
                "Round to the nearest half second. If an action falls between two of the "
                "keyframes you were given, take the one where it is already visible — an event "
                "written late reads as a slower build-up, an event written early makes the "
                "action rush to meet it.\n"
                "Dialogue is not bound to this grid: speech is generated, not copied from the "
                "reference, so keep the exact time you were given for every <d> tag.\n\n"
            )

        motion_note = ""
        if motion_from_video_only:
            motion_note = (
                "# The video owns the timing\n\n"
                "@Video 1 already fixes when every physical action happens, frame by frame. "
                "Your description must not fix it a second time.\n"
                "- Write NO timestamp on a physical action, a gesture, a camera move or an "
                "object's behaviour. Not '00:07.333', not 'at 7 seconds', not 'after three "
                "seconds', not 'first ... then ... finally'.\n"
                "- Describe what happens ONCE, as a continuous whole, and say it comes from "
                "@Video 1. One sentence naming the action and its owner beats five timed beats: "
                "the beats are estimates, the video is exact.\n"
                "- Ordering words that carry no clock are fine ('as', 'while', 'throughout'). "
                "What is banned is anything that pins an action to a moment.\n"
                "- DIALOGUE IS THE EXCEPTION. Every <d> tag keeps its timestamp, because the "
                "model generates the audio and the reference video does not specify it.\n"
                "- If the observations give a time for an action, drop the time and keep the "
                "action. A time you cannot verify against the footage is worse than no time: "
                "it competes with footage that is right.\n\n"
            )

        user_b = (
            "# Target render\n\n"
            f"duration: {duration_seconds:.2f} seconds\n"
            f"ratio: {aspect_ratio}\n\n"
            + window_note
            + grid_note
            + motion_note
            + (f"# Asset roles\n\n{roles_note}\n\n" if roles_note else "")
            + (_win_speech + "Every <d> tag you write must start at the time given above. "
                              "Those times are already on this render's clock — do not shift a "
                              "line to make a beat fit, and do not write a line that is not "
                              "listed.\n\n"
               if _win_speech else "")
            + "# Grounded observations\n\n"
            f"{grounding}\n\n"
            "# User intent\n\n"
            f"{intent.strip()}\n\n"
            f"Write the {target_model} prompt for a {duration_seconds:.2f} second render. "
            f"Plan the shots and their timing to fit exactly that length. {label_note}"
            + (f" Every timestamp is relative to the start of this render, not to the source."
               if _w0 > 0.0 else "")
            + (f" Nothing may be described before {_lead:.3f}s: that part is already rendered."
               if _lead > 0.0 and _lead < float(duration_seconds) else "")
        )
        if extra_prompt_rules:
            system_b = system_b + "\n\n" + _EXTRA_RULES
        print(f"🛠️  [H3 Context-IR] Pass B — compiling "
              f"(guide {len(system_b)} chars) via {provider}/{model}...")
        h3_prompt = _strip_fence(_dispatch(
            provider, gemini_api_key, vertex_json_folder, model,
            system_b, [("text", user_b)], max_tokens, media_resolution,
            grok_api_key,
        ))

        print(f"✅ [H3 Context-IR] Done — prompt {len(h3_prompt)} chars, "
              f"grounding {len(grounding)} chars")
        return (h3_prompt, grounding)


NODE_CLASS_MAPPINGS = {"OnyxH3ContextIR": OnyxH3ContextIR}
NODE_DISPLAY_NAME_MAPPINGS = {"OnyxH3ContextIR": "H3 Context-IR (Gemini)"}
