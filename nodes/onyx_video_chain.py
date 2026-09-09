# -*- coding: utf-8 -*-
"""
ComfyUI nodes - Onyx Video Chain (Prepare / Commit).

Generate a long video as a chain of H3-length segments, without cutting and
re-plugging anything between runs.

    Prepare  ->  [ your normal H3 motion-control graph ]  ->  Commit

Prepare hands out, for the current segment: the tail of the previous segment
(to be pinned with MiniMaxH3AddGuide at index 0), and the slice of the source
reference video and audio that carries the motion for THIS segment. Commit
stores the new tail and advances the counter.

Why two nodes and not one: ComfyUI is a DAG. A node placed after the sampler
cannot re-trigger a node placed before it, so no single node can loop a graph
it does not itself contain. What a node CAN do is remember. The state lives in
_cache/video_chain/, every queued run reads it, and ComfyUI's own batch count
supplies the repetition - set it to 4 and press Queue once.

Grid: segment_frames and overlap_frames both sit on H3's 17k+5 video grid, and
both are multiples of 3, so every joint also lands exactly on the 40 Hz audio
clock. That is what keeps lip sync from stepping at each seam.
"""

import io
import os
import json
import shutil
import logging

import numpy as np
import torch

# ExecutionBlocker : le seul moyen propre, dans ComfyUI, de dire "cette branche
# n'a pas lieu d'etre". Tout ce qui en depend est saute, y compris les nodes de
# SORTIE - ce que l'evaluation paresseuse du Join ne peut pas faire, puisque
# ComfyUI evalue tous les output nodes quoi qu'il arrive.
try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:
    ExecutionBlocker = None

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_ROOT = os.path.join(_HERE, "_cache", "video_chain")

# Longueurs valides pour les DEUX grilles : video 17k+5 et audio 40 Hz contre
# 24 fps (frames divisibles par 3). C'est la suite 39 + 51k.
_AV_GRID = [39, 90, 141, 192, 243, 294, 345]
_SEGMENT_CHOICES = [f"{n} frames ({n / 24.0:.3f} s)" for n in _AV_GRID]
_DEFAULT_SEGMENT = f"345 frames ({345 / 24.0:.3f} s)"

# Valeurs de recouvrement du workflow de reference. 39 est le seul qui soit
# aussi divisible par 3 : avec 5 ou 22, l'avance par segment cesse de tomber
# sur l'horloge audio et le raccord derive.
_OVERLAP_CHOICES = ["39 (audio-exact)", "22", "5"]


def _frames_of(label: str) -> int:
    return int(str(label).split()[0])


def _state_paths(session: str):
    session = "".join(c for c in (session or "chain").strip() if c.isalnum() or c in "-_ ").strip()
    session = session or "chain"
    folder = os.path.join(_STATE_ROOT, session)
    return folder, os.path.join(folder, "state.json"), os.path.join(folder, "tail.npy")


def _read_state(session):
    _folder, meta_path, tail_path = _state_paths(session)
    if not os.path.isfile(meta_path):
        return None, None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as e:
        logging.warning("Video Chain: unreadable state (%s) — starting over.", e)
        return None, None
    tail = None
    if os.path.isfile(tail_path):
        arr = np.load(tail_path)
        tail = torch.from_numpy(arr).float() / 255.0
    return meta, tail


def _write_state(session, meta, tail_tensor):
    folder, meta_path, tail_path = _state_paths(session)
    os.makedirs(folder, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if tail_tensor is not None:
        arr = (tail_tensor.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        np.save(tail_path, arr)


def reset_session(session):
    folder, _m, _t = _state_paths(session)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        return True
    return False


def _slice_audio(audio, start_s, end_s):
    """Cut an AUDIO dict to [start_s, end_s). Returns None when out of range."""
    if audio is None:
        return None
    wav = audio["waveform"]
    sr = int(audio["sample_rate"])
    n = int(wav.shape[-1])
    a = max(0, int(round(start_s * sr)))
    b = min(n, int(round(end_s * sr)))
    if b <= a:
        return None
    return {"waveform": wav[..., a:b].clone(), "sample_rate": sr}


class OnyxVideoChainSegment:
    """One link of a chain laid out in the graph: N samplers side by side.

    Stateless on purpose. A DAG cannot go backwards, but it goes forwards very
    well: segment 2 reads segment 1's output because it sits after it. Nothing
    needs to be remembered between runs, so nothing is - the whole chain is one
    Queue press, and what happens is visible on the canvas instead of hidden in
    a cache folder.

    Wire `next_index` into the following node's `segment_index_in` and the
    numbering takes care of itself.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment": (_SEGMENT_CHOICES, {"default": _DEFAULT_SEGMENT}),
                "overlap": (_OVERLAP_CHOICES, {"default": "39 (audio-exact)"}),
                "segment_index": ("INT", {
                    "default": 1, "min": 1, "max": 999, "step": 1,
                    "tooltip": "Which link this is, 1-based. Ignored when segment_index_in is "
                               "connected — chain that instead and never touch this."}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "reference_video": ("IMAGE", {
                    "tooltip": "The whole source video. This node hands out only the window that "
                               "belongs to this segment."}),
                "reference_audio": ("AUDIO",),
                "previous_segment": ("IMAGE", {
                    "tooltip": "The decoded output of the previous sampler. Its tail becomes this "
                               "segment's guide. Leave empty on segment 1."}),
                "initial_frames": ("IMAGE", {
                    "tooltip": "Segment 1 only: what to pin at index 0 — typically your swapped "
                               "first frame."}),
                "segment_index_in": ("INT", {
                    "forceInput": True,
                    "tooltip": "Connect the previous node's next_index."}),
            },
        }

    # `guide_seconds` est ajoute EN DERNIER : les sorties, comme les widgets, sont
    # reperees par position dans les workflows deja enregistres.
    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "INT", "INT", "FLOAT", "INT", "BOOLEAN",
                    "STRING", "INT", "FLOAT", "AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("guide_frames", "reference_slice", "audio_slice",
                    "next_index", "segment_frames", "segment_seconds", "overlap_frames",
                    "is_first", "info", "segments_needed", "source_seconds", "guide_audio",
                    "window_start_seconds", "guide_seconds")
    FUNCTION = "segment_for"
    CATEGORY = "video"

    def segment_for(self, segment, overlap, segment_index, video_fps,
                    reference_video=None, reference_audio=None,
                    previous_segment=None, initial_frames=None, segment_index_in=None):

        seg = _frames_of(segment)
        ov = _frames_of(overlap)
        if ov >= seg:
            raise ValueError(f"[Video Chain] overlap ({ov}) must be smaller than the segment ({seg}).")

        index = int(segment_index_in) if segment_index_in is not None else int(segment_index)
        index = max(1, index)
        advance = seg - ov
        fps = video_fps if video_fps > 0 else 24.0

        # ── Guide ────────────────────────────────────────────────────────────
        is_first = previous_segment is None or int(previous_segment.shape[0]) == 0
        if is_first:
            if initial_frames is None or int(initial_frames.shape[0]) == 0:
                # NE JAMAIS renvoyer une image de remplissage ici.
                #
                # Ce que cette sortie alimente, c'est MiniMaxH3AddGuide, et un
                # guide n'est pas une suggestion : le latent est re-injecte a
                # CHAQUE etape et n'est jamais debruite. Une frame noire de
                # remplacement devient donc un carre noir que le modele est
                # force de reproduire pendant les 50 etapes, et il passe le
                # debut du rendu a s'en extraire - ce qui ruine l'ouverture,
                # decale toute la suite, et donne l'impression que la video de
                # reference n'est pas suivie.
                #
                # Cette version rendait torch.zeros((1, 64, 64, 3)) en silence,
                # et le journal affichait "guide: 1 frame(s) [first segment]",
                # ce qui avait l'air normal.
                raise ValueError(
                    "[Video Chain] segment 1 has nothing to pin: `initial_frames` is empty "
                    "and there is no previous segment.\n"
                    "\n"
                    "`guide_frames` feeds MiniMaxH3AddGuide, whose latent is re-injected at "
                    "every sampling step and never denoised. A placeholder here is not a "
                    "harmless blank — it is a hard constraint the model must reproduce.\n"
                    "\n"
                    "-> Connect your corrected first frame to `initial_frames` (the same image "
                    "you already feed to ref_image_0),\n"
                    "-> or, if segment 1 should start free, bypass its MiniMaxH3AddGuide and "
                    "wire the sampler's conditioning straight from MiniMaxH3ReferenceToVideo — "
                    "which is what a single-segment workflow does."
                )
            guide = initial_frames
        else:
            have = int(previous_segment.shape[0])
            if have < ov:
                raise ValueError(
                    f"[Video Chain] the previous segment has {have} frame(s), fewer than the "
                    f"{ov}-frame overlap. Either it did not render fully, or overlap is too large."
                )
            guide = previous_segment[-ov:]

        # ── Fenetre de reference ─────────────────────────────────────────────
        # L'avance vaut seg - ov : les ov premieres frames du nouveau segment
        # refabriquent la fin du precedent. Avancer de seg decalerait le
        # mouvement de ov frames a chaque joint - l'image raccorderait, le geste
        # sauterait, et ca se voit beaucoup plus qu'une couture.
        start = (index - 1) * advance
        source_total = int(reference_video.shape[0]) if reference_video is not None else 0
        shrunk_from = 0
        segments_needed = 0

        if source_total > 0:
            segments_needed = 1 + max(0, -(-(source_total - seg) // advance))

            if start >= source_total:
                msg = (f"[Video Chain] segment {index} is beyond the source: it would start at "
                       f"frame {start}, and the reference video has {source_total} "
                       f"({source_total / fps:.2f}s). This source needs {segments_needed} "
                       f"segment(s).")
                if ExecutionBlocker is not None:
                    print(f"⛓️  {msg}\n     → branch skipped, nothing downstream runs.")
                    return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)
                raise ValueError(
                    msg + "\n-> Bypass this branch (Onyx Group Toggle), or use a longer "
                          "reference video."
                )

            available = source_total - start
            if available < seg:
                # Derniere tranche : le segment se reduit a ce que la reference
                # peut reellement conditionner. Generer 294 frames avec 225 frames
                # de reference laisse 69 frames sans conditionnement, et c'est
                # exactement la ou le modele invente une fin.
                fits = [n for n in _AV_GRID if n <= available]
                if not fits or fits[-1] <= ov:
                    msg = (f"[Video Chain] segment {index} has only {available} reference "
                           f"frame(s) left ({available / fps:.2f}s), not enough for a segment "
                           f"longer than the {ov}-frame overlap. This source needs "
                           f"{segments_needed} segment(s).")
                    if ExecutionBlocker is not None:
                        print(f"⛓️  {msg}\n     → branch skipped, nothing downstream runs.")
                        return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)
                    raise ValueError(msg + "\n-> Bypass this branch.")
                shrunk_from, seg = seg, fits[-1]
                end = start + seg

        end = start + seg

        if source_total > 0:
            ref_slice = reference_video[start:start + seg].clone()
        else:
            ref_slice = torch.zeros((1, 64, 64, 3))

        audio_slice = _slice_audio(reference_audio, start / fps, end / fps)
        if audio_slice is None:
            audio_slice = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": 44100}

        lines = [
            f"segment {index} of {segments_needed or '?'} — timeline frames {start}..{end - 1} "
            f"({start / fps:.3f}s → {end / fps:.3f}s)",
            f"guide: {int(guide.shape[0])} frame(s)" + ("  [first segment]" if is_first else ""),
            f"reference slice: {int(ref_slice.shape[0])} frame(s)",
            f"video length once this segment lands: {(start + seg) / fps:.3f}s",
        ]
        if shrunk_from:
            lines.append(
                f"** last segment: shrunk {shrunk_from} -> {seg} frames to match the "
                f"{source_total - start} reference frame(s) left. `length` follows automatically."
            )
        if segments_needed and index == 1 and segments_needed > 1:
            lines.append(f"** this source needs {segments_needed} segment(s) in total "
                         f"({source_total} frames / {source_total / fps:.2f}s).")
        info = "\n".join(lines)

        print(f"⛓️  [Video Chain #{index}] " + info.replace("\n", "\n     "))
        # segment_seconds sort ici pour que duration_seconds du Context-IR vienne
        # de la MEME source que length. Les laisser sur deux nodes distincts, c'est
        # une desynchronisation silencieuse a chaque changement de segment.
        # source_seconds decrit la SOURCE ENTIERE, pas le segment.
        #
        # C'est ce qu'attend le grounding d'un Context-IR partage : il analyse la
        # video complete, donc lui annoncer la duree d'un segment lui fait
        # comprimer - ou supprimer - tout ce qui n'y rentre pas. Un dialogue situe
        # dans la seconde moitie de la source disparait alors du prompt, et le
        # modele n'a plus aucune raison d'articuler.
        # guide_audio couvre EXACTEMENT la meme fenetre que guide_frames.
        #
        # MiniMaxH3AddGuide epingle image et audio ensemble a un meme index. Lui
        # donner 39 frames d'image (1,6 s) et 12 s de son le laisse decider seul
        # de la correspondance, et le resultat est un decalage de plusieurs
        # secondes entre la bouche et la parole. Les deux guides doivent durer
        # la meme chose.
        guide_audio = None
        if not is_first and reference_audio is not None:
            guide_frames_n = int(guide.shape[0])
            guide_audio = _slice_audio(reference_audio,
                                       start / fps,
                                       (start + guide_frames_n) / fps)
        if guide_audio is None:
            guide_audio = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": 44100}

        # Duree reellement epinglee par AddGuide, a brancher sur le
        # `pinned_lead_seconds` du Context-IR. Zero sur le premier segment : son
        # guide ne fait qu'une frame, il n'y a rien de deja rendu a proteger.
        guide_seconds = 0.0 if is_first else guide_frames_n / fps

        return (guide, ref_slice, audio_slice, index + 1, seg, seg / fps, ov, is_first, info,
                segments_needed or 1, (source_total / fps) if source_total else seg / fps,
                guide_audio, start / fps, guide_seconds)


class OnyxVideoChainJoin:
    """Concatenate the finished segments, and never run the ones the source cannot fill.

    Two jobs in one node.

    1. Trim and join. Every segment after the first opens with `overlap` frames
       that re-render the tail of the previous one. Kept, they appear twice and
       the joint stutters; this node drops them.

    2. Decide how many branches actually run. `image_2` and beyond are LAZY
       inputs: ComfyUI evaluates an input only when the node asks for it, and
       this node asks for exactly `segments_needed`. A branch that is not asked
       for is never executed - its sampler does not run, does not load the
       model, does not spend twenty minutes producing frames that would be
       thrown away.

    That is what replaces the manual bypassing. `segments_needed` comes from
    segment 1, which knows the length of the source video, so the graph adapts
    to the footage instead of the other way round.

    Segment 1 stays eager: it always runs, and it is what tells this node how
    many of the others to ask for.
    """

    MAX_SEGMENTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "image_1": ("IMAGE", {"tooltip": "Segment 1, whole. Always evaluated."}),
            "audio_1": ("AUDIO", {"tooltip": "Optional. Connect each sampler's audio output to "
                                             "get one continuous generated soundtrack."}),
        }
        for i in range(2, cls.MAX_SEGMENTS + 1):
            optional[f"image_{i}"] = ("IMAGE", {
                "lazy": True,
                "tooltip": f"Segment {i}. Only evaluated when segments_needed >= {i}; "
                           f"otherwise this whole branch is skipped."})
            optional[f"audio_{i}"] = ("AUDIO", {"lazy": True})
        return {
            "required": {
                "segments_needed": ("INT", {
                    "forceInput": True,
                    "tooltip": "Connect segment 1's `segments_needed`. It is computed from the "
                               "length of the reference video."}),
                "overlap_frames": ("INT", {
                    "forceInput": True,
                    "tooltip": "Connect segment 1's `overlap_frames`. Dropped from the head of "
                               "every segment after the first."}),
                "video_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Used to convert the overlap into seconds when trimming audio."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "frame_count", "info")
    FUNCTION = "join"
    CATEGORY = "video"

    @classmethod
    def _linked_depth(cls, kwargs, wanted):
        """How many segments are actually wired, counting from 1 without a gap.

        `key in kwargs` is the test that distinguishes the two cases: ComfyUI
        puts a connected-but-unevaluated lazy input in the dict as None, while an
        input with no link at all is absent from it entirely.

        The count stops at the first gap on purpose. Segments are a timeline: if
        4 is wired but 3 is not, there is no video to make from 1, 2 and 4 - the
        middle is missing and the rest cannot be shifted to cover it.
        """
        depth = 0
        for i in range(1, wanted + 1):
            if f"image_{i}" not in kwargs:
                break
            depth = i
        return depth

    @classmethod
    def check_lazy_status(cls, segments_needed=None, overlap_frames=None, video_fps=24.0,
                          **kwargs):
        """Name the branches to evaluate. Everything unnamed is never executed.

        Only WIRED branches are ever asked for. Asking for an unlinked input
        would never be satisfied, and the graph would wait on it forever.

        This is also where a short wiring gets caught. The old code left the
        check to `join()`, which runs only after every requested branch has
        finished: a source needing 4 segments with 3 wired sampled all three -
        forty-odd minutes - and only then raised. Clamping the plan here means
        the branches that would be discarded are never run at all.
        """
        wanted = max(1, min(int(segments_needed or 1), cls.MAX_SEGMENTS))
        depth = cls._linked_depth(kwargs, wanted)
        if depth < wanted:
            print(f"⚠️  [Video Chain Join] the source needs {wanted} segment(s) but only "
                  f"{depth} is/are wired.\n"
                  f"     → rendering {depth} and stopping there. Segment(s) "
                  f"{', '.join(str(i) for i in range(depth + 1, wanted + 1))} will not run.\n"
                  f"     → wire image_{depth + 1} (and its audio) to get the whole source.")

        needed = []
        for i in range(1, max(1, depth) + 1):
            for key in (f"image_{i}", f"audio_{i}"):
                if key in kwargs and kwargs[key] is None:
                    needed.append(key)
        return needed

    def join(self, segments_needed, overlap_frames, video_fps=24.0, **images):
        asked = max(1, min(int(segments_needed or 1), self.MAX_SEGMENTS))
        ov = max(0, int(overlap_frames or 0))
        fps = video_fps if video_fps > 0 else 24.0

        # Ce qui est reellement arrive, pas ce qui etait prevu. Un raise ici
        # jetterait tout le rendu deja produit : sur une chaine de segments cela
        # represente des dizaines de minutes de GPU, pour un cablage incomplet
        # que l'on peut parfaitement livrer tronque en le disant. Le plan a deja
        # ete ramene a la bonne taille dans check_lazy_status ; ceci est le
        # filet, pour le cas ou un segment reviendrait vide malgre tout.
        wanted = 0
        for i in range(1, asked + 1):
            if images.get(f"image_{i}") is None:
                break
            wanted = i

        if wanted == 0:
            raise ValueError(
                "[Video Chain Join] no segment reached this node — image_1 is empty.\n"
                "-> Connect segment 1's sampler output to image_1."
            )
        if wanted < asked:
            print(f"⚠️  [Video Chain Join] {asked} segment(s) planned, {wanted} received — "
                  f"joining what exists. The video will be short of the source.")

        parts, report = [], []
        for i in range(1, wanted + 1):
            part = images.get(f"image_{i}")
            total = int(part.shape[0])
            if i == 1:
                kept = part
                report.append(f"segment 1: {total} frame(s), whole")
            else:
                if total <= ov:
                    raise ValueError(
                        f"[Video Chain Join] segment {i} has {total} frame(s), not more than the "
                        f"{ov}-frame overlap. Nothing new in it."
                    )
                kept = part[ov:]
                report.append(f"segment {i}: {total} - {ov} = {int(kept.shape[0])} frame(s)")
            parts.append(kept.cpu())

        result = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        count = int(result.shape[0])
        # Compte sur `asked`, pas sur `wanted` : les branches au-dela du besoin
        # de la source sont legitimement inutilisees, tandis qu'un segment
        # manquant par cablage, lui, manque vraiment - il est signale a part.
        skipped = self.MAX_SEGMENTS - asked

        audio_out, audio_note = self._join_audio(images, wanted, ov, fps)
        if audio_note:
            report.append(audio_note)

        info = (f"joined {wanted} segment(s) -> {count} frame(s) "
                f"({count / fps:.3f}s)\n  "
                + "\n  ".join(report)
                + (f"\n  {skipped} branch(es) never executed — the source does not need them."
                   if skipped else "")
                + (f"\n  ⚠️  {asked - wanted} segment(s) missing from the wiring: the source "
                   f"is longer than this render."
                   if wanted < asked else ""))
        print(f"⛓️  [Video Chain Join] " + info.replace("\n", "\n     "))
        return (result, audio_out, count, info)

    @staticmethod
    def _join_audio(sources, wanted, overlap_frames, fps):
        """Concatenate the segments' audio, dropping the same overlap as the video.

        The trim is expressed in SAMPLES derived from the overlap in frames, so
        picture and sound are cut at the same instant. Rounding them separately
        would drift the soundtrack by a few milliseconds at every joint, and the
        drift accumulates.
        """
        tracks = [sources.get(f"audio_{i}") for i in range(1, wanted + 1)]
        if not any(t is not None for t in tracks):
            return None, ""
        if any(t is None for t in tracks):
            return None, ("audio: skipped — connect every segment's audio output, or none "
                          "(some are missing)")

        rates = {int(t["sample_rate"]) for t in tracks}
        if len(rates) > 1:
            return None, f"audio: skipped — mixed sample rates {sorted(rates)}"
        sr = rates.pop()
        trim = int(round(overlap_frames / fps * sr))

        pieces = []
        for i, track in enumerate(tracks):
            wav = track["waveform"]
            if i and trim:
                if int(wav.shape[-1]) <= trim:
                    return None, (f"audio: skipped — segment {i + 1} is shorter than the "
                                  f"{overlap_frames}-frame overlap")
                wav = wav[..., trim:]
            pieces.append(wav)

        joined = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
        seconds = int(joined.shape[-1]) / sr
        return ({"waveform": joined, "sample_rate": sr},
                f"audio: {len(tracks)} track(s) joined -> {seconds:.3f}s at {sr} Hz "
                f"({trim} samples trimmed per joint)")


class OnyxVideoChainPrepare:
    """Hand out the guide tail and the reference slice for the current segment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_name": ("STRING", {"default": "chain", "multiline": False,
                                            "tooltip": "One chain per name. Two workflows with "
                                                       "different names advance independently."}),
                "segment": (_SEGMENT_CHOICES, {"default": _DEFAULT_SEGMENT,
                                               "tooltip": "Length of ONE generation. Every value here "
                                                          "is on both H3 grids, so the audio stays in "
                                                          "phase at the joints. 345 is H3's ceiling."}),
                "overlap": (_OVERLAP_CHOICES, {"default": "39 (audio-exact)",
                                               "tooltip": "How many frames of the previous segment are "
                                                          "regenerated as a guide. More overlap = a "
                                                          "steadier joint and less new footage per run."}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "reference_video": ("IMAGE", {
                    "tooltip": "The full source video that carries the motion. It is sliced "
                               "automatically: each segment receives the part that matches its own "
                               "position on the timeline."}),
                "reference_audio": ("AUDIO", {"tooltip": "Sliced on the same window as the video."}),
                "initial_frames": ("IMAGE", {
                    "tooltip": "What segment 1 uses as its guide — typically your swapped first "
                               "frame. Left empty, segment 1 runs with no guide."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("guide_frames", "reference_slice", "audio_slice",
                    "segment_index", "segment_frames", "overlap_frames", "is_first", "info")
    FUNCTION = "prepare"
    CATEGORY = "video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # L'etat vit sur disque et change d'un lancement a l'autre : sans ca,
        # ComfyUI reservirait le cache et la chaine n'avancerait jamais.
        return float("nan")

    def prepare(self, session_name, segment, overlap, video_fps,
                reference_video=None, reference_audio=None, initial_frames=None):

        seg = _frames_of(segment)
        ov = _frames_of(overlap)
        if ov >= seg:
            raise ValueError(f"[Video Chain] overlap ({ov}) must be smaller than the segment ({seg}).")
        advance = seg - ov

        meta, tail = _read_state(session_name)
        index = int(meta.get("index", 0)) if meta else 0

        if meta and (meta.get("segment") != seg or meta.get("overlap") != ov):
            raise ValueError(
                f"[Video Chain] session '{session_name}' was started with segment="
                f"{meta.get('segment')} / overlap={meta.get('overlap')}, and you are now asking for "
                f"{seg} / {ov}.\n-> Reset the chain, or put the previous values back. Changing them "
                f"mid-chain would misalign every later reference slice."
            )

        is_first = index == 0

        # ── Guide ────────────────────────────────────────────────────────────
        if is_first:
            # Meme mine que dans Segment : un guide alimente AddGuide, dont le
            # latent est re-injecte a chaque etape et jamais debruite. Une frame
            # de remplissage y devient une contrainte dure.
            if initial_frames is None or int(initial_frames.shape[0]) == 0:
                raise ValueError(
                    "[Video Chain] segment 1 has nothing to pin: `initial_frames` is empty.\n"
                    "-> Connect your corrected first frame, or bypass MiniMaxH3AddGuide for "
                    "this segment and feed the sampler straight from "
                    "MiniMaxH3ReferenceToVideo."
                )
            guide = initial_frames
        else:
            if tail is None:
                raise RuntimeError(
                    f"[Video Chain] session '{session_name}' is at segment {index + 1} but its tail "
                    f"is missing. Reset the chain."
                )
            guide = tail

        # ── Tranche de reference ─────────────────────────────────────────────
        # L'avance est seg - ov, pas seg : les ov premieres frames du nouveau
        # segment refabriquent la fin du precedent. Faire avancer de seg
        # decalerait le mouvement de ov frames a chaque joint, ce qui se voit
        # comme un saut alors que l'image, elle, raccorde proprement.
        start = index * advance
        end = start + seg
        ref_slice = None
        exhausted = False
        if reference_video is not None and int(reference_video.shape[0]) > 0:
            total = int(reference_video.shape[0])
            if start >= total:
                exhausted = True
                ref_slice = reference_video[-1:].clone()
            else:
                ref_slice = reference_video[start:min(end, total)].clone()
        else:
            ref_slice = torch.zeros((1, 64, 64, 3))

        fps = video_fps if video_fps > 0 else 24.0
        audio_slice = _slice_audio(reference_audio, start / fps, end / fps)
        if audio_slice is None:
            audio_slice = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": 44100}

        produced = index * advance
        info_lines = [
            f"segment {index + 1} — frames {start}..{end - 1} of the timeline "
            f"({start / fps:.3f}s → {end / fps:.3f}s)",
            f"guide: {int(guide.shape[0])} frame(s)"
            + ("  [segment 1, no previous tail]" if is_first else ""),
            f"reference slice: {int(ref_slice.shape[0])} frame(s)",
            f"finished so far: {produced} frame(s) = {produced / fps:.3f}s",
            f"this run adds {advance} frame(s) = {advance / fps:.3f}s",
        ]
        if exhausted:
            info_lines.append("!! the reference video is exhausted — this segment has no motion source")
        info = "\n".join(info_lines)

        print(f"⛓️  [Video Chain: {session_name}] " + info.replace("\n", "\n     "))
        if exhausted:
            print(f"⚠️  [Video Chain: {session_name}] reference video exhausted at segment {index + 1}.")

        return (guide, ref_slice, audio_slice, index + 1, seg, ov, is_first, info)


class OnyxVideoChainCommit:
    """Store the tail of the segment that was just generated and advance the counter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The segment that just came out of the sampler."}),
                "session_name": ("STRING", {"default": "chain", "multiline": False}),
                "overlap": (_OVERLAP_CHOICES, {"default": "39 (audio-exact)"}),
            },
            "optional": {
                "trim_leading_overlap": ("BOOLEAN", {
                    "default": True,
                    "label_on": "drop the regenerated head",
                    "label_off": "keep every frame",
                    "tooltip": "On: the first `overlap` frames are removed from `new_frames`, "
                               "because they are a re-render of footage you already have. This is "
                               "what makes the segments concatenate without a stutter.\n"
                               "Off: useful only to inspect how well the joint was reproduced."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("new_frames", "segment_index", "info")
    FUNCTION = "commit"
    CATEGORY = "video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def commit(self, images, session_name, overlap, trim_leading_overlap=True):
        ov = _frames_of(overlap)
        total = int(images.shape[0])
        if total <= ov:
            raise ValueError(
                f"[Video Chain] the segment has {total} frame(s), which is not more than the "
                f"{ov}-frame overlap. Nothing new was produced."
            )

        meta, _tail = _read_state(session_name)
        index = int(meta.get("index", 0)) if meta else 0
        seg = int(meta.get("segment", total)) if meta else total

        # La queue devient le guide du segment suivant.
        new_tail = images[-ov:].clone()
        new_frames = images[ov:] if trim_leading_overlap else images

        _write_state(session_name,
                     {"index": index + 1, "segment": seg, "overlap": ov, "last_total": total},
                     new_tail)

        info = (f"segment {index + 1} committed — {total} generated, "
                f"{int(new_frames.shape[0])} kept, {ov} stored as the next guide.\n"
                f"Queue again for segment {index + 2}.")
        print(f"⛓️  [Video Chain: {session_name}] " + info.replace("\n", "\n     "))
        return (new_frames, index + 1, info)


# ── Route de remise a zero, appelee par le bouton du node ────────────────────
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/onyx/video_chain/reset")
    async def _video_chain_reset(request):
        try:
            data = await request.json()
            session = (data.get("session") or "").strip()
            if not session:
                return web.json_response({"error": "session_name is empty"}, status=400)
            existed = reset_session(session)
            return web.json_response({"reset": existed, "session": session})
        except Exception as e:
            logging.exception("Video Chain: reset failed")
            return web.json_response({"error": str(e)}, status=500)
except Exception:
    logging.warning("Onyx Video Chain: could not register the reset route (server not available)")
