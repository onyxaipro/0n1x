"""
Onyx HD Detailer nodes
===========================

Copies patchees de deux nodes d'Impact Pack / Impact Subpack, deplacees dans le
pack onyx pour ne jamais toucher aux fichiers originaux (qui sont ecrases a
chaque mise a jour des packs Impact) :

1. OnyxEyeBBoxDetectorProvider  ("Onyx HD Ultralytic BBox Loader")
   -> copie de `UltralyticsDetectorProvider` (Impact Subpack) dont l'inference
      YOLO force `imgsz=1280` (au lieu du 640x384 par defaut) pour detecter les
      petits objets comme les yeux.

2. OnyxDetailer  ("Onyx Detailer")
   -> copie exacte de `FaceDetailer` (Impact Pack), memes options (bbox +
      segm + sam integres, pas besoin d'un node SEGS separe), avec un seul
      champ ajoute : `interpolation`, qui choisit le resampler utilise pour
      le resize retour (paste-back, apres decode VAE, juste avant le
      recollage dans l'image pleine resolution). Defaut : bicubic (+
      antialias). Impact Pack original n'expose pas ce choix.

Les deux nodes n'heritent PAS des classes originales : elles dupliquent la
logique en dur (duck-typing) pour ne jamais dependre de l'ordre de chargement
des custom_nodes au demarrage de ComfyUI. Les modules `impact.*` (Impact Pack)
et `subcore` (Impact Subpack) sont recherches paresseusement (a l'execution,
jamais a l'import) via `sys.modules`, ce qui a pour effet pratique de toujours
patcher la copie du pack Impact reellement chargee par ComfyUI, meme s'il en
existe plusieurs en double dans custom_nodes.
"""

import logging
import time
import inspect
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import nodes
import comfy.samplers

try:
    from comfy_extras import nodes_differential_diffusion
except Exception:
    nodes_differential_diffusion = None


# ---------------------------------------------------------------------------
# Patch 0 - register the "beta57" scheduler (variante de `beta` avec
# alpha=0.5, beta=0.7 au lieu de 0.6/0.6, popularisee par RES4LYF) dans le
# registre global de ComfyUI, pour qu'elle apparaisse dans le menu deroulant
# "scheduler" de ce node (qui liste dynamiquement comfy.samplers.SCHEDULER_HANDLERS
# via `core.get_schedulers()`). Idempotent : ne fait rien si deja enregistree
# (ex: par un autre pack comme RES4LYF).
# ---------------------------------------------------------------------------

if "beta57" not in comfy.samplers.SCHEDULER_HANDLERS:
    comfy.samplers.SCHEDULER_HANDLERS["beta57"] = comfy.samplers.SchedulerHandler(
        lambda model_sampling, steps: comfy.samplers.beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)
    )
    comfy.samplers.SCHEDULER_NAMES.append("beta57")


# ---------------------------------------------------------------------------
# Lazy lookup helpers - resolve the *actually loaded* Impact modules at call
# time so import order between custom_nodes folders never matters.
# ---------------------------------------------------------------------------

_subcore_cache = None


def _get_subcore():
    """Return the live `subcore` module from ComfyUI-Impact-Subpack, whichever
    of the (possibly duplicated) install folders ComfyUI actually loaded."""
    global _subcore_cache
    if _subcore_cache is not None:
        return _subcore_cache

    for name, mod in list(sys.modules.items()):
        if mod is None or not name.endswith("subcore"):
            continue
        if hasattr(mod, "UltraBBoxDetector") and hasattr(mod, "load_yolo") and hasattr(mod, "inference_bbox"):
            _subcore_cache = mod
            return mod

    raise RuntimeError(
        "[Onyx] Impossible de trouver le module 'subcore' de ComfyUI-Impact-Subpack. "
        "Verifie que le pack Impact Subpack est installe et active."
    )


def _impact_modules():
    """Return (core, utils, wildcards, impact_sampling) from Impact Pack."""
    import impact.core as core
    import impact.utils as utils
    import impact.wildcards as wildcards
    from impact import impact_sampling
    return core, utils, wildcards, impact_sampling


# ---------------------------------------------------------------------------
# Patch 1 - HD Ultralytics BBox loader with forced YOLO inference resolution
# ---------------------------------------------------------------------------

def _inference_bbox_hd(model, image, confidence, imgsz, device=""):
    """Copie de `subcore.inference_bbox` avec `imgsz` ajoute a l'appel YOLO."""
    pred = model(image, conf=confidence, device=device, imgsz=imgsz)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()
    else:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        segms.append(cv2_mask.astype(bool))

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


class OnyxHDBBoxDetector:
    """Objet BBOX_DETECTOR duck-type compatible (detect / detect_combined /
    setAux) qui force la resolution d'inference YOLO via `imgsz`."""

    def __init__(self, bbox_model, imgsz=1280):
        self.bbox_model = bbox_model
        self.imgsz = imgsz

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        subcore = _get_subcore()
        utils = subcore.utils

        drop_size = max(drop_size, 1)
        detected_results = _inference_bbox_hd(self.bbox_model, utils.tensor2pil(image), threshold, self.imgsz)
        segmasks = subcore.create_segmasks(detected_results)

        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:
                crop_region = utils.make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = utils.crop_image(image, crop_region)
                cropped_mask = utils.crop_ndarray2(item_mask, crop_region)
                confidence = x[2]

                item = subcore.SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)
                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        subcore = _get_subcore()
        utils = subcore.utils

        detected_results = _inference_bbox_hd(self.bbox_model, utils.tensor2pil(image), threshold, self.imgsz)
        segmasks = subcore.create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)

        return utils.combine_masks(segmasks)

    def setAux(self, x):
        pass


class OnyxEyeBBoxDetectorProvider:
    """Copie patchee de `UltralyticsDetectorProvider` (Impact Subpack) :
    ajoute un widget `imgsz` (defaut 1280) transmis a l'inference YOLO pour
    detecter correctement les petits elements comme les yeux."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {
            "required": {
                "model_name": (bboxs + segms,),
                "imgsz": ("INT", {
                    "default": 1280, "min": 320, "max": 2048, "step": 32,
                    "tooltip": "Resolution d'inference YOLO. Plus haut = meilleure detection des petits objets (yeux) mais plus lent. Impact Pack par defaut n'expose pas ce parametre et infere en 640x384."
                }),
            }
        }

    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"
    CATEGORY = "Onyx/Detailer"
    DESCRIPTION = "Copie patchee de Ultralytics Detector Provider : force imgsz=1280 sur l'inference YOLO pour detecter les deux yeux."

    def doit(self, model_name, imgsz=1280):
        import folder_paths
        subcore = _get_subcore()

        model_path = folder_paths.get_full_path("ultralytics", model_name)

        if model_path is None:
            if model_name.startswith('bbox/'):
                model_path = folder_paths.get_full_path("ultralytics_bbox", model_name[5:])
            elif model_name.startswith('segm/'):
                model_path = folder_paths.get_full_path("ultralytics_segm", model_name[5:])

        if model_path is None:
            raise ValueError(f"[Onyx] model file '{model_name}' introuvable.")

        model = subcore.load_yolo(model_path)

        if model_name.startswith("bbox"):
            return OnyxHDBBoxDetector(model, imgsz), subcore.NO_SEGM_DETECTOR()
        else:
            segm_detector_cls = getattr(subcore, "UltraSegmDetector", None)
            segm_detector = segm_detector_cls(model) if segm_detector_cls else subcore.NO_SEGM_DETECTOR()
            return OnyxHDBBoxDetector(model, imgsz), segm_detector


# ---------------------------------------------------------------------------
# Patch 2 - FaceDetailer clone with a selectable paste-back interpolation mode
# ---------------------------------------------------------------------------

# Modes compatibles avec torch.nn.functional.interpolate pour une image RGB.
# antialias n'est supporte que par bilinear/bicubic ; align_corners uniquement
# par les modes qui l'acceptent.
INTERPOLATION_MODES = ["bicubic", "bilinear", "nearest-exact", "nearest", "area"]
_ALIGN_CORNERS_MODES = {"bilinear", "bicubic"}
_ANTIALIAS_MODES = {"bilinear", "bicubic"}


def _resize_interp(image, w, h, mode="bicubic"):
    """Resize d'un tensor image (B,H,W,C) dans [0,1] avec le resampler choisi."""
    img = image.permute(0, 3, 1, 2)
    kwargs = {"size": (h, w), "mode": mode}
    if mode in _ALIGN_CORNERS_MODES:
        kwargs["align_corners"] = False
    if mode in _ANTIALIAS_MODES:
        kwargs["antialias"] = True
    img = F.interpolate(img, **kwargs)
    img = img.clamp(0.0, 1.0)
    img = img.permute(0, 2, 3, 1)
    return img.contiguous()


# ITU-R BT.601 luma weights.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def _unsharp_mask_luma(image, amount, kernel_size=5, sigma=1.0):
    """Nettete (unsharp mask) appliquee uniquement sur la luminance, puis
    reinjectee a poids egal sur les 3 canaux RGB pour preserver la teinte et
    la saturation d'origine.

    Pourquoi luminance-only plutot qu'un unsharp mask naif applique
    independamment sur R/G/B : un unsharp RGB accentue aussi le bruit
    chromatique et cree un liseret de couleur (fringing) sur les contours,
    ce qui se voit particulierement sur une petite zone feathered (grain qui
    se detache visuellement de la peau adjacente). En ne sharpenant que la
    luminance, on evite cet artefact pour un cout quasi identique : le flou
    gaussien tourne sur 1 canal au lieu de 3.

    image: tensor (B,H,W,C) dans [0,1]. amount: intensite (0 = desactive).
    """
    if amount <= 0:
        return image

    img = image.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W

    weights = img.new_tensor(_LUMA_WEIGHTS).view(1, 3, 1, 1)
    luma = (img * weights).sum(dim=1, keepdim=True)  # (B,1,H,W)

    blurred_luma = TF.gaussian_blur(luma, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    high_freq = luma - blurred_luma

    delta = amount * high_freq  # (B,1,H,W), broadcast sur les 3 canaux RGB
    img = (img + delta).clamp(0.0, 1.0)

    return img.permute(0, 2, 3, 1).contiguous()  # B,C,H,W -> B,H,W,C


def _color_match_stats(refined, original, strength=1.0):
    """Réaligne moyenne/écart-type par canal du crop raffiné (`refined`) sur
    ceux du crop d'origine (`original`), pour neutraliser toute dérive
    globale de teinte/luminosité (VAE ou sampler) tout en gardant le détail
    régénéré. Les deux tensors sont (B,H,W,C) dans [0,1] et de même taille.

    `strength` (0-1) mélange linéairement entre l'image non modifiée (0) et
    le match statistique complet (1).
    """
    if strength <= 0:
        return refined

    matched = refined.clone()
    for c in range(matched.shape[-1]):
        src = original[..., c]
        ref = matched[..., c]
        matched[..., c] = (ref - ref.mean()) / (ref.std() + 1e-6) * src.std() + src.mean()
    matched = matched.clamp(0.0, 1.0)

    if strength < 1.0:
        matched = refined * (1.0 - strength) + matched * strength
        matched = matched.clamp(0.0, 1.0)

    return matched


def enhance_detail_hd(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                      sampler_name, scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                      wildcard_opt=None, wildcard_opt_concat_mode=None,
                      detailer_hook=None,
                      refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                      refiner_negative=None, control_net_wrapper=None, cycle=1,
                      inpaint_model=False, noise_mask_feather=0, scheduler_func=None,
                      vae_tiled_encode=False, vae_tiled_decode=False, interpolation="bicubic", sharpness=0.5,
                      color_match_strength=1.0):
    """Copie de `impact.core.enhance_detail` : seul le resize retour (downscale
    apres decode VAE, juste avant le paste-back dans l'image pleine resolution)
    utilise le resampler choisi (`interpolation`) au lieu du resampler fixe
    d'origine."""

    core, utils, wildcards, impact_sampling = _impact_modules()

    if noise_mask is not None:
        noise_mask = utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

        if noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = wildcards.process_with_loras(wildcard_opt, model, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive
            positive = [positive[0].copy()]
            if 'pooled_output' in wildcard_positive[0][1]:
                positive[0][1]['pooled_output'] = wildcard_positive[0][1]['pooled_output']
            elif 'pooled_output' in positive[0][1]:
                del positive[0][1]['pooled_output']

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        logging.info("Onyx Detailer: segment skip (enough big)")
        return None, None

    if guide_size_for_bbox:
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if 'aitemplate_keep_loaded' in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            logging.info(f"Onyx Detailer: segment skip [determined upscale factor={upscale}]")
            return None, None

        if new_w == 0 or new_h == 0:
            logging.info(f"Onyx Detailer: segment skip [zero size={new_w, new_h}]")
            return None, None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            logging.info("Onyx Detailer: force inpaint")
            upscale = 1.0
            new_w = w
            new_h = h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    logging.info(f"Onyx Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    upscaled_image = utils.tensor_resize(image, new_w, new_h)

    if detailer_hook is not None:
        upscaled_image = detailer_hook.post_upscale(upscaled_image, noise_mask)

    cnet_pils = None
    if control_net_wrapper is not None:
        positive, negative, cnet_pils = control_net_wrapper.apply(positive, negative, upscaled_image, noise_mask)
        model, cnet_pils2 = control_net_wrapper.doit_ipadapter(model)
        cnet_pils.extend(cnet_pils2)

    if detailer_hook is None or not detailer_hook.get_skip_sampling():
        if noise_mask is not None and inpaint_model:
            imc_encode = nodes.InpaintModelConditioning().encode
            if 'noise_mask' in inspect.signature(imc_encode).parameters:
                positive, negative, latent_image = imc_encode(positive, negative, upscaled_image, vae, mask=noise_mask, noise_mask=True)
            else:
                logging.warning("[Onyx Detailer] ComfyUI is an outdated version.")
                positive, negative, latent_image = imc_encode(positive, negative, upscaled_image, vae, noise_mask)
        else:
            latent_image = utils.to_latent_image(upscaled_image, vae, vae_tiled_encode=vae_tiled_encode)
            if noise_mask is not None:
                latent_image['noise_mask'] = noise_mask

        if detailer_hook is not None:
            latent_image = detailer_hook.post_encode(latent_image)

        refined_latent = latent_image

        sampler_opt = None
        if detailer_hook is not None:
            sampler_opt = detailer_hook.get_custom_sampler()

        for i in range(0, cycle):
            if detailer_hook is not None:
                detailer_hook.set_steps((i, cycle))
                refined_latent = detailer_hook.cycle_latent(refined_latent)

                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = \
                    detailer_hook.pre_ksample(model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
                noise, is_touched = detailer_hook.get_custom_noise(seed + i, torch.zeros(latent_image['samples'].size()), is_touched=False)
                if not is_touched:
                    noise = None
            else:
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, _, denoise2 = \
                    model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
                noise = None

            refined_latent = impact_sampling.ksampler_wrapper(
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2,
                refined_latent, denoise2, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative,
                noise=noise, scheduler_func=scheduler_func, sampler_opt=sampler_opt)

        if detailer_hook is not None:
            refined_latent = detailer_hook.pre_decode(refined_latent)

        start = time.time()
        if vae_tiled_decode:
            (refined_image,) = nodes.VAEDecodeTiled().decode(vae, refined_latent, 512)
            logging.info(f"[Onyx Detailer] vae decoded (tiled) in {time.time() - start:.1f}s")
        else:
            try:
                refined_image = vae.decode(refined_latent['samples'])
            except Exception:
                logging.warning(f"[Onyx Detailer] failed after {time.time() - start:.1f}s, doing vae.decode_tiled 64...")
                refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64)
            logging.info(f"[Onyx Detailer] vae decoded in {time.time() - start:.1f}s")
    else:
        refined_image = upscaled_image

    if detailer_hook is not None:
        refined_image = detailer_hook.post_decode(refined_image)

    if len(refined_image.shape) == 5:
        refined_image = refined_image.squeeze(0)

    # --- patch: resize retour (paste-back) avec le resampler choisi ---
    print(">>> PATCH RESCALE ACTIF: bicubic antialias")
    refined_image = _resize_interp(refined_image, w, h, interpolation)

    # --- patch: color match statistique (moyenne/ecart-type par canal) qui
    # realigne la teinte du crop raffine sur celle du crop d'origine (`image`,
    # meme resolution w x h a ce stade). ---
    refined_image = _color_match_stats(refined_image, image, color_match_strength)

    # --- patch: nettete (unsharp mask luminance-only) sur le crop regenere,
    # appliquee APRES le downscale (pas avant : sharpener a la resolution
    # d'inference puis redescendre attenue/efface l'effet). Ne touche que ce
    # crop, donc uniquement la partie regeneree par le detailer. ---
    refined_image = _unsharp_mask_luma(refined_image, sharpness)

    refined_image = refined_image.cpu()

    return refined_image, cnet_pils


class OnyxDetailer:
    """Copie exacte de `FaceDetailer` (Impact Pack) : bbox + segm + sam
    integres (aucun node SEGS separe requis), memes options, avec un seul
    champ ajoute : `interpolation` (defaut bicubic) pour le resize retour
    (paste-back)."""

    @classmethod
    def INPUT_TYPES(s):
        core, utils, wildcards, impact_sampling = _impact_modules()
        return {"required": {
                     "image": ("IMAGE",),
                     "model": ("MODEL", {"tooltip": "Si `ImpactDummyInput` est connecte au model, l'etape d'inference est sautee."}),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (core.get_schedulers(),),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

                     "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                     "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                     "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                     "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                     "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                     "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),

                     "drop_size": ("INT", {"min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "default": 10}),

                     "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),

                     "interpolation": (INTERPOLATION_MODES, {
                         "default": "bicubic",
                         "tooltip": "Resampler utilise pour le resize retour (paste-back) apres decode VAE. Impact Pack original n'expose pas ce choix."
                     }),
                     "sharpness": ("FLOAT", {
                         "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                         "tooltip": "Nettete (unsharp mask, luminance uniquement) appliquee apres le resize retour, sur la zone regeneree seulement. 0 = desactive. 0.3 = subtil, 0.5-0.6 = marque, 1.0+ = agressif."
                     }),
                     "color_match_strength": ("FLOAT", {
                         "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                         "tooltip": "Force du color-match statistique (moyenne/ecart-type par canal) qui realigne la teinte/luminosite de la zone regeneree sur celle du crop d'origine. 0 = desactive, 1 = match complet."
                     }),
                     },
                "optional": {
                    "bbox_detector": ("BBOX_DETECTOR", {
                        "tooltip": "Detecteur bbox (visages/yeux). Facultatif si `segs` est connecte : dans ce cas, la detection bbox/sam/segm est entierement sautee et ces SEGS sont utilisees telles quelles."
                    }),
                    "segs": ("SEGS", {
                        "tooltip": "SEGS deja calculees (ex: un masque manuel converti en SEGS). Si connecte, remplace entierement la detection bbox/sam/segm : `bbox_detector` et les reglages bbox_*/sam_* sont ignores."
                    }),
                    "sam_model_opt": ("SAM_MODEL",),
                    "segm_detector_opt": ("SEGM_DETECTOR",),
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    "tiled_encode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "tiled_decode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "Onyx/Detailer"

    DESCRIPTION = "Copie exacte de FaceDetailer (Impact Pack), meme options bbox/segm/sam, avec quatre ajouts : 'interpolation' (resampler du resize retour, defaut bicubic+antialias), 'sharpness' (unsharp mask luminance-only sur la zone regeneree), 'color_match_strength' (0-1, realigne moyenne/ecart-type par canal sur le crop d'origine) et une entree optionnelle 'segs' (SEGS deja calculees en amont, ex: par un masque manuel) qui remplace entierement la detection bbox/sam/segm quand elle est connectee -- 'bbox_detector' devient alors facultatif."

    @staticmethod
    def do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard_opt=None, detailer_hook=None,
                  refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None,
                  cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None, tiled_encode=False, tiled_decode=False,
                  interpolation="bicubic", sharpness=0.5, color_match_strength=1.0):

        core, utils, wildcards, impact_sampling = _impact_modules()
        SEG = core.SEG

        if len(image) > 1:
            raise Exception('[Onyx Detailer] ERROR: image batches are not supported by do_detail.\n'
                            'Please refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        image = image.clone()
        enhanced_alpha_list = []
        enhanced_list = []
        cropped_list = []
        cnet_pil_list = []

        segs = core.segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        if wildcard_opt is not None:
            if wildcard_opt.startswith('[CONCAT]'):
                wildcard_concat_mode = 'concat'
                wildcard_opt = wildcard_opt[8:]
            wmode, wildcard_chooser = wildcards.process_wildcard_for_segs(wildcard_opt)
        else:
            wmode, wildcard_chooser = None, None

        if wmode in ['ASC', 'DSC', 'ASC-SIZE', 'DSC-SIZE']:
            if wmode == 'ASC':
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[0], x.bbox[1]))
            elif wmode == 'DSC':
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[0], x.bbox[1]), reverse=True)
            elif wmode == 'ASC-SIZE':
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            else:
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        else:
            ordered_segs = segs[1]

        if not (isinstance(model, str) and model == "DUMMY") and noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

        for i, seg in enumerate(ordered_segs):
            cropped_image = utils.crop_ndarray4(image.cpu().numpy(), seg.crop_region)
            cropped_image = utils.to_tensor(cropped_image)
            mask = utils.to_tensor(seg.cropped_mask)
            mask = utils.tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                logging.info("Onyx Detailer: segment skip [empty mask]")
                continue

            if noise_mask:
                cropped_mask = seg.cropped_mask
            else:
                cropped_mask = None

            if wildcard_chooser is not None and wmode != "LAB":
                seg_seed, wildcard_item = wildcard_chooser.get(seg)
            elif wildcard_chooser is not None and wmode == "LAB":
                seg_seed, wildcard_item = None, wildcard_chooser.get(seg)
            else:
                seg_seed, wildcard_item = None, None

            seg_seed = seed + i if seg_seed is None else seg_seed

            if not isinstance(positive, str):
                cropped_positive = [
                    [condition, {
                        k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                        for k, v in details.items()
                    }]
                    for condition, details in positive
                ]
            else:
                cropped_positive = positive

            if not isinstance(negative, str):
                cropped_negative = [
                    [condition, {
                        k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                        for k, v in details.items()
                    }]
                    for condition, details in negative
                ]
            else:
                cropped_negative = negative

            if wildcard_item and wildcard_item.strip() == '[SKIP]':
                continue

            if wildcard_item and wildcard_item.strip() == '[STOP]':
                break

            orig_cropped_image = cropped_image.clone()
            if not (isinstance(model, str) and model == "DUMMY"):
                enhanced_image, cnet_pils = enhance_detail_hd(
                    cropped_image, model, clip, vae, guide_size, guide_size_for_bbox, max_size,
                    seg.bbox, seg_seed, steps, cfg, sampler_name, scheduler,
                    cropped_positive, cropped_negative, denoise, cropped_mask, force_inpaint,
                    wildcard_opt=wildcard_item, wildcard_opt_concat_mode=wildcard_concat_mode,
                    detailer_hook=detailer_hook,
                    refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                    refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                    refiner_negative=refiner_negative, control_net_wrapper=seg.control_net_wrapper,
                    cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                    scheduler_func=scheduler_func_opt, vae_tiled_encode=tiled_encode,
                    vae_tiled_decode=tiled_decode, interpolation=interpolation, sharpness=sharpness,
                    color_match_strength=color_match_strength)
            else:
                enhanced_image = cropped_image
                cnet_pils = None

            if cnet_pils is not None:
                cnet_pil_list.extend(cnet_pils)

            if enhanced_image is not None:
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                utils.tensor_paste(image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), mask)
                enhanced_list.append(enhanced_image)

                if detailer_hook is not None:
                    image = detailer_hook.post_paste(image)

            if enhanced_image is not None:
                enhanced_image_alpha = utils.tensor_convert_rgba(enhanced_image)
                new_seg_image = enhanced_image.numpy()

                mask = utils.tensor_resize(mask, *utils.tensor_get_size(enhanced_image))
                utils.tensor_putalpha(enhanced_image_alpha, mask)
                enhanced_alpha_list.append(enhanced_image_alpha)
            else:
                new_seg_image = None

            cropped_list.append(orig_cropped_image)

            new_seg = SEG(new_seg_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
            new_segs.append(new_seg)

        image_tensor = utils.tensor_convert_rgb(image)

        cropped_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

        return image_tensor, cropped_list, enhanced_list, enhanced_alpha_list, cnet_pil_list, (segs[0], new_segs)

    @staticmethod
    def enhance_face(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name, scheduler,
                     positive, negative, denoise, feather, noise_mask, force_inpaint,
                     bbox_threshold, bbox_dilation, bbox_crop_factor,
                     sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                     sam_mask_hint_use_negative, drop_size,
                     bbox_detector=None, segm_detector=None, sam_model_opt=None, wildcard_opt=None, detailer_hook=None,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, cycle=1,
                     inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None, tiled_encode=False, tiled_decode=False,
                     interpolation="bicubic", sharpness=0.5, color_match_strength=1.0, segs_opt=None):

        core, utils, wildcards, impact_sampling = _impact_modules()

        if segs_opt is not None:
            # SEGS deja fournies (ex: par un node de masque externe) : on saute
            # entierement la detection bbox/sam/segm et on utilise ces SEGS
            # telles quelles, comme le fait `DetailerForEach` (Impact Pack) avec
            # son entree `segs` requise.
            segs = core.segs_scale_match(segs_opt, image.shape)
        else:
            if bbox_detector is None:
                raise Exception("[Onyx Detailer] ERROR: connecte soit `bbox_detector`, soit l'entree optionnelle `segs`.")

            # make default prompt as 'face' if empty prompt for CLIPSeg
            bbox_detector.setAux('face')
            segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size, detailer_hook=detailer_hook)
            bbox_detector.setAux(None)

            # bbox + sam combination
            if sam_model_opt is not None:
                sam_mask = core.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                              sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                              sam_mask_hint_use_negative, )
                segs = core.segs_bitwise_and_mask(segs, sam_mask)

            elif segm_detector is not None:
                segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

                if (hasattr(segm_detector, 'override_bbox_by_segm') and segm_detector.override_bbox_by_segm and
                        not (detailer_hook is not None and not hasattr(detailer_hook, 'override_bbox_by_segm'))):
                    segs = segm_segs
                else:
                    segm_mask = core.segs_to_combined_mask(segm_segs)
                    segs = core.segs_bitwise_and_mask(segs, segm_mask)

        if len(segs[1]) > 0:
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
                OnyxDetailer.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                                           sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                           force_inpaint, wildcard_opt, detailer_hook,
                                           refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                           refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                           refiner_negative=refiner_negative,
                                           cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                                           scheduler_func_opt=scheduler_func_opt, tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                                           interpolation=interpolation, sharpness=sharpness,
                                           color_match_strength=color_match_strength)
        else:
            enhanced_img = image
            cropped_enhanced = []
            cropped_enhanced_alpha = []
            cnet_pil_list = []

        # Mask Generator
        mask = core.segs_to_combined_mask(segs)

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [utils.empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [utils.empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [utils.empty_pil_tensor()]

        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

    def doit(self, image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             sam_mask_hint_use_negative, drop_size, bbox_detector=None, wildcard="", cycle=1,
             interpolation="bicubic", sharpness=0.5, color_match_strength=1.0,
             sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0,
             scheduler_func_opt=None, tiled_encode=False, tiled_decode=False, segs=None):

        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []

        # Garde-fou : si le frontend ComfyUI a envoye une entree mal mappee
        # (ex: une string sur le socket `image`), c'est presque toujours un
        # cache de schema perime cote navigateur apres une modif de ce node.
        # On leve une erreur claire au lieu du cryptique 'str'/'unsqueeze'.
        if not torch.is_tensor(image):
            raise Exception(
                "[Onyx Detailer] L'entree `image` n'est pas une image (recu: "
                f"{type(image).__name__}). C'est un cache de schema perime dans le "
                "navigateur apres la mise a jour du node. Correctif : rechargement "
                "force de la page (Ctrl+Shift+R), puis SUPPRIME et re-ajoute ce node. "
                "Si ca persiste, desactive le rendu 'Nodes 2.0' (nouveau moteur de "
                "rendu) dans les Settings ComfyUI, qui re-serialise mal les nodes modifies."
            )

        if len(image) > 1:
            logging.warning("[Onyx Detailer] WARN: this node is not designed for video detailing. If you intend to perform video detailing, use Impact Pack's Detailer For AnimateDiff.")

        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = OnyxDetailer.enhance_face(
                single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for, max_size, seed + i, steps, cfg, sampler_name, scheduler,
                positive, negative, denoise, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector, segm_detector_opt, sam_model_opt, wildcard, detailer_hook,
                cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt,
                tiled_encode=tiled_encode, tiled_decode=tiled_decode, interpolation=interpolation, sharpness=sharpness,
                color_match_strength=color_match_strength, segs_opt=segs)

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None)
        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images


# NOTE: l'enregistrement (NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS) se
# fait dans __init__.py du pack, pas ici.
