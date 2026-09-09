/**
 * OnyxNanoBananaAIO — Automation Face Swap UI + Image/Video Mode Toggle
 * Bloc visuel distinctif avec toggle ON/OFF, verrouillage des widgets,
 * restriction du modèle à NBPro/NB2, et option Breast Refiner.
 * + Toggle IMAGE / VIDEO avec support Veo 3 (Google & Vertex).
 */
import { app } from "../../scripts/app.js";
// ─── Constantes mode vidéo ─────────────────────────────────────────────────
const VIDEO_MODELS         = ["Veo 3.1 Lite", "Veo 3.1", "Veo 3.1 Fast", "Kling 3.0", "Kling 2.6", "Kling 3.0 Motion Control", "Seedance 2.0", "Seedance 2.5", "Omni Flash"];
const IMAGE_MODELS_DEFAULT = ["Nano Banana Pro", "Nano Banana 2", "Seedream 4.5", "Seedream 5 Pro", "GPT Image 2.0"];
const VIDEO_ASPECT_RATIOS  = ["16:9", "9:16"];
const IMAGE_ASPECT_RATIOS  = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"];
const VIDEO_RESOLUTIONS    = ["480p", "720p", "1080p", "4K"];
const KLING30_MODELS       = ["Kling 3.0"];
const IMAGE_RESOLUTIONS    = ["1K", "2K", "4K"];
const SEEDANCE_MODELS      = ["Seedance 2.0", "Seedance 2.5"];
// La 2.5 double la longueur d'un seul passage et monte au 4K sur WaveSpeed.
const SEEDANCE25_MODELS    = ["Seedance 2.5"];
// Omni Flash : GOOGLE uniquement (API Gemini directe), et le seul modèle vidéo
// du pack qui n'expose ni résolution ni durée — les deux sont décidées par le
// modèle, la durée se pilote dans le prompt ("[0-3s] …").
const OMNI_MODELS          = ["Omni Flash"];
// Modèles qui utilisent le widget video_input_mode. Les autres l'ignorent.
const VIDEO_INPUT_MODE_MODELS = [...SEEDANCE_MODELS, ...OMNI_MODELS];
// ─── Widgets verrouillés en mode automation ────────────────────────────────
const LOCKABLE_WIDGETS = [
    "prompt",
    "negative_prompt",
    "use_search",
    "system_instructions",
    "aspect_ratio",
    "temperature",
    "top_p",
    "fal_safety_tolerance",
    "fal_enable_web_search",
];
const AUTOMATION_MODELS = ["Nano Banana Pro", "Nano Banana 2", "Seedream 4.5", "Seedream 5 Pro"];
// ─── Helpers ───────────────────────────────────────────────────────────────
function getWidget(node, name) {
    return node.widgets?.find(w => w.name === name);
}
/** Synchronise un widget Python caché avec une valeur JS. */
function syncNodeWidget(node, name, value) {
    const w = getWidget(node, name);
    if (w) {
        w.value = value;
        if (w.callback) w.callback(value);
    }
}
/** Masque ou affiche un widget standard (computeSize trick). */
function setWidgetVisible(node, widgetName, visible) {
    const w = getWidget(node, widgetName);
    if (!w) return;
    if (visible) {
        if (w._modeOrigComputeSize !== undefined) {
            w.computeSize = w._modeOrigComputeSize;
            delete w._modeOrigComputeSize;
        } else {
            delete w.computeSize;
        }
        w.disabled = false;
        w.hidden   = false;
    } else {
        if (w._modeOrigComputeSize === undefined) {
            w._modeOrigComputeSize = w.computeSize;
        }
        w.computeSize = () => [0, -4];
        w.disabled    = true;
        w.hidden      = true;
    }
    if (w.inputEl) w.inputEl.style.display = visible ? "" : "none";
    if (w.element) w.element.style.display = visible ? "" : "none";
}
function setWidgetLocked(node, widgetName, locked) {
    const w = getWidget(node, widgetName);
    if (!w) return;
    w.disabled = locked;
    const applyStyle = (el) => {
        if (!el || !el.style) return;
        el.style.opacity       = locked ? "0.3" : "1.0";
        el.style.pointerEvents = locked ? "none" : "auto";
        el.style.filter        = locked ? "grayscale(60%)" : "none";
    };
    if (w.inputEl) {
        w.inputEl.disabled = locked;
        applyStyle(w.inputEl);
        // Les widgets multiline (negative_prompt, system_instructions) sont
        // wrappés dans un parent <div> ; on applique aussi l'effet dessus.
        applyStyle(w.inputEl.parentElement);
    }
    applyStyle(w.element);
}
/**
 * Redirige les événements wheel du container DOM vers le canvas LiteGraph,
 * pour que le zoom ComfyUI fonctionne même quand la souris survole un widget DOM.
 */
function forwardWheelToCanvas(el) {
    el.addEventListener("wheel", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const canvas = app.canvas?.canvas;
        if (!canvas) return;
        canvas.dispatchEvent(new WheelEvent("wheel", {
            bubbles: true, cancelable: true, view: window,
            deltaX: e.deltaX, deltaY: e.deltaY, deltaZ: e.deltaZ, deltaMode: e.deltaMode,
            clientX: e.clientX, clientY: e.clientY, screenX: e.screenX, screenY: e.screenY,
            ctrlKey: e.ctrlKey, altKey: e.altKey, shiftKey: e.shiftKey, metaKey: e.metaKey,
        }));
    }, { passive: false });
}
/**
 * Applique / retire l'état automation au node (verrouille les widgets
 * et restreint la liste des modèles à NBPro/NB2).
 */
function applyAutomationState(node, enabled) {
    const applyLocks = () => {
        for (const name of LOCKABLE_WIDGETS) setWidgetLocked(node, name, enabled);
    };
    // Passe immédiate, puis plusieurs passes différées pour rattraper les
    // widgets multiline (negative_prompt, system_instructions) dont l'inputEl
    // n'est parfois rattaché au DOM qu'après un ou deux frames.
    applyLocks();
    requestAnimationFrame(applyLocks);
    setTimeout(applyLocks, 100);
    setTimeout(applyLocks, 400);
    const modelWidget = getWidget(node, "model");
    if (!modelWidget) return;
    if (enabled) {
        if (modelWidget.options?.values) {
            modelWidget.options.values = [...AUTOMATION_MODELS];
        }
        if (!AUTOMATION_MODELS.includes(modelWidget.value)) {
            modelWidget.value = "Nano Banana Pro";
            if (modelWidget.callback) modelWidget.callback("Nano Banana Pro");
        }
    } else {
        // Restaurer la liste selon le mode courant — Python déclare image+vidéo
        // donc on ne peut plus se fier à _fsOrigValues comme source de vérité.
        const correctModels = node._nbIsVideoMode ? VIDEO_MODELS : IMAGE_MODELS_DEFAULT;
        if (modelWidget.options?.values) modelWidget.options.values = [...correctModels];
        // S'assurer que la valeur courante est dans la liste restaurée
        if (!correctModels.includes(modelWidget.value)) {
            modelWidget.value = correctModels[0];
            if (modelWidget.callback) modelWidget.callback(correctModels[0]);
        }
    }
    modelWidget.disabled = false;
    app.graph.setDirtyCanvas(true, true);
}
// ─── Sync aspect_ratio : grisé si Kling (any) + WAVESPEED en mode vidéo ────────
// KLING_MODELS — conservé pour référence future, plus utilisé pour le verrouillage aspect_ratio
// (Kling supporte 16:9 et 9:16, les deux valeurs de VIDEO_ASPECT_RATIOS)
const KLING_MODELS = ["Kling 3.0", "Kling 2.6", "Kling 3.0 Motion Control"];
// GPT Image 2.0 — aspect ratios supportés (filtre automatique du dropdown)
const GPT2_ASPECT_RATIOS = ["auto", "1:1", "4:3", "3:4", "16:9", "9:16"];
const GPT2_NEAREST_AR = {
    "2:3": "9:16", "3:2": "16:9", "4:5": "3:4", "5:4": "4:3", "21:9": "16:9",
};
function setupKlingAspectRatioSync(node) {
    // Kling supporte 16:9 ET 9:16 — le dropdown est déjà restreint à VIDEO_ASPECT_RATIOS
    // en mode vidéo, donc aucun verrouillage supplémentaire n'est nécessaire.
    // La fonction reste en place pour compatibilité avec les appels existants.
    node._nbSyncKlingAspectRatio = () => {};
}
// ─── Contraintes de durée par modèle vidéo ────────────────────────────────────
const DURATION_CONSTRAINTS = {
    "Kling 2.6":                 { min: 5,  max: 10, step: 5, tooltip: "⚠️ Kling 2.6 only supports 5s or 10s. Any other value will be rounded automatically." },
    "Kling 3.0":                 { min: 3,  max: 15, step: 1, tooltip: "" },
    "Kling 3.0 Motion Control":  { min: 3,  max: 15, step: 1, tooltip: "ℹ️ Motion Control: output duration equals the reference video's duration." },
    "Seedance 2.0":              { min: 4,  max: 15, step: 1, tooltip: "⚠️ Seedance 2.0 : image-to-video uniquement." },
    "Seedance 2.5":              { min: 4,  max: 30, step: 1, tooltip: "Seedance 2.5 : jusqu'à 30 s en un seul passage (contre 15 s en 2.0)." },
    "Omni Flash":                { min: 1,  max: 60, step: 1, tooltip: "ℹ️ Omni Flash: duration is not an API parameter — it is ignored. Steer it from the prompt instead (\"After 3 seconds…\", \"[0-3s] … [3-6s] …\")." },
    _veo:                        { min: 4,  max: 8,  step: 1, tooltip: "" },
};
function getDurationConstraint(model) {
    return DURATION_CONSTRAINTS[model] ?? DURATION_CONSTRAINTS._veo;
}

// ─── Sync video_duration selon le modèle ──────────────────────────────────────
function setupKlingDurationSync(node) {
    const modelWidget    = getWidget(node, "model");
    const durationWidget = getWidget(node, "video_duration");
    if (!modelWidget || !durationWidget) return;

    /** Clamp une valeur dans [min, max] en respectant step. */
    const clampDuration = (v, { min, max, step }) => {
        let clamped = Math.max(min, Math.min(max, parseInt(v) || min));
        // Pour step > 1 (ex. Kling 2.6 : step=5) : snap sur le multiple le plus proche
        if (step > 1) {
            const steps = Math.round((clamped - min) / step);
            clamped = min + steps * step;
        }
        return Math.max(min, Math.min(max, clamped));
    };

    /** Recalibre min/max/step et snappe la valeur si besoin. */
    const syncDuration = () => {
        if (!node._nbIsVideoMode) return;
        const c = getDurationConstraint(modelWidget.value);
        if (durationWidget.options) {
            durationWidget.options.min  = c.min;
            durationWidget.options.max  = c.max;
            durationWidget.options.step = c.step;
        }
        durationWidget.tooltip = c.tooltip;
        const clamped = clampDuration(durationWidget.value, c);
        if (durationWidget.value !== clamped) {
            durationWidget.value = clamped;
            if (durationWidget.callback) durationWidget.callback(clamped);
        }
        node.setDirtyCanvas(true);
    };

    node._nbSyncKlingDuration = syncDuration;
    syncDuration();

    // Hooker le changement de modèle → recalibrer la durée
    const origModelCb = modelWidget.callback;
    modelWidget.callback = function (...args) {
        if (origModelCb) origModelCb.apply(this, args);
        syncDuration();
    };
}
// ─── Sync batch_size selon provider et mode vidéo ─────────────────────────
function setupProviderBatchSync(node) {
    const providerWidget  = getWidget(node, "provider");
    const batchSizeWidget = getWidget(node, "batch_size");
    if (!providerWidget || !batchSizeWidget) return;
    const syncBatchState = () => {
        const isVideo   = node._nbIsVideoMode === true;
        // VERTEX image mode : batch autorisé (multi-projets JSON)
        // Seul le mode vidéo verrouille batch_size à 1
        const shouldLock = isVideo;
        batchSizeWidget.disabled = shouldLock;
        if (shouldLock) batchSizeWidget.value = 1;
        node.setDirtyCanvas(true);
    };
    // Expose pour que les fonctions activateVideoMode / activateImageMode
    // puissent déclencher une resynchronisation sans passer par le callback
    // du widget provider.
    node._nbSyncBatchState = syncBatchState;
    syncBatchState();
    const origCallback = providerWidget.callback;
    providerWidget.callback = function (...args) {
        if (origCallback) origCallback.apply(this, args);
        syncBatchState();
        // Le plafond de résolution de Seedance dépend du provider : passer de
        // WaveSpeed à Fal en 2.5 fait tomber le maximum de 4K à 720p. Sans ce
        // rappel, le dropdown resterait sur une valeur que Python re-snapperait
        // silencieusement — l'utilisateur croirait générer en 1080p.
        node._nbSyncVideoResolution?.();
    };
}
// ─── Sync image_size (résolution) selon le modèle vidéo ─────────────────────
// Seedance 2.0 : 480p / 720p / 1080p (pas de 4K)
// Autres modèles vidéo : 720p / 1080p uniquement (pas de 480p)
function setupVideoResolutionSync(node) {
    const modelWidget    = getWidget(node, "model");
    const imgSizeWidget  = getWidget(node, "image_size");
    if (!modelWidget || !imgSizeWidget) return;

    const syncResolution = () => {
        if (!node._nbIsVideoMode) return;
        const model      = modelWidget.value;
        const isSeedance = SEEDANCE_MODELS.includes(model);
        const isOmni     = OMNI_MODELS.includes(model);
        const isKling30  = KLING30_MODELS.includes(model);

        let allowedRes, defaultRes;
        if (isOmni) {
            // Omni Flash ne prend aucun paramètre de résolution. On laisse le
            // dropdown lisible plutôt que de le vider, mais la valeur n'est
            // jamais envoyée — le tooltip plus bas le dit explicitement.
            allowedRes = ["720p", "1080p"];
            defaultRes = "1080p";
        } else if (isSeedance) {
            // Le plafond dépend du COUPLE modèle/provider, pas du modèle seul.
            // WaveSpeed vend du 4K sur les deux versions ; Fal et Kie plafonnent
            // à 720p sur la 2.5. Doit rester aligné sur _SEEDANCE_MAX_RES côté
            // Python — celui-ci re-snappe de toute façon, donc un écart ici ne
            // produit qu'un choix grisé, jamais une requête invalide.
            const providerWidget = getWidget(node, "provider");
            const prov = providerWidget?.value || "WAVESPEED";
            const is25 = SEEDANCE25_MODELS.includes(model);
            if (prov === "WAVESPEED") {
                allowedRes = ["480p", "720p", "1080p", "4K"];
                defaultRes = "720p";
            } else if (is25 && prov === "FAL") {
                allowedRes = ["480p", "720p"];          // Fal : 720p max en 2.5
                defaultRes = "720p";
            } else {
                // Kie en 2.5, et Fal/Kie en 2.0 : jusqu'au 1080p.
                allowedRes = ["480p", "720p", "1080p"];
                defaultRes = "720p";
            }
        } else if (isKling30) {
            allowedRes = ["720p", "1080p", "4K"];
            defaultRes = "1080p";
        } else {
            allowedRes = ["720p", "1080p"];
            defaultRes = "1080p";
        }

        if (imgSizeWidget.options?.values) imgSizeWidget.options.values = [...allowedRes];
        if (!allowedRes.includes(imgSizeWidget.value)) {
            imgSizeWidget.value = defaultRes;
            if (imgSizeWidget.callback) imgSizeWidget.callback(defaultRes);
        }
        // Tooltip sur le widget modèle
        if (modelWidget.options) {
            modelWidget.options.tooltip = isOmni
                ? "ℹ️ Omni Flash: GOOGLE provider only (direct Gemini API). Resolution and duration are not API parameters and are ignored."
                : isSeedance
                    ? "⚠️ Image-to-video uniquement"
                    : "";
        }
        if (isOmni) {
            imgSizeWidget.options = imgSizeWidget.options || {};
            imgSizeWidget.options.tooltip =
                "ℹ️ Ignored by Omni Flash — the model decides its own output size.";
        } else if (imgSizeWidget.options) {
            imgSizeWidget.options.tooltip = "";
        }
        node.setDirtyCanvas(true);
    };

    node._nbSyncVideoResolution = syncResolution;

    // Hooker le changement de modèle → recalibrer la résolution
    const origModelCb = modelWidget.callback;
    modelWidget.callback = function (...args) {
        if (origModelCb) origModelCb.apply(this, args);
        syncResolution();
    };
}
// ─── Sync video_input_mode selon le modèle ────────────────────────────────
// Le widget ne concerne que Seedance et Omni Flash. Sur les autres modèles il
// est verrouillé plutôt que masqué : le masquer décalerait la position des
// widgets, et ComfyUI sérialise les valeurs par position.
function setupVideoInputModeSync(node) {
    const modelWidget = getWidget(node, "model");
    const modeWidget  = getWidget(node, "video_input_mode");
    if (!modelWidget || !modeWidget) return;

    const EDIT_MODE = "Edit (video, Omni)";

    const syncMode = () => {
        const model    = modelWidget.value;
        const relevant = VIDEO_INPUT_MODE_MODELS.includes(model);
        const isOmni   = OMNI_MODELS.includes(model);

        setWidgetLocked(node, "video_input_mode", !(node._nbIsVideoMode && relevant));

        // Le mode Edit n'existe que sur Omni. Le laisser sélectionnable sur
        // Seedance produirait une erreur côté serveur au moment de générer,
        // ce qui coûte un aller-retour pour rien.
        if (modeWidget.options) {
            modeWidget.options.values = isOmni
                ? ["First & Last Frame", "Reference", EDIT_MODE]
                : ["First & Last Frame", "Reference"];
        }
        if (!isOmni && modeWidget.value === EDIT_MODE) {
            modeWidget.value = "First & Last Frame";
            if (modeWidget.callback) modeWidget.callback(modeWidget.value);
        }
        node.setDirtyCanvas(true);
    };

    node._nbSyncVideoInputMode = syncMode;

    const origModelCb = modelWidget.callback;
    modelWidget.callback = function (...args) {
        if (origModelCb) origModelCb.apply(this, args);
        syncMode();
    };
}
// ─── Sync aspect_ratio et gpt2_image_quality selon le modèle ─────────────
function setupGPT2Sync(node) {
    const modelWidget  = getWidget(node, "model");
    const arWidget     = getWidget(node, "aspect_ratio");
    if (!modelWidget || !arWidget) return;

    /** Retourne le ratio le plus proche supporté par GPT2. */
    function nearestGPT2AR(value) {
        if (GPT2_ASPECT_RATIOS.includes(value)) return value;
        return GPT2_NEAREST_AR[value] || "1:1";
    }

    /** Applique ou retire le filtre GPT2 sur le dropdown aspect_ratio. */
    const syncAR = () => {
        const isGPT2 = (modelWidget.value === "GPT Image 2.0");
        if (isGPT2) {
            if (arWidget.options?.values) arWidget.options.values = [...GPT2_ASPECT_RATIOS];
            const snapped = nearestGPT2AR(arWidget.value);
            if (arWidget.value !== snapped) {
                arWidget.value = snapped;
                if (arWidget.callback) arWidget.callback(snapped);
            }
        } else {
            // Restaure la liste complète selon le mode courant
            const fullList = node._nbIsVideoMode ? VIDEO_ASPECT_RATIOS : IMAGE_ASPECT_RATIOS;
            if (arWidget.options?.values) arWidget.options.values = [...fullList];
        }
        node.setDirtyCanvas(true);
    };

    node._nbSyncGPT2AR = syncAR;
    syncAR();

    // Hooker le changement de modèle
    const origCb = modelWidget.callback;
    modelWidget.callback = function (...args) {
        if (origCb) origCb.apply(this, args);
        syncAR();
    };
}
// ─── Factorisation des styles des boutons ON/OFF ──────────────────────────
// Évite de dupliquer 8 lignes de style à chaque toggle (Face Swap, Breast
// Refiner, Low Neck). Chaque bouton a sa propre palette "ON".
const TOGGLE_OFF_CSS = {
    text: "OFF",
    background: "#3a3a3a",
    color: "#888",
    border: "1.5px solid #666",
    boxShadow: "none",
};
/**
 * Applique l'état visuel ON/OFF à un bouton bascule.
 * @param {HTMLButtonElement} btn
 * @param {boolean} on
 * @param {{bg:string, text?:string, border:string, glow:string}} onStyle
 */
function setToggleVisual(btn, on, onStyle) {
    if (on) {
        btn.textContent       = onStyle.text || "ON";
        btn.style.background  = onStyle.bg;
        btn.style.color       = "#fff";
        btn.style.border      = `1.5px solid ${onStyle.border}`;
        btn.style.boxShadow   = `0 0 8px ${onStyle.glow}`;
    } else {
        btn.textContent       = TOGGLE_OFF_CSS.text;
        btn.style.background  = TOGGLE_OFF_CSS.background;
        btn.style.color       = TOGGLE_OFF_CSS.color;
        btn.style.border      = TOGGLE_OFF_CSS.border;
        btn.style.boxShadow   = TOGGLE_OFF_CSS.boxShadow;
    }
}
const FS_ON_STYLE = { bg: "#00c853", border: "#00e676", glow: "rgba(0,200,83,0.5)"   };
const BR_ON_STYLE = { bg: "#c2185b", border: "#f50057", glow: "rgba(245,0,87,0.4)"   };
const LN_ON_STYLE = { bg: "#6a1b9a", border: "#ab47bc", glow: "rgba(171,71,188,0.4)" };
const AM_ON_STYLE = { bg: "#00838f", border: "#00bcd4", glow: "rgba(0,188,212,0.4)"  };
// CSS de base partagé par les 3 boutons ON/OFF (même taille, même rayon)
const TOGGLE_BTN_BASE_CSS = `
    padding: 4px 18px;
    border-radius: 14px;
    border: 1.5px solid #666;
    background: #3a3a3a;
    color: #888;
    font-weight: 700;
    font-size: 11px;
    cursor: pointer;
    min-width: 55px;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
`;
// ─── Image / Video mode toggle ────────────────────────────────────────────
function buildModeToggle(node) {
    // ── Outer container ───────────────────────────────────────────────────
    const container = document.createElement("div");
    container.style.cssText = `
        box-sizing: border-box;
        border: 2px solid #1565c0;
        border-radius: 9px;
        padding: 8px 10px 11px 10px;
        background: linear-gradient(135deg,
            rgba(21,101,192,0.14) 0%,
            rgba(79,195,247,0.06) 100%);
        font-family: 'Segoe UI', Arial, sans-serif;
        width: 100%;
        overflow: hidden;
        transition: border-color 0.25s, background 0.25s;
    `;
    forwardWheelToCanvas(container);
    // ── Titre ─────────────────────────────────────────────────────────────
    const titleEl = document.createElement("div");
    titleEl.textContent = "⚡  GENERATION MODE";
    titleEl.style.cssText = `
        font-weight: 700;
        font-size: 11.5px;
        color: #4fc3f7;
        letter-spacing: 2px;
        text-align: center;
        text-shadow: 0 0 10px rgba(79,195,247,0.45);
        margin-bottom: 7px;
        user-select: none;
        transition: color 0.25s, text-shadow 0.25s;
    `;
    container.appendChild(titleEl);
    // ── Séparateur ────────────────────────────────────────────────────────
    const sep = document.createElement("div");
    sep.style.cssText = `
        height: 1px;
        background: linear-gradient(to right, transparent, #1565c0 40%, #1565c0 60%, transparent);
        opacity: 0.55;
        margin-bottom: 8px;
    `;
    container.appendChild(sep);
    // ── Boutons IMAGE / VIDÉO côte à côte ─────────────────────────────────
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex; gap:8px;";
    function makeChoiceBtn(label, active) {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.style.cssText = `
            flex: 1;
            padding: 6px 4px;
            border-radius: 8px;
            border: 1.5px solid ${active ? "#4fc3f7" : "#444"};
            background: ${active ? "rgba(79,195,247,0.18)" : "#252525"};
            color: ${active ? "#4fc3f7" : "#555"};
            font-weight: 700;
            font-size: 11px;
            cursor: pointer;
            letter-spacing: 1px;
            box-shadow: ${active ? "0 0 7px rgba(79,195,247,0.25)" : "none"};
            transition: all 0.2s ease;
        `;
        return btn;
    }
    const imgBtn = makeChoiceBtn("🖼  IMAGES", true);
    const vidBtn = makeChoiceBtn("🎬  VIDÉO", false);
    btnRow.appendChild(imgBtn);
    btnRow.appendChild(vidBtn);
    container.appendChild(btnRow);
    // ── État ──────────────────────────────────────────────────────────────
    let isVideoMode = false;
    const GREY_IN_VIDEO = [
        "temperature", "top_p", "negative_prompt",
        "fal_safety_tolerance", "fal_enable_web_search",
        "use_search", "system_instructions", "batch_size",
        "gpt2_image_quality",
    ];
    // ── Grise/dégrise le bloc Automation Face Swap ───────────────────────
    function setFaceSwapGreyed(greyed) {
        const fsW = node.widgets?.find(w => w.name === "nb_faceswap_automation");
        const el  = fsW?.element;
        if (!el) return;
        el.style.opacity       = greyed ? "0.22" : "1";
        el.style.pointerEvents = greyed ? "none" : "auto";
        el.style.filter        = greyed ? "grayscale(80%) brightness(0.35)" : "none";
        app.graph.setDirtyCanvas(true, true);
    }
    function updateCombo(name, newValues, newDefault) {
        const w = getWidget(node, name);
        if (!w) return;
        if (w.options?.values) w.options.values = [...newValues];
        if (!newValues.includes(w.value)) {
            w.value = newDefault;
            if (w.callback) w.callback(newDefault);
        }
    }
    // ── Visibilité des widgets vidéo-only ─────────────────────────────────
    // Ces widgets sont visibles SI ET SEULEMENT SI on est en mode VIDEO.
    function updateVideoOnlyWidgets(visible) {
        setWidgetVisible(node, "video_duration",       visible);
        setWidgetVisible(node, "video_generate_audio", visible);
    }
    function updateVideoDurationVisibility() { updateVideoOnlyWidgets(node._nbIsVideoMode === true); }
    node._nbUpdateVideoDuration = updateVideoDurationVisibility;
    // ── Activer le mode IMAGE ─────────────────────────────────────────────
    function activateImageMode(initial = false) {
        isVideoMode          = false;
        node._nbIsVideoMode  = false;
        imgBtn.style.border     = "1.5px solid #4fc3f7";
        imgBtn.style.background = "rgba(79,195,247,0.18)";
        imgBtn.style.color      = "#4fc3f7";
        imgBtn.style.boxShadow  = "0 0 7px rgba(79,195,247,0.25)";
        vidBtn.style.border     = "1.5px solid #444";
        vidBtn.style.background = "#252525";
        vidBtn.style.color      = "#555";
        vidBtn.style.boxShadow  = "none";
        container.style.border     = "2px solid #1565c0";
        container.style.background = "linear-gradient(135deg, rgba(21,101,192,0.14) 0%, rgba(79,195,247,0.06) 100%)";
        titleEl.style.color        = "#4fc3f7";
        titleEl.style.textShadow   = "0 0 10px rgba(79,195,247,0.45)";
        // IMAGE_MODELS_DEFAULT = source de vérité (Python déclare image+vidéo
        // pour la validation serveur, donc options.values n'est plus fiable)
        const modelW = getWidget(node, "model");
        if (modelW) {
            const vals = IMAGE_MODELS_DEFAULT;
            if (modelW.options?.values) modelW.options.values = [...vals];
            if (!vals.includes(modelW.value)) {
                modelW.value = vals[0];
                if (modelW.callback) modelW.callback(vals[0]);
            }
            if (modelW._imgOrigTooltip !== undefined) {
                if (modelW.options) modelW.options.tooltip = modelW._imgOrigTooltip;
                delete modelW._imgOrigTooltip;
            }
        }
        updateCombo("aspect_ratio", IMAGE_ASPECT_RATIOS, "1:1");
        updateCombo("image_size",   IMAGE_RESOLUTIONS,   "2K");
        const imgSizeW = getWidget(node, "image_size");
        if (imgSizeW) imgSizeW.label = "image_size";
        for (const name of GREY_IN_VIDEO) setWidgetLocked(node, name, false);
        // Re-synchronise batch_size et aspect_ratio (Kling) en mode image
        node._nbSyncBatchState?.();
        node._nbSyncKlingAspectRatio?.();
        node._nbSyncVideoInputMode?.();
        updateVideoDurationVisibility();
        setTimeout(() => setFaceSwapGreyed(false), 60);
        if (!initial) {
            syncNodeWidget(node, "video_mode_enabled", false);
            app.graph.setDirtyCanvas(true, true);
        }
    }
    // ── Activer le mode VIDÉO ─────────────────────────────────────────────
    function activateVideoMode() {
        isVideoMode         = true;
        node._nbIsVideoMode = true;
        vidBtn.style.border     = "1.5px solid #ce93d8";
        vidBtn.style.background = "rgba(206,147,216,0.18)";
        vidBtn.style.color      = "#ce93d8";
        vidBtn.style.boxShadow  = "0 0 7px rgba(206,147,216,0.3)";
        imgBtn.style.border     = "1.5px solid #444";
        imgBtn.style.background = "#252525";
        imgBtn.style.color      = "#555";
        imgBtn.style.boxShadow  = "none";
        container.style.border     = "2px solid #6a1b9a";
        container.style.background = "linear-gradient(135deg, rgba(106,27,154,0.14) 0%, rgba(206,147,216,0.06) 100%)";
        titleEl.style.color        = "#ce93d8";
        titleEl.style.textShadow   = "0 0 10px rgba(206,147,216,0.45)";
        // VIDEO_MODELS = source de vérité — pas de sauvegarde _imgOrigValues
        // car options.values contient maintenant image+vidéo côté Python
        const modelW = getWidget(node, "model");
        if (modelW) {
            if (modelW.options?.values) modelW.options.values = [...VIDEO_MODELS];
            if (!VIDEO_MODELS.includes(modelW.value)) {
                modelW.value = VIDEO_MODELS[0];
                if (modelW.callback) modelW.callback(VIDEO_MODELS[0]);
            }
            if (modelW.options?.tooltip) {
                modelW._imgOrigTooltip = modelW.options.tooltip;
                modelW.options.tooltip = "";
            }
        }
        updateCombo("aspect_ratio", VIDEO_ASPECT_RATIOS, "16:9");
        updateCombo("image_size",   VIDEO_RESOLUTIONS,   "1080p");
        const imgSizeW = getWidget(node, "image_size");
        if (imgSizeW) imgSizeW.label = "video_size";
        for (const name of GREY_IN_VIDEO) setWidgetLocked(node, name, true);
        // Force batch_size = 1, synchronise aspect_ratio, durée et résolution selon le modèle
        node._nbSyncBatchState?.();
        node._nbSyncKlingAspectRatio?.();
        node._nbSyncKlingDuration?.();
        node._nbSyncVideoResolution?.();
        node._nbSyncVideoInputMode?.();
        updateVideoDurationVisibility();
        setTimeout(() => setFaceSwapGreyed(true), 60);
        syncNodeWidget(node, "video_mode_enabled", true);
        app.graph.setDirtyCanvas(true, true);
    }
    // ── Listeners ─────────────────────────────────────────────────────────
    imgBtn.addEventListener("click", () => { if (isVideoMode)  activateImageMode(); });
    vidBtn.addEventListener("click", () => { if (!isVideoMode) activateVideoMode(); });
    node._modeToggleAPI = { activateImageMode, activateVideoMode };
    // ── État initial SYNCHRONE (mode image par défaut) ────────────────────
    // Masquer video_duration AVANT d'ajouter le DOM widget évite un
    // redimensionnement visible du node ~250 ms après le démarrage.
    node._nbIsVideoMode = false;
    updateVideoOnlyWidgets(false);
    // ── Ajout DOM widget ──────────────────────────────────────────────────
    const domWidget = node.addDOMWidget(
        "nb_mode_toggle", "mode_toggle_div", container,
        { serialize: false, hideOnZoom: false }
    );
    if (domWidget) domWidget.computeSize = (w) => [w, 95];
    // ── Placer en PREMIER ─────────────────────────────────────────────────
    if (domWidget) {
        const idx = node.widgets.indexOf(domWidget);
        if (idx > 0) { node.widgets.splice(idx, 1); node.widgets.unshift(domWidget); }
        app.graph.setDirtyCanvas(true, true);
    }
    // ── Filtrage initial du dropdown "model" (mode image par défaut) ─────
    // On ne fait QUE filtrer la liste — pas d'appel à activateImageMode()
    // pour éviter de déclencher des callbacks qui perturbent la visibilité
    // des widgets DOM (inputEl créés après le premier rendu).
    setTimeout(() => {
        const modelW = getWidget(node, "model");
        if (!modelW || node._nbIsVideoMode) return;
        if (modelW.options?.values) modelW.options.values = [...IMAGE_MODELS_DEFAULT];
        if (!IMAGE_MODELS_DEFAULT.includes(modelW.value)) modelW.value = IMAGE_MODELS_DEFAULT[0];
    }, 200);
    // ── Sync UI ↔ widget "video_mode_enabled" après chargement workflow ──
    // Au chargement d'un workflow, ComfyUI restaure la valeur du widget
    // caché `video_mode_enabled`, mais la variable locale isVideoMode reste
    // à false. Résultat : l'UI montre IMAGES mais le backend reçoit
    // video_mode_enabled=true → "Modèle vidéo inconnu : Nano Banana Pro".
    // On fixe ça en synchronisant l'UI avec la valeur réelle du widget ~350ms
    // après le onNodeCreated (délai suffisant pour que configure() ait injecté
    // les valeurs sauvegardées).
    setTimeout(() => {
        const videoModeWidget = getWidget(node, "video_mode_enabled");
        const savedIsVideo = videoModeWidget?.value === true;
        if (savedIsVideo && !node._nbIsVideoMode) {
            // Workflow sauvegardé en mode vidéo → bascule l'UI en vidéo
            activateVideoMode();
        } else {
            // Force explicitement le widget à false pour éviter qu'une
            // valeur "truthy" résiduelle (1, "true"…) passe côté Python
            syncNodeWidget(node, "video_mode_enabled", false);
        }
    }, 350);
}
// ─── Automation Face Swap UI ──────────────────────────────────────────────
function buildFaceSwapUI(node) {
    // ── Outer container ───────────────────────────────────────────────────
    const container = document.createElement("div");
    container.style.cssText = `
        box-sizing: border-box;
        border: 2px solid #1497b8;
        border-radius: 9px;
        padding: 9px 10px 16px 10px;
        background: linear-gradient(135deg,
            rgba(20, 151, 184,0.13) 0%,
            rgba(46, 196, 224,0.06) 100%);
        font-family: 'Segoe UI', Arial, sans-serif;
        width: 100%;
        transition: border-color 0.25s, background 0.25s;
    `;
    forwardWheelToCanvas(container);
    // ── Title ─────────────────────────────────────────────────────────────
    const titleEl = document.createElement("div");
    titleEl.textContent = "⚡  AUTOMATION FACE SWAP";
    titleEl.style.cssText = `
        font-weight: 700;
        font-size: 11.5px;
        color: #2ec4e0;
        letter-spacing: 2px;
        text-align: center;
        text-shadow: 0 0 10px rgba(46, 196, 224,0.45);
        margin-bottom: 7px;
        transition: color 0.25s, text-shadow 0.25s;
        user-select: none;
    `;
    container.appendChild(titleEl);
    // ── Separator ─────────────────────────────────────────────────────────
    const sep = document.createElement("div");
    sep.style.cssText = `
        height: 1px;
        background: linear-gradient(to right, transparent, #1497b8 40%, #1497b8 60%, transparent);
        opacity: 0.55;
        margin-bottom: 8px;
    `;
    container.appendChild(sep);
    // ── Info hint ─────────────────────────────────────────────────────────
    const hintEl = document.createElement("div");
    hintEl.textContent = "ℹ️  Image 1 = reference face   ·   Image 2 = face to swap";
    hintEl.style.cssText = `
        font-size: 10px;
        color: #999;
        font-style: italic;
        text-align: center;
        margin-bottom: 8px;
        user-select: none;
    `;
    container.appendChild(hintEl);
    // ── Helper pour construire une ligne "label + bouton ON/OFF" ──────────
    function makeToggleRow(labelText, tooltip, hidden = false) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: ${hidden ? "none" : "flex"};
            align-items: center;
            gap: 8px;
            ${hidden ? "margin-top: 8px; padding-top: 7px; border-top: 1px solid rgba(20, 151, 184,0.3);" : ""}
        `;
        const label = document.createElement("span");
        label.textContent = labelText;
        label.title = tooltip;
        label.style.cssText = `
            font-size: 12px;
            color: #ccc;
            flex: 1;
            user-select: none;
        `;
        const btn = document.createElement("button");
        btn.textContent = "OFF";
        btn.title = tooltip;
        btn.style.cssText = TOGGLE_BTN_BASE_CSS;
        row.appendChild(label);
        row.appendChild(btn);
        return { row, btn };
    }
    // ── Face Swap toggle ──────────────────────────────────────────────────
    const { row: toggleRow, btn: toggleBtn } = makeToggleRow(
        "Face Swap",
        "Activate to run Automation Face Swap.\nImage 1 = reference face · Image 2 = face to swap",
        false
    );
    container.appendChild(toggleRow);
    // ── Breast Refiner row (hidden until automation ON) ──────────────────
    const { row: brRow, btn: brBtn } = makeToggleRow(
        "Breast Refiner",
        "This tool enhances breast shape by defining the upper curve for a more natural look. May trigger NSFW restrictions.",
        true
    );
    container.appendChild(brRow);
    // ── Low Neck row (hidden until automation ON) ────────────────────────
    const { row: lnRow, btn: lnBtn } = makeToggleRow(
        "Low Neck",
        "Add a low neck, might be flagged as sensitive content.",
        true
    );
    container.appendChild(lnRow);
    // ── Amateur Mode row (hidden until automation ON) ─────────────────────
    const { row: amRow, btn: amBtn } = makeToggleRow(
        "Amateur, Low quality input match",
        "Overrides the default face swap prompt for low quality or amateur input photos.",
        true
    );
    container.appendChild(amRow);
    // ── Facial Expression section (hidden until automation ON) ───────────
    const EXPRESSIONS = [
        "Manual", "Neutral", "Sensual", "Playful", "Subtle smile", "Smile",
    ];
    const exprSection = document.createElement("div");
    exprSection.style.cssText = `
        display: none;
        flex-direction: column;
        gap: 6px;
        margin-top: 8px;
        padding-top: 7px;
        border-top: 1px solid rgba(20, 151, 184,0.3);
    `;
    const exprTitleEl = document.createElement("div");
    exprTitleEl.textContent = "Facial Expression";
    exprTitleEl.style.cssText = `
        font-size: 11px;
        font-weight: 600;
        color: #aaa;
        letter-spacing: 0.8px;
        user-select: none;
    `;
    exprSection.appendChild(exprTitleEl);
    const exprGrid = document.createElement("div");
    exprGrid.style.cssText = `
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 4px;
    `;
    const exprBtns = {};
    function selectExpression(value) {
        for (const [v, btn] of Object.entries(exprBtns)) {
            const active = (v === value);
            btn.style.background = active ? "rgba(46, 196, 224,0.35)" : "#2a2a2a";
            btn.style.color      = active ? "#2ec4e0" : "#888";
            btn.style.border     = active ? "1.5px solid #1497b8" : "1.5px solid #444";
            btn.style.fontWeight = active ? "700" : "400";
        }
        syncNodeWidget(node, "face_expression", value);
    }
    for (const value of EXPRESSIONS) {
        const btn = document.createElement("button");
        btn.textContent = value;
        btn.style.cssText = `
            padding: 4px 6px;
            border-radius: 10px;
            border: 1.5px solid #444;
            background: #2a2a2a;
            color: #888;
            font-size: 10.5px;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;
        btn.addEventListener("click", () => selectExpression(value));
        exprBtns[value] = btn;
        exprGrid.appendChild(btn);
    }
    exprSection.appendChild(exprGrid);
    container.appendChild(exprSection);
    selectExpression("Manual");
    // ── Custom Prompt section (hidden until automation ON) ─────────────────
    const promptSection = document.createElement("div");
    promptSection.style.cssText = `
        display: none;
        flex-direction: column;
        gap: 5px;
        margin-top: 8px;
        padding-top: 7px;
        border-top: 1px solid rgba(20, 151, 184,0.3);
    `;
    const promptLabel = document.createElement("div");
    promptLabel.textContent = "✏️  Additional Prompt";
    promptLabel.style.cssText = `
        font-size: 11px;
        font-weight: 600;
        color: #aaa;
        letter-spacing: 0.8px;
        user-select: none;
    `;
    const promptTextarea = document.createElement("textarea");
    promptTextarea.placeholder = "Describe extra details to append to the face swap prompt…";
    promptTextarea.rows = 3;
    promptTextarea.style.cssText = `
        width: 100%;
        box-sizing: border-box;
        background: #1a1a1a;
        border: 1.5px solid #444;
        border-radius: 6px;
        color: #ddd;
        font-size: 11px;
        font-family: 'Segoe UI', Arial, sans-serif;
        padding: 6px 8px;
        resize: vertical;
        outline: none;
        transition: border-color 0.2s;
        min-height: 54px;
    `;
    promptTextarea.addEventListener("focus", () => promptTextarea.style.borderColor = "#1497b8");
    promptTextarea.addEventListener("blur",  () => promptTextarea.style.borderColor = "#444");
    promptTextarea.addEventListener("keydown", e => e.stopPropagation());
    promptTextarea.addEventListener("input", () => {
        syncNodeWidget(node, "face_swap_custom_prompt", promptTextarea.value);
    });
    promptSection.appendChild(promptLabel);
    promptSection.appendChild(promptTextarea);
    container.appendChild(promptSection);
    // ── State ─────────────────────────────────────────────────────────────
    let automationActive    = false;
    let breastRefinerActive = false;
    let lowNeckActive       = false;
    let amateurModeActive   = false;
    const FS_HEIGHT_OFF = 115;
    const FS_HEIGHT_ON  = 400;
    let _blockH = FS_HEIGHT_OFF;
    // ── Face Swap toggle ──────────────────────────────────────────────────
    toggleBtn.addEventListener("click", () => {
        automationActive = !automationActive;
        _blockH          = automationActive ? FS_HEIGHT_ON : FS_HEIGHT_OFF;
        setToggleVisual(toggleBtn, automationActive, FS_ON_STYLE);
        if (automationActive) {
            container.style.border     = "2px solid #00c853";
            container.style.background = "linear-gradient(135deg, rgba(0,200,83,0.13) 0%, rgba(0,230,118,0.06) 100%)";
            titleEl.style.color        = "#00e676";
            titleEl.style.textShadow   = "0 0 10px rgba(0,230,118,0.55)";
            brRow.style.display           = "flex";
            lnRow.style.display           = "flex";
            amRow.style.display           = "flex";
            exprSection.style.display     = "flex";
            promptSection.style.display   = "flex";
        } else {
            container.style.border     = "2px solid #1497b8";
            container.style.background = "linear-gradient(135deg, rgba(20, 151, 184,0.13) 0%, rgba(46, 196, 224,0.06) 100%)";
            titleEl.style.color        = "#2ec4e0";
            titleEl.style.textShadow   = "0 0 10px rgba(46, 196, 224,0.45)";
            brRow.style.display           = "none";
            lnRow.style.display           = "none";
            amRow.style.display           = "none";
            exprSection.style.display     = "none";
            promptSection.style.display   = "none";
            // Reset sub-toggles when automation is switched off
            if (breastRefinerActive) {
                breastRefinerActive = false;
                setToggleVisual(brBtn, false, BR_ON_STYLE);
                syncNodeWidget(node, "breast_refiner_enabled", false);
            }
            if (lowNeckActive) {
                lowNeckActive = false;
                setToggleVisual(lnBtn, false, LN_ON_STYLE);
                syncNodeWidget(node, "low_neck_enabled", false);
            }
            if (amateurModeActive) {
                amateurModeActive = false;
                setToggleVisual(amBtn, false, AM_ON_STYLE);
                syncNodeWidget(node, "amateur_mode_enabled", false);
            }
        }
        applyAutomationState(node, automationActive);
        syncNodeWidget(node, "face_swap_enabled", automationActive);
        // Force la taille immédiatement (pas de latence au premier clic)
        const newSize = node.computeSize();
        node.size[0] = Math.max(node.size[0], newSize[0]);
        node.size[1] = newSize[1];
        app.graph.setDirtyCanvas(true, true);
    });
    // ── Breast Refiner toggle ─────────────────────────────────────────────
    brBtn.addEventListener("click", () => {
        breastRefinerActive = !breastRefinerActive;
        setToggleVisual(brBtn, breastRefinerActive, BR_ON_STYLE);
        syncNodeWidget(node, "breast_refiner_enabled", breastRefinerActive);
        app.graph.setDirtyCanvas(true);
    });
    // ── Low Neck toggle ───────────────────────────────────────────────────
    lnBtn.addEventListener("click", () => {
        lowNeckActive = !lowNeckActive;
        setToggleVisual(lnBtn, lowNeckActive, LN_ON_STYLE);
        syncNodeWidget(node, "low_neck_enabled", lowNeckActive);
        app.graph.setDirtyCanvas(true);
    });
    // ── Amateur Mode toggle ───────────────────────────────────────────────
    amBtn.addEventListener("click", () => {
        amateurModeActive = !amateurModeActive;
        setToggleVisual(amBtn, amateurModeActive, AM_ON_STYLE);
        syncNodeWidget(node, "amateur_mode_enabled", amateurModeActive);
        app.graph.setDirtyCanvas(true);
    });
    // ── Add DOM widget to node ────────────────────────────────────────────
    const domWidget = node.addDOMWidget("nb_faceswap_automation", "automation_div", container, {
        serialize:  false,
        hideOnZoom: false,
    });
    if (domWidget) domWidget.computeSize = (w) => [w, _blockH];
    // ── Restaure l'état sauvegardé dans le workflow ───────────────────────
    // Le DOM démarre à OFF par défaut ; on le réaligne sur les widgets Python
    // pour que l'état visuel corresponde toujours à ce qui sera exécuté.
    const _savedFS = getWidget(node, "face_swap_enabled")?.value === true;
    if (_savedFS) {
        automationActive = true;
        _blockH = FS_HEIGHT_ON;
        if (domWidget) domWidget.computeSize = (w) => [w, _blockH];
        setToggleVisual(toggleBtn, true, FS_ON_STYLE);
        container.style.border     = "2px solid #00c853";
        container.style.background = "linear-gradient(135deg, rgba(0,200,83,0.13) 0%, rgba(0,230,118,0.06) 100%)";
        titleEl.style.color        = "#00e676";
        titleEl.style.textShadow   = "0 0 10px rgba(0,230,118,0.55)";
        brRow.style.display           = "flex";
        lnRow.style.display           = "flex";
        amRow.style.display           = "flex";
        exprSection.style.display     = "flex";
        promptSection.style.display   = "flex";
        applyAutomationState(node, true);
    }
    // Restaure la valeur sauvegardée du custom prompt
    const _savedCustomPrompt = getWidget(node, "face_swap_custom_prompt")?.value;
    if (_savedCustomPrompt) promptTextarea.value = _savedCustomPrompt;
    // Breast Refiner et Low Neck : toujours OFF au démarrage.
    // On réinitialise le widget Python même si le workflow les avait sauvegardés
    // à true, pour éviter qu'ils se réactivent silencieusement entre sessions.
    syncNodeWidget(node, "breast_refiner_enabled", false);
    syncNodeWidget(node, "low_neck_enabled", false);
    syncNodeWidget(node, "amateur_mode_enabled", false);
    // breastRefinerActive, lowNeckActive et amateurModeActive restent false, boutons restent OFF.
    // Masque les widgets Python cachés (IMMÉDIATEMENT, pas en setTimeout, pour
    // éviter que le node soit dessiné grand puis rétréci juste après).
    const HIDDEN_PY_WIDGETS = [
        "face_swap_enabled", "breast_refiner_enabled", "low_neck_enabled",
        "amateur_mode_enabled", "face_expression", "video_mode_enabled", "face_swap_custom_prompt",
    ];
    for (const name of HIDDEN_PY_WIDGETS) {
        const w = getWidget(node, name);
        if (!w) continue;
        w._fsOrigComputeSize = w.computeSize;
        w.computeSize = () => [0, -4];
        w.hidden = true;
        if (w.inputEl) w.inputEl.style.display = "none";
        if (w.element) w.element.style.display = "none";
    }
    app.graph.setDirtyCanvas(true, true);
}
// ─── Preset panel (10 slots, mode-aware, avec champ nom) ──────────────────
function buildPresetUI(node) {
    const NUM_PRESETS = 10;
    const container = document.createElement("div");
    container.style.cssText = `
        box-sizing: border-box;
        border: 2px solid #37474f;
        border-radius: 9px;
        padding: 8px 10px 10px 10px;
        background: linear-gradient(135deg,
            rgba(55,71,79,0.14) 0%,
            rgba(96,125,139,0.06) 100%);
        font-family: 'Segoe UI', Arial, sans-serif;
        width: 100%;
        overflow: hidden;
    `;
    forwardWheelToCanvas(container);
    const titleEl = document.createElement("div");
    titleEl.textContent = "💾  PRESETS";
    titleEl.style.cssText = `
        font-weight: 700; font-size: 11.5px; color: #90a4ae;
        letter-spacing: 2px; text-align: center;
        margin-bottom: 7px; user-select: none;
    `;
    container.appendChild(titleEl);
    const sep = document.createElement("div");
    sep.style.cssText = `
        height: 1px;
        background: linear-gradient(to right, transparent, #37474f 40%, #37474f 60%, transparent);
        opacity: 0.55; margin-bottom: 7px;
    `;
    container.appendChild(sep);
    const list = document.createElement("div");
    list.style.cssText = "display: flex; flex-direction: column; gap: 4px;";
    container.appendChild(list);
    const slotRows   = {};
    const slotInputs = {};
    function markSlotFilled(n, filled, name) {
        const row   = slotRows[n];
        const input = slotInputs[n];
        if (!row) return;
        row.style.border     = filled ? "1px solid rgba(79,195,247,0.3)" : "1px solid #2a2a2a";
        row.style.background = filled ? "rgba(79,195,247,0.04)"          : "#1a1a1a";
        if (input && name !== undefined) input.value = name || "";
    }
    function buildSlot(n) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex; align-items: center; gap: 4px;
            padding: 3px 5px; border-radius: 6px;
            border: 1px solid #2a2a2a; background: #1a1a1a;
            transition: border 0.2s, background 0.2s;
        `;
        slotRows[n] = row;
        const label = document.createElement("span");
        label.textContent = `${n < 10 ? " " : ""}${n}`;
        label.style.cssText = `
            font-size: 9.5px; color: #555; min-width: 14px;
            user-select: none; font-weight: 700; font-variant-numeric: tabular-nums;
        `;
        const nameInput = document.createElement("input");
        nameInput.type = "text";
        nameInput.placeholder = `preset ${n}`;
        nameInput.style.cssText = `
            flex: 1; min-width: 0;
            background: #111; border: 1px solid #333; border-radius: 4px;
            color: #aaa; font-size: 9.5px; padding: 2px 5px;
            outline: none; font-family: 'Segoe UI', Arial, sans-serif;
        `;
        nameInput.addEventListener("focus", () => nameInput.style.borderColor = "#546e7a");
        nameInput.addEventListener("blur",  () => nameInput.style.borderColor = "#333");
        nameInput.addEventListener("keydown", e => e.stopPropagation());
        slotInputs[n] = nameInput;
        const btnBase = `
            padding: 2px 8px; border-radius: 4px; font-size: 9px;
            font-weight: 700; cursor: pointer; white-space: nowrap;
            transition: all 0.15s ease; letter-spacing: 0.4px;
        `;
        const saveBtn = document.createElement("button");
        saveBtn.textContent = "SAVE";
        saveBtn.title = `Sauvegarder dans le slot ${n}`;
        saveBtn.style.cssText = btnBase + "border: 1px solid #444; background: #252525; color: #666;";
        const loadBtn = document.createElement("button");
        loadBtn.textContent = "LOAD";
        loadBtn.title = `Charger le slot ${n}`;
        loadBtn.style.cssText = btnBase + "border: 1px solid #444; background: #252525; color: #666;";
        row.appendChild(label);
        row.appendChild(nameInput);
        row.appendChild(saveBtn);
        row.appendChild(loadBtn);
        list.appendChild(row);
        saveBtn.addEventListener("click", async () => {
            const params = { preset_name: nameInput.value.trim() };
            for (const w of node.widgets || []) {
                if (w.name && w.value !== undefined
                    && w.type !== "button" && w.type !== "dom_widget"
                    && !w.name.startsWith("nb_")) {
                    params[w.name] = w.value;
                }
            }
            try {
                const res = await fetch("/onyx_nb_aio/save_preset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ slot: n, params }),
                });
                const data = await res.json();
                if (data.success) {
                    markSlotFilled(n, true, params.preset_name);
                    saveBtn.style.cssText = btnBase + "border:1px solid #00c853; background:#1b5e20; color:#69f0ae;";
                    setTimeout(() => {
                        saveBtn.style.cssText = btnBase + "border:1px solid #444; background:#252525; color:#666;";
                    }, 900);
                }
            } catch (e) { console.error("[NB Presets] Save error:", e); }
        });
        loadBtn.addEventListener("click", async () => {
            try {
                const res = await fetch("/onyx_nb_aio/load_presets");
                const data = await res.json();
                if (!data.success) return;
                const preset = data.presets?.[String(n)];
                if (!preset) return;
                const isVideo = preset.video_mode_enabled === true;
                if (isVideo && !node._nbIsVideoMode) {
                    node._modeToggleAPI?.activateVideoMode();
                } else if (!isVideo && node._nbIsVideoMode) {
                    node._modeToggleAPI?.activateImageMode();
                }
                if (preset.preset_name !== undefined) {
                    nameInput.value = preset.preset_name || "";
                }
                markSlotFilled(n, true, preset.preset_name);
                const SKIP = new Set(["video_mode_enabled", "preset_name"]);
                for (const [key, value] of Object.entries(preset)) {
                    if (SKIP.has(key)) continue;
                    syncNodeWidget(node, key, value);
                }
                loadBtn.style.cssText = btnBase + "border:1px solid #2962ff; background:#0d47a1; color:#82b1ff;";
                setTimeout(() => {
                    loadBtn.style.cssText = btnBase + "border:1px solid #444; background:#252525; color:#666;";
                }, 900);
                app.graph.setDirtyCanvas(true, true);
            } catch (e) { console.error("[NB Presets] Load error:", e); }
        });
    }
    for (let i = 1; i <= NUM_PRESETS; i++) buildSlot(i);
    const PANEL_H = 368;
    const domWidget = node.addDOMWidget("nb_presets_ui", "presets_div", container, {
        serialize: false, hideOnZoom: false,
    });
    if (domWidget) domWidget.computeSize = (w) => [w, PANEL_H];
    fetch("/onyx_nb_aio/load_presets")
        .then(r => r.json())
        .then(data => {
            if (!data.success) return;
            for (let i = 1; i <= NUM_PRESETS; i++) {
                const p = data.presets?.[String(i)];
                if (p) markSlotFilled(i, true, p.preset_name);
            }
        })
        .catch(() => {});
}
// ─── Prompt widget colour coding ──────────────────────────────────────────
function stylePromptWidgets(node) {
    setTimeout(() => {
        const pos = getWidget(node, "prompt");
        if (pos?.inputEl) {
            pos.inputEl.style.borderLeft   = "3px solid #00c853";
            pos.inputEl.style.paddingLeft  = "6px";
            pos.inputEl.style.borderRadius = "3px";
        }
        const neg = getWidget(node, "negative_prompt");
        if (neg?.inputEl) {
            neg.inputEl.style.borderLeft   = "3px solid #e53935";
            neg.inputEl.style.paddingLeft  = "6px";
            neg.inputEl.style.borderRadius = "3px";
        }
    }, 200);
}
// ─── Extension registration ───────────────────────────────────────────────
app.registerExtension({
    name: "OnyxNanoBananaAIO.FaceSwap",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "OnyxNanoBananaAIO") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            const node = this;
            stylePromptWidgets(node);
            // Mode toggle en premier (synchrone) — place le widget tout en haut
            buildModeToggle(node);
            setTimeout(() => setupKlingAspectRatioSync(node),   500);
            setTimeout(() => setupKlingDurationSync(node),     500);
            setTimeout(() => setupProviderBatchSync(node),     500);
            setTimeout(() => setupGPT2Sync(node),              500);
            setTimeout(() => setupVideoResolutionSync(node),   500);
            setTimeout(() => setupVideoInputModeSync(node),    500);
            setTimeout(() => buildFaceSwapUI(node),          400);
            setTimeout(() => buildPresetUI(node),            450);
            // Nettoyage tardif des widgets preset hérités (autre JS)
            setTimeout(() => {
                if (!node.widgets) return;
                const KEEP = new Set([
                    "nb_faceswap_automation", "nb_presets_ui", "nb_mode_toggle",
                    "face_expression", "face_swap_enabled", "breast_refiner_enabled",
                    "low_neck_enabled", "amateur_mode_enabled", "video_mode_enabled",
                ]);
                const isLegacy = (w) => {
                    if (!w.name) return false;
                    if (KEEP.has(w.name)) return false;
                    if (w.name.startsWith("nb_")) return false;
                    const n = w.name.toLowerCase();
                    return (
                        w.type === "button" ||
                        n.includes("preset") ||
                        n.includes("slot")
                    );
                };
                for (let i = node.widgets.length - 1; i >= 0; i--) {
                    const w = node.widgets[i];
                    if (!isLegacy(w)) continue;
                    if (w.inputEl) w.inputEl.style.display = "none";
                    if (w.element) w.element.style.display = "none";
                    node.widgets.splice(i, 1);
                }
                // Force-hide des widgets Python automation (canvas widgets LiteGraph)
                for (const name of ["face_expression", "face_swap_enabled",
                                    "breast_refiner_enabled", "low_neck_enabled",
                                    "amateur_mode_enabled", "video_mode_enabled", "face_swap_custom_prompt"]) {
                    const w = getWidget(node, name);
                    if (!w) continue;
                    w.hidden      = true;
                    w.computeSize = () => [0, -4];
                    if (w.inputEl) { try { w.inputEl.remove(); } catch (_) {} w.inputEl = null; }
                    if (w.element) { try { w.element.remove(); } catch (_) {} w.element = null; }
                }
                app.graph.setDirtyCanvas(true, true);
            }, 1200);
            // ── Restauration de la taille sauvegardée (workflow.json) ───────────
            // Les setTimeouts ci-dessus ajoutent des DOM widgets et appellent
            // setDirtyCanvas, ce qui peut écraser la taille restaurée par
            // configure(). On ré-applique la taille sauvegardée après le dernier
            // setTimeout (1200ms) avec une petite marge.
            setTimeout(() => {
                if (node._nbSavedSize && Array.isArray(node._nbSavedSize)) {
                    node.size[0] = node._nbSavedSize[0];
                    node.size[1] = node._nbSavedSize[1];
                    app.graph.setDirtyCanvas(true, true);
                } else {
                    // Nouveau node (jamais sauvegardé) — applique une taille
                    // par défaut généreuse pour que les prompt boxes soient
                    // utilisables sans avoir à redimensionner manuellement.
                    const DEFAULT_W = 520;
                    const DEFAULT_H = 900;
                    if (node.size[0] < DEFAULT_W) node.size[0] = DEFAULT_W;
                    if (node.size[1] < DEFAULT_H) node.size[1] = DEFAULT_H;
                    app.graph.setDirtyCanvas(true, true);
                }
            }, 1300);
        };
        // ── Hook onConfigure : capture la taille sauvée dans le workflow ─────
        // configure(info) est appelé par LiteGraph lors de la restauration d'un
        // workflow. info.size contient la taille que l'utilisateur avait à la
        // session précédente. On la stocke pour la ré-appliquer après que les
        // DOM widgets aient fini leur setup.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            try {
                if (info && Array.isArray(info.size) && info.size.length >= 2) {
                    this._nbSavedSize = [info.size[0], info.size[1]];
                }
            } catch (_) { /* noop */ }
            return r;
        };
    },
});