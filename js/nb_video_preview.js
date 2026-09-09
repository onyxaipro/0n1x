/**
 * nb_video_preview.js
 * Injecte un vrai player <video> HTML dans les nodes Onyx qui retournent
 * des vidéos MP4.
 *
 * Piège du nouveau ComfyUI (Vue) : la `computeSize` d'un DOM widget est lue
 * UNE FOIS au mount et cachée. De plus, le wrapper `.dom-widget` reçoit un
 * inline style `height: 0px` que Vue remet après chaque render.
 *
 * Stratégie :
 *  1. `computeSize` retourne TOUJOURS une hauteur réelle (jamais 0).
 *  2. On épingle la hauteur du container ET du wrapper .dom-widget en inline.
 *  3. Un MutationObserver sur le wrapper ré-applique la hauteur si Vue la
 *     remet à 0.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const VIDEO_PREVIEW_NODES = [
    "OnyxPreviewImageWithoutMetadata",
    "SaveImageWithoutMetadata",
];

// Hauteur minimale au repos (tant qu'aucune vidéo n'est chargée).
// On ne peut pas mettre 0 sinon Vue cache la valeur et le wrapper reste
// verrouillé à 0 même après avoir reçu une vidéo.
const IDLE_HEIGHT = 1;

/** Redirige les événements wheel vers le canvas LiteGraph (zoom graph). */
function forwardWheelToCanvas(el) {
    el.addEventListener("wheel", (e) => {
        const canvas = app.canvas?.canvas;
        if (!canvas) return;
        e.preventDefault();
        e.stopPropagation();
        canvas.dispatchEvent(new WheelEvent("wheel", {
            bubbles: true, cancelable: true, view: window,
            deltaX: e.deltaX, deltaY: e.deltaY, deltaZ: e.deltaZ, deltaMode: e.deltaMode,
            clientX: e.clientX, clientY: e.clientY, screenX: e.screenX, screenY: e.screenY,
            ctrlKey: e.ctrlKey, altKey: e.altKey, shiftKey: e.shiftKey, metaKey: e.metaKey,
        }));
    }, { passive: false });
}

/** Remonte jusqu'au premier ancêtre qui matche un sélecteur. */
function findAncestor(el, selector) {
    let cur = el?.parentElement;
    while (cur) {
        if (cur.matches?.(selector)) return cur;
        cur = cur.parentElement;
    }
    return null;
}

app.registerExtension({
    name: "onyx.VideoPreview",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!VIDEO_PREVIEW_NODES.includes(nodeData.name)) return;

        // ─── Création du player à la création du node ────────────────
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            const node = this;

            node._onyxIdleH    = IDLE_HEIGHT;
            node._onyxCurrentH = IDLE_HEIGHT;

            // ── Container ───────────────────────────────────────────
            // Fond transparent + 0 bordure tant qu'aucune vidéo n'est
            // chargée : rien ne s'affiche à l'écran.
            const container = document.createElement("div");
            container.style.cssText = `
                box-sizing: border-box;
                width: 100%;
                background: transparent;
                border-radius: 0;
                overflow: hidden;
                position: relative;
                display: block;
            `;
            container.style.height = node._onyxCurrentH + "px";
            forwardWheelToCanvas(container);

            // ── Élément video ───────────────────────────────────────
            const video = document.createElement("video");
            video.controls    = true;
            video.loop        = true;
            video.autoplay    = true;
            video.muted       = true;
            video.playsInline = true;
            video.preload     = "auto";
            video.style.cssText = `
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: none;
                background: #000;
                border-radius: 4px;
            `;
            container.appendChild(video);

            node._onyxVideoEl        = video;
            node._onyxVideoContainer = container;
            node._onyxVideoAspect    = 16 / 9;

            // ── DOM widget ComfyUI ──────────────────────────────────
            const domWidget = node.addDOMWidget(
                "onyx_video_preview",
                "onyx_video",
                container,
                { serialize: false, hideOnZoom: false }
            );
            node._onyxVideoWidget = domWidget;

            // computeSize retourne TOUJOURS la hauteur courante (jamais 0).
            if (domWidget) {
                domWidget.computeSize = function (width) {
                    return [width, node._onyxCurrentH];
                };
            }

            // ── Surveillance du wrapper .dom-widget ─────────────────
            // Vue peut remettre height:0 après un render — on ré-écrit.
            const pinWrapper = () => {
                if (node._onyxWrapperObs) return;
                const wrapper = findAncestor(container, ".dom-widget")
                             || container.parentElement;
                if (!wrapper) return;
                node._onyxWrapperEl = wrapper;
                node._onyxApplyWrapperHeight();

                // Debounce l'observer via rAF : Vue peut émettre plusieurs
                // mutations de style par frame, on n'en traite qu'une seule.
                node._onyxWrapperObs = new MutationObserver(() => {
                    if (node._onyxObsRaf) return;
                    node._onyxObsRaf = requestAnimationFrame(() => {
                        node._onyxObsRaf = null;
                        node._onyxApplyWrapperHeight();
                    });
                });
                node._onyxWrapperObs.observe(wrapper, {
                    attributes: true,
                    attributeFilter: ["style"],
                });
            };
            // Un seul essai immédiat + un fallback si le wrapper n'est
            // pas encore monté. Plus de cascade 0/100/500ms.
            node._onyxPinTimers = [];
            node._onyxPinTimers.push(setTimeout(pinWrapper, 0));
            node._onyxPinTimers.push(setTimeout(pinWrapper, 200));

            // ── Événements vidéo ────────────────────────────────────
            video.addEventListener("loadedmetadata", () => {
                if (!video.videoWidth || !video.videoHeight) return;
                node._onyxVideoAspect = video.videoWidth / video.videoHeight;
                node._onyxResizeToVideo();
            });
            video.addEventListener("error", () => {
                console.warn("[Onyx Video Preview] Erreur chargement :",
                             video.src, "code:", video.error?.code);
            });

            // ── Hook du redimensionnement manuel du node ───────────
            // Quand l'utilisateur agrandit/rapetisse le node à la souris,
            // on recalcule la hauteur du widget pour que la vidéo suive.
            // Debounced via requestAnimationFrame : coalesce les appels
            // multiples par frame pour éviter les pics de layout.
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                if (origOnResize) origOnResize.apply(this, arguments);
                if (!node._onyxVideoHasSource) return;
                if (node._onyxResizeRaf) return;
                node._onyxResizeRaf = requestAnimationFrame(() => {
                    node._onyxResizeRaf = null;
                    const aspect = node._onyxVideoAspect || (16 / 9);
                    const w = Math.max(80, node.size?.[0] || 360);
                    node._onyxCurrentH = Math.max(40,
                        Math.round(w / Math.max(0.1, aspect)));
                    node._onyxApplyWrapperHeight();
                });
            };

            // ── Cleanup à la suppression du node ───────────────────
            // CRITIQUE : sans ça, le MutationObserver reste actif sur un
            // DOM orphelin et l'élément <video> continue à décoder en
            // mémoire. À l'échelle de quelques reload de workflow, ça
            // cause le freeze + duplication du front ComfyUI.
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                try {
                    if (node._onyxWrapperObs) {
                        node._onyxWrapperObs.disconnect();
                        node._onyxWrapperObs = null;
                    }
                } catch (_) {}
                try {
                    if (Array.isArray(node._onyxPinTimers)) {
                        for (const t of node._onyxPinTimers) clearTimeout(t);
                        node._onyxPinTimers = null;
                    }
                    if (node._onyxResizeRaf) {
                        cancelAnimationFrame(node._onyxResizeRaf);
                        node._onyxResizeRaf = null;
                    }
                    if (node._onyxObsRaf) {
                        cancelAnimationFrame(node._onyxObsRaf);
                        node._onyxObsRaf = null;
                    }
                    if (node._onyxResizeReapply) {
                        clearTimeout(node._onyxResizeReapply);
                        node._onyxResizeReapply = null;
                    }
                } catch (_) {}
                try {
                    const v = node._onyxVideoEl;
                    if (v) {
                        v.pause();
                        v.removeAttribute("src");
                        v.load();
                        v.onloadedmetadata = null;
                        v.onerror = null;
                    }
                } catch (_) {}
                node._onyxVideoEl        = null;
                node._onyxVideoContainer = null;
                node._onyxWrapperEl      = null;
                node._onyxVideoWidget    = null;
                node._onyxVideoHasSource = false;
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };
        };

        // ─── Ré-applique la hauteur sur le wrapper .dom-widget ──────────
        nodeType.prototype._onyxApplyWrapperHeight = function () {
            const node = this;
            const h = node._onyxCurrentH || node._onyxIdleH || IDLE_HEIGHT;
            const wrapper = node._onyxWrapperEl;
            const pxH = h + "px";
            if (wrapper) {
                // Garde séparé sur chaque propriété pour éviter les
                // mutations DOM inutiles qui re-déclencheraient l'observer.
                if (wrapper.style.height !== pxH) {
                    wrapper.style.setProperty("height", pxH, "important");
                }
                if (wrapper.style.minHeight !== pxH) {
                    wrapper.style.setProperty("min-height", pxH, "important");
                }
            }
            if (node._onyxVideoContainer
                && node._onyxVideoContainer.style.height !== pxH) {
                node._onyxVideoContainer.style.height = pxH;
            }
        };

        // ─── Resize quand une vidéo est chargée ─────────────────────
        // Calibre une taille "correcte" en respectant l'aspect, mais
        // l'utilisateur peut ensuite agrandir/rapetisser librement à la
        // souris (voir node.onResize).
        nodeType.prototype._onyxResizeToVideo = function () {
            const node = this;
            // Garde anti-récursion : loadedmetadata peut refirer après un
            // re-src, et setSize ci-dessous ré-émettrait onResize.
            if (node._onyxResizing) return;
            node._onyxResizing = true;
            try {
                const aspect = node._onyxVideoAspect || (16 / 9);
                const w = Math.max(360, node.size?.[0] || 360);
                // Hauteur initiale = largeur / ratio, avec un plancher à 200px
                // et un plafond dur pour éviter qu'un mauvais aspect ratio
                // renvoyé par le <video> ne crée un node géant qui couvre
                // tout le canvas (cause de "la fenêtre part en vrille").
                const h = Math.max(
                    200,
                    Math.min(4000, Math.round(w / Math.max(0.1, aspect)))
                );
                node._onyxCurrentH = h;

                node._onyxApplyWrapperHeight();

                // Force le re-layout du node à une taille qui affiche tout
                const total = node.computeSize();
                const newW = Math.max(node.size?.[0] || 0, total[0]);
                const newH = Math.max(node.size?.[1] || 0, total[1]);
                if (typeof node.setSize === "function") {
                    node.setSize([newW, newH]);
                } else {
                    node.size[0] = newW;
                    node.size[1] = newH;
                }
                app.graph?.change?.();
                app.graph?.setDirtyCanvas?.(true, true);

                // Re-force après le prochain tick au cas où Vue ait touché.
                // Tracké pour cleanup dans onRemoved.
                if (node._onyxResizeReapply) {
                    clearTimeout(node._onyxResizeReapply);
                }
                node._onyxResizeReapply = setTimeout(() => {
                    node._onyxResizeReapply = null;
                    node._onyxApplyWrapperHeight();
                }, 100);
            } finally {
                node._onyxResizing = false;
            }
        };

        // ─── onExecuted : affiche vidéo ou masque ───────────────────────
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            if (origOnExecuted) origOnExecuted.call(this, output);

            const node = this;
            const gifs = output?.gifs ?? [];
            const videoGif = gifs.find(
                (g) => g.format?.startsWith("video/") ||
                       (g.filename && /\.(mp4|webm|mov|m4v)$/i.test(g.filename))
            );

            if (!videoGif) {
                node._onyxHideVideo();
                return;
            }

            node.imgs = null;
            node.imageIndex = 0;

            const url = api.apiURL(
                `/view?filename=${encodeURIComponent(videoGif.filename)}` +
                `&type=${encodeURIComponent(videoGif.type ?? "temp")}` +
                `&subfolder=${encodeURIComponent(videoGif.subfolder ?? "")}` +
                `&t=${Date.now()}`
            );
            node._onyxShowVideo(url);
        };

        // ─── Affiche la vidéo ───────────────────────────────────────────
        nodeType.prototype._onyxShowVideo = function (url) {
            const node = this;
            if (!node._onyxVideoEl) return;
            node._onyxVideoHasSource = true;
            node._onyxVideoEl.style.display = "block";
            node._onyxVideoEl.src = url;
            node._onyxVideoEl.load();
            node._onyxVideoEl.play().catch(() => {});
            // loadedmetadata déclenchera _onyxResizeToVideo
        };

        // ─── Cache la vidéo (retour à l'état idle, invisible) ──────────
        nodeType.prototype._onyxHideVideo = function () {
            const node = this;
            if (!node._onyxVideoEl) return;
            try { node._onyxVideoEl.pause(); } catch (_) {}
            node._onyxVideoEl.removeAttribute("src");
            node._onyxVideoEl.load();
            node._onyxVideoEl.style.display = "none";
            node._onyxVideoHasSource = false;
            node._onyxCurrentH = node._onyxIdleH;
            node._onyxApplyWrapperHeight();
            app.graph?.setDirtyCanvas?.(true, true);
        };
    },
});
