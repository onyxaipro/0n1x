/**
 * Onyx Image and Video Batch Loader — Frontend Widget
 * Large-icon grid, drag & drop upload, sequential Queue All.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


// ── Queue All en paquets ────────────────────────────────────────────────────
//
// Le frontend ComfyUI plafonne le nombre de taches ajoutees en un seul clic
// (100 par defaut en local). Un queuePrompt(0, 420) est donc silencieusement
// tronque : rien n'echoue, il manque simplement des executions, et on ne s'en
// apercoit qu'a la fin d'un lot.
//
// D'ou l'envoi par paquets, en attendant que la file redescende entre chacun.
// L'attente sur la file plutot qu'un delai fixe : un rendu peut prendre une
// seconde ou vingt minutes, et un sleep arbitraire serait faux dans les deux
// sens.
const QUEUE_CHUNK = 50;
const QUEUE_LOW_WATER = 8;      // on renvoie quand il reste moins que ca
const QUEUE_POLL_MS = 1500;

async function pendingCount() {
    try {
        const q = await (await api.fetchApi("/queue")).json();
        return (q.queue_running?.length || 0) + (q.queue_pending?.length || 0);
    } catch (e) {
        // Sans lecture de la file on ne peut plus reguler : on renvoie une
        // valeur haute pour que l'appelant attende plutot que d'inonder.
        return QUEUE_CHUNK;
    }
}

async function waitUntil(predicate) {
    while (!(await predicate())) {
        await new Promise(r => setTimeout(r, QUEUE_POLL_MS));
    }
}

async function queueInChunks(total, chunkSize, onProgress) {
    let sent = 0;
    while (sent < total) {
        const batch = Math.min(chunkSize, total - sent);
        await app.queuePrompt(0, batch);
        sent += batch;
        onProgress?.(sent, total);
        if (sent >= total) break;
        await waitUntil(async () => (await pendingCount()) <= QUEUE_LOW_WATER);
    }
    return sent;
}

/**
 * Consume mode: submit a batch, wait for the queue to EMPTY, then submit again.
 *
 * The full drain is not caution, it is a requirement. Every queued prompt
 * carries its own snapshot of batch_data taken at submission time, and each run
 * removes one entry from the live list. Submitting the next batch before the
 * current one has finished would build it from a list that is still being
 * rewritten — and the same photo would be sent twice.
 *
 * The loop reads getRemaining() afresh each turn rather than counting down from
 * a total, because the list is the authority: an item that failed to load stays
 * in it, and the loop must come back to it rather than skip it.
 */
async function queueConsuming(getRemaining, chunkSize, onProgress) {
    let sent = 0, guard = 0;
    while (getRemaining() > 0 && guard < 10000) {
        const batch = Math.min(chunkSize, getRemaining());
        const before = getRemaining();
        await app.queuePrompt(0, batch);
        sent += batch;
        onProgress?.(sent, before);
        await waitUntil(async () => (await pendingCount()) === 0);
        if (getRemaining() >= before) {
            // Rien n'a ete consomme : consume_on_load est off, ou le
            // chargement a echoue. Continuer bouclerait sur le meme item.
            console.warn("[Onyx Batch] nothing was consumed — stopping to avoid a loop. "
                       + "Is consume_on_load enabled on the node?");
            break;
        }
        guard++;
    }
    return sent;
}

// ── Theme colours (matching Onyx fire theme) ────────────────────────────
const T = {
    bg:        "#071a21",
    panel:     "#04202a",
    border:    "#082a33",
    borderHot: "#22b8d4",
    accent:    "#22b8d4",
    accentDim: "rgba(34, 184, 212,0.25)",
    text:      "#5fd8ef",
    textDim:   "#1a8aa3",
    btnBg:     "#082a33",
    btnHover:  "#0a3c4a",
    dangerBg:  "#04222b",
    dangerClr: "#1aa6c4",
    successBg: "#0a2a0a",
    successClr:"#40c050",
    cardActive:"#082a33",
    cardBorder:"#22b8d4",
};

// ── Helpers ─────────────────────────────────────────────────────────────────
function css(el, styles) {
    Object.assign(el.style, styles);
}

function btn(label, bgCol, textCol, title = "") {
    const b = document.createElement("button");
    b.innerHTML = label;
    b.title     = title;
    css(b, {
        background:   bgCol,
        color:        textCol,
        border:       `1px solid ${textCol}55`,
        borderRadius: "5px",
        padding:      "5px 11px",
        fontSize:     "12px",
        fontWeight:   "700",
        cursor:       "pointer",
        whiteSpace:   "nowrap",
        transition:   "background 0.15s, transform 0.1s",
        fontFamily:   "inherit",
        letterSpacing:"0.3px",
    });
    b.addEventListener("mouseenter", () => css(b, { background: T.btnHover, transform: "scale(1.03)" }));
    b.addEventListener("mouseleave", () => css(b, { background: bgCol,     transform: "scale(1)" }));
    return b;
}

// ── Main setup ───────────────────────────────────────────────────────────────
function setupBatchLoader(node) {
    /** Live state: {images: [...], order: [...]} */
    let imageData    = { images: [], order: [] };
    let currentIndex = -1;
    let isUploading  = false;

    // ── Find / hide the batch_data STRING widget ──────────────────────────
    // ComfyUI creates a textarea widget for our "batch_data" required STRING.
    // We make it invisible (0-height) and manage its value ourselves.
    let batchWidget = null;
    const patchBatchWidget = () => {
        batchWidget = node.widgets?.find(w => w.name === "batch_data");
        if (batchWidget) {
            // Zero out its height so it takes no vertical space
            batchWidget.computeSize = () => [0, -4];
            // Prevent canvas-drawn widget from rendering anything
            batchWidget.draw = () => {};
            // Hide DOM element if it exists (multiline textarea)
            if (batchWidget.element) {
                batchWidget.element.style.cssText =
                    "display:none!important;height:0!important;width:0!important;" +
                    "overflow:hidden!important;position:absolute!important;opacity:0!important;pointer-events:none!important;";
            }
        }
    };
    // Try immediately; also retry after a tick in case widgets aren't ready yet.
    patchBatchWidget();
    setTimeout(patchBatchWidget, 50);
    setTimeout(patchBatchWidget, 300);

    const syncBatchData = () => {
        if (batchWidget) batchWidget.value = JSON.stringify(imageData);
        app.graph.setDirtyCanvas(true, true);
    };

    // ── Root container ────────────────────────────────────────────────────
    const root = document.createElement("div");
    css(root, {
        display:       "flex",
        flexDirection: "column",
        width:         "100%",
        height:        "100%",
        background:    T.bg,
        borderRadius:  "6px",
        overflow:      "hidden",
        fontFamily:    "'Segoe UI', system-ui, sans-serif",
        boxSizing:     "border-box",
        userSelect:    "none",
    });

    // ── Toolbar ───────────────────────────────────────────────────────────
    const toolbar = document.createElement("div");
    css(toolbar, {
        display:      "flex",
        alignItems:   "center",
        gap:          "7px",
        padding:      "7px 9px",
        background:   T.panel,
        borderBottom: `1px solid ${T.border}`,
        flexShrink:   "0",
    });

    // File input (hidden)
    const fileInput = document.createElement("input");
    fileInput.type     = "file";
    fileInput.multiple = true;
    fileInput.accept   = "image/*,video/*";
    css(fileInput, { display: "none" });
    root.appendChild(fileInput);

    const uploadBtn = btn("📁 Upload", T.btnBg, T.accent, "Add images to the batch");
    const queueBtn  = btn("▶ Queue All (0)", T.successBg, T.successClr, "Queue one execution per image");
    const clearBtn  = btn("✕ Clear", T.dangerBg, T.dangerClr, "Remove all images");
    css(clearBtn, { marginLeft: "auto" });
    queueBtn.disabled = true;

    toolbar.append(uploadBtn, queueBtn, clearBtn);

    // ── Drop zone (shown when list is empty) ──────────────────────────────
    const dropZone = document.createElement("div");
    css(dropZone, {
        display:         "flex",
        flexDirection:   "column",
        alignItems:      "center",
        justifyContent:  "center",
        flex:            "1",
        margin:          "10px",
        border:          `2px dashed ${T.border}`,
        borderRadius:    "8px",
        cursor:          "pointer",
        transition:      "border-color 0.2s, background 0.2s",
        padding:         "24px",
        gap:             "10px",
    });
    dropZone.innerHTML = `
        <div style="font-size:44px;line-height:1">🖼️</div>
        <div style="color:${T.accent};font-size:14px;font-weight:700">Drop images here</div>
        <div style="color:${T.textDim};font-size:11px">or click <b style="color:${T.text}">Upload</b> above</div>
    `;
    dropZone.addEventListener("click", () => fileInput.click());

    // ── Grid wrapper ───────────────────────────────────────────────────────
    const gridWrap = document.createElement("div");
    css(gridWrap, {
        flex:        "1",
        overflowY:   "auto",
        overflowX:   "hidden",
        padding:     "8px",
        scrollbarWidth: "thin",
        scrollbarColor: `${T.border} ${T.bg}`,
    });

    const grid = document.createElement("div");
    css(grid, {
        display:             "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap:                 "8px",
    });
    gridWrap.appendChild(grid);

    // ── Status bar ─────────────────────────────────────────────────────────
    const status = document.createElement("div");
    css(status, {
        padding:     "4px 10px",
        background:  T.panel,
        borderTop:   `1px solid ${T.border}`,
        fontSize:    "11px",
        color:       T.textDim,
        flexShrink:  "0",
    });
    status.textContent = "No images loaded";

    // ── Assemble ───────────────────────────────────────────────────────────
    root.append(toolbar, dropZone, status);

    // ── Drag-and-drop wiring ───────────────────────────────────────────────
    function wireDropEvents(target) {
        target.addEventListener("dragover", e => {
            e.preventDefault(); e.stopPropagation();
            css(root, { outline: `2px solid ${T.accent}` });
            css(dropZone, { borderColor: T.accent, background: T.accentDim });
        });
        target.addEventListener("dragleave", e => {
            css(root, { outline: "none" });
            css(dropZone, { borderColor: T.border, background: "transparent" });
        });
        target.addEventListener("drop", e => {
            e.preventDefault(); e.stopPropagation();
            css(root, { outline: "none" });
            css(dropZone, { borderColor: T.border, background: "transparent" });
            // Certains navigateurs laissent f.type vide sur un glisser-deposer
            // depuis l'explorateur Windows ; l'extension sert alors de repli,
            // sinon le fichier est rejete sans aucun message.
            const VIDEO_RE = /\.(mp4|mov|webm|mkv|avi|m4v|mpe?g|wmv|flv)$/i;
            const files = [...e.dataTransfer.files].filter(
                f => f.type.startsWith("image/") || f.type.startsWith("video/") || VIDEO_RE.test(f.name)
            );
            if (files.length) handleFiles(files);
        });
    }
    wireDropEvents(root);
    wireDropEvents(dropZone);

    // ── Button handlers ────────────────────────────────────────────────────
    uploadBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", e => {
        handleFiles([...e.target.files]);
        fileInput.value = "";
    });

    clearBtn.addEventListener("click", () => {
        if (!imageData.order.length) return;
        if (!confirm(`Remove all ${imageData.order.length} image(s) from the batch?`)) return;
        // Fire-and-forget deletes
        for (const id of [...imageData.order]) {
            fetch(`/onyx/batch_delete/${id}`, { method: "DELETE" }).catch(() => {});
        }
        imageData    = { images: [], order: [] };
        currentIndex = -1;
        refresh();
        syncBatchData();
    });

    const widgetValue = (name, fallback) => {
        const w = node.widgets?.find(x => x.name === name);
        return w === undefined ? fallback : w.value;
    };

    queueBtn.addEventListener("click", async () => {
        const n = imageData.order.length;
        if (!n) return;
        if (queueBtn.dataset.busy === "1") return;   // double-clic = double lot

        const consume = !!widgetValue("consume_on_load", false);
        const chunk   = Math.max(1, Math.min(QUEUE_CHUNK,
                                             parseInt(widgetValue("queue_batch_size", QUEUE_CHUNK), 10)
                                             || QUEUE_CHUNK));

        queueBtn.dataset.busy = "1";
        const label = queueBtn.innerHTML;
        const progress = (sent, total) => {
            queueBtn.innerHTML = sent >= total ? `⏳ ${sent} / ${total}`
                                               : `⏳ ${sent} / ${total} — waiting`;
        };
        try {
            if (consume) {
                queueBtn.innerHTML = `⏳ 0 / ${n}`;
                const sent = await queueConsuming(() => imageData.order.length, chunk, progress);
                console.log(`[Onyx Batch] consume mode: queued ${sent} run(s) `
                          + `${chunk} at a time, ${imageData.order.length} left in the list.`);
            } else if (n <= chunk) {
                await app.queuePrompt(0, n);
            } else {
                queueBtn.innerHTML = `⏳ 0 / ${n}`;
                await queueInChunks(n, chunk, progress);
                console.log(`[Onyx Batch] queued ${n} run(s) in chunks of ${chunk}`);
            }
        } catch(e) {
            console.error("[Onyx Batch] queuePrompt error:", e);
        } finally {
            queueBtn.dataset.busy = "0";
            queueBtn.innerHTML = label;
            refresh();
        }
    });

    // ── Upload ─────────────────────────────────────────────────────────────
    async function handleFiles(files) {
        if (isUploading) return;
        isUploading = true;
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = "⏳ Uploading…";

        const form = new FormData();
        for (const f of files) form.append("files", f);

        try {
            const resp = await fetch("/onyx/batch_upload", { method: "POST", body: form });
            const json = await resp.json();
            if (json.success) {
                for (const img of json.images) {
                    if (!imageData.images.find(i => i.id === img.id)) {
                        imageData.images.push(img);
                        imageData.order.push(img.id);
                    }
                }
                refresh();
                syncBatchData();
            } else {
                console.error("[Onyx Batch] Upload failed:", json.error);
            }
        } catch(e) {
            console.error("[Onyx Batch] Upload error:", e);
        } finally {
            isUploading = false;
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = "📁 Upload";
        }
    }

    // ── Delete single image ────────────────────────────────────────────────
    async function deleteImage(imgId) {
        await fetch(`/onyx/batch_delete/${imgId}`, { method: "DELETE" }).catch(() => {});
        imageData.images = imageData.images.filter(i => i.id !== imgId);
        imageData.order  = imageData.order.filter(id => id !== imgId);
        refresh();
        syncBatchData();
    }

    // ── Rebuild UI ─────────────────────────────────────────────────────────
    function refresh() {
        const count = imageData.order.length;

        // Toggle drop zone / grid
        if (count === 0) {
            if (root.contains(gridWrap)) root.removeChild(gridWrap);
            if (!root.contains(dropZone)) root.insertBefore(dropZone, status);
            queueBtn.disabled    = true;
            queueBtn.innerHTML   = "▶ Queue All (0)";
            status.textContent   = "No images loaded";
        } else {
            if (root.contains(dropZone)) root.removeChild(dropZone);
            if (!root.contains(gridWrap)) root.insertBefore(gridWrap, status);
            queueBtn.disabled    = false;
            queueBtn.innerHTML   = `▶ Queue All (${count})`;
            status.textContent   = `${count} image${count !== 1 ? "s" : ""} • sequential mode`;
        }

        // Rebuild grid cards
        grid.innerHTML = "";
        imageData.order.forEach((imgId, i) => {
            const meta    = imageData.images.find(m => m.id === imgId);
            if (!meta) return;
            const isActive = i === currentIndex;

            // Card
            const card = document.createElement("div");
            css(card, {
                position:     "relative",
                background:   isActive ? T.cardActive : T.panel,
                border:       `2px solid ${isActive ? T.cardBorder : T.border}`,
                borderRadius: "6px",
                overflow:     "hidden",
                display:      "flex",
                flexDirection:"column",
                aspectRatio:  "1",
                transition:   "border-color 0.2s, box-shadow 0.2s",
                boxShadow:    isActive ? `0 0 10px ${T.accentDim}` : "none",
            });

            // Thumbnail
            const thumb = document.createElement("img");
            const thumbSrc = meta.thumbnail || meta.filename;
            thumb.src = `/onyx/view/${thumbSrc}`;
            css(thumb, {
                width:      "100%",
                height:     "85%",
                objectFit:  "cover",
                display:    "block",
                flexShrink: "0",
            });
            thumb.onerror = () => {
                thumb.style.display = "none";
                css(card, { background: T.border });
            };

            // Label
            const raw   = (meta.original_name || meta.filename).replace(/\.[^.]+$/, "");
            const label = document.createElement("div");
            label.textContent = raw.length > 15 ? raw.slice(0, 13) + "…" : raw;
            css(label, {
                height:       "15%",
                display:      "flex",
                alignItems:   "center",
                justifyContent: "center",
                fontSize:     "10px",
                color:        isActive ? T.text : T.textDim,
                padding:      "0 4px",
                overflow:     "hidden",
                whiteSpace:   "nowrap",
                textOverflow: "ellipsis",
                fontWeight:   isActive ? "700" : "400",
            });

            // Active badge ▶
            if (isActive) {
                const badge = document.createElement("div");
                badge.textContent = "▶";
                css(badge, {
                    position:     "absolute",
                    top:          "4px",
                    left:         "4px",
                    background:   T.accent,
                    color:        "#fff",
                    borderRadius: "50%",
                    width:        "18px",
                    height:       "18px",
                    fontSize:     "8px",
                    display:      "flex",
                    alignItems:   "center",
                    justifyContent: "center",
                    fontWeight:   "bold",
                    boxShadow:    "0 0 6px rgba(0,0,0,0.6)",
                });
                card.appendChild(badge);
            }

            // Index badge
            const idxBadge = document.createElement("div");
            idxBadge.textContent = i + 1;
            css(idxBadge, {
                position:     "absolute",
                bottom:       "calc(15% + 3px)",
                right:        "4px",
                background:   "rgba(0,0,0,0.55)",
                color:        T.textDim,
                borderRadius: "3px",
                padding:      "1px 4px",
                fontSize:     "9px",
                fontWeight:   "600",
            });

            // Delete button ✕
            const delBtn = document.createElement("button");
            delBtn.textContent = "✕";
            css(delBtn, {
                position:     "absolute",
                top:          "4px",
                right:        "4px",
                background:   "rgba(0,0,0,0.65)",
                color:        T.dangerClr,
                border:       "none",
                borderRadius: "50%",
                width:        "18px",
                height:       "18px",
                fontSize:     "11px",
                lineHeight:   "1",
                display:      "none",
                alignItems:   "center",
                justifyContent: "center",
                cursor:       "pointer",
                padding:      "0",
                fontWeight:   "bold",
            });
            delBtn.addEventListener("click", e => {
                e.stopPropagation();
                deleteImage(meta.id);
            });

            card.addEventListener("mouseenter", () => {
                delBtn.style.display = "flex";
                if (!isActive) css(card, { borderColor: T.textDim });
            });
            card.addEventListener("mouseleave", () => {
                delBtn.style.display = "none";
                if (!isActive) css(card, { borderColor: T.border });
            });

            card.append(thumb, label, idxBadge, delBtn);
            grid.appendChild(card);
        });

        // Auto-size node height: toolbar(42) + grid + status(26)
        const cols      = Math.max(1, Math.floor((node.size[0] - 16) / 148));
        const rows      = Math.ceil(count / cols);
        const gridH     = count === 0 ? 320 : Math.min(Math.max(rows * 160, 320), 900);
        node.size[1]    = 42 + gridH + 26;
        app.graph.setDirtyCanvas(true, true);
    }

    // ── WebSocket: highlight current image during processing ───────────────
    const wsHandler = ({ detail }) => {
        if (String(detail?.node_id) !== String(node.id)) return;
        currentIndex = detail.current_index ?? -1;
        refresh();
        // Scroll active card into view
        const cards = grid.children;
        const active = cards[currentIndex];
        if (active) active.scrollIntoView({ block: "nearest", behavior: "smooth" });
    };
    api.addEventListener("onyx_batch_loader_update", wsHandler);

    // ── WebSocket: an item was loaded and must leave the list ──────────────
    // Only the list is touched. The file stays in the pool folder, so nothing
    // is destroyed by a mode whose entire purpose is to survive a crash.
    const consumeHandler = ({ detail }) => {
        if (String(detail?.node_id) !== String(node.id)) return;
        const id = String(detail.image_id || "");
        if (!id) return;
        const before = imageData.order.length;
        imageData.order  = imageData.order.filter(x => x !== id);
        imageData.images = imageData.images.filter(i => String(i.id) !== id);
        if (imageData.order.length === before) return;   // deja retire
        currentIndex = -1;
        refresh();
        syncBatchData();
    };
    api.addEventListener("onyx_batch_loader_consume", consumeHandler);

    // ── DOM widget (display only — does not serialize) ─────────────────────
    node.addDOMWidget("images_display", "ONYX_BATCH_DISPLAY", root, {
        serialize: false,
        getValue:  () => undefined,
        setValue:  () => {},
    });

    // ── Serialisation: persist imageData in batch_data widget ──────────────
    // Override serialize() so the JSON always reflects current imageData.
    const origSerialize = node.serialize?.bind(node);
    node.serialize = function() {
        const out = origSerialize ? origSerialize() : {};
        if (out.widgets_values) {
            // batch_data is the first (and only) required widget → index 0
            out.widgets_values[0] = JSON.stringify(imageData);
        }
        if (out.inputs) {
            for (const inp of Object.values(out.inputs)) {
                if (inp.name === "batch_data") inp.widget.value = JSON.stringify(imageData);
            }
        }
        return out;
    };

    // Also keep the batch_data widget value in sync at all times
    const origOnConfigure = node.onConfigure?.bind(node);
    node.onConfigure = function(info) {
        origOnConfigure?.(info);
        // Restore state from saved widgets_values[0]
        const saved = info?.widgets_values?.[0];
        if (saved) {
            try {
                imageData = JSON.parse(saved) || { images: [], order: [] };
            } catch { imageData = { images: [], order: [] }; }
        }
        patchBatchWidget();
        refresh();
        syncBatchData();
    };

    // ── Cleanup ────────────────────────────────────────────────────────────
    const origOnRemoved = node.onRemoved?.bind(node);
    node.onRemoved = function() {
        api.removeEventListener("onyx_batch_loader_update", wsHandler);
        origOnRemoved?.();
    };

    // ── Initial render ─────────────────────────────────────────────────────
    node.size = [460, 540];
    refresh();
}

// ── Register extension ───────────────────────────────────────────────────────
app.registerExtension({
    name: "Onyx.ImageBatchLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "OnyxImageBatchLoader") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            setupBatchLoader(this);
        };
    },
});
