/**
 * OnyxNanoBananaAIO — Preset UI
 * Bouton Save/Load presets directement dans le node.
 */

import { app } from "../../scripts/app.js";

const PRESET_KEYS = [
    "provider",
    "prompt",
    "image_size",
    "model",
    "batch_size",
    "use_search",
    "system_instructions",
    "aspect_ratio",
    "temperature",
    "top_p",
    "fal_safety_tolerance",
    "fal_enable_web_search",
    "disable_safety_threshold",
    "vertex_location",
];

let _presetsCache = {};

async function loadPresetsFromServer() {
    try {
        const resp = await fetch("/onyx_nb_aio/load_presets");
        const data = await resp.json();
        if (data.success) {
            _presetsCache = data.presets || {};
        }
    } catch (e) {
        console.warn("[NB AIO Presets] Impossible de charger :", e);
    }
}

loadPresetsFromServer();

function getWidgetValue(node, name) {
    return node.widgets?.find(w => w.name === name)?.value;
}

function setWidgetValue(node, name, value) {
    const w = node.widgets?.find(w => w.name === name);
    if (!w) return;
    w.value = value;
    if (w.callback) w.callback(value);
    if (w.inputEl) w.inputEl.value = value;
}

function collectNodeParams(node) {
    const params = {};
    for (const key of PRESET_KEYS) {
        const val = getWidgetValue(node, key);
        if (val !== undefined) params[key] = val;
    }
    return params;
}

function applyPresetToNode(node, data) {
    for (const [key, value] of Object.entries(data)) {
        if (PRESET_KEYS.includes(key)) {
            setWidgetValue(node, key, value);
        }
    }
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "OnyxNanoBananaAIO.Presets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "OnyxNanoBananaAIO") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                buildPresetUI(node);
            }, 200);
        };
    },
});

function buildPresetUI(node) {

    // ── PRESET SLOT selector ──────────────────────────────────────
    const slotWidget = node.addWidget(
        "combo",
        "Preset Slot",
        "1",
        (value) => {},
        {
            values: ["1", "2", "3", "4", "5"],
            serialize: false,
        }
    );

    // ── BOUTON SAVE ───────────────────────────────────────────────
    node.addWidget(
        "button",
        "💾  Save current settings to preset slot",
        null,
        async () => {
            const slot   = parseInt(slotWidget.value);
            const params = collectNodeParams(node);

            showStatus(node, "⏳  Saving...", "#aaaaaa");

            try {
                const resp = await fetch("/onyx_nb_aio/save_preset", {
                    method:  "POST",
                    headers: { "Content-Type": "application/json" },
                    body:    JSON.stringify({ slot, params }),
                });
                const data = await resp.json();

                if (data.success) {
                    _presetsCache[String(slot)] = data.params;
                    showStatus(node, `✅  Preset ${slot} saved !`, "#4caf50");
                } else {
                    showStatus(node, `❌  Error: ${data.error}`, "#f44336");
                }
            } catch (e) {
                showStatus(node, "❌  Network error", "#f44336");
                console.error("[NB AIO Presets]", e);
            }
        },
        { serialize: false }
    );

    // ── BOUTON LOAD ───────────────────────────────────────────────
    node.addWidget(
        "button",
        "📂  Load preset slot into node",
        null,
        async () => {
            const slot   = slotWidget.value;
            let   cached = _presetsCache[slot];

            if (!cached) {
                await loadPresetsFromServer();
                cached = _presetsCache[slot];
            }

            if (!cached) {
                showStatus(node, `⚠️  Slot ${slot} is empty`, "#33cbe6");
                return;
            }

            applyPresetToNode(node, cached);
            const preview = (cached.prompt || "").slice(0, 40);
            showStatus(node, `📂  Preset ${slot} loaded: ${preview}…`, "#2196f3");
        },
        { serialize: false }
    );

    // ── WIDGET STATUS ─────────────────────────────────────────────
    const statusWidget = node.addWidget(
        "text",
        "nb_preset_status",
        "",
        () => {},
        {
            serialize: false,
            multiline: false,
        }
    );
    statusWidget.label    = "⬆ preset info";
    statusWidget.disabled = true;

    node._nbStatusWidget = statusWidget;
    node.setSize([node.size[0], node.computeSize()[1]]);
}

function showStatus(node, msg, color = "#aaa") {
    if (!node._nbStatusWidget) return;
    node._nbStatusWidget.value = msg;

    if (node._nbStatusWidget.inputEl) {
        node._nbStatusWidget.inputEl.style.color = color;
    }

    clearTimeout(node._nbStatusTimer);
    node._nbStatusTimer = setTimeout(() => {
        if (node._nbStatusWidget) {
            node._nbStatusWidget.value = "";
        }
    }, 4000);

    app.graph.setDirtyCanvas(true);
}