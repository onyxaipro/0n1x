/**
 * Onyx Grok Prompt Generator — Preset UI.
 *
 * Same mechanism as nb_aio_presets.js: numbered slots stored server-side in a
 * JSON file next to the node, so the presets survive a workflow reload and are
 * shared by every copy of the node.
 *
 * The API key is deliberately absent from PRESET_KEYS — it is cached by the
 * Python side already, and a preset file is the last place a key should live.
 */

import { app } from "../../scripts/app.js";

const PRESET_KEYS = [
    "prompt",
    "trigger_word",
    "model",
    "temperature",
    "max_tokens",
];

let _presetsCache = {};

async function loadPresetsFromServer() {
    try {
        const resp = await fetch("/onyx_grok_prompt/load_presets");
        const data = await resp.json();
        if (data.success) _presetsCache = data.presets || {};
    } catch (e) {
        console.warn("[Grok Presets] could not load:", e);
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
        if (PRESET_KEYS.includes(key)) setWidgetValue(node, key, value);
    }
    app.graph.setDirtyCanvas(true, true);
}

function showStatus(node, msg, color = "#aaa") {
    if (!node._grokStatusWidget) return;
    node._grokStatusWidget.value = msg;
    if (node._grokStatusWidget.inputEl) {
        node._grokStatusWidget.inputEl.style.color = color;
    }
    clearTimeout(node._grokStatusTimer);
    node._grokStatusTimer = setTimeout(() => {
        if (node._grokStatusWidget) node._grokStatusWidget.value = "";
    }, 4000);
    app.graph.setDirtyCanvas(true);
}

function buildPresetUI(node) {
    const slotWidget = node.addWidget(
        "combo", "Preset Slot", "1", () => {},
        { values: ["1", "2", "3", "4", "5"], serialize: false }
    );

    node.addWidget("button", "💾  Save current settings to preset slot", null, async () => {
        const slot = parseInt(slotWidget.value);
        const params = collectNodeParams(node);
        showStatus(node, "⏳  Saving...", "#aaaaaa");
        try {
            const resp = await fetch("/onyx_grok_prompt/save_preset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ slot, params }),
            });
            // Lu en texte d'abord : une route absente renvoie "404: Not Found",
            // que JSON.parse signale comme un caractere inattendu en position 3
            // — un message qui cache completement la vraie cause.
            const raw = await resp.text();
            let data;
            try {
                data = JSON.parse(raw);
            } catch (_) {
                throw new Error(resp.status === 404
                    ? "route missing — restart ComfyUI (a browser refresh reloads this "
                      + "button but not the Python side)"
                    : `HTTP ${resp.status} — ${raw.slice(0, 120)}`);
            }
            if (!data.success) throw new Error(data.error || resp.statusText);
            _presetsCache[String(slot)] = data.params;
            showStatus(node, `✅  Preset ${slot} saved!`, "#4caf50");
        } catch (e) {
            showStatus(node, `❌  ${e.message}`, "#f44336");
            console.error("[Grok Presets]", e);
        }
    }, { serialize: false });

    node.addWidget("button", "📂  Load preset slot into node", null, async () => {
        const slot = slotWidget.value;
        let cached = _presetsCache[slot];
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
        showStatus(node, `📂  Preset ${slot}: ${preview}…`, "#2196f3");
    }, { serialize: false });

    const statusWidget = node.addWidget(
        "text", "grok_preset_status", "", () => {},
        { serialize: false, multiline: false }
    );
    statusWidget.label = "⬆ preset info";
    statusWidget.disabled = true;

    node._grokStatusWidget = statusWidget;
    node.setSize([node.size[0], node.computeSize()[1]]);
}

app.registerExtension({
    name: "onyx.GrokPromptPresets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "OnyxGrokPromptNode") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            const node = this;
            // Meme delai que le node NanoBanana : les widgets Python ne sont pas
            // encore tous poses au moment ou onNodeCreated se declenche.
            setTimeout(() => buildPresetUI(node), 200);
        };
    },
});
