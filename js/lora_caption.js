import { app } from "../../scripts/app.js";

let modePrompts = {};

async function loadPrompts() {
    try {
        const resp = await fetch("/lora_caption/get_prompts");
        if (resp.ok) modePrompts = await resp.json();
    } catch (e) {
        console.warn("LoraCaptionGenerator: could not load mode prompts", e);
    }
}

loadPrompts();

app.registerExtension({
    name: "onyx.LoraCaptionGenerator",
    nodeCreated(node) {
        if (node.comfyClass !== "OnyxLoraCaptionGeneratorNode") return;

        const providerWidget = node.widgets.find(w => w.name === "provider");
        const geminiModelWidget = node.widgets.find(w => w.name === "gemini_model");
        const grokModelWidget = node.widgets.find(w => w.name === "grok_model");
        const modeWidget = node.widgets.find(w => w.name === "mode");
        const instructionWidget = node.widgets.find(w => w.name === "caption_instruction");

        // ── provider toggle (existing behavior) ──
        if (providerWidget) {
            function applyProvider(provider) {
                const isGemini = provider === "gemini";
                const isGrok = provider === "grok";
                if (geminiModelWidget) geminiModelWidget.disabled = !isGemini;
                if (grokModelWidget) grokModelWidget.disabled = !isGrok;
                app.graph.setDirtyCanvas(true);
            }
            const origCallback = providerWidget.callback;
            providerWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                applyProvider(value);
            };
            applyProvider(providerWidget.value);
        }

        // ── mode → caption_instruction sync ──
        if (modeWidget && instructionWidget) {
            async function applyMode(mode) {
                if (!modePrompts || Object.keys(modePrompts).length === 0) {
                    await loadPrompts();
                }
                const prompt = modePrompts[mode];
                if (typeof prompt === "string") {
                    instructionWidget.value = prompt;
                    if (instructionWidget.callback) instructionWidget.callback(prompt);
                    app.graph.setDirtyCanvas(true);
                }
            }
            const origModeCallback = modeWidget.callback;
            modeWidget.callback = function (value) {
                if (origModeCallback) origModeCallback.call(this, value);
                applyMode(value);
            };
            // Populate on creation if textbox is empty
            if (!instructionWidget.value || instructionWidget.value.trim() === "") {
                applyMode(modeWidget.value);
            }
        }

        // ── Rename files ──
        // Un bouton et non un widget booleen : renommer est destructif et ne
        // doit pas partir tout seul au prochain lancement du graphe.
        const btn = node.addWidget("button", "Rename files", null, async () => {
            const folder  = node.widgets.find(w => w.name === "input_folder")?.value;
            const trigger = (node.widgets.find(w => w.name === "trigger_word")?.value || "").trim();

            if (!folder || folder === "(no folders found)") {
                alert("Rename files: pick an input folder first.");
                return;
            }
            if (!trigger) {
                alert("Rename files: trigger_word is empty — nothing to rename the files to.");
                return;
            }
            if (!confirm(
                `Rename every image in "${folder}" to ${trigger}_01, ${trigger}_02, ...\n\n` +
                `Any .txt caption sitting beside an image is renamed with it.\n` +
                `This cannot be undone. Continue?`
            )) return;

            btn.name = "Renaming...";
            node.setDirtyCanvas(true);
            try {
                const resp = await fetch("/lora_caption/rename_files", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ folder, trigger }),
                });
                // On lit en texte avant de parser : une route absente renvoie
                // "404: Not Found", que JSON.parse signale comme un caractere
                // inattendu en position 3 — un message qui cache la vraie panne.
                const raw = await resp.text();
                let data;
                try {
                    data = JSON.parse(raw);
                } catch (_) {
                    if (resp.status === 404) {
                        throw new Error(
                            "the /lora_caption/rename_files route does not exist on the server.\n\n" +
                            "Restart ComfyUI: a browser refresh reloads this button but not the " +
                            "Python side that answers it."
                        );
                    }
                    throw new Error(`HTTP ${resp.status} — ${raw.slice(0, 200)}`);
                }
                if (!resp.ok) throw new Error(data.error || resp.statusText);
                console.log(`[Lora Caption] renamed ${data.renamed} image(s), ` +
                            `${data.captions} caption(s) in ${data.folder}`);
                alert(`Renamed ${data.renamed} image(s)` +
                      (data.captions ? ` and ${data.captions} caption file(s)` : "") +
                      ` to ${trigger}_01 ...`);
            } catch (e) {
                console.error("[Lora Caption] rename failed", e);
                alert("Rename failed: " + e.message);
            } finally {
                btn.name = "Rename files";
                node.setDirtyCanvas(true);
            }
        });
        // Un bouton n'a pas de valeur a sauvegarder, et l'inscrire decalerait
        // les widgets des workflows deja enregistres.
        btn.serialize = false;
    },
});
