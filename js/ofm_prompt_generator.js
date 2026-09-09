import { app } from "../../scripts/app.js";

let modePrompts = {};

async function loadPrompts() {
    try {
        const resp = await fetch("/ofm_prompt_generator/get_prompts");
        if (resp.ok) {
            modePrompts = await resp.json();
        }
    } catch (e) {
        console.warn("OFM Prompt Generator: could not load prompts", e);
    }
}

loadPrompts();

app.registerExtension({
    name: "onyx.OFMPromptGenerator",
    nodeCreated(node) {
        if (node.comfyClass !== "OnyxGeminiPromptNode") return;

        // --- Fire theme styling (matching Onyx Prompt Selector) ---
        node.color = "#082a33";
        node.bgcolor = "#071a21";
        node.boxcolor = "#22b8d4";
        node.title_color = "#5fd8ef";

        // Fire particles
        const flames = [];
        for (let i = 0; i < 35; i++) {
            flames.push({
                x: Math.random(),
                y: 1.0 + Math.random() * 0.3,
                size: Math.random() * 4 + 2,
                speed: Math.random() * 0.0022 + 0.0009,
                wobble: Math.random() * 0.0075,
                phase: Math.random() * Math.PI * 2,
                life: Math.random(),
            });
        }

        function resetFlame(f) {
            f.x = Math.random();
            f.y = 1.0 + Math.random() * 0.1;
            f.life = 1.0;
            f.size = Math.random() * 4 + 2;
            f.speed = Math.random() * 0.008 + 0.004;
        }

        const origDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origDrawForeground) origDrawForeground.call(this, ctx);

            const t = performance.now() / 1000;
            const w = node.size[0];
            const h = node.size[1];

            // Warm glow border
            ctx.save();
            const pulse = 0.3 + Math.sin(t * 2) * 0.15;
            ctx.shadowColor = "#22b8d4";
            ctx.shadowBlur = 8 + Math.sin(t * 3) * 4;
            ctx.strokeStyle = `rgba(34, 184, 212, ${pulse})`;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(0, 0, w, h);
            ctx.restore();

            // Fire gradient accent line
            ctx.save();
            const grad = ctx.createLinearGradient(0, 0, w, 0);
            const s = (t * 0.5) % 1;
            grad.addColorStop(0, "#0b7f9c");
            grad.addColorStop(Math.abs((s) % 1), "#22b8d4");
            grad.addColorStop(Math.abs((s + 0.3) % 1), "#5fd8ef");
            grad.addColorStop(Math.abs((s + 0.6) % 1), "#8ee9f7");
            grad.addColorStop(1, "#0b7f9c");
            ctx.fillStyle = grad;
            ctx.fillRect(0, -1, w, 3);
            ctx.restore();

            // Fire embers
            ctx.save();
            for (const f of flames) {
                f.y -= f.speed;
                f.x += Math.sin(t * 0.9 + f.phase) * f.wobble;
                f.life -= f.speed * 0.55;

                if (f.life <= 0 || f.y < -0.1) {
                    resetFlame(f);
                    continue;
                }

                const px = f.x * w;
                const py = f.y * h;
                const life = f.life;
                const sz = f.size * life;

                let r, g, b;
                if (life > 0.7) {
                    // Coeur clair, presque blanc
                    r = 190 + (life - 0.7) * 180; g = 240; b = 255;
                } else if (life > 0.4) {
                    // Cyan
                    r = 34; g = 184; b = 212;
                } else {
                    // Bleu profond en fin de vie
                    r = 12; g = 90; b = 130;
                }

                ctx.globalAlpha = life * 0.6;
                const glowGrad = ctx.createRadialGradient(px, py, 0, px, py, sz * 3);
                glowGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${life * 0.5})`);
                glowGrad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
                ctx.fillStyle = glowGrad;
                ctx.beginPath();
                ctx.arc(px, py, sz * 3, 0, Math.PI * 2);
                ctx.fill();

                ctx.globalAlpha = life * 0.9;
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.beginPath();
                ctx.arc(px, py, sz * 0.6, 0, Math.PI * 2);
                ctx.fill();

                f.phase += 0.02;
            }
            ctx.restore();

            // Badge
            ctx.save();
            const badgeText = "ONYX";
            ctx.font = "bold 9px sans-serif";
            const textWidth = ctx.measureText(badgeText).width;
            const badgeX = w - textWidth - 12;
            const badgeY = -LiteGraph.NODE_TITLE_HEIGHT + 6;

            ctx.fillStyle = `rgba(8, 42, 51, 0.9)`;
            ctx.fillRect(badgeX - 4, badgeY - 1, textWidth + 8, 14);

            ctx.fillStyle = "#5fd8ef";
            ctx.fillText(badgeText, badgeX, badgeY + 10);
            ctx.restore();

            node.setDirtyCanvas(true, false);
        };

        // --- Mode switching logic ---
        const modeWidget = node.widgets.find(w => w.name === "mode");
        const customPromptWidget = node.widgets.find(w => w.name === "custom_prompt");
        if (!modeWidget || !customPromptWidget) return;

        function applyMode(mode) {
            const promptText = modePrompts[mode];
            if (promptText != null) {
                customPromptWidget.value = promptText;
            } else {
                customPromptWidget.value = "";
            }
            app.graph.setDirtyCanvas(true);
        }

        const origCallback = modeWidget.callback;
        modeWidget.callback = function (value) {
            if (origCallback) origCallback.call(this, value);
            applyMode(value);
        };

        loadPrompts().then(() => {
            applyMode(modeWidget.value);
        });

        // --- Provider switching: swap the model list, enable/disable fields ---
        //
        // Python declares `model` with every id at once, because ComfyUI validates
        // the widget value server-side and would refuse a Grok id if the list only
        // held Gemini ones. Narrowing the list is purely a frontend concern — and
        // run() re-checks the pairing anyway, for workflows driven by the API.
        const GEMINI_MODELS = [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ];
        const GROK_MODELS = [
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
            "grok-2-vision-1212",
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-3-mini-fast",
        ];
        // temperature is deprecated on these — see _GEMINI_NO_SAMPLING in the
        // Python side. Kept in sync by hand; the node stops sending it either way,
        // so a stale entry here only costs a missing hint, never a wrong request.
        const NO_SAMPLING = ["gemini-3.6-flash", "gemini-3.5-flash-lite"];

        const providerWidget = node.widgets.find(w => w.name === "provider");
        const geminiApiKeyWidget = node.widgets.find(w => w.name === "gemini_api_key");
        const grokApiKeyWidget = node.widgets.find(w => w.name === "grok_api_key");
        const modelWidget = node.widgets.find(w => w.name === "model");
        const thinkingWidget = node.widgets.find(w => w.name === "thinking_level");
        const temperatureWidget = node.widgets.find(w => w.name === "temperature");
        const safetyWidget = node.widgets.find(w => w.name === "safety_threshold");

        /** Greys out temperature on the models that ignore it, and says why. */
        function applyModel(model) {
            const isGrok = String(model || "").startsWith("grok");
            const deprecated = NO_SAMPLING.includes(model);

            if (temperatureWidget) {
                temperatureWidget.disabled = deprecated;
                if (temperatureWidget.options) {
                    temperatureWidget.options.tooltip = deprecated
                        ? `⚠️ ${model} ignores temperature — Google deprecated it. Use system_prompt and thinking_level instead.`
                        : "";
                }
            }
            // Grok has no equivalent knob, so the widget would be decorative there.
            if (thinkingWidget) thinkingWidget.disabled = isGrok;

            app.graph.setDirtyCanvas(true);
        }

        function applyProvider(provider) {
            const isGemini = provider === "Gemini";
            const isGrok = provider === "Grok";
            const isVertex = provider === "Vertex";

            if (geminiApiKeyWidget) geminiApiKeyWidget.disabled = !isGemini;
            if (grokApiKeyWidget) grokApiKeyWidget.disabled = !isGrok;
            // Vertex authenticates with the service-account JSON, not a key, but it
            // speaks to the same Gemini models.
            if (safetyWidget) safetyWidget.disabled = !isGemini;

            if (modelWidget) {
                const list = isGrok ? GROK_MODELS : GEMINI_MODELS;
                if (modelWidget.options?.values) modelWidget.options.values = [...list];
                // Switching provider almost always leaves the model on the other
                // vendor's id. Snap it, or the run fails on the mismatch guard.
                if (!list.includes(modelWidget.value)) {
                    modelWidget.value = list[0];
                    if (modelWidget.callback) modelWidget.callback(list[0]);
                }
                applyModel(modelWidget.value);
            }

            app.graph.setDirtyCanvas(true);
        }

        if (modelWidget) {
            const origModelCb = modelWidget.callback;
            modelWidget.callback = function (value) {
                if (origModelCb) origModelCb.call(this, value);
                applyModel(value);
            };
        }

        if (providerWidget) {
            const origProviderCallback = providerWidget.callback;
            providerWidget.callback = function (value) {
                if (origProviderCallback) origProviderCallback.call(this, value);
                applyProvider(value);
            };
            applyProvider(providerWidget.value);
        }

        // --- Text output display ---
        const outputWidget = node.addWidget("customtext", "output_text", "", () => {}, {
            multiline: true,
            serialize: false,
        });
        outputWidget.inputEl?.setAttribute("readonly", "true");

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origOnExecuted) origOnExecuted.call(this, output);
            if (output?.text?.[0] != null) {
                outputWidget.value = output.text[0];
                app.graph.setDirtyCanvas(true);
            }
        };
    },
});
