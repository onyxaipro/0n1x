import { app } from "../../scripts/app.js";

let promptMap = {};
let byFile = {};

async function loadData() {
    try {
        const [mapResp, byFileResp] = await Promise.all([
            fetch("/prompt_selector_monthly/get_prompts"),
            fetch("/prompt_selector_monthly/get_prompts_by_file"),
        ]);
        if (mapResp.ok) promptMap = await mapResp.json();
        if (byFileResp.ok) byFile = await byFileResp.json();
    } catch (e) {
        console.warn("PromptSelector: could not load prompt data", e);
    }
}

loadData();

app.registerExtension({
    name: "onyx.PromptSelectorMonthly",
    nodeCreated(node) {
        if (node.comfyClass !== "OnyxPromptSelectorNodeMonthly") return;

        // --- Custom styling: Fire theme from logo ---
        node.color = "#082a33";       // title bar
        node.bgcolor = "#071a21";     // body background
        node.boxcolor = "#22b8d4";    // selection outline
        // Override title text color
        node.title_color = "#5fd8ef";

        // Cold drift — slow motes rising through dark water
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

            // Warm glow border that pulses
            ctx.save();
            const pulse = 0.3 + Math.sin(t * 2) * 0.15;
            ctx.shadowColor = "#22b8d4";
            ctx.shadowBlur = 8 + Math.sin(t * 3) * 4;
            ctx.strokeStyle = `rgba(34, 184, 212, ${pulse})`;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(0, 0, w, h);
            ctx.restore();

            // Animated fire gradient line under title
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

            // Fire embers rising from bottom
            ctx.save();
            for (const f of flames) {
                // Rise upward
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

                // Color transitions: white-yellow core → orange → red → fade
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

                // Halo diffus
                ctx.globalAlpha = life * 0.6;
                const glowGrad = ctx.createRadialGradient(px, py, 0, px, py, sz * 3);
                glowGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${life * 0.5})`);
                glowGrad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
                ctx.fillStyle = glowGrad;
                ctx.beginPath();
                ctx.arc(px, py, sz * 3, 0, Math.PI * 2);
                ctx.fill();

                // Bright ember center
                ctx.globalAlpha = life * 0.9;
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.beginPath();
                ctx.arc(px, py, sz * 0.6, 0, Math.PI * 2);
                ctx.fill();

                f.phase += 0.02;
            }
            ctx.restore();

            // Fire badge
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

            // Keep animating
            node.setDirtyCanvas(true, false);
        };

        // --- Functional logic ---
        const fileWidget = node.widgets.find(w => w.name === "prompt_file");
        const exampleWidget = node.widgets.find(w => w.name === "example_prompt");
        const textWidget = node.widgets.find(w => w.name === "prompt_text");
        if (!fileWidget || !exampleWidget || !textWidget) return;

        let currentPrompts = [];
        let currentIndex = -1;

        function updateExampleOptions(fileStem) {
            const labels = byFile[fileStem] || [];
            currentPrompts = labels;
            if (exampleWidget.options) {
                exampleWidget.options.values = labels.length > 0 ? labels : ["(no prompts)"];
            }
            showPrompt(0);
        }

        function showPrompt(index) {
            if (currentPrompts.length === 0) {
                exampleWidget.value = "(no prompts)";
                textWidget.value = "";
                currentIndex = -1;
                app.graph.setDirtyCanvas(true);
                return;
            }
            currentIndex = ((index % currentPrompts.length) + currentPrompts.length) % currentPrompts.length;
            const label = currentPrompts[currentIndex];
            const fullText = promptMap[label] || "";
            exampleWidget.value = label;
            textWidget.value = fullText;
            app.graph.setDirtyCanvas(true);
        }

        const prevNextWidget = node.addWidget("button", "\u25C0  Prev Prompt", null, () => {
            showPrompt(currentIndex - 1);
        });

        const nextWidget = node.addWidget("button", "Next Prompt  \u25B6", null, () => {
            showPrompt(currentIndex + 1);
        });

        const origFileCallback = fileWidget.callback;
        fileWidget.callback = function (value) {
            if (origFileCallback) origFileCallback.call(this, value);
            updateExampleOptions(value.replace(/\.txt$/, ""));
        };

        const origExampleCallback = exampleWidget.callback;
        exampleWidget.callback = function (value) {
            if (origExampleCallback) origExampleCallback.call(this, value);
            const idx = currentPrompts.indexOf(value);
            if (idx >= 0) {
                currentIndex = idx;
                textWidget.value = promptMap[value] || value;
                app.graph.setDirtyCanvas(true);
            }
        };

        requestAnimationFrame(() => {
            node.setSize([450, 340]);
            app.graph.setDirtyCanvas(true);
        });

        loadData().then(() => {
            const stem = (fileWidget.value || "").replace(/\.txt$/, "");
            if (stem) updateExampleOptions(stem);
        });
    },
});
