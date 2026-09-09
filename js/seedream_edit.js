import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "onyx.SeedreamEdit",
    nodeCreated(node) {
        if (node.comfyClass !== "SeedreamEditAPINode" && node.comfyClass !== "SeedreamEditFalAPINode") return;

        const sizeWidget = node.widgets.find(w => w.name === "size");
        const customToggle = node.widgets.find(w => w.name === "use_custom_size");
        const customWidth = node.widgets.find(w => w.name === "custom_width");
        const customHeight = node.widgets.find(w => w.name === "custom_height");

        if (!sizeWidget || !customToggle || !customWidth || !customHeight) return;

        function updateVisibility() {
            const isCustom = customToggle.value;
            sizeWidget.disabled = isCustom;
            customWidth.disabled = !isCustom;
            customHeight.disabled = !isCustom;
        }

        // Set initial state
        updateVisibility();

        // Watch for toggle changes
        const origCallback = customToggle.callback;
        customToggle.callback = function (...args) {
            if (origCallback) origCallback.apply(this, args);
            updateVisibility();
        };
    },
});
