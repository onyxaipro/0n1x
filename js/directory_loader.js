import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "onyx.DirectoryImageLoader",
    nodeCreated(node) {
        if (node.comfyClass !== "OnyxDirectoryImageLoaderNode") return;

        const toggleWidget = node.widgets.find(w => w.name === "use_absolute_path");
        const folderWidget = node.widgets.find(w => w.name === "input_folder");
        const pathWidget = node.widgets.find(w => w.name === "absolute_path");

        if (!toggleWidget || !folderWidget || !pathWidget) return;

        function updateVisibility() {
            const useAbsolute = toggleWidget.value;
            folderWidget.disabled = useAbsolute;
            pathWidget.disabled = !useAbsolute;
        }

        updateVisibility();

        const origCallback = toggleWidget.callback;
        toggleWidget.callback = function (...args) {
            if (origCallback) origCallback.apply(this, args);
            updateVisibility();
        };
    },

    async setup() {
        // Track remaining count from directory loader's output
        let pendingRemaining = 0;
        let hadError = false;

        // Capture remaining count when directory loader node executes
        api.addEventListener("executed", ({ detail }) => {
            if (!detail || !detail.output) return;

            const output = detail.output;
            if (output.remaining === undefined || output.run_count === undefined) return;

            const remaining = output.remaining[0];
            const current = output.current_index[0];
            const filename = output.filename ? output.filename[0] : "";

            console.log(`[Directory Loader] Processed ${current + 1}: ${filename}`);
            pendingRemaining = remaining;
        });

        // On any error, stop the batch
        api.addEventListener("execution_error", () => {
            if (pendingRemaining > 0) {
                console.warn("[Directory Loader] Error during batch — stopping. " + pendingRemaining + " remaining images skipped.");
                fetch("/directory_loader/reset_batch", { method: "POST" }).catch(() => {});
            }
            pendingRemaining = 0;
            hadError = true;
        });

        // Only auto-queue after the ENTIRE workflow completes successfully
        api.addEventListener("status", ({ detail }) => {
            // status fires with detail.exec_info.queue_remaining === 0 when idle
            if (!detail || !detail.exec_info) return;
            if (detail.exec_info.queue_remaining !== 0) return;

            // Workflow just finished — check if we should queue next
            if (hadError) {
                hadError = false;
                pendingRemaining = 0;
                return;
            }

            if (pendingRemaining > 0) {
                console.log(`[Directory Loader] ${pendingRemaining} remaining, auto-queuing next...`);
                const rem = pendingRemaining;
                pendingRemaining = 0;
                setTimeout(() => {
                    app.queuePrompt(0, 1);
                }, 500);
            }
        });
    },
});
