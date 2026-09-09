/**
 * Onyx Video Chain — reset button.
 *
 * The chain advances on disk, so there has to be a way back to segment 1 that
 * is not "remember to flip a boolean before the first run and back after it".
 */

import { app } from "../../scripts/app.js";

const CHAIN_NODES = ["OnyxVideoChainPrepare", "OnyxVideoChainCommit"];

app.registerExtension({
    name: "onyx.VideoChain",

    nodeCreated(node) {
        if (!CHAIN_NODES.includes(node.comfyClass)) return;

        const btn = node.addWidget("button", "⟳  Reset chain", null, async () => {
            const session = (node.widgets?.find(w => w.name === "session_name")?.value || "").trim();
            if (!session) {
                alert("Reset chain: session_name is empty.");
                return;
            }
            if (!confirm(`Restart chain "${session}" at segment 1?\n\n` +
                         `The stored tail is discarded. Video already rendered and saved is untouched.`)) return;

            try {
                const resp = await fetch("/onyx/video_chain/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session }),
                });
                const raw = await resp.text();
                let data;
                try {
                    data = JSON.parse(raw);
                } catch (_) {
                    if (resp.status === 404) {
                        throw new Error(
                            "the reset route is not on the server.\n\n" +
                            "Restart ComfyUI: a browser refresh reloads this button but not the " +
                            "Python side that answers it."
                        );
                    }
                    throw new Error(`HTTP ${resp.status} — ${raw.slice(0, 200)}`);
                }
                if (!resp.ok) throw new Error(data.error || resp.statusText);
                alert(data.reset
                    ? `Chain "${session}" reset — the next run is segment 1.`
                    : `Chain "${session}" had no saved state; the next run is segment 1 anyway.`);
            } catch (e) {
                console.error("[Video Chain] reset failed", e);
                alert("Reset failed: " + e.message);
            }
        });
        btn.serialize = false;
    },
});
