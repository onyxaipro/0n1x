/**
 * Onyx Lora Guard — a safety net for rgthree's Power Lora Loader.
 *
 * The problem it solves: when rgthree's nodes fail to register in time, the
 * workflow opens with them as missing-node placeholders. Refreshing usually
 * fixes the registration — but by then the lora stack is gone, and rebuilding
 * a dozen loras with their weights by hand is the expensive part.
 *
 * This is NOT a patch of rgthree. It lives in this pack, reads and writes only
 * through rgthree's own public surface, and therefore survives every rgthree
 * update:
 *
 *   read     node.widgets, taking the entries that carry a `.lora` value
 *   write    node.configure({widgets_values}) — the exact call rgthree itself
 *            uses in configureFromApiJson()
 *
 * Storage is localStorage on purpose: the failure window is a browser refresh,
 * and localStorage is what survives one. No server, no file, nothing to break.
 *
 * Restore is MANUAL, by design. Auto-restoring would fight you the day you
 * deliberately empty a stack — the node would silently put the loras back and
 * you would have no idea why.
 */

import { app } from "../../scripts/app.js";

const TARGET_TYPES = [
    "Power Lora Loader (rgthree)",
    "Power Lora Loader",
];

const STORE_PREFIX = "onyx.loraGuard.";
const MAX_SNAPSHOTS = 60;

function isTarget(node) {
    return TARGET_TYPES.includes(node?.comfyClass) || TARGET_TYPES.includes(node?.type);
}

/** The loras currently held by a node, in rgthree's own value shape. */
function readLoras(node) {
    const out = [];
    for (const w of node.widgets || []) {
        const v = w?.value;
        if (v && typeof v === "object" && v.lora !== undefined) out.push({ ...v });
    }
    return out;
}

function keyFor(node) {
    // Node id alone is not enough: two workflows both have a node 12. The graph
    // id keeps their snapshots apart, and falls back to a constant when the
    // frontend does not expose one.
    const graphId = app.graph?.id ?? app.graph?.extra?.workflowId ?? "graph";
    return `${STORE_PREFIX}${graphId}.${node.id}`;
}

function saveSnapshot(node) {
    const loras = readLoras(node);
    if (!loras.length) return;            // never overwrite a good snapshot with nothing
    try {
        localStorage.setItem(keyFor(node), JSON.stringify({
            at: Date.now(),
            title: node.title,
            loras,
        }));
        pruneStore();
    } catch (e) {
        console.warn("[Lora Guard] could not save snapshot", e);
    }
}

function loadSnapshot(node) {
    try {
        const raw = localStorage.getItem(keyFor(node));
        return raw ? JSON.parse(raw) : null;
    } catch (e) {
        return null;
    }
}

/** Keep the store bounded: localStorage is a few MB shared with everything else. */
function pruneStore() {
    try {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k?.startsWith(STORE_PREFIX)) keys.push(k);
        }
        if (keys.length <= MAX_SNAPSHOTS) return;
        const dated = keys.map(k => {
            let at = 0;
            try { at = JSON.parse(localStorage.getItem(k))?.at || 0; } catch (_) {}
            return { k, at };
        }).sort((a, b) => a.at - b.at);
        for (const { k } of dated.slice(0, keys.length - MAX_SNAPSHOTS)) {
            localStorage.removeItem(k);
        }
    } catch (_) {}
}

function restore(node, snapshot) {
    // rgthree's own path: configure() with only widgets_values rebuilds the lora
    // widgets and skips super.configure(), leaving the rest of the node alone.
    node.configure({ widgets_values: snapshot.loras.map(l => ({ ...l })) });
    app.graph.setDirtyCanvas(true, true);
}

function describe(loras) {
    const on = loras.filter(l => l.on !== false).length;
    return `${loras.length} lora(s), ${on} enabled`;
}

function addGuardUI(node) {
    if (node._loraGuardReady) return;
    node._loraGuardReady = true;

    const btn = node.addWidget("button", "🛟  Lora Guard", null, () => {
        const snap = loadSnapshot(node);
        const current = readLoras(node);

        if (!snap?.loras?.length) {
            alert(current.length
                ? "Lora Guard: nothing saved yet for this node.\n\n" +
                  `${describe(current)} currently loaded — a snapshot is taken automatically ` +
                  "every 20 s and whenever the graph is serialised."
                : "Lora Guard: no snapshot and no loras. Nothing to do.");
            return;
        }

        const when = new Date(snap.at).toLocaleString();
        if (current.length) {
            if (!confirm(
                "Lora Guard\n\n" +
                `Snapshot: ${describe(snap.loras)} — saved ${when}\n` +
                `Current:  ${describe(current)}\n\n` +
                "Replace the current stack with the snapshot?"
            )) return;
        }
        restore(node, snap);
        console.log(`[Lora Guard] restored ${snap.loras.length} lora(s) into node ${node.id}`);
    });
    btn.serialize = false;

    // Snapshot periodique : couvre le cas ou l'onglet est ferme ou rafraichi
    // sans passage par une serialisation du graphe.
    const timer = setInterval(() => {
        if (!app.graph?._nodes?.includes(node)) {
            clearInterval(timer);
            return;
        }
        saveSnapshot(node);
    }, 20000);

    saveSnapshot(node);
}

app.registerExtension({
    name: "onyx.LoraGuard",

    nodeCreated(node) {
        if (!isTarget(node)) return;
        // Le bouton est pose apres que rgthree a fini de reconstruire ses
        // widgets : au moment de nodeCreated ils n'existent pas encore.
        setTimeout(() => addGuardUI(node), 400);
    },
});

// Snapshot a chaque serialisation du graphe — sauvegarde, export, envoi en
// queue. C'est le moment ou l'etat est certainement complet.
const graphProto = app.graph?.constructor?.prototype;
if (graphProto?.serialize && !graphProto.__loraGuardPatched) {
    graphProto.__loraGuardPatched = true;
    const origSerialize = graphProto.serialize;
    graphProto.serialize = function (...args) {
        try {
            for (const n of this._nodes || []) if (isTarget(n)) saveSnapshot(n);
        } catch (_) {}
        return origSerialize.apply(this, args);
    };
}
