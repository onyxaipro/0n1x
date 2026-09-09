/**
 * Onyx Resolution (MP) — live preview of the resolution actually produced.
 *
 * The figure comes from the server, not from a copy of the algorithm rewritten
 * here. _snap_to_budget searches a grid and balances two competing errors; a
 * JavaScript twin would drift from it the first time either side is edited, and
 * the node would then advertise a resolution it does not produce — exactly the
 * failure this node exists to prevent.
 *
 * The round trip is to 127.0.0.1, so it costs about a millisecond. Requests are
 * debounced anyway: dragging a slider fires a callback per pixel of travel.
 */

import { app } from "../../scripts/app.js";

const WATCHED = ["megapixels", "aspect_ratio", "multiple_of"];
const DEBOUNCE_MS = 120;

function widget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

async function refresh(node) {
    const preview = node._resMpPreview;
    if (!preview) return;

    const mp = widget(node, "megapixels")?.value;
    const ar = widget(node, "aspect_ratio")?.value;
    const mult = widget(node, "multiple_of")?.value;
    if (mp === undefined || ar === undefined) {
        // Sortir en silence laisserait le champ vide sans dire pourquoi : c'est
        // exactement le cas ou l'on croit que rien ne marche.
        preview.value = "widgets not found";
        console.warn("[Resolution MP] widget names not found on node", node.id,
                     (node.widgets || []).map(w => w.name));
        node.setDirtyCanvas(true);
        return;
    }

    // Un jeton par requete : sur un glissement de slider les reponses peuvent
    // revenir dans le desordre, et la derniere arrivee n'est pas la plus
    // recente demandee. Sans ce test l'affichage se fige sur une valeur perimee.
    const token = (node._resMpToken = (node._resMpToken || 0) + 1);

    try {
        const url = `/onyx/resolution_mp/preview`
            + `?mp=${encodeURIComponent(mp)}`
            + `&ar=${encodeURIComponent(ar)}`
            + `&mult=${encodeURIComponent(mult)}`;
        const resp = await fetch(url);
        const raw = await resp.text();
        if (token !== node._resMpToken) return;
        console.debug("[Resolution MP]", url, "->", resp.status, raw.slice(0, 120));

        let data;
        try {
            data = JSON.parse(raw);
        } catch (_) {
            preview.value = resp.status === 404
                ? "route missing — restart ComfyUI"
                : `HTTP ${resp.status}`;
            node.setDirtyCanvas(true);
            return;
        }
        preview.value = data.success ? data.text : `error: ${data.error}`;
    } catch (e) {
        preview.value = "fetch failed — see console";
        console.error("[Resolution MP]", e);
    }
    node.setDirtyCanvas(true);
}

function attach(node) {
    if (node._resMpReady) return;
    node._resMpReady = true;

    // Volontairement PAS disabled : selon la version du frontend, un widget
    // desactive se dessine en gris sans jamais peindre sa valeur - ce qui donne
    // une ligne vide impossible a diagnostiquer.
    const preview = node.addWidget("text", "output", "computing...", () => {}, {
        serialize: false,
        multiline: false,
    });
    node._resMpPreview = preview;

    let timer = null;
    const schedule = () => {
        clearTimeout(timer);
        timer = setTimeout(() => refresh(node), DEBOUNCE_MS);
    };

    for (const name of WATCHED) {
        const w = widget(node, name);
        if (!w) continue;
        const prev = w.callback;
        w.callback = function (...args) {
            const r = prev?.apply(this, args);
            schedule();
            return r;
        };
    }

    refresh(node);
    node.setSize([node.size[0], node.computeSize()[1]]);
}

app.registerExtension({
    name: "onyx.ResolutionMPPreview",

    nodeCreated(node) {
        if (node.comfyClass !== "OnyxResolutionMP") return;
        // Les widgets Python ne sont pas encore tous poses a cet instant.
        setTimeout(() => attach(node), 200);
    },
});
