/**
 * Onyx Group Toggle — one switch for one group.
 *
 * LiteGraph modes: 0 = ALWAYS, 2 = NEVER (mute), 4 = BYPASS.
 */

import { app } from "../../scripts/app.js";

const MODE_ALWAYS = 0;
const MODE_MUTE   = 2;
const MODE_BYPASS = 4;

function groupTitles() {
    return (app.graph?._groups ?? app.graph?.groups ?? []).map(g => g.title).filter(Boolean);
}

function findGroup(title) {
    return (app.graph?._groups ?? app.graph?.groups ?? []).find(g => g.title === title);
}

/**
 * Nodes inside a group.
 *
 * Membership is purely geometric: a group holds whatever currently sits inside
 * its rectangle. recomputeInsideNodes() is the frontend's own way of refreshing
 * that, but it does not exist in every version, and when it is missing `_nodes`
 * comes back empty — the toggle then finds the group and moves nothing, which
 * looks exactly like "it does not work".
 *
 * So the geometric test is done here as a fallback, using LiteGraph's own rule:
 * a node belongs when its top-left corner is inside the group rectangle.
 */
function nodesOf(group) {
    try {
        group.recomputeInsideNodes?.();
        const found = group._nodes ?? group.nodes;
        if (found?.length) return found;
    } catch (e) { /* frontends without the method */ }

    const b = group._bounding ?? group.bounding ?? group.pos?.concat(group.size);
    if (!b || b.length < 4) return [];
    const [gx, gy, gw, gh] = b;

    return (app.graph?._nodes ?? app.graph?.nodes ?? []).filter(n => {
        const [nx, ny] = n.pos ?? [0, 0];
        const titleH = n.constructor?.title_mode === 1 ? 0 : (window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30);
        return nx >= gx && nx <= gx + gw && (ny - titleH) >= gy && (ny - titleH) <= gy + gh;
    });
}

function applyToggle(node) {
    const w = n => node.widgets?.find(x => x.name === n);
    const title = w("group")?.value;
    if (!title) return;

    const group = findGroup(title);
    if (!group) {
        console.warn(`[Onyx Group Toggle] No group titled "${title}" in this workflow.`);
        return;
    }

    const on      = w("enabled")?.value !== false;
    const offMode = w("off_mode")?.value === "bypass" ? MODE_BYPASS : MODE_MUTE;
    const target  = on ? MODE_ALWAYS : offMode;

    let count = 0;
    for (const n of nodesOf(group)) {
        // Le toggle lui-meme est exclu : s'il est pose dans le groupe qu'il
        // pilote, il se couperait avec lui et on ne pourrait plus le rallumer.
        if (n === node) continue;
        n.mode = target;
        count++;
    }

    node.color   = on ? undefined : "#04222b";
    node.bgcolor = on ? undefined : "#1a0505";
    app.graph.setDirtyCanvas(true, true);
    console.log(`[Onyx Group Toggle] "${title}": ${count} node(s) -> ${on ? "active" : w("off_mode")?.value}`);
}

app.registerExtension({
    name: "onyx.GroupToggle",

    nodeCreated(node) {
        if (node.comfyClass !== "OnyxGroupToggle") return;

        const groupW = node.widgets?.find(w => w.name === "group");
        if (groupW) {
            // Le widget est DEJA un combo cote Python ; on ne remplace que sa
            // source de valeurs. Les groupes n'existent qu'a l'ouverture d'un
            // workflow, donc aucune liste figee a l'import ne peut etre juste.
            groupW.options = groupW.options || {};
            groupW.options.values = () => {
                const titles = groupTitles();
                return titles.length ? titles : ["(no group in this workflow)"];
            };
            const prev = groupW.callback;
            groupW.callback = function (...args) {
                prev?.apply(this, args);
                applyToggle(node);
            };
        }

        // Bouton de diagnostic : dit ce que le node voit reellement, ce qui
        // distingue "aucun groupe trouve" de "groupe trouve mais vide" — deux
        // pannes identiques a l'ecran et de causes opposees.
        const dbg = node.addWidget("button", "⟳  Refresh / check", null, () => {
            const titles = groupTitles();
            const title = node.widgets?.find(w => w.name === "group")?.value;
            const g = findGroup(title);
            console.log(`[Onyx Group Toggle] groups in graph: ${JSON.stringify(titles)}`);
            console.log(`[Onyx Group Toggle] selected: ${JSON.stringify(title)} -> ${g ? "found" : "NOT FOUND"}`);
            if (g) console.log(`[Onyx Group Toggle] nodes inside: ${nodesOf(g).length}`);
            applyToggle(node);
            node.setDirtyCanvas(true, true);
        });
        dbg.serialize = false;

        for (const name of ["enabled", "off_mode"]) {
            const w = node.widgets?.find(x => x.name === name);
            if (!w) continue;
            const prev = w.callback;
            w.callback = function (...args) {
                prev?.apply(this, args);
                applyToggle(node);
            };
        }

        // Reapplique l'etat apres chargement du workflow : sans ca, un graphe
        // sauvegarde avec le toggle sur off se rouvre avec le groupe actif,
        // et l'affichage ment sur ce qui va reellement s'executer.
        setTimeout(() => applyToggle(node), 400);
    },
});
