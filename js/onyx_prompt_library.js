/**
 * Onyx Prompt Library — Save button and thumbnail gallery.
 *
 * The gallery writes into the node's ordinary multiline `prompt` widget rather
 * than holding a reference to a library entry. That matters: the value is then
 * serialised with the workflow, so a graph shared with someone else — or opened
 * after the library folder has moved — still carries its prompt.
 */

import { app } from "../../scripts/app.js";

const API = "/onyx/prompt_library";

function widget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function setPrompt(node, text) {
    const w = widget(node, "prompt");
    if (!w) return;
    w.value = text;
    if (w.callback) w.callback(text);
    if (w.inputEl) w.inputEl.value = text;
    app.graph.setDirtyCanvas(true, true);
}

async function readJson(resp) {
    // Lu en texte d'abord : une route absente renvoie "404: Not Found", que
    // JSON.parse signale comme un caractere inattendu en position 3 — un
    // message qui cache completement la vraie cause.
    const raw = await resp.text();
    try {
        return JSON.parse(raw);
    } catch (_) {
        throw new Error(resp.status === 404
            ? "route missing — restart ComfyUI (a browser refresh reloads this button "
              + "but not the Python side)"
            : `HTTP ${resp.status} — ${raw.slice(0, 140)}`);
    }
}

// ── Save ─────────────────────────────────────────────────────────────────────

function addSaveButton(node) {
    if (node._promptSaverReady) return;
    node._promptSaverReady = true;

    const btn = node.addWidget("button", "💾  Save prompt to library", null, async () => {
        const prompt = (widget(node, "prompt")?.value || "").trim();
        if (!prompt) {
            alert("Prompt Saver: the prompt field is empty.");
            return;
        }
        btn.name = "⏳  Saving...";
        node.setDirtyCanvas(true);
        try {
            const resp = await fetch(`${API}/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(node.id),
                    prompt,
                    name: widget(node, "name")?.value || "",
                    tags: widget(node, "tags")?.value || "",
                }),
            });
            const data = await readJson(resp);
            if (!data.success) throw new Error(data.error || resp.statusText);
            const e = data.entry;
            alert(`Saved: "${e.name}"\n\n${e.chars} characters` +
                  (e.thumb ? "\nThumbnail included."
                           : "\nNo thumbnail — run the graph once with an image connected, " +
                             "then save again."));
        } catch (e) {
            alert("Save failed: " + e.message);
            console.error("[Prompt Library]", e);
        } finally {
            btn.name = "💾  Save prompt to library";
            node.setDirtyCanvas(true);
        }
    });
    btn.serialize = false;
}

// ── Gallery ──────────────────────────────────────────────────────────────────

function styleOnce() {
    if (document.getElementById("onyx-pl-style")) return;
    const css = document.createElement("style");
    css.id = "onyx-pl-style";
    css.textContent = `
.apl-back{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;
  display:flex;align-items:center;justify-content:center}
.apl-box{background:#071a21;border:1px solid #082a33;border-radius:10px;
  width:min(1100px,92vw);height:min(760px,88vh);display:flex;flex-direction:column;
  color:#eee;font:13px sans-serif;box-shadow:0 12px 48px rgba(0,0,0,.6)}
.apl-head{display:flex;gap:10px;align-items:center;padding:12px 14px;
  border-bottom:1px solid #082a33}
.apl-head h3{margin:0;color:#5fd8ef;font-size:15px;flex:0 0 auto}
.apl-warn{color:#ffb4a2;font-size:11px;flex:0 0 auto;max-width:40%}
.apl-head input{flex:1;background:#040f14;border:1px solid #082a33;border-radius:6px;
  color:#eee;padding:7px 10px;font:13px sans-serif}
.apl-head button{background:#082a33;border:1px solid #22b8d4;color:#5fd8ef;
  border-radius:6px;padding:7px 14px;cursor:pointer;font:13px sans-serif}
/* min-height:0 est ce qui manquait. Un enfant flex a min-height:auto par
   defaut : il refuse de devenir plus petit que son contenu, donc la grille
   grandissait au lieu de defiler, et les lignes se comprimaient pour tenir.
   C'est ce qui aplatissait les images a chaque "show more". */
.apl-grid{flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;padding:14px;
  display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
  grid-auto-rows:min-content;align-content:start}
.apl-card{background:#040f14;border:1px solid #082a33;border-radius:8px;
  overflow:hidden;cursor:pointer;display:flex;flex-direction:column;position:relative}
.apl-card:hover{border-color:#22b8d4}
/* Hauteur en PIXELS, pas en aspect-ratio.
   Dans une grille en minmax(190px,1fr), la largeur de la carte est resolue
   APRES la mesure de son contenu. Une image en width:100% + aspect-ratio se
   mesure donc a une largeur de zero, donne une hauteur de zero, et la ligne
   s'ecrase. Une hauteur fixe ne depend de rien et ne peut pas se resoudre a
   zero.
   contain plutot que cover : l'image entiere, jamais rognee, quel que soit
   son format. flex:0 0 auto pour qu'elle ne se laisse pas comprimer. */
.apl-card img,.apl-noimg{width:100%;height:240px;object-fit:contain;
  display:block;background:#000;flex:0 0 auto}
.apl-noimg{display:flex;align-items:center;justify-content:center;color:#3a5a63;font-size:11px}
.apl-name{padding:7px 9px;font-weight:600;color:#5fd8ef;font-size:12px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.apl-prev{padding:0 9px 8px;color:#7a9aa3;font-size:11px;line-height:1.35;
  max-height:48px;overflow:hidden}
.apl-tags{padding:0 9px 8px;color:#22b8d4;font-size:10px}
.apl-del{position:absolute;top:6px;right:6px;background:rgba(0,0,0,.75);
  border:1px solid #663;color:#f88;border-radius:5px;width:24px;height:24px;
  cursor:pointer;line-height:1;font-size:13px}
.apl-empty{padding:40px;text-align:center;color:#5a7a83;grid-column:1/-1}
.apl-more{grid-column:1/-1;padding:18px;text-align:center;color:#7a9aa3;font-size:12px}
.apl-status{padding:6px 14px;border-top:1px solid #082a33;color:#7a9aa3;font-size:11px}
`;
    document.head.appendChild(css);
}

/**
 * Ask the server for one thumbnail before drawing 479 <img> tags.
 *
 * A broken <img> tells the user nothing: the browser shows the same empty box
 * whether the route is missing, the id was rejected, or the file is not there.
 * One probe turns that into a sentence, and it costs a single request.
 */
async function probeThumbs(entries) {
    const first = entries.find(e => e.thumb);
    if (!first) return { ok: true, note: "no entry claims a thumbnail" };
    const url = `${API}/thumb?id=${encodeURIComponent(first.id)}`;
    try {
        const r = await fetch(url, { cache: "no-store" });
        if (r.ok) {
            const type = r.headers.get("content-type") || "";
            if (!type.startsWith("image/")) {
                return { ok: false, note: `server answered ${r.status} but sent "${type}", not an image` };
            }
            return { ok: true };
        }
        if (r.status === 404) return { ok: false, note: `404 — the server has no file for id ${first.id}` };
        if (r.status === 400) return { ok: false, note: `400 — the server rejected the id ${first.id}` };
        return { ok: false, note: `HTTP ${r.status} on ${url}` };
    } catch (e) {
        return { ok: false, note: `the request failed (${e.message}) — is the route registered? Restart ComfyUI.` };
    }
}

async function openGallery(node) {
    styleOnce();
    let entries = [];
    try {
        entries = (await readJson(await fetch(`${API}/list`))).entries || [];
    } catch (e) {
        alert("Could not read the library: " + e.message);
        return;
    }
    const probe = await probeThumbs(entries);
    if (!probe.ok) console.warn("[Prompt Library] thumbnails unavailable:", probe.note);

    const back = document.createElement("div");
    back.className = "apl-back";
    back.innerHTML = `
<div class="apl-box">
  <div class="apl-head">
    <h3>Prompt library</h3>
    <span class="apl-warn"></span>
    <input placeholder="search — name, tag or prompt text" />
    <button data-close>Close</button>
  </div>
  <div class="apl-grid"></div>
  <div class="apl-status"></div>
</div>`;
    document.body.appendChild(back);

    const grid = back.querySelector(".apl-grid");
    const status = back.querySelector(".apl-status");
    if (!probe.ok) back.querySelector(".apl-warn").textContent = "⚠ thumbnails: " + probe.note;
    const search = back.querySelector("input");
    // Declare avant close(), qui le capture : `let` n'est pas hisse, et une
    // declaration plus bas ne tient que tant que close n'est pas appele plus tot.
    let sentinelObserver = null;

    const close = () => {
        if (sentinelObserver) { sentinelObserver.disconnect(); sentinelObserver = null; }
        back.remove();
    };

    back.addEventListener("click", e => { if (e.target === back) close(); });
    back.querySelector("[data-close]").addEventListener("click", close);
    // Escape ferme aussi : une modale sans sortie clavier est un piege.
    const onKey = e => { if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); } };
    document.addEventListener("keydown", onKey);

    // Combien de cartes on dessine d'un coup.
    //
    // Sans limite, une bibliotheque de 479 entrees cree 479 <img> dans le meme
    // tour de boucle. loading="lazy" ne differe une image que si le navigateur
    // la juge hors ecran AU MOMENT du calcul de mise en page ; dans une modale
    // qu'on vient d'inserer, ce calcul n'a pas encore eu lieu, et il part
    // souvent des centaines de requetes en meme temps. Le serveur ComfyUI sert
    // ces fichiers sur la meme boucle que le reste, et les dernieres n'arrivent
    // jamais - une case vide, sans erreur nulle part.
    //
    // Et a chaque frappe dans la recherche, tout etait reconstruit.
    const PAGE = 60;
    const MAX_AUTO_LOADS = 4;      // 4 x 60 = de quoi remplir un grand ecran
    let limit = PAGE;
    let autoLoads = 0;
    let lastScrollTop = 0;

    function render(resetLimit = true, manual = false) {
        if (resetLimit) { limit = PAGE; autoLoads = 0; lastScrollTop = 0; }
        const q = search.value.trim().toLowerCase();
        const shown = entries.filter(e => !q
            || e.name.toLowerCase().includes(q)
            || e.prompt.toLowerCase().includes(q)
            || (e.tags || []).some(t => t.toLowerCase().includes(q)));

        grid.innerHTML = "";
        if (!shown.length) {
            grid.innerHTML = `<div class="apl-empty">${entries.length
                ? "Nothing matches that search."
                : "The library is empty. Use a Prompt Saver node to add to it."}</div>`;
            return;
        }
        const page = shown.slice(0, limit);
        for (const e of page) {
            const card = document.createElement("div");
            card.className = "apl-card";
            const when = new Date(e.created * 1000).toLocaleDateString();
            card.innerHTML =
                (e.thumb ? `<img loading="lazy" alt="" src="${API}/thumb?id=${encodeURIComponent(e.id)}">`
                         : `<div class="apl-noimg">no thumbnail</div>`) +
                `<div class="apl-name" title="${e.name.replace(/"/g, "&quot;")}">${e.name}</div>` +
                `<div class="apl-prev">${e.prompt.slice(0, 150).replace(/</g, "&lt;")}…</div>` +
                `<div class="apl-tags">${(e.tags || []).join(" · ")}${e.tags?.length ? " — " : ""}${when} — ${e.chars} ch.</div>` +
                `<button class="apl-del" title="Delete">🗑</button>`;

            // Une image cassee laisse une case vide identique a "pas de
            // vignette". On remplace par la raison, et on la journalise une
            // fois : 479 lignes de console pour la meme cause n'aident personne.
            const img = card.querySelector("img");
            if (img) img.addEventListener("error", () => {
                const ph = document.createElement("div");
                ph.className = "apl-noimg";
                ph.textContent = "thumbnail failed";
                img.replaceWith(ph);
                if (!openGallery._loggedImgError) {
                    openGallery._loggedImgError = true;
                    console.warn("[Prompt Library] a thumbnail failed to load:", img.src);
                }
            });

            card.addEventListener("click", ev => {
                if (ev.target.classList.contains("apl-del")) return;
                setPrompt(node, e.prompt);
                close();
                document.removeEventListener("keydown", onKey);
            });
            card.querySelector(".apl-del").addEventListener("click", async ev => {
                ev.stopPropagation();
                if (!confirm(`Delete "${e.name}" from the library?\nThis cannot be undone.`)) return;
                await fetch(`${API}/delete`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ id: e.id }),
                });
                entries = entries.filter(x => x.id !== e.id);
                render(false);   // garde la position dans la liste apres une suppression
            });
            grid.appendChild(card);
        }

        // Chargement au defilement.
        //
        // Une sentinelle placee apres la derniere carte : des qu'elle entre
        // dans la zone visible de la grille, on ajoute une page. Le seuil
        // rootMargin de 400px la declenche un peu avant qu'elle soit vraiment
        // a l'ecran, pour que les images aient le temps d'arriver.
        //
        // Un IntersectionObserver plutot qu'un écouteur de scroll : le
        // navigateur ne le reveille que lorsque la position change vraiment,
        // au lieu d'executer du code a chaque pixel de molette.
        if (sentinelObserver) { sentinelObserver.disconnect(); sentinelObserver = null; }
        if (shown.length > page.length) {
            const sentinel = document.createElement(manual ? "button" : "div");
            sentinel.className = "apl-more";
            sentinel.textContent = manual
                ? `Show ${Math.min(PAGE, shown.length - page.length)} more (${page.length} of ${shown.length})`
                : `loading… (${page.length} of ${shown.length})`;
            grid.appendChild(sentinel);
            if (manual) {
                sentinel.style.cursor = "pointer";
                sentinel.addEventListener("click", () => { limit += PAGE; render(false, true); });
            } else {
            sentinelObserver = new IntersectionObserver((items) => {
                if (!items.some(i => i.isIntersecting)) return;
                sentinelObserver.disconnect();
                sentinelObserver = null;

                // Garde-fou contre l'emballement.
                //
                // Si les cartes ont une hauteur nulle - un defaut de mise en
                // page, comme celui qui a fait charger 513 entrees d'un coup -
                // la sentinelle reste visible apres chaque ajout et la
                // prochaine page se declenche aussitot. On compte les
                // chargements qui s'enchainent sans que la grille ait defile,
                // et au-dela on rend la main a l'utilisateur.
                if (grid.scrollTop <= lastScrollTop + 4) {
                    autoLoads += 1;
                } else {
                    autoLoads = 0;
                }
                lastScrollTop = grid.scrollTop;

                limit += PAGE;
                if (autoLoads > MAX_AUTO_LOADS) {
                    console.warn("[Prompt Library] stopped auto-loading after "
                        + `${MAX_AUTO_LOADS} pages without scrolling — cards may have `
                        + "collapsed. Showing a button instead.");
                    render(false, true);
                } else {
                    render(false);
                }
            }, { root: grid, rootMargin: "400px" });
            sentinelObserver.observe(sentinel);
            }
        }
        status.textContent = `${page.length} shown of ${shown.length}`
                           + (entries.length !== shown.length ? ` — ${entries.length} in the library` : "");
    }

    // render(true) explicitement : passer `render` directement lui transmet
    // l'evenement comme premier argument, qui vaut vrai par hasard. Ca marche,
    // et personne ne comprend pourquoi en relisant.
    search.addEventListener("input", () => render(true));
    render();
    search.focus();
}

function addGalleryButton(node) {
    if (node._promptGalleryReady) return;
    node._promptGalleryReady = true;
    const btn = node.addWidget("button", "🖼  Browse library", null, () => openGallery(node));
    btn.serialize = false;
}

app.registerExtension({
    name: "onyx.PromptLibrary",

    nodeCreated(node) {
        if (node.comfyClass === "OnyxPromptSaver") {
            setTimeout(() => addSaveButton(node), 200);
        } else if (node.comfyClass === "OnyxPromptGallery") {
            setTimeout(() => addGalleryButton(node), 200);
        }
    },
});
