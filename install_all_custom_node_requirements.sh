#!/usr/bin/env bash
# Onyx — install every custom node's requirements (macOS / Linux)
#
# Companion to install_all_custom_node_requirements.bat. Same job, but the
# assumptions are different enough that this is not a translation:
#
#   Windows  ComfyUI ships as "portable" with its own python_embeded, so the
#            interpreter is found by walking up until that folder appears.
#   macOS    there is no python_embeded. ComfyUI is a cloned repo run from a
#            virtual environment, or the desktop app. So the interpreter has to
#            be discovered, and — more importantly — the script must refuse to
#            install into the system Python, which macOS protects and which pip
#            will reject with "externally-managed-environment" anyway.
#
# Usage:   ./install_all_custom_node_requirements.sh [--force] [--dry-run]
#          --force    reinstall this pack's packages even if already satisfied
#          --dry-run  print what would be installed, install nothing

set -uo pipefail

# Le nom du pack se deduit du dossier ou vit ce script, il n'est pas ecrit en
# dur : renommer le dossier - ou le cloner sous un autre nom - ne doit pas
# desactiver silencieusement le traitement particulier reserve a ce pack.
PACK_NAME="$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)")"
FORCE=0
DRY=0
for arg in "$@"; do
    case "$arg" in
        --force)   FORCE=1 ;;
        --dry-run) DRY=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '%s\n' "$*"; }
rule() { printf '============================================================\n'; }

# ── 1. Locate custom_nodes ───────────────────────────────────────────────────
# The script normally sits inside it, so walking up is enough. Resolving the
# real path first matters because macOS users often launch this through a
# symlink from the Desktop app's bundle.
SELF="${BASH_SOURCE[0]}"
while [ -L "$SELF" ]; do SELF="$(readlink "$SELF")"; done
DIR="$(cd "$(dirname "$SELF")" && pwd -P)"

CUSTOM_NODES=""
probe="$DIR"
for _ in 1 2 3 4 5 6; do
    if [ -d "$probe/custom_nodes" ]; then CUSTOM_NODES="$probe/custom_nodes"; break; fi
    if [ "$(basename "$probe")" = "custom_nodes" ]; then CUSTOM_NODES="$probe"; break; fi
    probe="$(dirname "$probe")"
    [ "$probe" = "/" ] && break
done

if [ -z "$CUSTOM_NODES" ]; then
    say "[ERROR] Could not find a custom_nodes folder above:"
    say "        $DIR"
    say "Put this script inside ComfyUI/custom_nodes/$PACK_NAME/ and run it again."
    exit 1
fi
COMFY_ROOT="$(dirname "$CUSTOM_NODES")"

# ── 2. Locate the interpreter ────────────────────────────────────────────────
# Order matters: an already-active venv wins, because that is the one ComfyUI
# was most likely started from. A venv sitting next to ComfyUI comes next.
# Bare python3 is last and only with consent — see the guard below.
PYEXE=""
PY_SOURCE=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYEXE="$VIRTUAL_ENV/bin/python"; PY_SOURCE="active virtualenv"
else
    for cand in "$COMFY_ROOT/venv" "$COMFY_ROOT/.venv" "$COMFY_ROOT/../venv" "$COMFY_ROOT/../.venv"; do
        if [ -x "$cand/bin/python" ]; then
            PYEXE="$cand/bin/python"; PY_SOURCE="venv at $cand"; break
        fi
    done
fi
if [ -z "$PYEXE" ] && command -v python3 >/dev/null 2>&1; then
    PYEXE="$(command -v python3)"; PY_SOURCE="python3 on PATH"
fi
if [ -z "$PYEXE" ]; then
    say "[ERROR] No Python found. Install Python 3, or activate the environment"
    say "        you run ComfyUI from, then run this again."
    exit 1
fi

# Installing into the Python that macOS ships breaks it, and Homebrew's Python
# refuses outright (PEP 668). Both are worth stopping for rather than letting
# pip fail forty lines later.
IN_VENV="$("$PYEXE" -c 'import sys; print(int(sys.prefix != sys.base_prefix))' 2>/dev/null || echo 0)"
if [ "$IN_VENV" != "1" ]; then
    say ""
    say "[WARNING] $PYEXE is not a virtual environment."
    if [ "$PYEXE" = "/usr/bin/python3" ] && [ "$(uname -s)" = "Darwin" ]; then
        say "          This is the Python that ships with macOS. Installing into it"
        say "          can break system tools, and pip will most likely refuse."
    elif [ "$PYEXE" = "/usr/bin/python3" ]; then
        say "          This is the system Python. Most distributions mark it"
        say "          externally managed, and pip will refuse (PEP 668)."
    else
        say "          Packages would land in a shared environment."
    fi
    say ""
    say "          Activate the environment you start ComfyUI from, then re-run:"
    say "              source /path/to/venv/bin/activate"
    say ""
    printf "          Continue anyway? [y/N] "
    read -r reply
    case "$reply" in [yY]*) ;; *) say "Aborted."; exit 1 ;; esac
fi

rule
say " Onyx — custom node requirements installer"
rule
say " Python:       $PYEXE"
say "               ($PY_SOURCE, $("$PYEXE" --version 2>&1))"
say " custom_nodes: $CUSTOM_NODES"
[ "$FORCE" = 1 ] && say " Mode:         --force (reinstall $PACK_NAME's packages)"
[ "$DRY" = 1 ]   && say " Mode:         --dry-run (nothing will be installed)"
rule
say ""

# ── 3. Clone the extra node the pack expects ─────────────────────────────────
clone_node() {
    local url="$1" name="$2"
    if [ -d "$CUSTOM_NODES/$name" ]; then
        say "[skip] $name already present"; return 0
    fi
    if ! command -v git >/dev/null 2>&1; then
        say "[warn] git not found — cannot clone $name"; return 0
    fi
    say ""
    say "=== Cloning $name ==="
    [ "$DRY" = 1 ] && { say "  (dry-run) git clone $url"; return 0; }
    git clone "$url" "$CUSTOM_NODES/$name" || say "[warn] clone of $name failed"
}
clone_node "https://github.com/pythongosssss/ComfyUI-Custom-Scripts" "ComfyUI-Custom-Scripts"

# ── 4. Install every requirements.txt found ──────────────────────────────────
TOTAL=0; OK=0; FAIL=0; FAILED_NODES=""
for d in "$CUSTOM_NODES"/*/; do
    req="${d}requirements.txt"
    [ -f "$req" ] || continue
    name="$(basename "$d")"
    TOTAL=$((TOTAL + 1))
    say ""
    say "=== [$TOTAL] $name ==="

    args=(-m pip install -r "$req")
    # --force-reinstall is deliberately NOT the default here, unlike the .bat.
    # Without --no-deps it also reinstalls transitive dependencies — numpy and
    # protobuf among them — and a numpy swapped underneath an already-installed
    # torch is a broken ComfyUI. Pass --force when you actually need it.
    if [ "$name" = "$PACK_NAME" ] && [ "$FORCE" = 1 ]; then
        args+=(--force-reinstall --no-deps)
        say "    (--force: reinstalling this pack's own packages, dependencies untouched)"
    fi

    if [ "$DRY" = 1 ]; then
        say "    (dry-run) $PYEXE ${args[*]}"
        OK=$((OK + 1)); continue
    fi
    if "$PYEXE" "${args[@]}"; then
        OK=$((OK + 1))
    else
        FAIL=$((FAIL + 1)); FAILED_NODES="$FAILED_NODES $name"
    fi
done

say ""
rule
say " Done.   processed=$TOTAL   ok=$OK   failed=$FAIL"
[ "$FAIL" -gt 0 ] && say " Failed:$FAILED_NODES"
rule

# ── 5. What actually matters: can the pack be imported? ──────────────────────
# A green pip run is not the same as a working pack. These two modules are
# imported at load time, so a missing one stops the node from registering at
# all — and ComfyUI reports that as a bare traceback at startup, far from here.
say ""
say "Checking the imports this pack needs at load time:"
for mod in "google.genai" "cv2"; do
    if "$PYEXE" -c "import $mod" >/dev/null 2>&1; then
        say "  OK       $mod"
    else
        say "  MISSING  $mod   <- the pack will not load"
    fi
done
say ""
say "Optional (a missing one only breaks its own node, at run time):"
for mod in mediapipe colour pywt imageio fal_client gallery_dl; do
    if "$PYEXE" -c "import $mod" >/dev/null 2>&1; then
        say "  OK       $mod"
    else
        say "  missing  $mod"
    fi
done
say ""
say "Restart ComfyUI to pick up the new packages."
