"""
Onyx render profile — per-install sync state.

Deliberately named/worded around "render profile" and "sync" rather than
"license": the goal is a small speed bump against a casual grep or an AI told
"remove the license check", not real security — Python shipped to end users
is always readable by someone determined enough.

Two halves, kept apart on purpose:
  - _sync_with_server() / _write_cache() : only the Onyx Session node calls
    these, on every queue (it's OUTPUT_NODE=True so it always runs). They do
    the network round-trip and persist the result to disk.
  - ensure_profile_ready() : called from every other gated node. It NEVER
    talks to the network and NEVER contains the validity decision itself —
    it only reads what was already decided and cached. Finding this function
    doesn't show how "valid" gets computed.
"""

import os
import json
import time
import uuid

import requests

_HERE = os.path.dirname(__file__)
_NODE_ID_FILE = os.path.join(_HERE, ".render_profile_id")
_CACHE_FILE = os.path.join(_HERE, ".render_profile_cache.json")

_SYNC_URL = "https://onyx-sync.onyx-ai-pro.workers.dev/sync"

_GRACE_SECONDS = 3 * 24 * 3600  # tolerance si le reseau est indisponible au moment du refresh
_REQUEST_TIMEOUT = 8


def _get_node_id() -> str:
    """Identifiant d'installation stable, genere une seule fois."""
    try:
        with open(_NODE_ID_FILE) as fh:
            nid = fh.read().strip()
            if nid:
                return nid
    except OSError:
        pass
    nid = uuid.uuid4().hex
    try:
        with open(_NODE_ID_FILE, "w") as fh:
            fh.write(nid)
    except OSError:
        pass
    return nid


def _read_cache() -> dict:
    try:
        with open(_CACHE_FILE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write_cache(token: str, valid: bool, reason: str, next_check_in: int) -> None:
    data = {
        "token": token,
        "valid": bool(valid),
        "reason": reason,
        "checked_at": time.time(),
        "next_check_in": int(next_check_in),
    }
    try:
        with open(_CACHE_FILE, "w") as fh:
            json.dump(data, fh)
    except OSError:
        pass


def sync_with_server(token: str) -> dict:
    """Appel reseau — UNIQUEMENT depuis le node Onyx Session. Ecrit le cache
    et renvoie {"valid": bool, "reason": str} pour l'affichage dans le node."""
    token = (token or "").strip()
    if not token:
        _write_cache("", False, "empty", 0)
        return {"valid": False, "reason": "empty"}

    node_id = _get_node_id()
    try:
        resp = requests.post(
            _SYNC_URL, json={"token": token, "node_id": node_id},
            timeout=_REQUEST_TIMEOUT,
        )
        data = resp.json()
    except Exception as e:
        # Reseau indisponible : on ne touche PAS au cache existant, pour que
        # la periode de grace (voir ensure_profile_ready) continue a jouer
        # sur la base du dernier check reussi, pas d'un echec reseau.
        return {"valid": False, "reason": f"network_error: {e}"}

    valid = bool(data.get("valid"))
    reason = data.get("reason", "ok" if valid else "invalid")
    next_check_in = data.get("next_check_in", 24 * 3600)
    _write_cache(token, valid, reason, next_check_in)
    return {"valid": valid, "reason": reason}


def ensure_profile_ready() -> None:
    """Appelee en premiere ligne de chaque node gated. Ne fait AUCUN appel
    reseau et ne decide de rien elle-meme — elle relit juste ce que le node
    Onyx Session a deja tranche et cache sur disque."""
    cache = _read_cache()
    if not cache.get("valid"):
        raise RuntimeError(
            "Sync error SS-204: place the Onyx Session node anywhere in this "
            "workflow and run it once, then retry."
        )

    checked_at = cache.get("checked_at", 0)
    next_check_in = cache.get("next_check_in", 0)
    if time.time() - checked_at > next_check_in + _GRACE_SECONDS:
        raise RuntimeError(
            "Sync error SS-204: the Onyx Session node hasn't refreshed in a "
            "while — run it again."
        )
