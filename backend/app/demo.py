"""Demo buildings: the addresses we show judges, and whether each one is ready to load instantly.

The list lives in the repo-root demo_buildings.json ({address, label, model_id?, note, self_test?, expect_room?}).
A building that is not cached costs minutes of downloading and embedding on its first call, which would sink a
2-minute live demo, so scripts/preflight_demo.py warms every entry beforehand and GET /api/demo reports what is
cached. Everything here reads the local cache only (no network), so the dashboard can call it on page load.
"""

import json
import logging
from pathlib import Path

from . import config, ingest, matterport, orientation

log = logging.getLogger(__name__)

DEMO_FILE = config.REPO_ROOT / "demo_buildings.json"


def load_entries(path: Path | None = None) -> list[dict]:
    """The demo list, or [] if the file is missing or unreadable (the dashboard then shows no picker)."""
    path = path or DEMO_FILE
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("No demo buildings (%s): %s", path, exc)
        return []
    return [e for e in entries if isinstance(e, dict) and e.get("address")]


def _cached_models() -> list[dict]:
    # Same as sessions.cached_models. Not imported from there because sessions loads open_clip/torch, and this module
    # must stay importable (and fast) without them, e.g. in tests.
    models = []
    for path in sorted(config.BUILDINGS_DIR.glob("*/building.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            models.append({"id": data["id"], "name": data["name"], "address": data["address"]})
        except (OSError, ValueError, KeyError) as exc:  # a half-written cache must not hide the other buildings
            log.warning("Skipping unreadable %s: %s", path, exc)
    return models


def compat_score(model_id: str) -> dict | None:
    """compat.score_building on the cached building, if that module exists. Optional: None when missing or failing."""
    try:
        from . import compat
    except ImportError:
        return None
    try:
        building = json.loads((ingest.building_dir(model_id) / "building.json").read_text(encoding="utf-8"))
        return compat.score_building(building, orientation.load(model_id))
    except Exception as exc:  # another team's module: a bug there must not break the demo list
        log.warning("compat.score_building failed for %s: %s", model_id, exc)
        return None


def status(entry: dict, cached: list[dict] | None = None) -> dict:
    """Readiness of one demo building, from the cache only.

    The model is the one a call would pick from the cache (sessions.find_matches puts cached buildings first),
    falling back to the entry's model_id. ready = that model is ingested with the current schema and embeddings, and
    is the entry's model_id when one is listed. A missing orientation (plan drawn model-up) or compat problems are
    reported but don't block."""
    cached = _cached_models() if cached is None else cached
    matches = matterport.rank_matches(entry["address"], cached)
    match = matches[0] if matches else None
    model_id = match["id"] if match else entry.get("model_id")
    problems = []
    wrong_model = bool(entry.get("model_id") and match and match["id"] != entry["model_id"])
    if wrong_model:  # a call would open a different house than the one we rehearsed
        problems.append(f"address matches cached model {match['id']}, not the listed {entry['model_id']}")
    is_cached = bool(model_id) and ingest.is_ingested(model_id)
    has_orientation = is_cached and orientation.path(model_id).exists()
    if not model_id:
        problems.append("not cached, and no model_id listed: run scripts/preflight_demo.py")
    elif not is_cached:
        problems.append("not cached: the first call will download and embed it (minutes)")
    elif not has_orientation:
        problems.append("no orientation cached: the plan will not be north-up")
    if match and not match["exact"]:
        problems.append(f"address is only a {round(match['score'] * 100)}% match to {match['address']}")
    result = {
        "address": entry["address"], "label": entry.get("label") or entry["address"], "note": entry.get("note", ""),
        "model_id": model_id, "ready": is_cached and not wrong_model, "cached": is_cached,
        "orientation": has_orientation, "match_score": match["score"] if match else None, "problems": problems,
    }
    compat = compat_score(model_id) if is_cached else None
    if compat is not None:
        result["score"] = compat.get("score")
        result["problems"] += [str(p) for p in compat.get("problems") or []]
    return result


def statuses(path: Path | None = None) -> list[dict]:
    """Every demo building's readiness, ready ones first (stable within each group)."""
    cached = _cached_models()
    results = [status(entry, cached) for entry in load_entries(path)]
    return sorted(results, key=lambda r: not r["ready"])
