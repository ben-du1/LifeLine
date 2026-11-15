"""Caller address -> Matterport model(s), with a band that says what to do next (contract; owner: workstream A).

Bands:
  match    one model, exact address: load it.
  clarify  one model is close (non-exact): ask "Did you mean <address>?" once.
  choose   two models are about equally close (neighbouring houses, units): ask "Is it <a> or <b>?".
  none     nothing similar enough: ask again, then hand off (handoff.py).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

import httpx

from . import config, matterport

log = logging.getLogger(__name__)


@dataclass
class Resolution:
    band: Literal["match", "clarify", "choose", "none"]
    candidates: list[dict]   # rank_matches entries {id, name, address, score, exact}, best first; 1 for match/clarify,
                             # 2 for choose, [] for none
    cleaned: str             # the address text after cleaning, for logs and the responder screen
    distinguishing: list[str] = field(default_factory=list)  # for "choose": the parts that differ ("13120", "13122")


def cached_models() -> list[dict]:
    """Already-ingested buildings as [{id, name, address}], so they can match without (or before) a search."""
    if not config.BUILDINGS_DIR.exists():
        return []
    models = []
    for path in config.BUILDINGS_DIR.glob("*/building.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        models.append({"id": data["id"], "name": data["name"], "address": data["address"]})
    return models


def find_matches(address: str) -> list[dict]:
    """Matterport models whose address is sufficiently similar to `address`, best first (see addresses.py).
    Cached buildings come first on ties since they load instantly; if the search API is unreachable, the cache alone
    is used so the demo still works offline."""
    models = cached_models()
    cached = matterport.rank_matches(address, models)
    if cached and cached[0]["exact"]:
        # A downloaded building with this exact address: no need to search. The search is 3 GraphQL calls with long
        # timeouts, and on venue Wi-Fi it kept a demo call on "Looking up the address..." for minutes (audit #5).
        return cached
    try:
        models += matterport.search_models(address)
    except (matterport.MatterportError, httpx.HTTPError) as exc:
        log.warning("Matterport search failed, matching against cached buildings only: %s", exc)
    unique, seen = [], set()
    for model in models:
        if model["id"] not in seen:
            seen.add(model["id"])
            unique.append(model)
    return matterport.rank_matches(address, unique)


def resolve(text: str, exclude: set[str] = frozenset()) -> Resolution:
    """Which model(s) a caller's address means. Sync (it may search Matterport): call it in a thread. `exclude` holds
    model ids the caller already rejected. Phase 0 stub: today's find_matches, exact -> match, else clarify."""
    cleaned = text.strip()
    matches = [m for m in find_matches(cleaned) if m["id"] not in exclude]
    if not matches:
        return Resolution("none", [], cleaned)
    top = matches[0]
    return Resolution("match" if top["exact"] else "clarify", [top], cleaned)
