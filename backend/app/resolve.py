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

from . import addresses, catalog, config, matterport

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


def _unique(models: list[dict]) -> list[dict]:
    unique, seen = [], set()
    for model in models:
        if model["id"] not in seen:
            seen.add(model["id"])
            unique.append(model)
    return unique


def _local_models(address: str) -> list[dict]:
    """Cached buildings (they load instantly) followed by the downloaded catalog's likely rows, deduplicated."""
    return _unique(cached_models() + catalog.candidates(address))


def _with_search(address: str, models: list[dict]) -> list[dict]:
    try:
        models = models + matterport.search_models(address)
    except (matterport.MatterportError, httpx.HTTPError) as exc:
        log.warning("Matterport search failed, matching against cached buildings and the catalog only: %s", exc)
    return _unique(models)


def find_matches(address: str) -> list[dict]:
    """Matterport models whose address is sufficiently similar to `address` (config.ADDRESS_MATCH_THRESHOLD), best
    first (see addresses.py). Cached buildings come first on ties since they load instantly; if the search API is
    unreachable, the cache and catalog alone are used so the demo still works offline."""
    models = _local_models(address)
    local = matterport.rank_matches(address, models)
    if local and local[0]["exact"]:
        # A downloaded building (or catalog row) with this exact address: no need to search. The search is 3 GraphQL
        # calls with long timeouts, and on venue Wi-Fi it kept a demo call on "Looking up the address..." for minutes
        # (audit #5).
        return local
    return matterport.rank_matches(address, _with_search(address, models))


def _same_place(a: dict, b: dict) -> bool:
    """Two models of one address (a re-scan, or the same house listed twice): not a choice worth asking about."""
    pa, pb = addresses.parse_matterport(a["address"]), addresses.parse_matterport(b["address"])
    return (pa.number, pa.street, pa.unit) == (pb.number, pb.street, pb.unit)


def _distinguishing(a: dict, b: dict) -> list[str]:
    """What differs between two models, for "Is it <a> or <b>?": house numbers, else units, else the street lines."""
    pa, pb = addresses.parse_matterport(a["address"]), addresses.parse_matterport(b["address"])
    if pa.number and pb.number and pa.number != pb.number:
        return [pa.number, pb.number]
    if pa.unit != pb.unit:
        return [f"unit {pa.unit}" if pa.unit else "no unit", f"unit {pb.unit}" if pb.unit else "no unit"]
    return [a["address"].split(",")[0].strip(), b["address"].split(",")[0].strip()]


def _band(ranked: list[dict], cleaned: str) -> Resolution:
    if not ranked:
        return Resolution("none", [], cleaned)
    top = ranked[0]
    # The closest model that is a different place from the top one.
    rival = next((m for m in ranked[1:] if not _same_place(top, m)), None)
    close = rival is not None and top["score"] - rival["score"] < config.ADDRESS_CLARIFY_MARGIN
    if top["exact"] and top["score"] >= config.ADDRESS_MATCH_THRESHOLD:
        # The caller said exactly this address. Only another exact one (another unit of the building, when the
        # caller gave none) makes it a question; a near neighbour never outranks the exact house.
        if close and rival["exact"]:
            return Resolution("choose", [top, rival], cleaned, _distinguishing(top, rival))
        return Resolution("match", [top], cleaned)
    if close:
        return Resolution("choose", [top, rival], cleaned, _distinguishing(top, rival))
    return Resolution("clarify", [top], cleaned)


def resolve(text: str, exclude: set[str] = frozenset()) -> Resolution:
    """Which model(s) a caller's address means. Sync (it may search Matterport): call it in a thread. `exclude` holds
    model ids the caller already rejected.

    The text is cleaned first (spoken numbers, filler, "in houston"; addresses.clean_caller). Candidates come from the
    cache and the downloaded catalog, then Matterport's search only when neither has an exact match, so a known
    building resolves in milliseconds without the network."""
    # TODO(geocode): standardize `cleaned` with the Census Geocoder (free, no key) here, and break "choose" ties by
    # distance between the geocoded point and each model's lat/lon.
    cleaned = addresses.clean_caller(text)
    threshold = config.ADDRESS_CLARIFY_THRESHOLD

    def ranked(models: list[dict]) -> list[dict]:
        return [m for m in matterport.rank_matches(cleaned, models, threshold) if m["id"] not in exclude]

    models = _local_models(cleaned)
    local = ranked(models)
    if local and local[0]["exact"] and local[0]["score"] >= config.ADDRESS_MATCH_THRESHOLD:
        return _band(local, cleaned)
    return _band(ranked(_with_search(cleaned, models)), cleaned)
