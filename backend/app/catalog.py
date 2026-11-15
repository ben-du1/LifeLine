"""Every address the hackathon key can reach, for the dashboard's address suggestions.

The track ships models_index.csv: one row per Matterport model the hackathon keys see (14,945), with its address,
segment and pano/room/floor counts. The Model API itself has no address autocomplete, so the dashboard searches this
list instead. It is downloaded once into data/ and read from there afterwards; without it, suggestions are simply
empty and typing a full address still works (the backend searches Matterport when the call starts).
"""

from __future__ import annotations

import csv
import logging
import re
import threading

import httpx

from . import addresses, config

log = logging.getLogger(__name__)

INDEX_URL = ("https://static.matterport.com/hackathon/26.15.39-main-39-gef8d668/assets/quickstart/"
             "models_index.csv")
INDEX_PATH = config.DATA_DIR / "models_index.csv"

# A real street address starts with a house number and a street word ("5317 Saturn St"); the index also holds
# listing titles like "1 Br, 1 Ba" and bare "US", which can't be called in.
STREET = re.compile(r"^\d+[A-Za-z]?\s+[A-Za-z]")

_entries: list[dict] | None = None
_lock = threading.Lock()


def _load() -> list[dict]:
    global _entries
    with _lock:
        if _entries is not None:
            return _entries
        if not INDEX_PATH.exists():
            try:
                INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
                response = httpx.get(INDEX_URL, timeout=60, follow_redirects=True)
                response.raise_for_status()
                INDEX_PATH.write_bytes(response.content)
            except Exception as exc:  # offline or moved: suggestions stay empty, typed addresses still work
                log.warning("Couldn't download the model index for address suggestions: %s", exc)
                _entries = []
                return _entries
        entries = []
        with INDEX_PATH.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                address = row.get("address") or ""
                street = address.split(",")[0].strip()
                # The city must be part of the address itself: "1 Br, 1 Ba" (a listing title) has a city column but
                # isn't a place anyone could call from.
                if not STREET.match(street) or not row.get("city") or row["city"].lower() not in address.lower():
                    continue
                full = f"{street}, {row['city']}, {row.get('state', '')} {row.get('postal_code', '')}".strip()
                # Matterport's own comma format, which addresses.parse_matterport reads (the display form above
                # puts state and ZIP in one part).
                parts = [street, row["city"], row.get("state") or "", row.get("postal_code") or "", "US"]
                entries.append({
                    "address": full,
                    "model_id": row["sid"],
                    "segment": row.get("segment") or "",
                    "rooms": _int(row.get("room_count")),
                    "floors": _int(row.get("floor_count")),
                    "panos": _int(row.get("pano_count")),
                    "lat": _float(row.get("lat")),   # for the call form's map preview
                    "lon": _float(row.get("lon")),
                    "_key": full.lower(),
                    "_matterport": ", ".join(p for p in parts if p),
                    "_name": row.get("name") or street,
                })
        # Houses first (the app is tuned on homes), then the best-photographed.
        entries.sort(key=lambda e: (e["segment"] != "residential", -(e["panos"] or 0)))
        _entries = entries
        return _entries


def _int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def search(query: str, limit: int = 8) -> list[dict]:
    """Addresses containing every word of the query (any order, case-insensitive), houses first."""
    words = [w for w in re.split(r"[\s,]+", query.lower()) if w]
    if not words:
        return []
    found = []
    for entry in _load():
        if all(w in entry["_key"] for w in words):
            found.append({k: v for k, v in entry.items() if not k.startswith("_")})
            if len(found) >= limit:
                break
    return found


# ---- candidates for resolve.py ------------------------------------------------------------------------------------

_street_index: dict[str, list[int]] | None = None   # street word -> positions in _entries
_indexed: list[dict] | None = None                  # the _entries list the index was built from


def _loaded_without_download() -> list[dict]:
    """The index if it is already on disk (or loaded); never downloads, because resolve() runs while a caller waits."""
    if _entries is None and not INDEX_PATH.exists():
        return []
    try:
        return _load()
    except Exception as exc:  # a corrupt or half-written file: match against the cache and search instead
        log.warning("Couldn't read the model index: %s", exc)
        return []


def _index(entries: list[dict]) -> dict[str, list[int]]:
    global _street_index, _indexed
    with _lock:
        if _indexed is not entries:
            index: dict[str, list[int]] = {}
            for i, entry in enumerate(entries):
                street = addresses.parse_matterport(entry["_matterport"]).street
                for word in set(street):
                    if word not in addresses.DIRECTIONS.values():
                        index.setdefault(word, []).append(i)
            _street_index, _indexed = index, entries
        return _street_index


def _close_words(word: str, vocabulary) -> list[str]:
    """Index words spelled like `word` (same similarity cut as the scorer), so "chapman" still finds "chipman"."""
    try:
        from rapidfuzz import fuzz, process
    except ImportError:  # pragma: no cover
        from difflib import get_close_matches
        return get_close_matches(word, list(vocabulary), n=10, cutoff=addresses.TOKEN_MATCH_MIN)
    found = process.extract(word, vocabulary, scorer=fuzz.ratio, limit=10,
                            score_cutoff=addresses.TOKEN_MATCH_MIN * 100)
    return [match for match, _score, _key in found]


def candidates(address: str, limit: int | None = None) -> list[dict]:
    """Catalog models worth scoring against a caller's address, as [{id, name, address}] in Matterport's format:
    those sharing the most street words (allowing typos), houses first. [] without a downloaded index."""
    entries = _loaded_without_download()
    if not entries:
        return []
    words = [w for w in addresses.parse_caller(address).street if w not in addresses.DIRECTIONS.values()
             and not w.isdigit()]
    if not words:
        return []
    index = _index(entries)
    hits: dict[int, int] = {}
    for word in set(words):
        for close in ([word] if word in index else []) + [w for w in _close_words(word, index.keys()) if w != word]:
            for i in index[close]:
                hits[i] = hits.get(i, 0) + 1
    limit = limit or config.ADDRESS_CATALOG_CANDIDATES
    # Most shared street words first; entries are already sorted houses-first, so position breaks ties.
    best = sorted(hits, key=lambda i: (-hits[i], i))[:limit]
    return [{"id": entries[i]["model_id"], "name": entries[i]["_name"], "address": entries[i]["_matterport"]}
            for i in best]
