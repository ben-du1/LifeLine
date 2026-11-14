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

from . import config

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
