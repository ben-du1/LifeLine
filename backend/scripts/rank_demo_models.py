"""Find the Matterport models the demo works best on, in two stages.

Stage 1 (offline, no key): filter the hackathon's models_index.csv (one row per model the key can see) to houses in
Houston / Los Angeles with a street address and a demo-sized scan, give each a rough pre-score, and write a
shortlist. The committed scripts/demo_shortlist.csv is this stage's output, so stage 2 can run without the index.

  .venv/Scripts/python scripts/rank_demo_models.py shortlist models_index.csv [--out scripts/demo_shortlist.csv]

Stage 2 (needs MATTERPORT_TOKEN_ID/SECRET in the repo-root .env): fetch each shortlisted model with a light GraphQL
query (room labels, floors, room outlines, which sweeps have a skybox; no image URLs), score it with app.compat (the
same rules as a cached building.json), and write data/demo_candidates.csv, best first.

  .venv/Scripts/python scripts/rank_demo_models.py check [--shortlist scripts/demo_shortlist.csv] [--limit 60]

Stage 2 is polite and resumable: 4 requests at a time, retries with backoff on 429/5xx, and every model's raw
response is cached in data/demo_rank_cache/<sid>.json (gitignored, as Matterport content is "hackathon use only"),
so a re-run only fetches what is missing and rescoring after a rule change needs no network (--refresh refetches).
Then ingest and try the top few: .venv/Scripts/python -m app.ingest <sid>.
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import addresses, compat, config, matterport  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_SHORTLIST = SCRIPTS_DIR / "demo_shortlist.csv"
CACHE_DIR = config.DATA_DIR / "demo_rank_cache"
DEFAULT_OUT = config.DATA_DIR / "demo_candidates.csv"

# Stage 1 defaults. Panos and rooms in the index include outdoor sweeps and Void/closet regions, so the ranges are
# wider than the ideal indoor counts. The sample house (13122 Chipman Glen) is 36 panos / 13 rooms / 1 floor.
CITIES = {"houston", "los angeles"}
PANO_RANGE = (12, 60)
ROOM_RANGE = (6, 25)
MAX_FLOORS = 3
HOUSE_NUMBER = re.compile(r"^\s*\d+[A-Za-z]?\s+[A-Za-z0-9]")
# "1219 W 25th St - B", "Unit 4": a unit shares its street address with its neighbours, so search may pick another.
UNIT_HINT = re.compile(r"(\s-\s*\w+\s*$|\b(unit|apt|suite|ste)\b|#)", re.I)

WORKERS = 4                 # concurrent GraphQL requests; the hackathon key is shared by many teams
PAUSE_SECONDS = 0.25        # per worker, between requests
RETRIES = 4

# Light query: counts, labels and outlines only. skybox `status` instead of `children` avoids 6 signed URLs per
# sweep. Fields not used elsewhere in the app (status, dimensions, generatedDescription, confidence) come first; if
# the API rejects any of them, FALLBACK_QUERY uses only fields ingest.MODEL_QUERY / GEO_QUERY already rely on.
LIGHT_QUERY = """
query DemoCheck($id: ID!, $res: String!) {
  model(id: $id) {
    id
    name
    address { address }
    floors { id label sequence }
    geocoordinates { latitude }
    generatedDescription { summary }
    rooms {
      id
      label
      classifications { label confidence }
      floor { id }
      dimensions(units: metric) { areaFloor }
      boundary { edges { vertices { position { x y } } } }
    }
    panoLocations { id anchor { room { id } } skybox(resolution: $res) { status } }
    assets { floorplans { format floor { id } } }
  }
}
"""
FALLBACK_QUERY = """
query DemoCheckFallback($id: ID!, $res: String!) {
  model(id: $id) {
    id
    name
    address { address }
    floors { id label sequence }
    geocoordinates { latitude }
    rooms {
      id
      label
      classifications { label }
      floor { id }
      boundary { edges { vertices { position { x y } } } }
    }
    panoLocations { id anchor { room { id } } skybox(resolution: $res) { children } }
    assets { floorplans { format floor { id } } }
  }
}
"""


# ---------------------------------------------------------------------------------------------------------- stage 1

def _street_key(text: str) -> tuple | None:
    parsed = addresses.parse_freeform(text or "")
    return (parsed.number, tuple(parsed.street)) if parsed.number and parsed.street else None


def prescore(row: dict) -> float:
    """0-100 from index columns only. Panos per room stands in for photo coverage (the sample has 2.8)."""
    panos, rooms, floors = int(row["pano_count"]), int(row["room_count"]), int(row["floor_count"] or 1)
    points = 30 * min(panos / max(rooms, 1) / 2.5, 1.0)
    points += 20 if 8 <= rooms <= 16 else 12 if 6 <= rooms <= 20 else 5
    points += {1: 10, 2: 15, 3: 5}.get(floors, 0)              # 2 floors show off the floor question
    points += 10 if row["ai_summary"] == "yes" else 0           # description available for the vacancy check
    points += 10 if row["lat"] and row["lon"] else 0            # geolocated: north-up plan and street
    points += 10 if 20 <= panos <= 45 else 5
    points += 5 if HOUSE_NUMBER.match(row["address"] or "") else 0   # full address field (with city), not just name
    return round(points, 1)


def shortlist(index_path: Path, out: Path, segment: str, limit: int) -> list[dict]:
    with index_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # How many models share each street address: a typed address then matches several models (README "Address
    # matching"), and the demo may open the wrong one.
    shared = Counter(k for k in (_street_key(r["address"] or r["name"]) for r in rows) if k)
    picked = []
    for row in rows:
        address = row["address"] or row["name"]
        if segment != "any" and row["segment"] != segment:
            continue
        if row["city"].strip().lower() not in CITIES:
            continue
        panos, rooms = int(row["pano_count"] or 0), int(row["room_count"] or 0)
        if not (PANO_RANGE[0] <= panos <= PANO_RANGE[1] and ROOM_RANGE[0] <= rooms <= ROOM_RANGE[1]):
            continue
        if int(row["floor_count"] or 1) > MAX_FLOORS or not HOUSE_NUMBER.match(address):
            continue
        key = _street_key(address)
        notes, score = [], prescore(row)
        if key and shared[key] > 1:
            notes.append(f"address shared by {shared[key]} models")
            score -= 15
        if UNIT_HINT.search(address) or addresses.parse_matterport(address).unit:
            notes.append("unit address")
            score -= 5
        if row["ai_summary"] != "yes":
            notes.append("no ai_summary")
        picked.append({"sid": row["sid"], "prescore": round(score, 1), "address": address, "city": row["city"],
                       "segment": row["segment"], "pano_count": panos, "room_count": rooms,
                       "floor_count": row["floor_count"], "panos_per_room": round(panos / rooms, 2),
                       "ai_summary": row["ai_summary"], "geolocated": "yes" if row["lat"] else "no",
                       "notes": "; ".join(notes)})
    picked.sort(key=lambda r: (-r["prescore"], -r["panos_per_room"]))
    picked = picked[:limit]
    _write_csv(out, picked)
    return picked


# ---------------------------------------------------------------------------------------------------------- stage 2

def _fetch(sid: str) -> dict:
    """The model from the light query, falling back to proven fields. Retries on rate limits and server errors."""
    last_error = None
    for query in (LIGHT_QUERY, FALLBACK_QUERY):
        for attempt in range(RETRIES):
            try:
                return matterport.graphql(query, {"id": sid, "res": config.SKYBOX_RESOLUTION})["model"]
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** attempt * 2)
            except httpx.TransportError as error:
                last_error = error
                time.sleep(2 ** attempt * 2)
            except matterport.MatterportError as error:
                last_error = error   # most likely an unknown field: try the fallback query
                break
    raise RuntimeError(str(last_error))


def _cached_model(sid: str, refresh: bool) -> dict:
    path = CACHE_DIR / f"{sid}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        entry = {"model": _fetch(sid), "error": None}
    except RuntimeError as error:
        entry = {"model": None, "error": str(error)[:300]}
    time.sleep(PAUSE_SECONDS)
    if entry["model"] is not None or "not found" in (entry["error"] or "").lower():
        path.write_text(json.dumps(entry), encoding="utf-8")   # transient errors are not cached, so a re-run retries
    return entry


def check(shortlist_path: Path, out: Path, limit: int, refresh: bool) -> list[dict]:
    with shortlist_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:limit]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def one(row: dict) -> dict:
        entry = _cached_model(row["sid"], refresh)
        model = entry["model"]
        if model is None:
            return {**_base(row), "score": -1, "problems": "", "error": entry["error"] or "model not found"}
        result = compat.score_model(model)
        c = result["checks"]
        return {**_base(row), "score": result["score"], "rooms": c.get("rooms"), "floors": c.get("floors"),
                "photo_coverage": c.get("photo_coverage"), "label_coverage": c.get("label_coverage"),
                "distinct_types": c.get("distinct_types"), "lookalike_rooms": c.get("lookalike_rooms"),
                "furnished": c.get("furnished"), "geolocation": c.get("geolocation"),
                "problems": "; ".join(result["problems"]), "error": ""}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, rows))
    results.sort(key=lambda r: (-r["score"], -float(r["prescore"])))
    _write_csv(out, results)
    return results


def _base(row: dict) -> dict:
    return {"sid": row["sid"], "address": row["address"], "prescore": row["prescore"], "notes": row.get("notes", "")}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("shortlist", help="stage 1: filter models_index.csv (offline)")
    s1.add_argument("index", type=Path)
    s1.add_argument("--out", type=Path, default=DEFAULT_SHORTLIST)
    s1.add_argument("--segment", default="residential", choices=["residential", "commercial", "any"])
    s1.add_argument("--limit", type=int, default=150)
    s2 = sub.add_parser("check", help="stage 2: query and score shortlisted models (needs the Matterport key)")
    s2.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    s2.add_argument("--out", type=Path, default=DEFAULT_OUT)
    s2.add_argument("--limit", type=int, default=60)
    s2.add_argument("--refresh", action="store_true", help="refetch models already in the cache")
    args = parser.parse_args()

    if args.stage == "shortlist":
        rows = shortlist(args.index, args.out, args.segment, args.limit)
        print(f"{len(rows)} models -> {args.out}")
        for r in rows[:25]:
            print(f"{r['prescore']:5.1f}  {r['sid']}  {r['address']}  ({r['pano_count']} panos, {r['room_count']} "
                  f"rooms, {r['floor_count']} fl) {r['notes']}")
    else:
        rows = check(args.shortlist, args.out, args.limit, args.refresh)
        print(f"{len(rows)} models -> {args.out}")
        for r in rows[:25]:
            print(f"{r['score']:4d}  {r['sid']}  {r['address']}  {r['error'] or r['problems'][:110]}")


if __name__ == "__main__":
    main()
