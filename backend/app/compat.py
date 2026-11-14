"""How well a Matterport model suits a live demo of the locator: a 0-100 score, the checks behind it, and problems.

Usage: python -m app.compat <model_id> [<model_id> ...]   (scores cached data/buildings/<id>/building.json)

The checks follow the README's "Evaluation": calls fail on identical rooms, rooms without photos, labels that
contradict photos, and vacant houses. So a good demo model has most rooms photographed, every room labelled, many
distinct room types, few look-alike rooms, and furniture. Two inputs share one scorer (`score_rooms`):
  - score_building(): a cached building.json from ingest.py (already filtered: no Void rooms, no outdoor regions).
  - score_model(): a raw GraphQL `model` from scripts/rank_demo_models.py's light query (no skybox URLs). It applies
    ingest.py's room filtering rules first, so both see the same rooms.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict

from . import ingest
from .followups import _polygon_area

# Rooms bigger than this are usually open-plan or commercial floors. The dot then says little ("somewhere in a
# 120 m² room"), and many sweeps in one room make CLIP noise win (see config.ROOM_KEYWORD_BOOST). The sample's
# kitchen/dining/living room is ~60 m².
VERY_LARGE_ROOM_M2 = 80.0
# Same-type rooms whose floor areas are within 20% of each other look alike to a caller ("a bedroom with a window")
# and to the size question. Chipman Glen's Bedrooms 2 and 3 (13 vs 14 m²) are the README's tied pair.
SIMILAR_AREA_RATIO = 1.2
# A 2-minute demo has time for one floor question. More than 3 floors means a long floor list and more questions.
MAX_DEMO_FLOORS = 3
# Room counts a dispatcher can scan on the plan. Fewer than 4 makes the demo trivial; more than 25 means many
# look-alike closets and slow convergence (3012 Houston Ave has 60).
GOOD_ROOM_COUNT = (5, 25)
# Key room -> substring of a classification ("Half Bath" counts as a bathroom).
KEY_ROOM_TYPES = {"kitchen": "kitchen", "bathroom": "bath", "bedroom": "bedroom"}
# Words in Matterport's generated description that mean an empty house. Empty bedrooms are the README's main tie.
VACANT_WORDS = re.compile(r"\b(vacant|unfurnished|unoccupied|empty|bare|no furniture|devoid of furniture)\b", re.I)
FURNISHED_WORDS = re.compile(r"\b(furnished|furniture|sofa|couch|bed frame|dining table|decor)\b", re.I)


def describe_furnishing(description: str | None) -> bool | None:
    """True if the description reads furnished, False if vacant, None if there is no description or no signal."""
    if not description:
        return None
    if VACANT_WORDS.search(description):
        return False
    return True if FURNISHED_WORDS.search(description) else None


def _area(polygon, dimensions_area=None) -> float:
    area = _polygon_area(polygon)
    if math.isnan(area) and dimensions_area:
        return float(dimensions_area)
    return area


def _kind(room: dict) -> str:
    """What a caller would call the room: its types, or its label, lower-cased. Unlabelled rooms are all 'room'."""
    return " / ".join(sorted(t.lower() for t in room["types"])) or (room.get("label") or "room").lower()


def _lookalike_groups(rooms: list[dict]) -> list[list[dict]]:
    """Groups of 2+ rooms with the same kind and areas within SIMILAR_AREA_RATIO. Rooms of unknown area match any
    same-kind room, because nothing tells them apart. Unlabelled rooms ("Room 1", "Room 2") form one group whatever
    their size: a caller can't name them, and size alone rarely settled a call in the README's evaluation."""
    by_kind = defaultdict(list)
    for room in rooms:
        by_kind[_kind(room)].append(room)
    unlabelled = [r for r in rooms if not r["types"] and not r.get("label")]
    groups = [unlabelled] if len(unlabelled) > 1 else []
    for same in by_kind.values():
        same = [r for r in same if r["types"] or r.get("label")]
        known = sorted((r for r in same if not math.isnan(r["area"])), key=lambda r: r["area"])
        unknown = [r for r in same if math.isnan(r["area"])]
        clusters = []
        for room in known:
            if clusters and room["area"] <= clusters[-1][0]["area"] * SIMILAR_AREA_RATIO:
                clusters[-1].append(room)
            else:
                clusters.append([room])
        if unknown:
            if clusters:
                clusters[0].extend(unknown)
            else:
                clusters.append(unknown)
        groups.extend(c for c in clusters if len(c) > 1)
    return groups


def score_rooms(rooms: list[dict], n_floors: int, geolocation: bool | None = None, floorplan: bool | None = None,
                description: str | None = None) -> dict:
    """Score already-filtered rooms. Each room: {id, name, types, label, floor, area (m², NaN if unknown),
    polygon (bool), photos (number of photographed sweeps)}. Returns {"score", "checks", "problems"}."""
    checks: dict = {"floors": n_floors, "rooms": len(rooms), "geolocation": geolocation, "floorplan": floorplan}
    problems: list[str] = []
    if not n_floors:
        problems.append("no floors")
    if not rooms:
        problems.append("no rooms")
    photographed = [r for r in rooms if r["photos"] > 0]
    if rooms and not photographed:
        problems.append("no room has photos")
    if problems:   # the same cases ingest.UnusableModel rejects: nothing to locate a caller in
        return {"score": 0, "checks": {**checks, "usable": False}, "problems": problems}
    checks["usable"] = True
    n = len(rooms)
    points = 0.0

    # Photo coverage (30): CLIP matching needs photos; photo-less rooms get a neutral score and are often missed.
    coverage = len(photographed) / n
    checks["photo_coverage"] = round(coverage, 2)
    points += 30 * coverage
    unphotographed = [r["name"] for r in rooms if r["photos"] == 0]
    if unphotographed:
        problems.append(f"{len(unphotographed)}/{n} rooms without photos: {', '.join(unphotographed[:6])}")
    # Same-kind rooms where only some have photos: the photographed one wins every "I'm in the bathroom".
    mixed = sorted({_kind(r) for r in rooms if r["photos"] == 0}
                   & {_kind(r) for r in photographed})
    checks["mixed_photo_kinds"] = mixed
    if mixed:
        problems.append(f"some but not all rooms photographed for: {', '.join(mixed)}")
        points -= min(9, 3 * len(mixed))

    # Labels (10): room-type words are the strongest cue (config.ROOM_KEYWORD_BOOST); "Room 3" gives none.
    labelled = sum(1 for r in rooms if r["types"] or r.get("label"))
    checks["label_coverage"] = round(labelled / n, 2)
    points += 10 * labelled / n
    if labelled < n:
        problems.append(f"{n - labelled}/{n} rooms unlabelled")

    # Distinct room types (15): more distinct types, more rooms a caller can name outright.
    types = {t.lower() for r in rooms for t in r["types"]}
    checks["distinct_types"] = len(types)
    points += 15 * min(len(types), 6) / 6
    if len(types) < 3:
        problems.append(f"only {len(types)} distinct room types")

    # Look-alike groups (15): rooms a truthful caller can't tell apart.
    groups = _lookalike_groups(rooms)
    tied = sum(len(g) for g in groups)
    checks["lookalike_groups"] = len(groups)
    checks["lookalike_rooms"] = tied
    points += 15 * (1 - tied / n)
    for group in groups[:4]:
        problems.append("look-alike rooms: " + ", ".join(r["name"] for r in group))

    # Key rooms (10): the demo script's natural rooms ("I'm in the kitchen", "the bathroom", "my bedroom").
    present = {k: any(word in t for t in types) for k, word in KEY_ROOM_TYPES.items()}
    checks["key_rooms"] = present
    points += 10 * sum(present.values()) / len(KEY_ROOM_TYPES)
    missing = [k for k, ok in present.items() if not ok]
    if missing:
        problems.append("no " + "/".join(missing))

    # Polygons (5): the dashboard shades the room; without one it draws walls only.
    with_polygon = sum(1 for r in rooms if r["polygon"])
    checks["polygon_coverage"] = round(with_polygon / n, 2)
    points += 5 * with_polygon / n

    # Geolocation (5): needed for the north-up plan and the street (orientation.py). Unknown gets half.
    points += {True: 5, False: 0, None: 2.5}[geolocation]
    if geolocation is False:
        problems.append("no geolocation (no north-up plan or street)")

    # Floors (5): two floors show off the floor question; more than MAX_DEMO_FLOORS is too long for a demo.
    points += {1: 4, 2: 5, 3: 3}.get(n_floors, 0)
    if n_floors > MAX_DEMO_FLOORS:
        problems.append(f"{n_floors} floors (long for a 2-minute demo)")

    # Room count (5).
    low, high = GOOD_ROOM_COUNT
    if low <= n <= high:
        points += 5
    else:
        problems.append(f"{n} rooms (best {low}-{high})")

    large = [r["name"] for r in rooms if not math.isnan(r["area"]) and r["area"] > VERY_LARGE_ROOM_M2]
    checks["very_large_rooms"] = large
    if large:
        problems.append(f"very large rooms (> {VERY_LARGE_ROOM_M2:.0f} m²): {', '.join(large[:4])}")
        points -= min(9, 3 * len(large))

    # Furnished (penalty 10): furniture is what callers describe; an empty bedroom looks like every other one.
    furnished = describe_furnishing(description)
    checks["furnished"] = furnished
    if furnished is False:
        problems.append("description says vacant/empty")
        points -= 10

    checks["floorplan"] = floorplan   # optional dashboard background only: reported, not scored
    return {"score": int(round(max(0.0, min(100.0, points)))), "checks": checks, "problems": problems}


def building_rooms(building: dict) -> list[dict]:
    """score_rooms() input from a cached building.json (rooms there are already filtered by ingest.py)."""
    photos = Counter(c["room_id"] for c in building.get("candidates") or [] if c.get("views"))
    return [{"id": r["id"], "name": r.get("name") or r["id"], "types": r.get("types") or [], "label": r.get("label"),
             "floor": r.get("floor", 0), "area": _area(r.get("polygon")), "polygon": bool(r.get("polygon")),
             "photos": photos[r["id"]]} for r in building.get("rooms") or []]


def score_building(building_json: dict, orientation_json: dict | None = None, description: str | None = None) -> dict:
    """Score a cached building.json. orientation.json exists only when Matterport had geocoordinates, so its absence
    means "unknown" (it may not have been fetched yet), not "no geolocation"."""
    geolocation = True if orientation_json else None
    return score_rooms(building_rooms(building_json), len(building_json.get("floors") or []), geolocation,
                       bool(building_json.get("floorplans")), description)


def _sweep_photographed(pano: dict) -> bool:
    """A sweep with a skybox. With children (ingest's query) it must have all 6 faces, as ingest.py requires. With
    only `status` (the light query), anything not obviously missing counts, since status values are undocumented."""
    skybox = pano.get("skybox") or {}
    if "children" in skybox:
        return len(skybox.get("children") or []) >= 6
    status = str(skybox.get("status") or "").lower()
    return bool(skybox) and status not in {"", "none", "failed", "error", "missing", "unavailable", "pending",
                                           "processing", "not_available", "notavailable"}


def model_rooms(model: dict) -> tuple[list[dict], int]:
    """score_rooms() input and floor count from a raw GraphQL model, using ingest.download_building's room rules:
    Void-only rooms are dropped; unlabelled regions are dropped in models that label rooms, and kept only if
    photographed in models that label nothing; only floors with kept rooms count."""
    floor_ids = {f["id"] for f in model.get("floors") or []}
    photos = Counter(((p.get("anchor") or {}).get("room") or {}).get("id")
                     for p in model.get("panoLocations") or [] if _sweep_photographed(p))
    labels_rooms = any(r.get("label") or r.get("classifications") for r in model.get("rooms") or [])
    kept = []
    for room in model.get("rooms") or []:
        if (room.get("floor") or {}).get("id") not in floor_ids:
            continue
        types = [c["label"] for c in (room.get("classifications") or [])]
        if types and all(t.lower() in ingest.IGNORED_ROOM_TYPES for t in types):
            continue
        if not types and not room.get("label") and (labels_rooms or not photos[room["id"]]):
            continue
        polygon = ingest._room_polygon(ingest._room_walls(room)) if room.get("boundary") else None
        kept.append({"id": room["id"], "label": room.get("label"), "types": types,
                     "floor": room["floor"]["id"], "area": _area(polygon, (room.get("dimensions") or {}).get("areaFloor")),
                     "polygon": bool(polygon), "photos": photos[room["id"]]})
    names = ingest._room_names(kept)
    for room in kept:
        room["name"] = names[room["id"]]
    return kept, len({r["floor"] for r in kept})


def score_model(model: dict, description: str | None = None) -> dict:
    """Score a raw GraphQL model (see scripts/rank_demo_models.py). Uses model.generatedDescription.summary for the
    vacancy check unless `description` is given."""
    rooms, n_floors = model_rooms(model)
    if "geocoordinates" in model:
        geolocation = (model.get("geocoordinates") or {}).get("latitude") is not None
    else:
        geolocation = None
    floorplans = ((model.get("assets") or {}).get("floorplans"))
    floorplan = None if floorplans is None else bool(floorplans)
    if description is None:
        description = (model.get("generatedDescription") or {}).get("summary")
    return score_rooms(rooms, n_floors, geolocation, floorplan, description)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_ids", nargs="+")
    args = parser.parse_args()
    for model_id in args.model_ids:
        d = ingest.building_dir(model_id)
        building = json.loads((d / "building.json").read_text(encoding="utf-8"))
        orientation = d / "orientation.json"
        result = score_building(building, json.loads(orientation.read_text(encoding="utf-8"))
                                if orientation.exists() else None)
        print(f"{model_id}  {result['score']:3d}  {building.get('address')}")
        for problem in result["problems"]:
            print(f"    - {problem}")


if __name__ == "__main__":
    main()
