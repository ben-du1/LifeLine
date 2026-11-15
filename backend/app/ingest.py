"""Download a Matterport model into data/buildings/<model_id>/ and embed its views.

Usage: python -m app.ingest <model_id> [--no-embed]

Output layout:
  building.json   rooms, candidates (sweeps), floorplan metadata
  views/<sweep>_<face>.jpg   horizontal skybox faces (1-4) per sweep
  floorplan_<floor>.jpg   per floor, only if the model has a raster floorplan
  embeddings.npz  view_embeddings (n_views x d), view_candidate (n_views,)  -- written by embed_building()
"""

import argparse
import json
import re
from tempfile import TemporaryDirectory
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from . import config, matterport

# Matterport skybox faces: 0 = up, 1-4 = horizontal ring, 5 = down. Only the horizontal faces show the room.
HORIZONTAL_FACES = (1, 2, 3, 4)
IGNORED_ROOM_TYPES = {"void"}
# Bumped when building.json changes shape or room filtering changes; 2 = all floors (rooms carry a floor index);
# 3 = unlabeled regions kept only in models that label no rooms.
SCHEMA_VERSION = 3


def building_dir(model_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        raise ValueError("Invalid Matterport model ID")
    return config.BUILDINGS_DIR / model_id


def is_ingested(model_id: str) -> bool:
    """True if the cache is complete and current. Older single-floor caches (no schema) are re-ingested; their
    downloaded images are reused because _download skips files that already exist."""
    return _valid_cache(building_dir(model_id))


def _valid_cache(directory: Path) -> bool:
    try:
        data = json.loads((directory / "building.json").read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA_VERSION or not data["rooms"] or not data["floors"]:
            return False
        owners = []
        room_ids = {room["id"] for room in data["rooms"]}
        if not data["candidates"]:
            return False
        for index, candidate in enumerate(data["candidates"]):
            if candidate["room_id"] not in room_ids:
                return False
            for rel in candidate["views"]:
                if not (directory / rel).is_file():
                    return False
                owners.append(index)
        if any(not (directory / plan["file"]).is_file() for plan in data["floorplans"].values()):
            return False
        with np.load(directory / "embeddings.npz") as arrays:
            vectors = arrays["view_embeddings"]
            return (vectors.ndim == 2 and vectors.shape[0] == len(owners)
                    and np.isfinite(vectors).all() and np.array_equal(arrays["view_candidate"], owners))
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        return False


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    dest.write_bytes(response.content)


def _room_walls(room: dict) -> list[list[float]]:
    """Wall segments [x1, y1, x2, y2] in model meters, from the room's boundary edges. The dashboard draws these as
    its schematic floor plan, because Matterport's raster floorplans are aerial photogrammetry that show the roof."""
    walls = []
    for edge in ((room.get("boundary") or {}).get("edges") or []):
        points = [(v["position"]["x"], v["position"]["y"]) for v in (edge.get("vertices") or [])]
        walls.extend([a[0], a[1], b[0], b[1]] for a, b in zip(points, points[1:]))
    return walls


def _room_polygon(walls: list[list[float]], tolerance: float = 1e-3) -> list[list[float]] | None:
    """Chain wall segments end-to-end into one closed outline (for shading the room on the dashboard). Returns None
    if the segments don't form a single chain; the dashboard then draws walls only."""
    close = lambda a, b: abs(a[0] - b[0]) < tolerance and abs(a[1] - b[1]) < tolerance
    remaining = [((w[0], w[1]), (w[2], w[3])) for w in walls]
    # Some rooms have stub walls that dead-end inside the room (seen in the sample's kitchen and a bathroom).
    # A dead end is an endpoint no other segment touches; drop those segments until only the loop is left.
    pruned = True
    while pruned and remaining:
        pruned = False
        for i, (a, b) in enumerate(remaining):
            others = remaining[:i] + remaining[i + 1:]
            if not all(any(close(p, q) for seg in others for q in seg) for p in (a, b)):
                remaining.pop(i)
                pruned = True
                break
    if not remaining:
        return None
    start, current = remaining.pop(0)
    polygon = [start, current]
    while remaining:
        for i, (a, b) in enumerate(remaining):
            if close(a, current) or close(b, current):
                current = b if close(a, current) else a
                remaining.pop(i)
                polygon.append(current)
                break
        else:
            return None
    return [list(p) for p in polygon[:-1]] if close(polygon[-1], start) else None


def _room_centroid(walls: list[list[float]]) -> list[float] | None:
    points = {(w[0], w[1]) for w in walls} | {(w[2], w[3]) for w in walls}
    if not points:
        return None
    xs, ys = zip(*points)
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _room_names(rooms: list[dict]) -> dict[str, str]:
    """Readable, unique names like 'Bedroom 1', 'Bedroom 2', 'Dining Room / Kitchen / Living Room'."""
    base = {r["id"]: (r["label"] or " / ".join(r["types"]) or "Room") for r in rooms}
    totals = Counter(base.values())
    seen: Counter = Counter()
    names = {}
    for r in rooms:
        name = base[r["id"]]
        if totals[name] > 1:
            seen[name] += 1
            name = f"{name} {seen[name]}"
        names[r["id"]] = name
    return names


class UnusableModel(RuntimeError):
    """The model has no interior we can locate a caller in (no rooms or no photos)."""


def _pick_floorplan(floorplans: list[dict], floor_id: str) -> dict | None:
    # Prefer the plain photographic JPG (no measurement/colour overlays) for the chosen floor. It is only an optional
    # dashboard background, so a model without one is still usable.
    on_floor = [f for f in floorplans if (f.get("floor") or {}).get("id") == floor_id and f.get("resolution")]
    for fp in on_floor:
        if fp["format"] == "jpg" and fp["flags"] == ["photogramy"]:
            return fp
    return on_floor[0] if on_floor else None


def download_building(model_id: str, *, destination: Path | None = None) -> dict:
    model = matterport.fetch_model(model_id)
    if model is None:
        raise UnusableModel(f"model {model_id} not found")
    floors = sorted(model.get("floors") or [], key=lambda f: f["sequence"])
    if not floors:
        raise UnusableModel("model has no floors")
    floor_of = {f["id"]: i for i, f in enumerate(floors)}
    out = destination or building_dir(model_id)
    (out / "views").mkdir(parents=True, exist_ok=True)

    # Rooms that contain at least one sweep with a full skybox (photographed from inside).
    photographed = Counter(((p.get("anchor") or {}).get("room") or {}).get("id")
                           for p in model["panoLocations"] or []
                           if len(((p.get("skybox") or {}).get("children")) or []) >= 6)
    # Does this model label its rooms at all? Some (many townhomes) label none.
    labels_rooms = any(r.get("label") or r.get("classifications") for r in model["rooms"] or [])

    rooms = []
    for room in model["rooms"] or []:
        floor_id = (room.get("floor") or {}).get("id")
        if floor_id not in floor_of:
            continue
        types = [c["label"] for c in (room.get("classifications") or [])]
        if types and all(t.lower() in IGNORED_ROOM_TYPES for t in types):
            continue
        # Unlabeled regions: in a model that labels its rooms they were outdoor areas (the sample's yard and
        # driveway, which do have outdoor sweeps), so they are dropped. In a model that labels nothing (many
        # townhomes), every region is unlabeled, so the photographed ones are kept as "Room 1", "Room 2", ...
        if not types and not room.get("label") and (labels_rooms or not photographed[room["id"]]):
            continue
        walls = _room_walls(room)
        rooms.append({"id": room["id"], "label": room.get("label"), "types": types, "floor": floor_of[floor_id],
                      "walls": walls, "polygon": _room_polygon(walls), "centroid": _room_centroid(walls)})

    # Keep only floors that have rooms, renumbered from 0 (lowest) upward.
    used = sorted({r["floor"] for r in rooms})
    renumber = {old: new for new, old in enumerate(used)}
    for room in rooms:
        room["floor"] = renumber[room["floor"]]
    floors = [floors[i] for i in used]
    names = _room_names(rooms)
    for room in rooms:
        room["name"] = names[room["id"]]
    room_ids = {r["id"] for r in rooms}

    candidates, downloads = [], []
    for pano in model["panoLocations"] or []:
        anchor = pano.get("anchor") or {}
        room_id = (anchor.get("room") or {}).get("id")
        if room_id not in room_ids:
            continue
        faces = ((pano.get("skybox") or {}).get("children")) or []
        if len(faces) < 6:
            continue
        views = []
        for face in HORIZONTAL_FACES:
            rel = f"views/{pano['id']}_{face}.jpg"
            downloads.append((faces[face], out / rel))
            views.append(rel)
        candidates.append({"id": pano["id"], "room_id": room_id,
                           "x": pano["position"]["x"], "y": pano["position"]["y"], "views": views})

    # Rooms with no sweeps get one image-less candidate at their centroid, so room-type words can still pick them.
    rooms_with_sweeps = {c["room_id"] for c in candidates}
    for room in rooms:
        if room["id"] not in rooms_with_sweeps and room["centroid"]:
            candidates.append({"id": f"room-{room['id']}", "room_id": room["id"],
                               "x": room["centroid"][0], "y": room["centroid"][1], "views": []})

    if not rooms or not any(c["views"] for c in candidates):
        raise UnusableModel(f"model {model_id} has no rooms with photos")

    floorplans = {}
    for index, floor in enumerate(floors):
        fp = _pick_floorplan(((model.get("assets") or {}).get("floorplans")) or [], floor["id"])
        if fp:
            downloads.append((fp["url"], out / f"floorplan_{index}.jpg"))
            floorplans[str(index)] = {"file": f"floorplan_{index}.jpg", "width": fp["width"],
                                      "height": fp["height"], "resolution": fp["resolution"], "origin": fp["origin"]}

    # Signed URLs expire after ~1 hour, so everything is downloaded now and served locally afterwards.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: _download(*job), downloads))

    building = {
        "schema": SCHEMA_VERSION,
        "id": model["id"],
        "name": model["name"],
        "address": (model.get("address") or {}).get("address") or model["name"],
        # Lowest first. Matterport labels are usually just "Floor 1", "Floor 2", ...
        "floors": [{"index": i, "id": f["id"], "label": f.get("label") or f"Floor {i + 1}"}
                   for i, f in enumerate(floors)],
        "floorplans": floorplans,   # floor index (as a string) -> aerial raster metadata, when Matterport has one
        "rooms": rooms,
        "candidates": candidates,
    }
    (out / "building.json").write_text(json.dumps(building, indent=1), encoding="utf-8")
    return building


def embed_building(model_id: str, *, destination: Path | None = None) -> None:
    from . import embeddings  # imported lazily: loading torch is slow and not needed for downloads

    out = destination or building_dir(model_id)
    building = json.loads((out / "building.json").read_text(encoding="utf-8"))
    paths, owners = [], []
    for index, candidate in enumerate(building["candidates"]):
        for rel in candidate["views"]:
            paths.append(out / rel)
            owners.append(index)
    vectors = embeddings.embed_images(paths)
    np.savez(out / "embeddings.npz", view_embeddings=vectors, view_candidate=np.array(owners, dtype=np.int32))


def ingest(model_id: str, embed: bool = True) -> None:
    target = building_dir(model_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{model_id}-", dir=target.parent) as temp:
        staged = Path(temp) / "new"
        previous = Path(temp) / "previous"
        download_building(model_id, destination=staged)
        if embed:
            embed_building(model_id, destination=staged)
            if not _valid_cache(staged):
                raise UnusableModel("Downloaded building cache did not validate")
        elif _valid_cache(target):
            return
        if target.exists():
            target.rename(previous)
        try:
            staged.rename(target)
        except OSError:
            if previous.exists() and not target.exists():
                previous.rename(target)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id")
    parser.add_argument("--no-embed", action="store_true", help="download only")
    args = parser.parse_args()
    ingest(args.model_id, embed=not args.no_embed)
    building = json.loads((building_dir(args.model_id) / "building.json").read_text(encoding="utf-8"))
    imaged = sum(1 for c in building["candidates"] if c["views"])
    print(f"{building['name']}: {len(building['rooms'])} rooms, {imaged} sweeps with views, "
          f"{len(building['candidates']) - imaged} image-less room candidates -> {building_dir(args.model_id)}")


if __name__ == "__main__":
    main()
