"""The building on the map: rooms and sweeps in lon/lat (GeoJSON), plus a Vexcel aerial photo around it.

The dashboard's map view (and the responders' approach card) draw the Matterport rooms over a real map, so the
dispatcher can see which corner of the actual house the caller is in. Model meters go to lon/lat with the same rigid
transform the north-up plan uses (orientation.model_to_enu), then a local degrees-per-meter scale: over a ~100 m
building the flat-Earth error is millimeters.

Matterport's position can be a few meters off (see matterport.fetch_geo), so the outline may not sit exactly on the
roof in the photo; that is why the old Vexcel overlay was removed from the floor plan. Here the photo is context for
the approach (street, driveway, neighbours), and the API returns bounds so the client can draw both honestly.
"""

import json
import logging
import math
import threading

import httpx
import numpy as np

from . import config, ingest, matterport, orientation

log = logging.getLogger(__name__)

# Vexcel layers, best first: the extract endpoint takes the first one that covers the polygon.
VEXCEL_LAYERS = "urban,urban-r,wide-area,urban-g"
VEXCEL_IMAGE_PX = 1400
# Why aerial imagery is missing for a model (no key, outside coverage, error), kept in memory so each reason is
# logged once. Models that got a 403 aren't asked again this run (coverage doesn't change); other failures are
# retried on the next request. Not written to disk, so a restart with a different key retries everything.
_aerial_unavailable: dict[str, str] = {}
_outside_coverage: set[str] = set()
# One fetch per model at a time: the session prefetch and a dashboard request can arrive together.
_aerial_locks: dict[str, threading.Lock] = {}


def model_to_lonlat(geo: dict, xy) -> np.ndarray:
    """Model meters (N, 2) -> [[lon, lat], ...] using Matterport's geocoordinates (from orientation.cached_geo)."""
    en = orientation.model_to_enu(geo, np.asarray(xy, dtype=float).reshape(-1, 2))
    lat = geo["lat"] + en[:, 1] / orientation.METERS_PER_DEG_LAT
    lon = geo["lon"] + en[:, 0] / (orientation.METERS_PER_DEG_LAT * math.cos(math.radians(geo["lat"])))
    return np.stack([lon, lat], axis=1)


def _load_building(model_id: str) -> dict | None:
    p = ingest.building_dir(model_id) / "building.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _room_points(room: dict) -> list[list[float]]:
    if room.get("polygon"):
        return room["polygon"]
    return [p for w in room.get("walls") or [] for p in ([w[0], w[1]], [w[2], w[3]])]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone chain, counter-clockwise. A hull, not the true footprint: Matterport gives rooms, not the outer
    walls, and a hull is enough to mark "this is the house" on a map."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _ring(lonlat: np.ndarray) -> list[list[float]]:
    ring = [[round(float(lon), 8), round(float(lat), 8)] for lon, lat in lonlat]  # 8 decimals ~ 1 mm
    return ring + ring[:1]  # GeoJSON rings repeat the first point


def _center(geo: dict, building: dict) -> tuple[float, float]:
    """(lon, lat) of the house: the middle of the scanned rooms and sweeps, as in orientation.fetch."""
    points = [p for room in building["rooms"] for p in _room_points(room)]
    points += [[c["x"], c["y"]] for c in building["candidates"]]
    lon, lat = model_to_lonlat(geo, points).mean(axis=0)
    return float(lon), float(lat)


def _geo_or_none(model_id: str) -> dict | None:
    try:
        return orientation.cached_geo(model_id)
    except (matterport.MatterportError, httpx.HTTPError, ValueError, KeyError, OSError) as exc:
        log.warning("No geocoordinates for %s: %s", model_id, exc)
        return None


def building_geojson(model_id: str) -> dict | None:
    """Rooms (polygon, or walls' hull when the walls don't close), the building outline and every sweep in lon/lat.
    None if the building isn't ingested or Matterport has no geolocation for it."""
    building = _load_building(model_id)
    geo = _geo_or_none(model_id) if building else None
    if building is None or geo is None:
        return None

    features, all_points = [], []
    for index, room in enumerate(building["rooms"]):
        points = _room_points(room)
        if len(points) < 3:
            continue
        all_points += points
        # A room without a closed polygon is drawn as the hull of its walls: close enough to click and shade.
        outline = np.array(room["polygon"], dtype=float) if room.get("polygon") else _convex_hull(np.array(points))
        if len(outline) < 3:
            continue
        features.append({"type": "Feature",
                         "geometry": {"type": "Polygon", "coordinates": [_ring(model_to_lonlat(geo, outline))]},
                         "properties": {"kind": "room", "index": index, "name": room.get("name") or room.get("label"),
                                        "floor": room.get("floor", 0), "types": room.get("types") or []}})

    all_points += [[c["x"], c["y"]] for c in building["candidates"]]
    if len(all_points) >= 3:
        hull = _convex_hull(np.array(all_points, dtype=float))
        if len(hull) >= 3:
            features.append({"type": "Feature",
                             "geometry": {"type": "Polygon", "coordinates": [_ring(model_to_lonlat(geo, hull))]},
                             "properties": {"kind": "building"}})

    room_index = {room["id"]: i for i, room in enumerate(building["rooms"])}
    candidates = []
    if building["candidates"]:
        lonlats = model_to_lonlat(geo, [[c["x"], c["y"]] for c in building["candidates"]])
        for i, (c, (lon, lat)) in enumerate(zip(building["candidates"], lonlats)):
            room = room_index.get(c["room_id"])
            candidates.append({"candidate": i, "lat": round(float(lat), 8), "lon": round(float(lon), 8), "room": room,
                               "floor": building["rooms"][room].get("floor", 0) if room is not None else None})

    coords = [p for f in features for p in f["geometry"]["coordinates"][0]] + [[c["lon"], c["lat"]] for c in candidates]
    if not coords:
        return None
    lons, lats = [p[0] for p in coords], [p[1] for p in coords]
    lon, lat = _center(geo, building)
    return {
        "type": "FeatureCollection",
        "center": {"lat": round(lat, 8), "lon": round(lon, 8)},
        # Same meaning as orientation.json's yaw_deg: rotate the model frame by this (counter-clockwise) for north-up.
        "yaw_deg": round(math.degrees(geo["yaw"]), 2),
        "features": features,
        "candidates": candidates,
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],   # [[south, west], [north, east]], Leaflet order
    }


# ---- Vexcel aerial imagery ----------------------------------------------------------------------------------------

def aerial_paths(model_id: str):
    out = ingest.building_dir(model_id)
    return out / "aerial.jpg", out / "aerial.json"


def aerial_reason(model_id: str) -> str | None:
    """Why aerial() returned None for this model, if it did."""
    return _aerial_unavailable.get(model_id)


def _unavailable(model_id: str, reason: str) -> None:
    if _aerial_unavailable.get(model_id) != reason:  # log each reason once, not on every dashboard poll
        log.info("No aerial imagery for %s: %s", model_id, reason)
    _aerial_unavailable[model_id] = reason


def square_bounds(lon: float, lat: float, size_m: float) -> list[list[float]]:
    """[[south, west], [north, east]] of a size_m square (in meters, so not square in degrees) centered on lon/lat."""
    half_lat = size_m / 2 / orientation.METERS_PER_DEG_LAT
    half_lon = size_m / 2 / (orientation.METERS_PER_DEG_LAT * math.cos(math.radians(lat)))
    return [[lat - half_lat, lon - half_lon], [lat + half_lat, lon + half_lon]]


def aerial(model_id: str, size_m: float | None = None) -> dict | None:
    """Vexcel orthophoto of a size_m square around the house, cached as aerial.jpg + aerial.json (bounds). Returns
    {"file", "bounds", "source"} or None when there is no key, the house is outside Vexcel's hackathon coverage
    (403), or anything else fails. Never raises: it runs as a background prefetch and imagery is optional."""
    with _aerial_locks.setdefault(model_id, threading.Lock()):
        return _aerial(model_id, size_m)


def _aerial(model_id: str, size_m: float | None) -> dict | None:
    jpg, meta = aerial_paths(model_id)
    if jpg.exists() and meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    if not config.VEXCEL_API_KEY:
        _unavailable(model_id, "VEXCEL_API_KEY is not set")
        return None
    if model_id in _outside_coverage:
        return None
    size_m = size_m or config.AERIAL_SIZE_M
    try:
        building = _load_building(model_id)
        geo = orientation.cached_geo(model_id) if building else None
        if building is None or geo is None:
            _unavailable(model_id, "building not loaded or no Matterport geolocation")
            return None
        lon, lat = _center(geo, building)
        (south, west), (north, east) = square_bounds(lon, lat, size_m)
        wkt = (f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))")
        response = httpx.post(config.VEXCEL_EXTRACT_URL, timeout=60, headers={"X-API-KEY": config.VEXCEL_API_KEY},
                              json={"wkt": wkt, "srid": 4326, "layer": VEXCEL_LAYERS, "crop": "bounding_box",
                                    "image-format": "jpeg",
                                    "destination-dimensions": f"{VEXCEL_IMAGE_PX},{VEXCEL_IMAGE_PX}",
                                    "attribution": "false"})
        if response.status_code == 403:
            _outside_coverage.add(model_id)
            _unavailable(model_id, "outside Vexcel coverage (403; hackathon keys cover Houston and Los Angeles)")
            return None
        response.raise_for_status()
        # Errors can come back as JSON with a 200; only a real JPEG (FF D8 magic) is cached.
        if not response.content.startswith(b"\xff\xd8"):
            _unavailable(model_id, f"Vexcel returned {response.headers.get('content-type', 'no content type')}")
            return None
        jpg.write_bytes(response.content)
        result = {"file": jpg.name, "bounds": [[south, west], [north, east]], "size_m": size_m, "source": "Vexcel"}
        meta.write_text(json.dumps(result, indent=1), encoding="utf-8")  # written last: it marks the cache complete
        _aerial_unavailable.pop(model_id, None)
        return result
    except Exception as exc:  # network, Matterport, disk: imagery is optional, so report and move on
        _unavailable(model_id, f"{type(exc).__name__}: {exc}")
        return None
