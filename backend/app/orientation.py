"""Which way the building faces: north, and the street out front (OpenStreetMap).

Firefighters arrive from the street, so the dashboard draws the floor plan north-up with the nearest street on the
side where it really is ("Pointer Street, 15 m east"). Two inputs:
  - Matterport's geocoordinates: the model's heading (rotation) and position on Earth.
  - OpenStreetMap Nominatim reverse geocoding at the house: the nearest named road and its line geometry.
No floor plan is laid over imagery, so a few meters of position error only shifts the street distance a little; it
cannot put the street on the wrong side of a house.

Nominatim was chosen over the Overpass API (which returns every nearby road) because Overpass timed out or was "at
capacity" on every attempt during development, while Nominatim answered in ~1 s. Its usage policy allows light use
with an identifying User-Agent and asks for caching: one request per building, cached in
data/buildings/<id>/orientation.json.
"""

import json
import logging
import math

import httpx
import numpy as np

from . import ingest, matterport

log = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
# Nominatim's policy requires an identifying User-Agent; no contact address is sent.
USER_AGENT = "caller-locator-hackathon/0.1 (HackPrinceton MVP)"
METERS_PER_DEG_LAT = 111_320.0
SIDES = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def path(model_id: str):
    return ingest.building_dir(model_id) / "orientation.json"


def load(model_id: str) -> dict | None:
    p = path(model_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def model_to_enu(geo: dict, xy: np.ndarray) -> np.ndarray:
    """Model meters (N, 2) -> east/north meters around (lat, lon)."""
    c, s = math.cos(geo["yaw"]), math.sin(geo["yaw"])
    return xy @ np.array([[c, s], [-s, c]]) + np.array([geo["tx"], geo["ty"]])


def side_of(bearing_deg: float) -> str:
    return SIDES[round(bearing_deg / 45) % 8]


def nearest_on_line(point: np.ndarray, line: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance from `point` to a polyline (east/north meters), the closest point, and the line's direction there."""
    best = (math.inf, line[0], np.array([1.0, 0.0]))
    for a, b in zip(line[:-1], line[1:]):
        d = b - a
        length2 = float(d @ d)
        t = 0.0 if length2 == 0 else min(1.0, max(0.0, float((point - a) @ d) / length2))
        closest = a + t * d
        dist = float(np.linalg.norm(point - closest))
        if dist < best[0]:
            best = (dist, closest, d / math.sqrt(length2) if length2 else best[2])
    return best


def street_from_nominatim(result: dict, center_lonlat: tuple[float, float]) -> dict | None:
    """Turn a Nominatim reverse result into {name, distance_m, bearing_deg, side, direction_deg}."""
    geometry = result.get("geojson") or {}
    lines = {"LineString": [geometry.get("coordinates")],
             "MultiLineString": geometry.get("coordinates")}.get(geometry.get("type")) or []
    name = result.get("name") or (result.get("address") or {}).get("road")
    if not name or not lines:
        return None
    lon0, lat0 = center_lonlat
    mx = METERS_PER_DEG_LAT * math.cos(math.radians(lat0))
    best = None
    for coords in lines:
        line = np.array([[(lon - lon0) * mx, (lat - lat0) * METERS_PER_DEG_LAT] for lon, lat in coords])
        if len(line) < 2:
            continue
        found = nearest_on_line(np.zeros(2), line)
        if best is None or found[0] < best[0]:
            best = found
    if best is None:
        return None
    dist, closest, direction = best
    bearing = math.degrees(math.atan2(closest[0], closest[1])) % 360   # clockwise from north
    return {
        "name": name,
        "distance_m": round(dist, 1),
        "bearing_deg": round(bearing, 1),
        "side": side_of(bearing),
        # Compass direction the road runs at its closest point (0-180), for drawing the street band.
        "direction_deg": round(math.degrees(math.atan2(direction[0], direction[1])) % 180, 1),
    }


def fetch(model_id: str) -> dict:
    building = json.loads((ingest.building_dir(model_id) / "building.json").read_text(encoding="utf-8"))
    geo = matterport.fetch_geo(model_id)
    if geo is None:
        raise ValueError("Matterport has no geolocation for this model")
    # The house center: middle of the scanned rooms, moved onto the map.
    points = [p for room in building["rooms"] for p in (room.get("polygon") or [])]
    points += [[c["x"], c["y"]] for c in building["candidates"]]
    center_en = model_to_enu(geo, np.array(points, dtype=float)).mean(axis=0)
    lat = geo["lat"] + center_en[1] / METERS_PER_DEG_LAT
    lon = geo["lon"] + center_en[0] / (METERS_PER_DEG_LAT * math.cos(math.radians(geo["lat"])))

    response = httpx.get(NOMINATIM_URL, timeout=15, headers={"User-Agent": USER_AGENT}, params={
        "lat": lat, "lon": lon, "zoom": 17, "format": "jsonv2", "polygon_geojson": 1, "layer": "address"})
    response.raise_for_status()
    result = {
        "source": "OpenStreetMap (Nominatim)",
        # Degrees to rotate the floor plan (model frame) so north is up; counter-clockwise like the yaw.
        "yaw_deg": round(math.degrees(geo["yaw"]), 2),
        "street": street_from_nominatim(response.json(), (lon, lat)),
    }
    path(model_id).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def ensure(model_id: str) -> dict | None:
    """Cached orientation, fetched once. Never raises: the floor plan works without it (drawn model-up)."""
    cached = load(model_id)
    if cached is not None:
        return cached
    try:
        return fetch(model_id)
    except (matterport.MatterportError, httpx.HTTPError, ValueError, KeyError, OSError) as exc:
        log.warning("No orientation for %s: %s", model_id, exc)
        return None
