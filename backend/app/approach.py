"""The approach card: which side of the building, which floor, and which outside wall the caller is behind.

Firefighters arrive from the street and see the building from outside, so "Bedroom 2" means little to them. This
turns the current estimate into "Floor 2, rear (north) side, away from Pointer Street", using only data we already
have: the room walls from ingest.py, the model's heading and nearest street from orientation.py, and (if cached) the
model's geolocation for a Street View link.

Outside walls are found per floor. A wall is "shared" where another room on the same floor is right behind it:
either a wall of that room runs along it (adjacent rooms are drawn with a gap of a wall's thickness, so the match
allows WALL_GAP_M) or a point just outside it lies inside that room's outline. What's left is the building outline.
Walls next to unscanned spaces (a closet nobody scanned) also count as outside; that errs toward the wall responders
can actually see.
"""

import json
import logging
import math

import numpy as np

from . import config, ingest, matterport, orientation

log = logging.getLogger(__name__)

WALL_GAP_M = 0.45          # adjacent rooms' walls sit a wall-thickness apart (~0.1-0.3 m in the sample models)
PROBE_M = 0.6              # how far outside a wall to look for a neighbouring room
SAMPLE_M = 0.2             # walls are tested every 20 cm, so a wall that is half shared splits correctly
MIN_PIECE_M = 0.6          # shorter outside pieces (door jambs, drawing noise) are ignored
PARALLEL_COS = math.cos(math.radians(12))
STREET_FACING_DEG = 60     # a wall "faces the street" if its outward bearing is within this of the street's
STREET_REAR_DEG = 120      # ...and is on the "rear" if it points at least this far away from it
STREETVIEW_STANDOFF_M = 18 # Street View viewpoint this far outside the wall; Google snaps to the nearest panorama


# ---- geometry (model meters) -----------------------------------------------------------------------------------

def _inside(point: np.ndarray, polygon: list | None) -> bool:
    """Even-odd point-in-polygon test; False for rooms without a closed outline."""
    if not polygon or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _outward(room: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit normal of wall a-b pointing out of `room`: the side whose nearby point is not inside the room."""
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([d[1], -d[0]])
    mid = (a + b) / 2
    if room.get("polygon"):
        if _inside(mid + 0.05 * n, room["polygon"]) and not _inside(mid - 0.05 * n, room["polygon"]):
            return -n
        if _inside(mid - 0.05 * n, room["polygon"]):
            return n
    center = room.get("centroid")
    if center is not None and float((mid - np.asarray(center)) @ n) < 0:
        return -n
    return n


MIN_USEFUL_WALL_M = 1.5   # an outside wall shorter than this rarely has a window or door to reach the room through

# A wall with more of the house within this distance behind it is not an outside wall, even when the space right
# behind it wasn't scanned (a small gap, a stair void, an unscanned closet). Without it the Pointer St pantry's east
# wall, which faces a 1.5 m unscanned gap and then the kitchen, was "outside" and the card sent responders to the
# street (east) side of a room in the north-west corner. Rooms are rarely more than this far apart inside a house.
BEYOND_M = 6.0


def _house_beyond(point: np.ndarray, normal: np.ndarray, others: list[dict]) -> bool:
    """Is another room of this floor straight out from this wall point, within BEYOND_M?"""
    for distance in np.arange(PROBE_M, BEYOND_M + 1e-9, 0.25):
        probe = point + distance * normal
        if any(_inside(probe, other.get("polygon")) for other in others):
            return True
    return False


def _covered(point: np.ndarray, direction: np.ndarray, normal: np.ndarray, others: list[dict]) -> bool:
    """Is another room right behind this point of a wall?"""
    probe = point + PROBE_M * normal
    for other in others:
        if _inside(probe, other.get("polygon")):
            return True
        for x1, y1, x2, y2 in other["walls"]:
            a, b = np.array([x1, y1]), np.array([x2, y2])
            length = float(np.linalg.norm(b - a))
            if length < 1e-6 or abs(float((b - a) @ direction)) / length < PARALLEL_COS:
                continue
            t = float((point - a) @ (b - a)) / length ** 2
            if -0.02 <= t <= 1.02 and np.linalg.norm(a + t * (b - a) - point) <= WALL_GAP_M:
                return True
    return False


def exterior_walls(rooms: list[dict], floor: int) -> list[dict]:
    """Outside wall pieces of one floor: [{room, a, b, normal}] (model meters, normal pointing outdoors)."""
    on_floor = [i for i, r in enumerate(rooms) if r.get("floor", 0) == floor]
    pieces = []
    for i in on_floor:
        room = rooms[i]
        others = [rooms[j] for j in on_floor if j != i]
        for x1, y1, x2, y2 in room.get("walls") or []:
            a, b = np.array([x1, y1], dtype=float), np.array([x2, y2], dtype=float)
            length = float(np.linalg.norm(b - a))
            if length < MIN_PIECE_M:
                continue
            direction = (b - a) / length
            normal = _outward(room, a, b)
            steps = max(2, int(math.ceil(length / SAMPLE_M)) + 1)
            ts = np.linspace(0, 1, steps)
            outside = [not _covered(p, direction, normal, others) and not _house_beyond(p, normal, others)
                       for p in (a + t * (b - a) for t in ts)]
            # Consecutive outside samples become one piece.
            start = None
            for k, out in enumerate(outside + [False]):
                if out and start is None:
                    start = k
                elif not out and start is not None:
                    pa, pb = a + ts[start] * (b - a), a + ts[k - 1] * (b - a)
                    if np.linalg.norm(pb - pa) >= MIN_PIECE_M:
                        pieces.append({"room": i, "a": pa, "b": pb, "normal": normal})
                    start = None
    return pieces


def _distance(point: np.ndarray, piece: dict) -> tuple[float, float]:
    """Distance from `point` to a wall piece and how far along it (meters from `a`) the closest point is."""
    a, b = piece["a"], piece["b"]
    length = float(np.linalg.norm(b - a))
    t = min(1.0, max(0.0, float((point - a) @ (b - a)) / length ** 2))
    return float(np.linalg.norm(a + t * (b - a) - point)), t * length


# ---- compass, map and wording ----------------------------------------------------------------------------------

def _bearing(vector: np.ndarray, yaw_deg: float) -> float:
    """Compass bearing (clockwise from north) of a model-frame direction. Same rotation as orientation.model_to_enu
    and the dashboard's makeProjector: model axes turned counter-clockwise by the yaw give east/north."""
    east, north = orientation.model_to_enu({"yaw": math.radians(yaw_deg), "tx": 0.0, "ty": 0.0}, vector[None])[0]
    return math.degrees(math.atan2(east, north)) % 360


def _angle_between(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def to_latlon(geo: dict, xy: np.ndarray) -> tuple[float, float]:
    east, north = orientation.model_to_enu(geo, np.asarray(xy, dtype=float)[None])[0]
    lat = geo["lat"] + north / orientation.METERS_PER_DEG_LAT
    lon = geo["lon"] + east / (orientation.METERS_PER_DEG_LAT * math.cos(math.radians(geo["lat"])))
    return lat, lon


def _street_view(geo: dict, wall_point: np.ndarray, normal: np.ndarray, heading: float) -> dict:
    lat, lon = to_latlon(geo, wall_point)
    vlat, vlon = to_latlon(geo, wall_point + STREETVIEW_STANDOFF_M * normal)
    links = {
        "lat": round(lat, 7), "lon": round(lon, 7),
        # No key needed: opens Google Maps' Street View at the nearest panorama, looking back at the wall.
        "streetview_url": (f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={vlat:.7f},{vlon:.7f}"
                           f"&heading={heading:.0f}"),
        "streetview_image_url": None,
    }
    if config.GOOGLE_MAPS_API_KEY:
        # The key goes to the browser in this URL, so it must be a key restricted to the Street View Static API
        # (and ideally to the dashboard's referrer), never a general-purpose one.
        links["streetview_image_url"] = (
            f"https://maps.googleapis.com/maps/api/streetview?size=640x400&location={vlat:.7f},{vlon:.7f}"
            f"&heading={heading:.0f}&fov=80&key={config.GOOGLE_MAPS_API_KEY}")
    return links


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def approach(building, room_index: int, xy, orientation_data: dict | None, geo: dict | None = None,
             window_probability: float | None = None) -> dict | None:
    """Approach card for a caller estimated at model point `xy` in `room_index`. None if the floor has no walls.

    `orientation_data` is orientation.json ({yaw_deg, street}); without it the card has model-frame facts only.
    `geo` is matterport.fetch_geo's result, needed only for the map position and Street View link."""
    room = building.rooms[room_index]
    floor = int(room.get("floor", 0))
    pieces = exterior_walls_cached(building, floor)
    if not pieces:
        return None
    point = np.asarray(xy, dtype=float)
    own = [p for p in pieces if p["room"] == room_index]
    interior = not own
    # A short stub of outside wall (a room poking 1 m past its neighbour) is a notch responders can't use; when the
    # room also has a substantial outside wall, pick among those (Pointer St pantry: its 0.9 m east stub had won
    # over its 4 m north wall because the scan point stood next to the stub).
    long_own = [p for p in own if np.linalg.norm(p["b"] - p["a"]) >= MIN_USEFUL_WALL_M]
    piece = min(long_own or own or pieces, key=lambda p: _distance(point, p)[0])
    distance, along = _distance(point, piece)
    a, b, normal = piece["a"], piece["b"], piece["normal"]
    length = float(np.linalg.norm(b - a))
    wall_point = a + (b - a) * (along / length)

    multi_floor = len(building.floors) > 1
    floor_label = building.floors[floor]["label"] if floor < len(building.floors) else f"Floor {floor + 1}"
    card = {
        "room": room_index, "room_name": room.get("name") or room.get("label"), "floor": floor,
        "floor_label": floor_label, "multi_floor": multi_floor,
        "floor_text": f"{_ordinal(floor + 1)} floor" if multi_floor else "Single floor",
        "interior": interior,
        "wall": {"x1": round(float(a[0]), 3), "y1": round(float(a[1]), 3), "x2": round(float(b[0]), 3),
                 "y2": round(float(b[1]), 3), "length_m": round(length, 1),
                 "normal": [round(float(normal[0]), 3), round(float(normal[1]), 3)],
                 "room": piece["room"], "room_name": building.rooms[piece["room"]].get("name")},
        "distance_m": round(distance, 1),
        "bearing_deg": None, "side": None, "street": None, "faces_street": None, "street_relation": None,
        "window_probability": None if window_probability is None else round(float(window_probability), 2),
        "has_window_likely": None if window_probability is None else bool(window_probability >= 0.5),
        "lat": None, "lon": None, "streetview_url": None, "streetview_image_url": None,
    }

    yaw = (orientation_data or {}).get("yaw_deg")
    if yaw is None:
        where = "an outside wall" if not interior else f"no outside wall; nearest outside wall is {distance:.0f} m away"
        card["headline"] = card["floor_text"]
        card["text"] = f"{card['floor_text']}. Caller's room has {where} (compass direction unknown)."
        return card

    bearing = _bearing(normal, yaw)
    side = orientation.side_of(bearing)
    card["bearing_deg"] = round(bearing, 1)
    card["side"] = side
    street = (orientation_data or {}).get("street")
    relation_text = ""
    if street:
        diff = _angle_between(bearing, street["bearing_deg"])
        card["street"] = street["name"]
        card["faces_street"] = diff <= STREET_FACING_DEG
        card["street_relation"] = "front" if diff <= STREET_FACING_DEG else "rear" if diff >= STREET_REAR_DEG else "side"
        relation_text = {"front": f", facing {street['name']}", "rear": f", away from {street['name']}",
                         "side": f", around the corner from {street['name']}"}[card["street_relation"]]
    # "rear (north)" when the street tells front from back; a plain compass side otherwise.
    label = f"{card['street_relation']} ({side})" if card["street_relation"] in ("front", "rear") else side
    card["headline"] = f"{card['floor_text']} · {label} side"

    # Which end of the wall the caller is nearer, as a compass word ("east end of the north wall").
    end_text = ""
    if length >= 2.0:
        near_end, from_end = (a, along) if along <= length / 2 else (b, length - along)
        far_end = b if near_end is a else a
        end_side = orientation.side_of(_bearing(near_end - far_end, yaw))
        end_text = f", about {max(1, round(from_end))} m from its {end_side} end"
    if interior:
        wall_text = (f"Caller's room has no outside wall (interior room). Nearest outside wall is the {side} wall of "
                     f"{card['wall']['room_name'] or 'the next room'}, about {max(1, round(distance))} m away"
                     f"{end_text}.")
    else:
        wall_text = f"Caller's room has an outside wall on the {side} side{end_text}."
    window_text = ""
    if card["has_window_likely"] is not None and not interior:
        window_text = " Window likely." if card["has_window_likely"] else " Possibly no window."
    card["text"] = f"{card['floor_text']}, {label} side{relation_text}. {wall_text}{window_text}"
    card["summary"] = f"{label} side, {card['floor_text'].lower()}"

    if geo is not None:
        card.update(_street_view(geo, wall_point, normal, (bearing + 180) % 360))
    return card


# ---- caches (snapshots are broadcast on every change, so keep this cheap) -------------------------------------

_exterior: dict[tuple[str, int], list[dict]] = {}
_cards: dict[tuple, dict | None] = {}


def exterior_walls_cached(building, floor: int) -> list[dict]:
    key = (building.id, floor)
    if key not in _exterior:
        _exterior[key] = exterior_walls(building.rooms, floor)
    return _exterior[key]


def geo_path(model_id: str):
    return ingest.building_dir(model_id) / "geo.json"


def load_geo(model_id: str) -> dict | None:
    p = geo_path(model_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ensure_geo(model_id: str) -> dict | None:
    """Matterport's geolocation for the model, via orientation.cached_geo (one shared geo.json, also used by geo.py).
    Never raises: without it the card just has no map position or Street View link."""
    try:
        return orientation.cached_geo(model_id)
    except Exception as exc:  # network, credentials or an odd model: the card works without a map position
        log.warning("No geolocation for %s: %s", model_id, exc)
        return None


def for_estimate(building, room_index: int, candidate: int, window_probability: float | None = None) -> dict | None:
    """Card for the current estimate, cached per (building, room, candidate) and whether orientation/geo exist yet
    (they arrive in the background after the building loads)."""
    orient = orientation.load(building.id) if orientation.path(building.id).exists() else None
    geo = load_geo(building.id)
    key = (building.id, room_index, candidate, orient is not None, geo is not None, window_probability)
    if key not in _cards:
        c = building.candidates[candidate]
        try:
            _cards[key] = approach(building, room_index, (c["x"], c["y"]), orient, geo, window_probability)
        except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:  # odd geometry must not break snapshots
            log.warning("No approach card for %s room %s: %s", building.id, room_index, exc)
            _cards[key] = None
    return _cards[key]
