"""Where inside a big room the caller is: a region (centroid + radius) and a short spoken description of it.

Open-plan "Dining Room / Kitchen / Living Room" spaces in these scans are 30-60 m² with 5-11 sweeps, so one dot at
the best sweep claims more precision than we have, and "the living room" alone leaves responders searching. The
posterior is already over sweeps, so the part of the room is read straight from it:
  - region(): the in-room sweep posterior's weighted centroid and RMS spread (model meters, like the floor plan).
  - zone_label(): "kitchen end, near the stove" / "west side, near the sliding door" from where that centroid sits in
    the north-up room and which landmark the likely sweeps' photos show more than the rest of the room.
  - needs_narrowing(): whether it is worth asking the caller more about where in the room they are.
Pure numpy, no I/O except reading the cached orientation when no yaw is passed.
"""

import math

import numpy as np

from . import config, orientation

# Landmarks worth naming to a responder, with the open-plan area each implies ("kitchen end"). Ordered by how
# fixed and visible they are: a stove or fireplace cannot move, a table can. Only features in followups.FEATURES.
LANDMARKS = {
    "stove": ("the stove", "kitchen", "kitchen end"),
    "fridge": ("the fridge", "kitchen", "kitchen end"),
    "sink": ("the sink", "kitchen", "kitchen end"),
    "fireplace": ("the fireplace", "living", "living area"),
    "sofa": ("the couch", "living", "living area"),
    "tv": ("the TV", "living", "living area"),
    "table": ("the table", "dining", "dining area"),
    "sliding_door": ("the sliding door", None, None),
    "front_door": ("the front door", None, None),
    "stairs": ("the stairs", None, None),
    "bed": ("the bed", None, None),
    "washer": ("the washer", None, None),
    "window": ("the window", None, None),
}


def _room_points(room: dict) -> np.ndarray:
    """Outline points of a room: its polygon, else its wall endpoints (some rooms' walls don't close a polygon)."""
    points = room.get("polygon") or [p for w in room.get("walls") or [] for p in (w[:2], w[2:])]
    return np.array(points, dtype=float).reshape(-1, 2)


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray | None:
    """Area centroid (shoelace). Not the vertex mean: an L-shaped room has more vertices around its corner."""
    if len(polygon) < 3:
        return None
    x, y = polygon.T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return None
    return np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6 * area)


def _polygon_area(polygon: np.ndarray) -> float | None:
    if len(polygon) < 3:
        return None
    x, y = polygon.T
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def region(posterior: np.ndarray, building, room: int) -> dict:
    """The part of `room` the caller is probably in: {x, y, radius_m, sweeps, area_m2, half_diagonal_m}.
    Sweeps carry their probability renormalised within the room, so they sum to 1 whatever the room's own share."""
    cands = np.flatnonzero(building.candidate_room == room)
    xy = np.array([[building.candidates[c]["x"], building.candidates[c]["y"]] for c in cands], dtype=float)
    outline = _room_points(building.rooms[room])
    extent_points = outline if len(outline) else xy
    half_diagonal = (float(np.linalg.norm(extent_points.max(axis=0) - extent_points.min(axis=0)) / 2)
                     if len(extent_points) else config.REGION_MIN_RADIUS_M)
    # A room is at least as wide as the minimum radius, so the cap can never undercut the floor.
    cap = max(half_diagonal, config.REGION_MIN_RADIUS_M)

    weights = posterior[cands].astype(float) if len(cands) else np.zeros(0)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(cands), 1.0 / max(len(cands), 1))

    if len(cands) <= 1:
        # One sweep (or one image-less centroid candidate) says nothing about where in the room: the whole room.
        center = _polygon_centroid(outline) if len(outline) >= 3 else None
        if center is None:
            center = np.array(building.rooms[room].get("centroid") or (xy[0] if len(xy) else outline.mean(axis=0)))
        spread = float(np.linalg.norm(outline - center, axis=1).max()) if len(outline) else 0.0  # circumradius
    else:
        center = weights @ xy
        spread = float(np.sqrt(weights @ ((xy - center) ** 2).sum(axis=1)))   # weighted RMS distance
    radius = min(max(spread, config.REGION_MIN_RADIUS_M), cap)
    area = _polygon_area(outline)
    return {
        "x": round(float(center[0]), 3), "y": round(float(center[1]), 3), "radius_m": round(radius, 2),
        "sweeps": [{"candidate": int(c), "x": building.candidates[c]["x"], "y": building.candidates[c]["y"],
                    "p": round(float(w), 4)} for c, w in sorted(zip(cands, weights), key=lambda t: -t[1])],
        "area_m2": None if area is None else round(area, 1),
        "half_diagonal_m": round(half_diagonal, 2),
    }


def needs_narrowing(room_area: float | None, region: dict) -> bool:
    """Worth asking where in the room? Only in a big room (config.BIG_ROOM_M2) whose region is still wide
    (config.NARROW_RADIUS_M): in a bedroom-sized room any door gets responders to the caller within seconds."""
    area = room_area if room_area is not None and not math.isnan(room_area) else region.get("area_m2")
    if area is None:
        return False
    return area >= config.BIG_ROOM_M2 and region["radius_m"] > config.NARROW_RADIUS_M


def _north_up(xy: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Model meters -> east/north, the same rotation as FloorPlan.jsx makeProjector and orientation.model_to_enu."""
    t = math.radians(yaw_deg)
    c, s = math.cos(t), math.sin(t)
    return np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], axis=1)


def _side(building, room: int, centroid: tuple[float, float], yaw_deg: float) -> str | None:
    """'west side', 'northeast corner', or None near the middle. An axis shorter than ZONE_MIN_AXIS_M is ignored:
    'the north end' of a 2 m wide strip is not a place anyone can find."""
    outline = _room_points(building.rooms[room])
    if len(outline) < 2:
        return None
    points = _north_up(outline, yaw_deg)
    point = _north_up(np.array([centroid], dtype=float), yaw_deg)[0]
    low, high = points.min(axis=0), points.max(axis=0)
    span = high - low
    # Position in the bounding box, -0.5 (west/south edge) .. 0.5 (east/north edge).
    rel = np.where(span >= config.ZONE_MIN_AXIS_M, (point - low) / np.maximum(span, 1e-9) - 0.5, 0.0)
    east_west = ("east" if rel[0] > 0 else "west") if abs(rel[0]) >= config.ZONE_SIDE_FRACTION else ""
    north_south = ("north" if rel[1] > 0 else "south") if abs(rel[1]) >= config.ZONE_SIDE_FRACTION else ""
    if east_west and north_south:
        return f"{north_south}{east_west} corner"
    if east_west or north_south:
        return f"{east_west or north_south} side"
    return None


def _landmark(building, room: int, centroid: tuple[float, float], features, posterior) -> str | None:
    """Landmark key whose P(yes) is highest at the likely sweeps compared with the room as a whole. Needs 2+
    photographed sweeps: one sweep's photos describe the whole room, not a part of it."""
    cands = np.flatnonzero((building.candidate_room == room) & building.has_views)
    if len(cands) < 2:
        return None
    if posterior is not None and posterior[cands].sum() > 0:
        weights = posterior[cands] / posterior[cands].sum()
    else:  # no posterior given: the sweeps nearest the centroid are taken as the likely ones
        xy = np.array([[building.candidates[c]["x"], building.candidates[c]["y"]] for c in cands], dtype=float)
        weights = np.exp(-((xy - np.asarray(centroid)) ** 2).sum(axis=1) / 2)
        weights = weights / weights.sum()
    best, best_contrast = None, config.ZONE_LANDMARK_MIN_CONTRAST
    for key in LANDMARKS:  # dict order breaks ties toward the more fixed landmark
        p = features.p_yes.get(key)
        if p is None:
            continue
        contrast = float(weights @ p[cands] - p[cands].mean())
        if contrast > best_contrast:
            best, best_contrast = key, contrast
    return best


def zone_label(building, room: int, centroid: tuple[float, float], features, posterior: np.ndarray | None = None,
               yaw_deg: float | None = None) -> str | None:
    """Short description of where in the room, e.g. "kitchen end, near the stove" or "west side, near the sliding
    door"; None when nothing distinguishes a part of the room. Compass sides are only used with a known yaw (from
    orientation.json when not passed): the model's own axes are arbitrary, and a wrong "west" sends crews the
    wrong way, so without north only the landmark is named."""
    if yaw_deg is None:
        try:
            yaw_deg = (orientation.load(building.id) or {}).get("yaw_deg")
        except (OSError, ValueError):
            yaw_deg = None
    landmark = _landmark(building, room, centroid, features, posterior)
    area = None
    if landmark is not None:
        kind, area_phrase = LANDMARKS[landmark][1:]
        room_kinds = " ".join(building.rooms[room].get("types") or []).lower()
        # "kitchen end" only means something in a room that is also something else (an open-plan space).
        if kind and kind in room_kinds and len(building.rooms[room].get("types") or []) > 1:
            area = area_phrase
    where = area or (_side(building, room, centroid, yaw_deg) if yaw_deg is not None else None)
    parts = [where] if where else []
    if landmark is not None:
        parts.append(f"near {LANDMARKS[landmark][0]}")
    return ", ".join(parts) or None
