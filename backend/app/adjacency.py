"""Where a room is in the house: which rooms share a wall with it, and whether it is at the front, on the street.

Rooms that look alike (three empty bedrooms, two bathrooms) can't be told apart by what is in them, but a caller
usually knows what is next door ("the kitchen is right outside") or whether their window faces the street. This
module turns the floor plan into yes/no "position" questions with P(yes | candidate) from geometry:
  - Neighbours: two rooms on the same floor share a wall when segments of their outlines run parallel, a wall's
    thickness apart, for at least a doorway's length (see config.ADJACENCY_*).
  - Front: the street's bearing (orientation.json) is turned into a direction in model meters; a room is at the
    front when part of its outline facing that way has no other room in front of it (a ray towards the street
    leaves the house) and is near the house's street-side edge.
Answers are soft (config.POSITION_P_YES / POSITION_P_NO): open-plan doorways and scan gaps make the geometry imperfect.

The questions are registered into the FeatureModel (model.p_yes under "adj:<room id>" / "front", text in
model.positions) so followups.choose and followups.apply_cues treat them like any yes/no feature.
"""

import math

import numpy as np

from . import config, followups

# Ray-casting step along a room's outline when looking for street-facing exterior walls.
_SAMPLE_STEP_M = 0.25


def room_segments(room: dict) -> np.ndarray:
    """(n, 4) wall segments [x1, y1, x2, y2] in model meters, from `walls`, or the polygon's edges if there are none."""
    walls = room.get("walls") or []
    if not walls and room.get("polygon") and len(room["polygon"]) >= 3:
        poly = room["polygon"]
        walls = [[*a, *b] for a, b in zip(poly, poly[1:] + poly[:1])]
    segs = np.array(walls, dtype=float).reshape(-1, 4)
    return segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) > 1e-6]


def shared_wall_length(a: np.ndarray, b: np.ndarray, gap: float = config.ADJACENCY_WALL_GAP_M) -> float:
    """Total length over which segments of `a` and `b` run parallel within `gap` meters of each other."""
    total = 0.0
    max_sin = math.sin(math.radians(15))   # walls in these scans are straight; 15° allows for scan error
    for x1, y1, x2, y2 in a:
        p, length = np.array([x1, y1]), math.hypot(x2 - x1, y2 - y1)
        d = np.array([x2 - x1, y2 - y1]) / length
        normal = np.array([-d[1], d[0]])
        for u1, v1, u2, v2 in b:
            e = np.array([u2 - u1, v2 - v1])
            if abs(d[0] * e[1] - d[1] * e[0]) / np.linalg.norm(e) > max_sin:
                continue
            q1, q2 = np.array([u1, v1]) - p, np.array([u2, v2]) - p
            if abs(q1 @ normal) > gap or abs(q2 @ normal) > gap:
                continue
            lo, hi = sorted((q1 @ d, q2 @ d))
            total += max(0.0, min(hi, length) - max(lo, 0.0))
    return total


def neighbours(building) -> list[set[int]]:
    """For each room, the indices of rooms on the same floor that share at least config.ADJACENCY_MIN_SHARED_M of
    wall with it (shorter contacts are corners touching, not rooms you can walk or see between)."""
    segs = [room_segments(r) for r in building.rooms]
    floor = building.room_floor
    out = [set() for _ in building.rooms]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if floor[i] != floor[j] or not len(segs[i]) or not len(segs[j]):
                continue
            if shared_wall_length(segs[i], segs[j]) >= config.ADJACENCY_MIN_SHARED_M:
                out[i].add(j)
                out[j].add(i)
    return out


def street_direction(orientation: dict | None) -> np.ndarray | None:
    """Unit vector from the house towards the street, in model meters. orientation.model_to_enu rotates model
    coordinates counter-clockwise by the yaw into east/north, so a compass bearing is rotated back clockwise."""
    street = (orientation or {}).get("street")
    if not street or street.get("bearing_deg") is None or orientation.get("yaw_deg") is None:
        return None
    b, yaw = math.radians(street["bearing_deg"]), math.radians(orientation["yaw_deg"])
    east, north = math.sin(b), math.cos(b)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([east * c + north * s, -east * s + north * c])


def _ray_hits(point: np.ndarray, direction: np.ndarray, segs: np.ndarray) -> bool:
    """Does the ray point + t·direction (t > 1 cm) cross any of the segments?"""
    a, b = segs[:, :2], segs[:, 2:]
    e = b - a
    denom = direction[0] * e[:, 1] - direction[1] * e[:, 0]
    ok = np.abs(denom) > 1e-9
    w = a - point
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / denom
        s = (w[:, 0] * direction[1] - w[:, 1] * direction[0]) / denom
    return bool(np.any(ok & (t > 0.01) & (s >= 0) & (s <= 1)))


def front_rooms(building, direction: np.ndarray) -> set[int]:
    """Rooms with at least config.FRONT_MIN_EXTERIOR_M of outline facing the street with no room in front of it, no
    further back than config.FRONT_MAX_SETBACK_M from the floor's street-side edge (so a back room seen past a
    narrower front room, or through a gap where an unscanned area was dropped, doesn't count)."""
    segs = [room_segments(r) for r in building.rooms]
    floor = building.room_floor
    front = set()
    for f in set(floor.tolist()):
        on_floor = [i for i in range(len(segs)) if floor[i] == f and len(segs[i])]
        if not on_floor:
            continue
        everything = np.vstack([segs[i] for i in on_floor])
        edge = max(float((everything[:, :2] @ direction).max()), float((everything[:, 2:] @ direction).max()))
        for i in on_floor:
            exterior = 0.0
            for x1, y1, x2, y2 in segs[i]:
                d = np.array([x2 - x1, y2 - y1])
                length = float(np.linalg.norm(d))
                if abs(d[0] * direction[1] - d[1] * direction[0]) / length < 0.5:
                    continue   # runs towards the street (a side wall): rays along it are degenerate
                n = max(1, int(length / _SAMPLE_STEP_M))
                for k in range(n):
                    point = np.array([x1, y1]) + d * (k + 0.5) / n
                    if edge - point @ direction <= config.FRONT_MAX_SETBACK_M and not _ray_hits(point, direction,
                                                                                                everything):
                        exterior += length / n
            if exterior >= config.FRONT_MIN_EXTERIOR_M:
                front.add(i)
    return front


def _reference_name(room: dict) -> str:
    return (room.get("label") or " / ".join(room.get("types") or [])).strip().lower()


def register(model: followups.FeatureModel, building, orientation: dict | None = None) -> None:
    """Add position questions to the feature model (idempotent; call again once orientation arrives).

    "Next to the <room>?" is only asked about rooms whose name is unique in the building ("the kitchen", not "Bedroom
    2", which a caller can't know), and only about rooms with at least one neighbour."""
    hi, lo, cand_room = config.POSITION_P_YES, config.POSITION_P_NO, building.candidate_room
    names = [_reference_name(r) for r in building.rooms]
    for j, near in enumerate(neighbours(building)):
        if not near or not names[j] or names[j] == "room" or names.count(names[j]) > 1:
            continue
        key = f"adj:{building.rooms[j]['id']}"
        by_room = np.array([hi if i in near else lo for i in range(len(building.rooms))])
        model.p_yes[key] = by_room[cand_room]
        model.positions[key] = f"Is the room you're in next to the {names[j]}, sharing a wall or a doorway?"

    direction = street_direction(orientation)
    if direction is not None:
        front = front_rooms(building, direction)
        if front and len(front) < len(building.rooms):
            street = orientation["street"]["name"]
            by_room = np.array([hi if i in front else lo for i in range(len(building.rooms))])
            model.p_yes["front"] = by_room[cand_room]
            model.positions["front"] = (f"Is the room at the front of the house, on the {street} side? For example, "
                                        f"can you see {street} from a window?")


def plans(building, model: followups.FeatureModel, posterior: np.ndarray,
          covered: set[str]) -> list[tuple[followups.Plan, np.ndarray]]:
    """The registered position questions not yet covered, each with its answer matrix (yes, no) x candidates.
    `building` and `posterior` are accepted for callers that want to rank them (see followups.choose)."""
    out = []
    for key, text in model.positions.items():
        if key not in covered:
            plan = followups.Plan("position", key, text)
            out.append((plan, followups.answer_matrix(plan, model)))
    return out
