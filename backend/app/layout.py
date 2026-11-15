"""Doors and windows of every room, from Matterport's room boundaries (one GraphQL call per building, cached).

Rooms that look alike in photos (empty bedrooms, closets, photo-less bathrooms) differ in where they are: what their
door opens onto, whether they have a window, whether that window faces the street. Matterport annotates each wall
("edge") of a room with typed openings (doorway, window, opening, divider) and a position along the wall, so these
facts come from the scan itself, not from guessing at photos. On the sample house they tell all four bedrooms and both
bathrooms apart (Bedroom 4 is the only bedroom without a window; only Bathroom 2 has one).

Cached in data/buildings/<id>/layout.json, separately from building.json, so adding this didn't require re-downloading
every building's photos. If the fetch fails (offline), followups simply has no layout questions for that building.

layout.json: {"rooms": {room_id: {"doors": [{"to": room_id | null, "to_kind": str, "x", "y", "width"}],
                                  "windows": [{"x", "y", "normal": [nx, ny], "width"}]}}}
  - "to" is the room on the other side (null: outside). "to_kind" is a coarse kind for questions: hallway, bedroom,
    bathroom, closet, living (kitchen, living, dining, family), stairs, garage, entry, outside, other.
  - "normal" points out of the room, in model coordinates (for "can you see the street from your window?").
"""

import json
import logging

import numpy as np

from . import ingest, matterport

log = logging.getLogger(__name__)

QUERY = """
query Layout($id: ID!) {
  model(id: $id) {
    rooms {
      id
      label
      classifications { label }
      floor { id }
      boundary { edges { id vertices { position { x y } } openings { type relativeCenter width } } }
    }
  }
}
"""
# A doorway on a wall that no other room shares leads to whichever room's wall is this close (meters) to the doorway's
# middle; if none is, it leads outside. Walls of neighbouring rooms are drawn a wall-thickness apart (~0.1-0.3 m).
NEIGHBOUR_DISTANCE_M = 0.45
# Unlabelled "Void" rooms smaller than this are closets (on the sample house they are the bedroom closets).
CLOSET_MAX_AREA_M2 = 5.0
KIND_WORDS = [  # first match wins
    ("bathroom", ("bath", "toilet", "powder", "restroom")),
    ("closet", ("closet", "pantry", "wardrobe", "storage")),
    ("bedroom", ("bed",)),
    ("hallway", ("hall", "corridor")),
    ("living", ("kitchen", "living", "dining", "family", "great room", "den")),
    ("stairs", ("stair", "landing")),
    ("garage", ("garage",)),
    ("entry", ("entry", "foyer", "porch", "mudroom")),
]


def path(model_id: str):
    return ingest.building_dir(model_id) / "layout.json"


def load(model_id: str) -> dict | None:
    p = path(model_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ensure(model_id: str) -> dict | None:
    """Cached layout, fetching it once if missing. None if Matterport can't be reached (no layout questions)."""
    cached = load(model_id)
    if cached is not None:
        return cached
    try:
        rooms = matterport.graphql(QUERY, {"id": model_id})["model"]["rooms"]
    except Exception as exc:  # offline or API error: the locator works without it
        log.warning("Layout fetch failed for %s: %s", model_id, exc)
        return None
    data = {"rooms": build(rooms)}
    path(model_id).write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def room_kind(text: str, area: float | None = None) -> str:
    text = text.lower()
    for kind, words in KIND_WORDS:
        if any(w in text for w in words):
            return kind
    if "void" in text and area is not None and area <= CLOSET_MAX_AREA_M2:
        return "closet"
    return "other"


def _area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points.T
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _distance_to_segments(p: np.ndarray, segments: np.ndarray) -> float:
    if not len(segments):
        return np.inf
    a, b = segments[:, 0], segments[:, 1]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t[:, None] * ab), axis=1).min())


def _cross(u, v):
    """z of the 2D cross product (positive: v is counter-clockwise from u). Works on arrays of vectors."""
    u, v = np.asarray(u), np.asarray(v)
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain)."""
    pts = sorted(map(tuple, points.round(3)))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and _cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return np.array(half(pts) + half(pts[::-1]))


def _inside_convex(p: np.ndarray, hull: np.ndarray) -> bool:
    if len(hull) < 3:
        return False
    edges = np.roll(hull, -1, axis=0) - hull
    return bool((_cross(edges, p - hull) >= 0).all())


# How far past a door to look when deciding whether it leads outside (meters).
BEYOND_PROBE_M = 1.0


def _beyond(a: np.ndarray, b: np.ndarray, p: np.ndarray, centroid: np.ndarray, hull: np.ndarray) -> str:
    """Kind of space behind a door that no mapped room touches. Many houses have spaces that were never scanned
    (garage, laundry, pantry, coat closets): on the sample house 8 doorways led to such spaces, 0.75-5 m from any
    mapped room. Those doors stay inside the building's footprint, so only a door whose far side falls outside the
    floor's convex hull counts as "outside"; the rest are "unmapped"."""
    normal = np.array([(b - a)[1], -(b - a)[0]]) / max(np.linalg.norm(b - a), 1e-9)
    if normal @ (p - centroid) < 0:
        normal = -normal
    return "unmapped" if _inside_convex(p + BEYOND_PROBE_M * normal, hull) else "outside"


def build(raw_rooms: list[dict]) -> dict:
    """Doors and windows per room id from the raw GraphQL rooms (pure; tested without the API)."""
    edges_of, kind_of, floor_of, centroid_of, segments_of = {}, {}, {}, {}, {}
    owners: dict[str, list[str]] = {}
    for room in raw_rooms:
        edges = (room.get("boundary") or {}).get("edges") or []
        rid = room["id"]
        segs = np.array([[[v["position"]["x"], v["position"]["y"]] for v in e["vertices"][:2]]
                         for e in edges if len(e.get("vertices") or []) >= 2], dtype=float).reshape(-1, 2, 2)
        points = segs[:, 0] if len(segs) else np.empty((0, 2))
        label = " ".join([room.get("label") or ""] + [c["label"] for c in room.get("classifications") or []])
        edges_of[rid], segments_of[rid] = edges, segs
        kind_of[rid] = room_kind(label, _area(points))
        floor_of[rid] = (room.get("floor") or {}).get("id")
        centroid_of[rid] = points.mean(axis=0) if len(points) else np.zeros(2)
        for e in edges:
            owners.setdefault(e["id"], []).append(rid)

    hull_of = {}
    for floor in set(floor_of.values()):
        points = [s.reshape(-1, 2) for o, s in segments_of.items() if floor_of[o] == floor and len(s)]
        hull_of[floor] = _convex_hull(np.vstack(points)) if points else np.empty((0, 2))

    out = {}
    for rid, edges in edges_of.items():
        doors, windows = [], []
        for e in edges:
            verts = e.get("vertices") or []
            if len(verts) < 2:
                continue
            a = np.array([verts[0]["position"]["x"], verts[0]["position"]["y"]])
            b = np.array([verts[1]["position"]["x"], verts[1]["position"]["y"]])
            for opening in e.get("openings") or []:
                # relativeCenter is the opening's middle as a fraction along the wall (0..1).
                p = a + (b - a) * float(opening.get("relativeCenter") or 0.5)
                if opening["type"] == "window":
                    normal = np.array([(b - a)[1], -(b - a)[0]]) / max(np.linalg.norm(b - a), 1e-9)
                    if normal @ (p - centroid_of[rid]) < 0:  # point it away from the room
                        normal = -normal
                    windows.append({"x": p[0], "y": p[1], "normal": normal.round(3).tolist(),
                                    "width": round(float(opening.get("width") or 0), 2)})
                elif opening["type"] in ("doorway", "opening"):
                    others = [o for o in owners[e["id"]] if o != rid]
                    if not others:  # unshared wall: the nearest other room's wall on the same floor
                        near = [(_distance_to_segments(p, segments_of[o]), o) for o in edges_of
                                if o != rid and floor_of[o] == floor_of[rid]]
                        others = [o for d, o in sorted(near) if d <= NEIGHBOUR_DISTANCE_M][:1]
                    to = others[0] if others else None
                    kind = kind_of[to] if to else _beyond(a, b, p, centroid_of[rid], hull_of[floor_of[rid]])
                    doors.append({"to": to, "to_kind": kind,
                                  "x": p[0], "y": p[1], "width": round(float(opening.get("width") or 0), 2)})
        out[rid] = {"kind": kind_of[rid], "doors": doors, "windows": windows}
    return out
