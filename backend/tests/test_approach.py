"""Approach card: outside walls, compass sides and the street (synthetic buildings, no network)."""

import math
import sys
import types

import numpy as np

from app import approach, orientation
from app.building import Building

GAP = 0.15  # adjacent rooms are drawn a wall's thickness apart, as in Matterport models


def rect(name: str, x0: float, y0: float, x1: float, y1: float, floor: int = 0) -> dict:
    poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    walls = [[*a, *b] for a, b in zip(poly, poly[1:] + poly[:1])]
    return {"id": name, "name": name, "label": name, "types": [], "floor": floor, "walls": walls, "polygon": poly,
            "centroid": [(x0 + x1) / 2, (y0 + y1) / 2]}


def make_building(rooms: list[dict], model_id: str) -> Building:
    candidates = [{"id": f"c{i}", "room_id": r["id"], "x": r["centroid"][0], "y": r["centroid"][1], "views": []}
                  for i, r in enumerate(rooms)]
    floors = sorted({r["floor"] for r in rooms})
    return Building(id=model_id, name=model_id, address="1 Test St", floorplans={},
                    floors=[{"index": f, "id": f"f{f}", "label": f"Floor {f + 1}"} for f in floors],
                    rooms=rooms, candidates=candidates, candidate_room=np.arange(len(rooms), dtype=np.int32),
                    has_views=np.zeros(len(rooms), bool), view_embeddings=np.zeros((0, 4), np.float32),
                    view_candidate=np.zeros(0, np.int32))


def l_shape() -> Building:
    # A B        (model frame, x to the right)
    # C          A = [0,4]x[0,4], B to its +x, C and D stacked toward +y.
    # D
    return make_building([rect("A", 0, 0, 4, 4), rect("B", 4 + GAP, 0, 8, 4),
                          rect("C", 0, 4 + GAP, 4, 8), rect("D", 0, 8 + 2 * GAP, 4, 12)], "l-shape")


# yaw 90°: model +x points north, +y west, -y east, -x south (orientation.model_to_enu).
YAW = 90.0


def orient(street_bearing: float, name: str = "Pointer Street") -> dict:
    return {"yaw_deg": YAW, "street": {"name": name, "distance_m": 20, "bearing_deg": street_bearing,
                                       "side": orientation.side_of(street_bearing), "direction_deg": 0}}


def outward_lines(pieces: list[dict], room: int) -> set[tuple]:
    """Each outside piece of a room as (axis, value, normal) e.g. ('x', 8.0, (1, 0))."""
    out = set()
    for p in pieces:
        if p["room"] != room:
            continue
        axis = "x" if abs(p["a"][0] - p["b"][0]) < 1e-6 else "y"
        value = round(float(p["a"][0] if axis == "x" else p["a"][1]), 2)
        out.add((axis, value, tuple(int(round(v)) for v in p["normal"])))
    return out


def test_exterior_walls_of_l_shape():
    pieces = approach.exterior_walls(l_shape().rooms, 0)
    assert outward_lines(pieces, 0) == {("x", 0.0, (-1, 0)), ("y", 0.0, (0, -1))}           # A: shared with B, C
    assert outward_lines(pieces, 1) == {("x", 8.0, (1, 0)), ("y", 0.0, (0, -1)), ("y", 4.0, (0, 1))}
    assert outward_lines(pieces, 2) == {("x", 0.0, (-1, 0)), ("x", 4.0, (1, 0))}            # C: between A and D
    assert outward_lines(pieces, 3) == {("x", 0.0, (-1, 0)), ("x", 4.0, (1, 0)), ("y", 12.0, (0, 1))}


def test_partly_shared_wall_keeps_its_outside_part():
    # A long room with a short neighbour along half of one wall: only the other half is outside.
    building = make_building([rect("Long", 0, 0, 8, 3), rect("Short", 0, 3 + GAP, 4, 6)], "partial")
    tops = [p for p in approach.exterior_walls(building.rooms, 0)
            if p["room"] == 0 and abs(p["a"][1] - 3) < 1e-6 and abs(p["b"][1] - 3) < 1e-6]
    assert len(tops) == 1
    xs = sorted([tops[0]["a"][0], tops[0]["b"][0]])
    assert 4.0 <= xs[0] <= 4.7 and xs[1] == 8


def test_sides_bearings_and_street():
    building = l_shape()
    # Caller near B's +x wall: model +x is north at yaw 90. Street to the east -> that wall is on the side.
    card = approach.approach(building, 1, (7.5, 1.0), orient(90))
    assert card["side"] == "north" and abs(card["bearing_deg"]) < 1e-6
    assert card["street_relation"] == "side" and card["faces_street"] is False
    assert not card["interior"] and card["distance_m"] == 0.5
    assert "east end" in card["text"] and "about 1 m" in card["text"]
    # Caller near B's -y wall: east, facing the street.
    card = approach.approach(building, 1, (6.0, 0.4), orient(90))
    assert card["side"] == "east" and card["faces_street"] and card["street_relation"] == "front"
    assert "facing Pointer Street" in card["text"]
    # D's +y wall is west; with the street to the east it is the rear.
    card = approach.approach(building, 3, (2.0, 11.8), orient(90))
    assert card["side"] == "west" and card["street_relation"] == "rear"
    assert card["text"].startswith("Single floor, rear (west) side, away from Pointer Street.")


def test_street_facing_tolerance_is_60_degrees():
    building = l_shape()
    assert approach.approach(building, 1, (6.0, 0.4), orient(90 + 55))["faces_street"] is True
    assert approach.approach(building, 1, (6.0, 0.4), orient(90 + 65))["faces_street"] is False


def test_interior_room_points_to_nearest_outside_wall():
    rooms = [rect(f"R{i}{j}", i * (3 + GAP), j * (3 + GAP), i * (3 + GAP) + 3, j * (3 + GAP) + 3)
             for i in range(3) for j in range(3)]
    building = make_building(rooms, "grid")
    middle = 4  # i = 1, j = 1
    assert not [p for p in approach.exterior_walls(rooms, 0) if p["room"] == middle]
    cx, cy = rooms[middle]["centroid"]
    card = approach.approach(building, middle, (cx + 1.0, cy), orient(90))   # nearer the +x side: north
    assert card["interior"] and card["side"] == "north"
    assert abs(card["distance_m"] - (3 + GAP + 1.5 - 1.0)) < 0.05
    assert "no outside wall (interior room)" in card["text"]


def test_upper_floor_label_and_window():
    rooms = [rect("Down", 0, 0, 4, 4, floor=0), rect("Up", 0, 0, 4, 4, floor=1)]
    building = make_building(rooms, "two-storey")
    card = approach.approach(building, 1, (3.8, 2), orient(90), window_probability=0.8)
    assert card["floor_text"] == "2nd floor" and card["multi_floor"]
    assert card["headline"] == "2nd floor · north side" and card["has_window_likely"]
    assert "Window likely." in card["text"]


def test_without_orientation_gives_model_frame_only():
    card = approach.approach(l_shape(), 1, (7.5, 1.0), None)
    assert card["side"] is None and card["wall"]["normal"] == [1.0, 0.0] and card["streetview_url"] is None


def test_street_view_link_looks_back_at_the_wall(monkeypatch):
    geo = {"lat": 40.35, "lon": -74.65, "yaw": math.radians(YAW), "tx": 0.0, "ty": 0.0}
    monkeypatch.setattr(approach.config, "GOOGLE_MAPS_API_KEY", "")
    card = approach.approach(l_shape(), 1, (7.5, 1.0), orient(90), geo)
    # The north wall is 8 m north of the anchor; the viewpoint is further north, looking south.
    assert card["lat"] > geo["lat"] and abs(card["lon"] - geo["lon"]) < 1e-3
    assert "heading=180" in card["streetview_url"] and card["streetview_image_url"] is None
    viewpoint_lat = float(card["streetview_url"].split("viewpoint=")[1].split(",")[0])
    assert abs((viewpoint_lat - card["lat"]) * orientation.METERS_PER_DEG_LAT - approach.STREETVIEW_STANDOFF_M) < 0.1
    monkeypatch.setattr(approach.config, "GOOGLE_MAPS_API_KEY", "test-key")
    card = approach.approach(l_shape(), 1, (7.5, 1.0), orient(90), geo)
    assert card["streetview_image_url"].startswith("https://maps.googleapis.com/maps/api/streetview?size=640x400")


def test_snapshot_includes_approach(monkeypatch):
    # sessions imports the CLIP/MiniLM stack; stub the heavy modules so the snapshot can be built here.
    for name in ("torch", "open_clip", "PIL", "sentence_transformers"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.Tensor = object   # used in an annotation
            stub.Image = object
            monkeypatch.setitem(sys.modules, name, stub)
    from app import locator, sessions

    building = l_shape()
    monkeypatch.setattr(approach.orientation, "path", lambda _id: type("P", (), {"exists": lambda self: True})())
    monkeypatch.setattr(approach.orientation, "load", lambda _id: orient(90))
    monkeypatch.setattr(approach, "load_geo", lambda _id: None)
    manager = sessions.SessionManager(notify=None)
    from app import followups
    # A real feature model (the snapshot also computes ties and the in-room region from it), window likely everywhere.
    features = followups.build_model(building, {p: np.zeros(len(building.candidates)) for p in followups.clip_prompts()})
    features.p_yes["window"] = np.full(4, 0.7)
    manager.buildings[building.id] = sessions.LoadedBuilding(building, features)
    session = sessions.Session(id="s1", phone="+15555550100", address="1 Test St", created=0.0,
                               model_id=building.id)
    session.posterior = np.array([0.02, 0.94, 0.02, 0.02])
    session.evidence_messages = 1   # the posterior above stands for one evidence-bearing caller message
    session.estimate = locator.estimate(session.posterior, building.candidate_room, 4)
    data = manager.snapshot(session)
    assert data["approach"]["room"] == 1 and data["approach"]["has_window_likely"]
    assert data["approach"]["side"] in {"north", "east", "west"}
