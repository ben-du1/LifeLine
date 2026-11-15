"""Doors and windows from Matterport room boundaries (layout.build), on a hand-made plan."""

from app import layout


def edge(eid, a, b, openings=()):
    return {"id": eid, "vertices": [{"position": {"x": a[0], "y": a[1]}}, {"position": {"x": b[0], "y": b[1]}}],
            "openings": [{"type": t, "relativeCenter": c, "width": 0.8} for t, c in openings]}


def room(rid, label, edges):
    return {"id": rid, "label": label, "classifications": [], "floor": {"id": "f"}, "boundary": {"edges": edges}}


def test_doors_windows_and_what_is_behind_them():
    # Bedroom (0..4 x 0..4) and hallway (4..6 x 0..4) share the wall x=4 (edge "s"), which has a doorway.
    # The bedroom's north wall has a window; its west wall a doorway that leads out of the footprint (outside).
    # The hallway's south wall has a doorway into a space nobody scanned but that lies inside the footprint,
    # because a mapped storage room (6..9 x -3..4) widens the floor's outline there.
    bedroom = room("b", "Bedroom", [edge("s", (4, 0), (4, 4), [("doorway", 0.5)]),
                                    edge("n", (4, 4), (0, 4), [("window", 0.5)]),
                                    edge("w", (0, 4), (0, 0), [("doorway", 0.5)]), edge("bs", (0, 0), (4, 0))])
    hall = room("h", "Hallway", [edge("s", (4, 0), (4, 4)), edge("hn", (4, 4), (6, 4)),
                                 edge("he", (6, 4), (6, 0)), edge("hs", (6, 0), (4, 0), [("doorway", 0.5)])])
    store = room("x", "Storage", [edge("x1", (6, -3), (9, -3)), edge("x2", (9, -3), (9, 4)),
                                  edge("x3", (9, 4), (6, 4)), edge("x4", (6, 4), (6, -3))])
    data = layout.build([bedroom, hall, store])
    assert sorted(d["to_kind"] for d in data["b"]["doors"]) == ["hallway", "outside"]
    assert data["b"]["windows"][0]["normal"] == [0.0, 1.0]                 # points out of the room (north)
    # The shared doorway is listed on the bedroom's side only (the hallway's copy of the wall has no opening);
    # the hallway's own south doorway stays inside the footprint, so it is "unmapped", not "outside".
    assert [d["to_kind"] for d in data["h"]["doors"]] == ["unmapped"]


def test_room_kinds():
    assert layout.room_kind("Master Bedroom") == "bedroom"
    assert layout.room_kind("Kitchen / Living Room") == "living"
    assert layout.room_kind("Void", area=2.0) == "closet"
    assert layout.room_kind("Void", area=20.0) == "other"
