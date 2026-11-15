import csv

from app import compat


def square(size: float, x0: float = 0.0) -> list[list[float]]:
    return [[x0, 0], [x0 + size, 0], [x0 + size, size], [x0, size]]


def room(rid, types=(), floor=0, size=4.0, label=None, polygon=True):
    return {"id": rid, "name": rid, "label": label, "types": list(types), "floor": floor,
            "polygon": square(size) if polygon else None}


def building(rooms, photographed, floors=1, floorplan=False):
    candidates = [{"id": f"s-{r}", "room_id": r, "views": ["v.jpg"]} for r in photographed]
    candidates += [{"id": f"room-{r['id']}", "room_id": r["id"], "views": []}
                   for r in rooms if r["id"] not in photographed]
    return {"id": "m", "address": "1 A St", "floors": [{"index": i} for i in range(floors)],
            "floorplans": {"0": {}} if floorplan else {}, "rooms": rooms, "candidates": candidates}


GOOD_ROOMS = [room("kitchen", ["Kitchen"], size=5), room("living", ["Living Room"], size=6),
              room("bed1", ["Bedroom"], size=4), room("bed2", ["Bedroom"], size=3.0),
              room("bath", ["Bathroom"], size=2), room("laundry", ["Laundry Room"], size=1.5),
              room("office", ["Office"], floor=1, size=3.5)]


def test_well_covered_furnished_house_scores_high():
    result = compat.score_building(building(GOOD_ROOMS, [r["id"] for r in GOOD_ROOMS], floors=2),
                                   orientation_json={"yaw_deg": 10}, description="A furnished home with a sofa.")
    assert result["score"] >= 90, result
    assert result["checks"]["photo_coverage"] == 1.0
    assert result["checks"]["lookalike_groups"] == 0          # bedrooms 16 vs 9 m² are told apart by size
    assert result["checks"]["key_rooms"] == {"kitchen": True, "bathroom": True, "bedroom": True}
    assert result["checks"]["furnished"] is True


def test_unusable_models_score_zero():
    assert compat.score_building(building([], []))["score"] == 0
    no_photos = compat.score_building(building(GOOD_ROOMS, []))
    assert no_photos["score"] == 0 and no_photos["checks"]["usable"] is False
    assert "no room has photos" in no_photos["problems"]


def test_photo_gaps_and_mixed_photo_kinds_are_flagged():
    rooms = GOOD_ROOMS + [room("bath2", ["Bathroom"], size=2.5)]
    full = compat.score_building(building(rooms, [r["id"] for r in rooms]))
    gappy = compat.score_building(building(rooms, [r["id"] for r in rooms if r["id"] != "bath2"]))
    assert gappy["score"] < full["score"]
    assert gappy["checks"]["mixed_photo_kinds"] == ["bathroom"]
    assert any("without photos: bath2" in p for p in gappy["problems"])


def test_identical_rooms_form_lookalike_groups():
    # The README's tie: Chipman Glen's Bedrooms 2 and 3 (13 vs 14 m²). 16 m² vs 14.4 m² is within 20%.
    rooms = [room("kitchen", ["Kitchen"]), room("b1", ["Bedroom"], size=4), room("b2", ["Bedroom"], size=3.8),
             room("b3", ["Bedroom"], size=3.0), room("c1", ["Closet"], polygon=False),
             room("c2", ["Closet"], polygon=False)]
    result = compat.score_building(building(rooms, [r["id"] for r in rooms]))
    assert result["checks"]["lookalike_groups"] == 2           # b1+b2, and the two closets of unknown area
    assert result["checks"]["lookalike_rooms"] == 4
    assert any(p == "look-alike rooms: b2, b1" for p in result["problems"])
    assert result["checks"]["polygon_coverage"] == round(4 / 6, 2)


def test_vacant_description_unlabelled_rooms_and_many_floors_are_penalised():
    base = building(GOOD_ROOMS, [r["id"] for r in GOOD_ROOMS])
    good = compat.score_building(base)["score"]
    vacant = compat.score_building(base, description="This vacant home has empty bedrooms.")
    assert vacant["checks"]["furnished"] is False and vacant["score"] == good - 10

    unlabelled = [room(f"r{i}", label=None, size=3 + i) for i in range(6)]
    result = compat.score_building(building(unlabelled, [r["id"] for r in unlabelled]))
    assert result["checks"]["label_coverage"] == 0.0 and result["checks"]["distinct_types"] == 0
    assert result["score"] < 50

    tall = compat.score_building(building(GOOD_ROOMS, [r["id"] for r in GOOD_ROOMS], floors=5))
    assert any("5 floors" in p for p in tall["problems"])


def test_very_large_rooms_are_flagged():
    rooms = GOOD_ROOMS + [room("hall", ["Great Room"], size=10)]   # 100 m²
    result = compat.score_building(building(rooms, [r["id"] for r in rooms]))
    assert result["checks"]["very_large_rooms"] == ["hall"]


def test_score_model_applies_ingest_room_rules():
    def gql_room(rid, types=(), floor="f0", area=None):
        edges = [{"vertices": [{"position": {"x": a[0], "y": a[1]}}, {"position": {"x": b[0], "y": b[1]}}]}
                 for a, b in zip(square(3), square(3)[1:] + square(3)[:1])]
        return {"id": rid, "label": None, "classifications": [{"label": t} for t in types], "floor": {"id": floor},
                "dimensions": {"areaFloor": area}, "boundary": {"edges": edges}}

    def pano(rid, status="available"):
        return {"id": f"p-{rid}", "anchor": {"room": {"id": rid}}, "skybox": {"status": status}}

    model = {"floors": [{"id": "f0"}, {"id": "f1"}], "geocoordinates": {"latitude": 29.7},
             "generatedDescription": {"summary": "A furnished two-story home."},
             "assets": {"floorplans": []},
             "rooms": [gql_room("k", ["Kitchen"]), gql_room("b", ["Bedroom"], "f1"), gql_room("ba", ["Bathroom"]),
                       gql_room("void", ["Void"]), gql_room("yard")],
             "panoLocations": [pano("k"), pano("b"), pano("ba", status="failed"), pano("yard")]}
    rooms, floors = compat.model_rooms(model)
    assert {r["id"] for r in rooms} == {"k", "b", "ba"}        # Void and the unlabelled yard are dropped
    assert floors == 2
    assert {r["id"]: r["photos"] for r in rooms} == {"k": 1, "b": 1, "ba": 0}
    assert abs(rooms[0]["area"] - 9.0) < 1e-6                   # from the outline
    result = compat.score_model(model)
    assert result["checks"]["geolocation"] is True and result["checks"]["floorplan"] is False
    assert result["checks"]["furnished"] is True
    assert result["checks"]["photo_coverage"] == round(2 / 3, 2)


def test_shortlist_filters_index(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "rank_demo_models", Path(__file__).resolve().parents[1] / "scripts" / "rank_demo_models.py")
    rank = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rank)

    columns = ["sid", "name", "segment", "lat", "lon", "address", "city", "pano_count", "room_count",
               "floor_count", "ai_summary"]
    rows = [
        ["good", "13122 Chipman Glen Dr", "residential", "29.7", "-95.6", "13122 Chipman Glen Dr, Houston, TX",
         "Houston", "36", "13", "1", "yes"],
        ["office", "Lobby", "commercial", "29.7", "-95.6", "", "Houston", "30", "10", "1", "yes"],
        ["tiny", "1 A St", "residential", "29.7", "-95.6", "1 A St, Houston, TX", "Houston", "5", "3", "1", "yes"],
        ["dallas", "2 B St", "residential", "32.7", "-96.8", "2 B St, Dallas, TX", "Dallas", "30", "10", "1", "yes"],
        ["tower", "3012 Houston Ave", "residential", "29.7", "-95.3", "3012 Houston Ave, Houston, TX", "Houston",
         "50", "20", "4", "yes"],
        ["unit", "915 Franklin St", "residential", "29.7", "-95.3", "915 Franklin St, 9L, Houston, TX, 77002, US",
         "Houston", "38", "14", "2", "yes"],
    ]
    index = tmp_path / "models_index.csv"
    with index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    picked = rank.shortlist(index, tmp_path / "out.csv", "residential", 10)
    assert [r["sid"] for r in picked] == ["good", "unit"]
    assert picked[1]["notes"] == "unit address"


def test_check_falls_back_scores_and_caches(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    from app import matterport

    spec = importlib.util.spec_from_file_location(
        "rank_demo_models", Path(__file__).resolve().parents[1] / "scripts" / "rank_demo_models.py")
    rank = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rank)
    monkeypatch.setattr(rank, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(rank, "PAUSE_SECONDS", 0)
    calls = []

    def fake_graphql(query, variables):
        calls.append("light" if "DemoCheck(" in query else "fallback")
        if calls[-1] == "light":
            raise matterport.MatterportError("Cannot query field 'status'")
        kitchen = {"id": "k", "label": None, "classifications": [{"label": "Kitchen"}], "floor": {"id": "f0"}}
        return {"model": {"floors": [{"id": "f0"}], "rooms": [kitchen],
                          "panoLocations": [{"id": "p", "anchor": {"room": {"id": "k"}},
                                             "skybox": {"children": ["u"] * 6}}]}}

    monkeypatch.setattr(matterport, "graphql", fake_graphql)
    shortlist = tmp_path / "shortlist.csv"
    shortlist.write_text("sid,address,prescore,notes\nabc,1 A St,90,\n", encoding="utf-8")
    rows = rank.check(shortlist, tmp_path / "out.csv", 10, refresh=False)
    assert calls == ["light", "fallback"]
    assert rows[0]["sid"] == "abc" and rows[0]["score"] > 0 and rows[0]["photo_coverage"] == 1.0
    rank.check(shortlist, tmp_path / "out.csv", 10, refresh=False)
    assert len(calls) == 2                                      # second run reads the cache
