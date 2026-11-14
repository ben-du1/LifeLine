import numpy as np

from app.building import Building


def make_building() -> Building:
    return Building(
        id="m", name="m", address="1 A St", rooms=[{"id": "r0"}, {"id": "r1"}],
        floors=[{"index": 0, "id": "f0", "label": "Floor 1"}],
        floorplans={"0": {"width": 1000, "height": 500, "resolution": 100.0, "origin": {"x": -2.0, "y": -1.0}}},
        candidates=[{"views": ["a", "b"]}, {"views": ["c"]}, {"views": []}],
        candidate_room=np.array([0, 1, 1]), has_views=np.array([True, True, False]),
        view_embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32),
        view_candidate=np.array([0, 0, 1]),
    )


def test_to_pixels_flips_y():
    b = make_building()
    assert b.to_pixels(-2.0, -1.0) == (0.0, 500.0)    # origin is the bottom-left corner
    assert b.to_pixels(0.0, 1.0) == (200.0, 300.0)


def test_candidate_scores_max_over_views_and_neutral_imageless():
    scores = make_building().candidate_scores(np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(scores[:2], [1.0, 0.8])        # best view per candidate
    assert np.isclose(scores[2], 0.9)                 # image-less gets the mean of imaged candidates


def test_ingest_keeps_every_floor_and_handles_unlabeled_regions(tmp_path, monkeypatch):
    import pytest
    from app import config, ingest, matterport

    def room(rid, floor, label=None, types=()):
        square = [((0, 0), (4, 0)), ((4, 0), (4, 4)), ((4, 4), (0, 4)), ((0, 4), (0, 0))]
        edges = [{"vertices": [{"position": {"x": a[0], "y": a[1]}}, {"position": {"x": b[0], "y": b[1]}}]}
                 for a, b in square]
        return {"id": rid, "label": label, "classifications": [{"label": t} for t in types],
                "floor": {"id": floor}, "boundary": {"edges": edges}}

    def sweep(sid, rid, floor):
        return {"id": sid, "position": {"x": 2, "y": 2, "z": 0}, "anchor": {"room": {"id": rid}, "floor": {"id": floor}},
                "skybox": {"children": [f"https://x/{sid}_{i}.jpg" for i in range(6)]}}

    floors = [{"id": "f1", "label": "Floor 2", "sequence": 1}, {"id": "f0", "label": "Floor 1", "sequence": 0}]
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "_download", lambda url, dest: None)

    # A model that labels its rooms: unlabeled regions are outdoor areas, even with sweeps in them (the sample's
    # yard had outdoor sweeps).
    model = {"id": "m", "name": "m", "address": None, "floors": floors, "assets": {},
             "rooms": [room("kitchen", "f0", types=["Kitchen"]), room("bed", "f1", types=["Bedroom"]),
                       room("yard", "f0")],
             "panoLocations": [sweep("s0", "kitchen", "f0"), sweep("s1", "bed", "f1"), sweep("s2", "yard", "f0")]}
    monkeypatch.setattr(matterport, "fetch_model", lambda model_id: model)
    building = ingest.download_building("m")
    assert [f["label"] for f in building["floors"]] == ["Floor 1", "Floor 2"]          # lowest first
    assert {r["id"]: r["floor"] for r in building["rooms"]} == {"kitchen": 0, "bed": 1}
    assert building["schema"] == ingest.SCHEMA_VERSION

    # A model that labels nothing (many townhomes): photographed regions are the rooms; unphotographed are dropped.
    townhome = {**model, "rooms": [room("a", "f0"), room("b", "f1"), room("c", "f1")],
                "panoLocations": [sweep("s0", "a", "f0"), sweep("s1", "b", "f1")]}
    monkeypatch.setattr(matterport, "fetch_model", lambda model_id: townhome)
    building = ingest.download_building("m")
    assert {r["id"]: (r["floor"], r["name"]) for r in building["rooms"]} == {"a": (0, "Room 1"), "b": (1, "Room 2")}

    empty = {**model, "rooms": [], "panoLocations": []}
    monkeypatch.setattr(matterport, "fetch_model", lambda model_id: empty)
    with pytest.raises(ingest.UnusableModel):
        ingest.download_building("m")
