"""Where inside a big room: regions, zone labels, and the within-room step before "located".

Synthetic building: an 8 m x 5 m open-plan kitchen/living room with 6 sweeps on a 3 x 2 grid, a 3 m x 4 m bedroom with
one sweep, and a bathroom with no photos (one image-less candidate at its centroid)."""

import asyncio
import importlib
import sys
import types

import numpy as np
import pytest

from app import config, followups, locator, subroom
from app.building import Building

OPEN_PLAN, BEDROOM, BATHROOM = 0, 1, 2
GRID = [(1, 1), (4, 1), (7, 1), (1, 4), (4, 4), (7, 4)]  # open-plan sweeps; x grows east when yaw is 0


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def make_building(stove_east: bool = False) -> tuple[Building, followups.FeatureModel]:
    rooms = [
        {"id": "o", "name": "Dining Room / Kitchen / Living Room", "types": ["Dining Room", "Kitchen", "Living Room"],
         "polygon": rect(0, 0, 8, 5), "walls": []},
        {"id": "b", "name": "Bedroom", "types": ["Bedroom"], "polygon": rect(8, 0, 11, 4), "walls": []},
        {"id": "w", "name": "Bathroom", "types": ["Bathroom"], "polygon": rect(8, 4, 10, 5), "walls": [],
         "centroid": [9, 4.5]},
    ]
    candidates = [{"x": x, "y": y, "views": [f"v{i}"]} for i, (x, y) in enumerate(GRID)]
    candidates += [{"x": 9.5, "y": 2.0, "views": ["vb"]}, {"x": 9.0, "y": 4.5, "views": []}]
    building = Building(
        id="no-such-model", name="m", address="1 A St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}],
        floorplans={}, rooms=rooms, candidates=candidates,
        candidate_room=np.array([0, 0, 0, 0, 0, 0, 1, 2]), has_views=np.array([True] * 7 + [False]),
        view_embeddings=np.eye(7, dtype=np.float32), view_candidate=np.arange(7),
    )
    scores = {p: np.zeros(8) for p in followups.clip_prompts()}
    if stove_east:  # the photos at the two east sweeps show the stove
        scores[followups.FEATURES_BY_KEY["stove"].prompt] = np.array([0, 0, 1, 0, 0, 1, 0, 0], dtype=float)
    return building, followups.build_model(building, scores)


def posterior_on(weights: dict[int, float], n: int = 8) -> np.ndarray:
    p = np.full(n, 1e-6)
    for c, w in weights.items():
        p[c] = w
    return p / p.sum()


def test_concentrated_on_two_adjacent_sweeps_gives_a_small_region_between_them():
    building, model = make_building()
    region = subroom.region(posterior_on({2: 0.5, 5: 0.5}), building, OPEN_PLAN)
    assert region["x"] == pytest.approx(7.0, abs=0.01) and region["y"] == pytest.approx(2.5, abs=0.01)
    assert region["radius_m"] == pytest.approx(1.5, abs=0.01)
    assert sum(s["p"] for s in region["sweeps"]) == pytest.approx(1.0, abs=1e-3)
    assert {s["candidate"] for s in region["sweeps"][:2]} == {2, 5}
    assert not subroom.needs_narrowing(float(model.room_area[OPEN_PLAN]), region)


def test_uniform_in_a_big_room_needs_narrowing():
    building, model = make_building()
    prior = locator.uniform_room_prior(building.candidate_room, 3)
    region = subroom.region(prior, building, OPEN_PLAN)
    assert region["x"] == pytest.approx(4.0) and region["y"] == pytest.approx(2.5)
    assert region["radius_m"] > config.NARROW_RADIUS_M
    assert region["radius_m"] <= region["half_diagonal_m"]
    assert region["area_m2"] == 40.0
    assert subroom.needs_narrowing(float(model.room_area[OPEN_PLAN]), region)
    # Nothing tells one part of the room from another: no zone is claimed.
    assert subroom.zone_label(building, OPEN_PLAN, (region["x"], region["y"]), model, prior, yaw_deg=0) is None


def test_one_sweep_room_uses_the_polygon():
    building, model = make_building()
    for room, center, radius in ((BEDROOM, (9.5, 2.0), 2.5), (BATHROOM, (9.0, 4.5), np.hypot(1, 0.5))):
        region = subroom.region(locator.uniform_room_prior(building.candidate_room, 3), building, room)
        assert (region["x"], region["y"]) == pytest.approx(center)
        assert region["radius_m"] == pytest.approx(max(radius, config.REGION_MIN_RADIUS_M), abs=0.01)
        assert len(region["sweeps"]) == 1 and region["sweeps"][0]["p"] == pytest.approx(1.0)
        assert not subroom.needs_narrowing(float(model.room_area[room]), region)


def test_zone_label_names_the_landmark_area_and_the_compass_side():
    building, model = make_building(stove_east=True)
    east = posterior_on({2: 0.5, 5: 0.5})
    assert subroom.zone_label(building, OPEN_PLAN, (7.0, 2.5), model, east, yaw_deg=0) == "kitchen end, near the stove"

    building, model = make_building()
    assert subroom.zone_label(building, OPEN_PLAN, (7.0, 2.5), model, east, yaw_deg=0) == "east side"
    # Model +x points west when the model is turned 180 degrees from north.
    assert subroom.zone_label(building, OPEN_PLAN, (7.0, 2.5), model, east, yaw_deg=180) == "west side"
    assert subroom.zone_label(building, OPEN_PLAN, (7.0, 4.5), model, east, yaw_deg=90) == "northwest corner"
    # Unknown north: no compass words (the model axes are arbitrary), and no landmark here either.
    assert subroom.zone_label(building, OPEN_PLAN, (7.0, 2.5), model, east) is None


# ---- the call flow (sessions._process) ------------------------------------------------------------------------


def _import_sessions():
    """sessions imports embeddings, which imports torch/open_clip/PIL. Stub them only when they are not installed,
    so a full environment still uses the real ones."""
    for name in ("torch", "open_clip", "PIL"):
        try:
            importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            module.Tensor = object  # embeddings.py uses torch.Tensor in an annotation
            if name == "PIL":
                module.Image = types.ModuleType("PIL.Image")
                sys.modules["PIL.Image"] = module.Image
            sys.modules[name] = module
    from app import sessions
    return sessions


NEAR = followups.Plan("nearest", "nearest", "Which are you closest to: the stove, the couch, or the sliding door?")


def _call(monkeypatch, posterior, stove_east=False, within=True):
    sessions = _import_sessions()
    monkeypatch.setattr(config, "MESSAGING_MODE", "simulated")
    monkeypatch.setattr(sessions.llm, "enabled", lambda: False)
    monkeypatch.setattr(sessions.semantics, "scores", lambda text: {})
    # Each reply carries no evidence: the test drives the posterior directly.
    monkeypatch.setattr(sessions.followups, "read_message",
                        lambda *a, **k: followups.Reading(None, set(), "", True))
    monkeypatch.setattr(sessions.SessionManager, "_evidence", staticmethod(lambda *a, **k: (None, None)))
    if within:
        monkeypatch.setattr(sessions.followups, "refinement", lambda *a: NEAR, raising=False)
    else:
        monkeypatch.setattr(sessions.followups, "refinement", lambda *a: None)

    async def notify(session):
        pass

    manager = sessions.SessionManager(notify)
    building, model = make_building(stove_east)
    manager.buildings[building.id] = sessions.LoadedBuilding(building, model)
    session = sessions.Session(id="s", phone="+1 555 000 0000", address="1 A St", created=0.0,
                               status="listening", model_id=building.id, posterior=posterior)
    # The posterior above stands for earlier caller messages: count one, so the located rule's evidence minimum
    # (locator.is_located, MIN_EVIDENCE_MESSAGES) is met by the message each test sends.
    session.evidence_messages = 1
    session.verified_room = int(np.argmax(locator.room_probabilities(posterior, building.candidate_room, len(building.rooms))))
    return manager, session


def test_verified_big_room_asks_within_room_questions(monkeypatch):
    uniform_in_room = posterior_on({c: 1.0 for c in range(6)})
    manager, session = _call(monkeypatch, uniform_in_room)
    asyncio.run(manager._process(session, "I'm in the living room"))
    assert session.status == "located" and session.pending is NEAR
    assert "narrowing down" in session.status_detail
    asyncio.run(manager._process(session, "the couch"))
    assert len(session.refined) == config.MAX_REFINEMENTS == 2
    asyncio.run(manager._process(session, "still the couch"))
    assert session.status == "located" and len(session.refined) == 2
    assert any("Responders will look for you in the Dining Room / Kitchen / Living Room" in m["text"] for m in session.messages)

    snap = manager.snapshot(session)["estimate"]
    assert set(snap["region"]) == {"x", "y", "radius_m", "sweeps"} and len(snap["region"]["sweeps"]) == 6
    assert snap["zone_label"] is None and snap["room_name"] == "Dining Room / Kitchen / Living Room"


def test_without_the_within_room_planner_the_flow_is_unchanged(monkeypatch):
    manager, session = _call(monkeypatch, posterior_on({c: 1.0 for c in range(6)}), within=False)
    asyncio.run(manager._process(session, "I'm in the living room"))
    assert session.status == "located" and not session.asked


def test_located_message_names_the_zone(monkeypatch):
    manager, session = _call(monkeypatch, posterior_on({2: 0.5, 5: 0.5}), stove_east=True)
    asyncio.run(manager._process(session, "I'm in the kitchen by the stove"))
    assert session.status == "located" and not session.asked   # region already narrow: nothing to ask
    assert any("Responders will look for you in the Dining Room / Kitchen / Living Room, kitchen end, near the stove" in m["text"] for m in session.messages)
    assert manager.snapshot(session)["estimate"]["zone_label"] == "kitchen end, near the stove"
