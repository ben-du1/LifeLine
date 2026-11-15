"""Ties and position questions on a synthetic house whose two bathrooms differ only by where they are.

    y
    4 +---------+ +-------+ +--------------+
      |         | | Bath 2| |              |      Bathroom 1 shares a wall with the kitchen, Bathroom 2 with the
      | Kitchen | +-------+ |   Bedroom    |      bedroom only; the street is to the south (y < 0) in the second
      |         | +-------+ |              |      test, where Bathroom 1 is at the front.
      |         | | Bath 1| |              |
    0 +---------+ +-------+ +--------------+
      0         4 4.1  6.1 6.2            10
"""

import math

import numpy as np

from app import adjacency, followups, locator, orientation, ties
from app.building import Building


def rect(x0, y0, x1, y1) -> dict:
    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {"polygon": polygon, "walls": [[*a, *b] for a, b in zip(polygon, polygon[1:] + polygon[:1])]}


def make_house(bath2_far: bool = True) -> tuple[Building, followups.FeatureModel]:
    # Bathroom 2 sits behind Bathroom 1 (north), so it touches neither the kitchen nor the street.
    bath2 = rect(4.1, 4.1, 6.1, 6.6) if bath2_far else rect(4.1, 2.6, 6.1, 4.0)
    rooms = [
        {"id": "k", "name": "Kitchen", "label": None, "types": ["Kitchen"], **rect(0, 0, 4, 4)},
        {"id": "b1", "name": "Bathroom 1", "label": None, "types": ["Bathroom"], **rect(4.1, 0, 6.1, 2.5)},
        {"id": "b2", "name": "Bathroom 2", "label": None, "types": ["Bathroom"], **bath2},
        {"id": "r", "name": "Bedroom", "label": None, "types": ["Bedroom"], **rect(6.2, 0, 10, 8)},
    ]
    building = Building(
        id="t", name="t", address="3 C St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms,
        candidates=[{"views": ["a"], "x": 2.0, "y": 2.0}, {"views": [], "x": 5.1, "y": 1.2},
                    {"views": [], "x": 5.1, "y": 5.3}, {"views": ["b"], "x": 8.0, "y": 4.0}],
        candidate_room=np.array([0, 1, 2, 3]), has_views=np.array([True, False, False, True]),
        view_embeddings=np.eye(2, dtype=np.float32), view_candidate=np.array([0, 3]),
    )
    # Flat CLIP: the bathrooms are identical to the feature model; only their position differs.
    model = followups.build_model(building, {p: np.zeros(4) for p in followups.clip_prompts()})
    return building, model


def bathroom_posterior(building, model):
    """After "there's a toilet, it's tiny in here": the two bathrooms hold nearly all the belief."""
    prior = locator.uniform_room_prior(building.candidate_room, 4)
    reading = followups.read_message("there's a toilet, it's tiny in here", model, None, "free")
    return locator.apply_likelihood(prior, reading.likelihood), reading.covered


def test_neighbours_from_shared_walls():
    building, _ = make_house()
    near = adjacency.neighbours(building)
    assert near[0] == {1}                 # kitchen: only Bathroom 1 (0.1 m wall gap); Bathroom 2 only touches a corner
    assert near[2] == {3} and near[1] == {3, 0}


def test_identical_bathrooms_are_a_tie():
    building, model = make_house()
    posterior, _ = bathroom_posterior(building, model)
    assert sorted(ties.tie_group(posterior, building, model)) == [1, 2]
    text = ties.describe(building, posterior, ties.tie_group(posterior, building, model))
    assert "Bathroom 1 or Bathroom 2" in text and "can't tell apart yet" in text
    snap = ties.snapshot(building, posterior, ties.tie_group(posterior, building, model))
    assert {r["name"] for r in snap} == {"Bathroom 1", "Bathroom 2"} and set(snap[0]) == {"index", "name",
                                                                                        "probability"}
    # Uniform prior: two look-alike bathrooms holding half the belief between them is the floor for a tie, but a
    # kitchen and a bedroom at the same probability are not alike.
    uniform = locator.uniform_room_prior(building.candidate_room, 4)
    assert ties.tie_group(np.array([0.1, 0.05, 0.05, 0.8]), building, model) == []
    assert 0 not in ties.tie_group(uniform, building, model)


def test_adjacency_question_separates_the_bathrooms():
    building, model = make_house()
    adjacency.register(model, building)
    assert "adj:k" in model.positions and "front" not in model.positions   # no orientation yet
    posterior, covered = bathroom_posterior(building, model)
    # Ask until the planner runs out of content questions, answering as the caller in Bathroom 1 would.
    for _ in range(6):
        plan = followups.choose(model, posterior, building.candidate_room, 4, covered, None)
        assert plan is not None
        if plan.kind == "position":
            break
        truth = followups.answer_matrix(plan, model)[:, 1]
        answer = int(np.argmax(truth))
        posterior = locator.apply_likelihood(posterior, followups.answer_matrix(plan, model)[answer])
        covered.add(plan.key)
    assert plan.kind == "position" and plan.key == "adj:k" and "next to the kitchen" in plan.text
    assert sorted(ties.tie_group(posterior, building, model)) == [1, 2]

    reading = followups.read_message("yes", model, plan, "yes")
    after = locator.room_probabilities(locator.apply_likelihood(posterior, reading.likelihood),
                                       building.candidate_room, 4)
    assert after[1] > 0.8 and after[1] > 4 * after[2] and "adj:k" in reading.covered
    assert ties.tie_group(locator.apply_likelihood(posterior, reading.likelihood), building, model) == []


def test_position_questions_wait_for_content_questions():
    building, model = make_house()
    adjacency.register(model, building)
    # Uniform prior: "is there a toilet" is worth far more than "next to the kitchen", so it comes first.
    plan = followups.choose(model, locator.uniform_room_prior(building.candidate_room, 4), building.candidate_room, 4,
                            set(), None)
    assert plan.kind != "position"


def test_front_of_house_from_street_bearing():
    building, model = make_house()
    street = {"yaw_deg": 0.0, "street": {"name": "Elm Street", "bearing_deg": 180.0}}   # street to the south
    assert adjacency.front_rooms(building, adjacency.street_direction(street)) == {0, 1, 3}
    adjacency.register(model, building, street)
    assert "Elm Street" in model.positions["front"]
    p = model.p_yes["front"]
    assert p[1] == 0.85 and p[2] == 0.15                     # Bathroom 1 faces the street, Bathroom 2 is behind it


def test_street_direction_matches_orientation_frame():
    # A unit step towards the street in model meters must point along the street's compass bearing on the map.
    for yaw_deg, bearing in [(0, 90), (90, 180), (37, 300), (-120, 15)]:
        info = {"yaw_deg": yaw_deg, "street": {"name": "S", "bearing_deg": bearing}}
        step = adjacency.street_direction(info)
        geo = {"yaw": math.radians(yaw_deg), "tx": 0.0, "ty": 0.0}
        east, north = orientation.model_to_enu(geo, step[None])[0]
        assert abs(east - math.sin(math.radians(bearing))) < 1e-9 and abs(north - math.cos(math.radians(bearing))) < 1e-9
    assert adjacency.street_direction({"yaw_deg": 0, "street": None}) is None


def test_rooms_without_geometry_get_no_position_questions():
    from test_followups import make_model
    building, model = make_model()
    for room in building.rooms:
        room["polygon"], room["walls"] = None, []
    adjacency.register(model, building)
    assert model.positions == {}


def _session_manager():
    """SessionManager with the torch/open_clip imports stubbed (pure functions only are exercised)."""
    import sys
    import types
    for name in ("open_clip", "torch", "PIL", "PIL.Image"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["PIL"].Image = sys.modules["PIL.Image"]
    sys.modules["torch"].Tensor = getattr(sys.modules["torch"], "Tensor", object)
    from app import sessions

    async def notify(_):
        return None
    return sessions, sessions.SessionManager(notify)


def test_best_guess_names_tied_rooms_and_snapshot_lists_them(monkeypatch):
    import asyncio
    sessions, manager = _session_manager()
    building, model = make_house()
    adjacency.register(model, building)
    manager.buildings["t"] = sessions.LoadedBuilding(building, model)
    monkeypatch.setattr(sessions.semantics, "scores", lambda text: {})
    monkeypatch.setattr(sessions.llm, "enabled", lambda: False)
    monkeypatch.setattr(sessions.embeddings, "embed_texts", lambda texts: np.zeros((len(texts), 2), np.float32))
    session = sessions.Session(id="s", phone="+15555550100", address="3 C St", created=0.0, status="listening",
                               model_id="t", posterior=locator.uniform_room_prior(building.candidate_room, 4))
    # The caller can't say whether the kitchen is next door, so the call ends as a best guess between the bathrooms.
    monkeypatch.setattr(sessions.config, "MAX_QUESTIONS", 0)
    asyncio.run(manager._process(session, "there's a toilet, it's tiny in here"))
    assert session.status == "best_guess"
    assert "Bathroom 1 or Bathroom 2" in session.status_detail and "can't tell apart yet" in session.status_detail
    assert any(m["role"] == "system" and "Bathroom 1 or Bathroom 2" in m["text"] for m in session.messages)
    snap = manager.snapshot(session)
    assert {r["name"] for r in snap["tie_group"]} == {"Bathroom 1", "Bathroom 2"}
    assert manager.snapshot(sessions.Session(id="x", phone="", address="", created=0.0))["tie_group"] == []
