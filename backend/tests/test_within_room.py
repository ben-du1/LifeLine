"""'Which are you closest to?' questions inside a big open-plan room (followups.choose_within_room)."""

import numpy as np

from app import followups, llm, locator
from app.building import Building


def square(side: float) -> list[list[float]]:
    return [[0, 0], [side, 0], [side, side], [0, side]]


def make_open_plan(flat: bool = False):
    """Room 0: open-plan kitchen/living with 4 sweeps (0, 1 at the kitchen end; 2, 3 at the living end).
    Room 1: a bedroom with one sweep."""
    rooms = [
        {"id": "o", "name": "Kitchen / Living", "types": ["Kitchen", "Living Room"], "polygon": square(8.0),
         "walls": []},
        {"id": "r", "name": "Bedroom", "types": ["Bedroom"], "polygon": square(3.8), "walls": []},
    ]
    building = Building(
        id="op", name="op", address="3 C St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms, candidates=[{"views": [str(i)]} for i in range(5)],
        candidate_room=np.array([0, 0, 0, 0, 1]), has_views=np.ones(5, dtype=bool),
        view_embeddings=np.eye(5, dtype=np.float32), view_candidate=np.arange(5),
    )
    scores = {p: np.zeros(5) for p in followups.clip_prompts()}
    if not flat:
        by_key = {f.key: f.prompt for f in followups.FEATURES}
        scores[by_key["stove"]] = np.array([2.0, 1.8, 0.0, 0.1, 0.0])
        scores[by_key["fireplace"]] = np.array([0.0, 0.1, 2.0, 1.9, 0.0])
        scores[by_key["sofa"]] = np.array([1.0, 1.0, 1.2, 1.3, 0.0])  # couch visible from everywhere: flat
    return building, followups.build_model(building, scores)


def in_room_posterior():
    return np.array([0.25, 0.25, 0.25, 0.25, 0.0])


def test_asks_stove_vs_fireplace_in_open_plan_room():
    building, model = make_open_plan()
    plan = followups.choose_within_room(model, in_room_posterior(), building.candidate_room, 0, set())
    assert plan is not None and plan.kind == "near" and plan.room == 0
    assert set(plan.items) == {"stove", "fireplace"}  # the couch is in every photo of the room: flat, skipped
    assert plan.text.startswith("Which are you closest to: ") and "the stove" in plan.text
    assert followups.answer_matrix(plan, model).shape == (len(plan.items) + 1, 5)


def test_fireplace_answer_concentrates_on_living_end():
    building, model = make_open_plan()
    plan = followups.Plan("near", "near", "Which are you closest to: the stove or the fireplace?",
                          ("stove", "fireplace"), room=0)
    for text in ("by the fireplace", "the second one"):
        reading = followups.read_message(text, model, plan, "free")
        post = locator.apply_within_rooms(in_room_posterior(), reading.position, building.candidate_room)
        assert post[2] + post[3] > 0.75, text   # NEAR_FLOOR keeps one "near" answer from being decisive
        assert {"near:stove", "near:fireplace"} <= reading.covered and reading.explained
    # Already asked: the same landmarks are not offered again.
    again = followups.choose_within_room(model, in_room_posterior(), building.candidate_room, 0, reading.covered)
    assert again is None or not ({"stove", "fireplace"} & set(again.items))


def test_not_sure_moves_nothing_but_is_covered():
    building, model = make_open_plan()
    plan = followups.Plan("near", "near", "Which?", ("stove", "fireplace"), room=0)
    reading = followups.read_message("not sure", model, plan, "unsure")
    assert reading.likelihood is None and "near:stove" in reading.covered


def test_flat_room_or_single_sweep_room_asks_nothing():
    building, model = make_open_plan(flat=True)
    assert followups.choose_within_room(model, in_room_posterior(), building.candidate_room, 0, set()) is None
    building, model = make_open_plan()
    assert followups.choose_within_room(model, np.full(5, 0.2), building.candidate_room, 1, set()) is None


def test_near_answer_does_not_move_other_rooms():
    building, model = make_open_plan()
    plan = followups.Plan("near", "near", "Which?", ("stove", "fireplace"), room=0)
    reading = followups.read_message("near the stove", model, plan, "free")
    prior = np.array([0.30, 0.05, 0.10, 0.05, 0.50])
    rooms = locator.room_probabilities(locator.apply_within_rooms(prior, reading.position, building.candidate_room), building.candidate_room, 2)
    assert np.allclose(rooms, [0.5, 0.5])


def test_llm_helpers_handle_near():
    plan = followups.Plan("near", "near", "Which?", ("stove", "sliding_door"), room=0)
    assert "closest to" in llm._pending_description(plan)
    assert llm._required_words(plan) == ["stove", "glass"]
    cues = followups.Cues(mentioned={"sliding_door": "yes"})
    _, model = make_open_plan()
    assert followups.apply_cues(cues, model, plan).covered >= {"near:stove", "near:sliding_door"}
