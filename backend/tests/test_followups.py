"""Follow-up planning on a small synthetic building: a closet, a bathroom (no photos), a bedroom and a kitchen."""

import numpy as np

from app import followups, locator
from app.building import Building


def square(side: float) -> list[list[float]]:
    return [[0, 0], [side, 0], [side, side], [0, side]]


def make_model() -> tuple[Building, followups.FeatureModel]:
    rooms = [
        {"id": "c", "name": "Closet", "types": ["Closet"], "polygon": square(1.2), "walls": []},
        {"id": "b", "name": "Bathroom", "types": ["Bathroom"], "polygon": square(2.3), "walls": []},
        {"id": "r", "name": "Bedroom", "types": ["Bedroom"], "polygon": square(3.8), "walls": []},
        {"id": "k", "name": "Kitchen", "types": ["Kitchen"], "polygon": square(4.5), "walls": []},
    ]
    building = Building(
        id="m", name="m", address="1 A St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms,
        candidates=[{"views": ["a"]}, {"views": []}, {"views": ["b"]}, {"views": ["c"]}],
        candidate_room=np.array([0, 1, 2, 3]), has_views=np.array([True, False, True, True]),
        view_embeddings=np.eye(3, dtype=np.float32), view_candidate=np.array([0, 2, 3]),
    )
    # Flat CLIP scores: every test below must work from room labels and geometry alone.
    scores = {p: np.zeros(4) for p in followups.clip_prompts()}
    return building, followups.build_model(building, scores)


def rooms_after(model, building, text, pending=None, kind="free"):
    reading = followups.read_message(text, model, pending, kind)
    prior = locator.uniform_room_prior(building.candidate_room, 4)
    post = locator.apply_likelihood(prior, reading.likelihood) if reading.likelihood is not None else prior
    return locator.room_probabilities(post, building.candidate_room, 4), reading


def test_cramped_favours_small_rooms_and_is_acknowledged():
    building, model = make_model()
    probs, reading = rooms_after(model, building, "I'm in a cramped space")
    assert probs[0] > probs[1] > probs[2] > probs[3]      # closet > bathroom > bedroom > kitchen by area
    assert "size" in reading.covered and reading.acknowledgement == "You said it's a tight space."


def test_mentioned_and_denied_features():
    building, model = make_model()
    probs, reading = rooms_after(model, building, "there's a toilet")
    assert probs.argmax() == 1 and "toilet" in reading.covered   # photo-less bathroom found from its label
    probs, _ = rooms_after(model, building, "there's no toilet here")
    assert probs[1] < 0.25


def test_follow_up_after_cramped_targets_small_rooms():
    building, model = make_model()
    probs, reading = rooms_after(model, building, "I'm in a cramped space")
    posterior = locator.apply_likelihood(locator.uniform_room_prior(building.candidate_room, 4), reading.likelihood)
    plan = followups.choose(model, posterior, building.candidate_room, 4, reading.covered, None)
    asked_about = set(plan.items) if plan.kind == "list" else {plan.key}
    # The follow-up must be about something that separates a closet from a bathroom, not about size again.
    assert asked_about & {"toilet", "sink", "mirror", "shelves"} and "size" not in asked_about
    assert followups.phrase(plan, reading.acknowledgement).startswith("You said it's a tight space.")


def test_answers_to_each_question_kind():
    building, model = make_model()
    yesno = followups.Plan("yesno", "shelves", "Are there shelves?")
    probs, _ = rooms_after(model, building, "yes", yesno, "yes")
    assert probs.argmax() == 0
    choice = followups.Plan("choice", "size", "How big?", options=followups.CHOICES_BY_KEY["size"].options)
    probs, reading = rooms_after(model, building, "bigger", choice)
    assert probs.argmax() == 3 and reading.explained
    listing = followups.Plan("list", "list", "What can you see?", ("toilet", "stove", "bed"))
    probs, _ = rooms_after(model, building, "I see a stove", listing)
    assert probs.argmax() == 3


def test_stops_when_nothing_separates_remaining_rooms():
    building, model = make_model()
    all_topics = {f.key for f in followups.FEATURES} | {c.key for c in followups.CHOICES}
    posterior = np.array([0.25, 0.25, 0.25, 0.25])
    assert followups.choose(model, posterior, building.candidate_room, 4, all_topics, None) is None


def make_two_floor_model():
    """Kitchen and living room downstairs; bedroom and bathroom upstairs."""
    rooms = [
        {"id": "k", "name": "Kitchen", "types": ["Kitchen"], "floor": 0, "polygon": square(4.0), "walls": []},
        {"id": "l", "name": "Living Room", "types": ["Living Room"], "floor": 0, "polygon": square(5.0), "walls": []},
        {"id": "b", "name": "Bedroom", "types": ["Bedroom"], "floor": 1, "polygon": square(3.8), "walls": []},
        {"id": "w", "name": "Bathroom", "types": ["Bathroom"], "floor": 1, "polygon": square(2.3), "walls": []},
    ]
    building = Building(
        id="m2", name="m2", address="2 B St", floorplans={}, rooms=rooms,
        floors=[{"index": 0, "id": "f0", "label": "Floor 1"}, {"index": 1, "id": "f1", "label": "Floor 2"}],
        candidates=[{"views": ["a"]}, {"views": ["b"]}, {"views": ["c"]}, {"views": []}],
        candidate_room=np.array([0, 1, 2, 3]), has_views=np.array([True, True, True, False]),
        view_embeddings=np.eye(3, dtype=np.float32), view_candidate=np.array([0, 1, 2]),
    )
    return building, followups.build_model(building, {p: np.zeros(4) for p in followups.clip_prompts()})


def floor_probs(building, reading):
    prior = locator.uniform_room_prior(building.candidate_room, 4)
    post = locator.apply_likelihood(prior, reading.likelihood) if reading.likelihood is not None else prior
    rooms = locator.room_probabilities(post, building.candidate_room, 4)
    return np.bincount(building.room_floor, weights=rooms, minlength=2)


def test_floor_words_in_free_text():
    building, model = make_two_floor_model()
    for text, floor in [("I'm upstairs", 1), ("I'm on the second floor", 1), ("hiding on floor 2", 1),
                        ("I'm downstairs", 0), ("ground floor, near the door", 0), ("I'm on the 1st floor", 0)]:
        reading = followups.read_message(text, model, None, "free")
        assert floor_probs(building, reading).argmax() == floor, text
        assert "floor" in reading.covered
    assert "floor" not in followups.read_message("there is a toilet", model, None, "free").covered


def test_floor_question_is_asked_first_and_its_answer_parsed():
    building, model = make_two_floor_model()
    posterior = locator.uniform_room_prior(building.candidate_room, 4)
    plan = followups.choose(model, posterior, building.candidate_room, 4, set(), None)
    assert plan.key == "floor" and plan.text == "Which floor are you on: ground floor or 2nd floor?"
    reading = followups.read_message("2nd floor", model, plan, "free")
    assert floor_probs(building, reading)[1] > 0.85 and reading.explained


def test_single_floor_buildings_never_ask_about_floors():
    building, model = make_model()
    assert model.floor_options == () and "floor" not in model.p_option
    posterior = locator.uniform_room_prior(building.candidate_room, 4)
    plan = followups.choose(model, posterior, building.candidate_room, 4, set(), None)
    assert plan.key != "floor"
