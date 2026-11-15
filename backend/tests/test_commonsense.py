"""Commonsense guard (commonsense.py): CLIP can't put a toilet in the garage or a bed in the kitchen."""

import numpy as np

from app import commonsense, followups, locator
from app.building import Building


def square(side: float) -> list[list[float]]:
    return [[0, 0], [side, 0], [side, side], [0, side]]


ROOMS = [
    {"id": "b", "name": "Bathroom", "types": ["Bathroom"], "polygon": square(2.3), "walls": []},
    {"id": "r", "name": "Bedroom", "types": ["Bedroom"], "polygon": square(3.8), "walls": []},
    {"id": "k", "name": "Kitchen", "types": ["Kitchen"], "polygon": square(4.5), "walls": []},
    {"id": "g", "name": "Garage", "types": ["Garage"], "polygon": square(6.0), "walls": []},
    {"id": "l", "name": "Living Room", "types": ["Living Room"], "polygon": square(5.5), "walls": []},
    {"id": "u", "name": "Room 1", "types": [], "polygon": square(4.0), "walls": []},
]
N = len(ROOMS)
TOILET = followups.FEATURES_BY_KEY["toilet"].prompt
BED = followups.FEATURES_BY_KEY["bed"].prompt


def make_house(guarded: bool = True):
    """Every room photographed, one sweep each. CLIP is wrong in the absurd direction: the garage scores highest
    for "toilet" (and the unlabelled room second), the kitchen highest for "bed"."""
    building = Building(
        id="h", name="h", address="1 A St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=ROOMS, candidates=[{"views": [str(i)]} for i in range(N)],
        candidate_room=np.arange(N), has_views=np.ones(N, dtype=bool),
        view_embeddings=np.eye(N, dtype=np.float32), view_candidate=np.arange(N),
    )
    scores = {p: np.zeros(N) for p in followups.clip_prompts()}
    scores[TOILET] = np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.04])
    scores[BED] = np.array([0.0, 0.02, 0.08, 0.0, 0.0, 0.0])
    if not guarded:
        original = commonsense.guard
        commonsense.guard = lambda *args: None
        try:
            return building, followups.build_model(building, scores)
        finally:
            commonsense.guard = original
    return building, followups.build_model(building, scores)


def rooms_after(building, model, text, pending=None, kind="free"):
    reading = followups.read_message(text, model, pending, kind)
    prior = locator.uniform_room_prior(building.candidate_room, N)
    return locator.room_probabilities(locator.apply_likelihood(prior, reading.likelihood), building.candidate_room, N)


def test_toilet_in_the_garage_is_overruled():
    building, unguarded = make_house(guarded=False)
    before = rooms_after(building, unguarded, "there's a toilet")
    assert before[3] > 0.15  # the failure this guards against: CLIP made the garage a real contender
    building, model = make_house()
    probs = rooms_after(building, model, "there's a toilet")
    assert probs.argmax() == 0 and probs[3] < 0.10


def test_yes_to_bed_does_not_favour_the_kitchen():
    building, model = make_house()
    bed = followups.Plan("yesno", "bed", "Is there a bed?")
    probs = rooms_after(building, model, "yes", bed, "yes")
    assert probs.argmax() == 1 and probs[2] < 0.10


def test_unlabelled_room_and_possible_features_unaffected():
    building, unguarded = make_house(guarded=False)
    _, model = make_house()
    assert model.p_yes["toilet"][5] == unguarded.p_yes["toilet"][5]       # "Room 1": no bounds
    assert np.array_equal(model.p_yes["window"], unguarded.p_yes["window"])  # window: possible everywhere
    assert np.allclose(model.p_option["flooring"].sum(axis=0), 1.0)


def test_bounds_table():
    assert commonsense.bounds("garage")["toilet"][1] <= 0.08
    assert commonsense.bounds("bathroom")["toilet"][0] >= 0.6
    assert commonsense.bounds("room 1") == {} and commonsense.bounds("") == {}
    assert "window" not in commonsense.bounds("closet")
    assert commonsense.bounds("closet")["large"][1] <= 0.08
    # Several types: the most permissive wins (a hall bathroom has a toilet; an open kitchen/living has a sofa).
    assert "toilet" not in commonsense.bounds("hall bathroom")
    assert "sofa" not in commonsense.bounds("kitchen living room")
    assert commonsense.room_kinds("Primary Bedroom") == {"bedroom"}
    assert commonsense.room_kinds("Half Bath") == {"bathroom"}
