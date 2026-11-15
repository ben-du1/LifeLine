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
    # The follow-up must be about something that separates a closet from a bathroom, not about size again. Flooring
    # qualifies: bathrooms are essentially never carpeted (followups._NEVER_CARPETED).
    assert asked_about & {"toilet", "sink", "mirror", "shelves", "flooring"} and "size" not in asked_about
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


# ---- calibrated CLIP presence (contrast against neutral prompts) -----------------------------------------------

WINDOW = followups.FEATURES_BY_KEY["window"].prompt  # no room types, so p_yes is CLIP alone
WASHER = followups.FEATURES_BY_KEY["washer"].prompt


def contrast_scores(feature_scores: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Per-candidate raw scores: every prompt at 0.20 (the neutral level) unless given; ensemble phrasings copy their
    original prompt."""
    scores = {p: np.full(4, 0.20) for p in followups.clip_prompts()}
    for prompt, values in feature_scores.items():
        for p in followups._ensemble(prompt):
            scores[p] = np.array(values)
    return scores


def test_feature_absent_everywhere_is_unlikely_everywhere(monkeypatch):
    monkeypatch.setattr(followups.config, "CLIP_CALIBRATED", True)
    building, _ = make_model()
    # Below or tied with the neutral prompts in every imaged room: the old z-score called the best of them ~0.8.
    scores = contrast_scores({WINDOW: [0.18, 0.20, 0.20, 0.17], WASHER: [0.19, 0.20, 0.16, 0.20]})
    model = followups.build_model(building, scores)
    for key in ("window", "washer"):
        assert model.p_yes[key][building.has_views].max() < 0.3, key  # photo-less rooms have no CLIP evidence


def test_feature_present_in_one_room_is_likely_there_only(monkeypatch):
    monkeypatch.setattr(followups.config, "CLIP_CALIBRATED", True)
    building, _ = make_model()
    model = followups.build_model(building, contrast_scores({WINDOW: [0.18, 0.20, 0.19, 0.28]}))
    p = model.p_yes["window"]
    assert p[3] > 0.8 and p[[0, 2]].max() < 0.3
    assert p[1] == 0.5  # the photo-less bathroom has no CLIP evidence and no room type for windows


def test_without_baseline_prompts_clip_presence_stays_relative():
    building, _ = make_model()
    raw = np.array([0.18, 0.20, 0.19, 0.28])
    scores = {p: raw for p in followups.clip_prompts() if p not in followups.BASELINE_PROMPTS}
    p = followups.build_model(building, scores).p_yes["window"]
    imaged = raw[[0, 2, 3]]
    expected = followups._sigmoid(1.5 * (imaged - imaged.mean()) / (imaged.std() + 1e-6))
    assert np.allclose(p[[0, 2, 3]], np.clip(expected, 0.05, 0.95))


def test_view_contrast_ignores_a_cluttered_view(monkeypatch):
    """Per-view baseline: a busy face that scores high against every prompt (neutral ones too) doesn't look like a
    window, while a face where only the window prompt is high does. One-hot view embeddings make each prompt
    vector's entries its per-view cosines."""
    monkeypatch.setattr(followups.config, "CLIP_CALIBRATED", True)
    building, _ = make_model()
    building.view_embeddings = np.eye(4, dtype=np.float32)
    building.view_candidate = np.array([0, 0, 2, 3])  # closet has a plain and a cluttered face
    vectors = {p: np.array([0.20, 0.30, 0.20, 0.20]) for p in followups.clip_prompts()}  # face 1 is cluttered
    for p in followups._ensemble(WINDOW):
        vectors[p] = np.array([0.19, 0.31, 0.19, 0.28])  # window only in the kitchen (face 3)
    contrast = followups.view_contrast(building, vectors)
    scores = {p: building.candidate_scores(v) for p, v in vectors.items()}
    p = followups.build_model(building, scores, contrast).p_yes["window"]
    assert p[3] > 0.8 and p[0] < 0.3 and p[2] < 0.3


def test_floor_destinations_and_negations_are_not_positive_floor_evidence():
    options = followups.floor_options(3)
    assert followups.stated_floors("not the second floor", options) == ()
    assert followups.stated_floors("I'm by the stairs to the second floor", options) == ()
    assert followups.stated_floors("I'm on the third floor", options) == (2,)
