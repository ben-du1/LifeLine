"""Follow-up planning on a small synthetic building: a closet, a bathroom (no photos), a bedroom and a kitchen."""

import numpy as np

from app import followups, locator, questions
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


def test_bare_floor_answers_count_only_as_answers():
    # A live caller answered "3" and "3rd" and was asked the floor again both times.
    building, model = make_two_floor_model()
    plan = followups.choose(model, locator.uniform_room_prior(building.candidate_room, 4),
                            building.candidate_room, 4, set(), None)
    for text in ("2", "2nd", "second", "the 2nd one", "top"):
        reading = followups.read_message(text, model, plan, questions.parse_reply(text).kind)
        assert floor_probs(building, reading)[1] > 0.85 and "floor" in reading.covered, text
    # Said unprompted, a bare number is not a floor.
    assert "floor" not in followups.read_message("2 windows", model, None, "free").covered


def test_floor_is_asked_even_when_descriptions_favour_one_floor():
    building, model = make_two_floor_model()
    posterior = locator.apply_likelihood(locator.uniform_room_prior(building.candidate_room, 4),
                                         np.array([0.95, 0.95, 0.05, 0.05]))   # ground floor at 95%
    assert followups.choose(model, posterior, building.candidate_room, 4, set(), None).key == "floor"


def test_dont_know_the_floor_leads_to_the_stairs_question():
    building, model = make_two_floor_model()
    posterior = locator.uniform_room_prior(building.candidate_room, 4)
    floor = followups.choose(model, posterior, building.candidate_room, 4, set(), None)
    text = questions.normalize("Don’t know")                 # phones send a curly apostrophe
    reading = followups.read_message(text, model, floor, questions.parse_reply(text).kind)
    assert "floor" in reading.covered and reading.acknowledgement == "That's okay."
    flights = followups.choose(model, posterior, building.candidate_room, 4, reading.covered, "choice")
    assert flights.key == "flights"
    for text, up in (("yes", 1), ("one flight", 1), ("no", 0), ("none", 0)):
        reading = followups.read_message(text, model, flights, questions.parse_reply(text).kind)
        assert floor_probs(building, reading).argmax() == up, text


def test_single_floor_buildings_never_ask_about_floors():
    building, model = make_model()
    assert model.floor_options == () and "floor" not in model.p_option
    posterior = locator.uniform_room_prior(building.candidate_room, 4)
    plan = followups.choose(model, posterior, building.candidate_room, 4, set(), None)
    assert plan.key != "floor"


def make_kitchen_with_three_scans():
    """One kitchen scanned at three spots: by the stove, by the sink, by the window."""
    rooms = [{"id": "k", "name": "Kitchen", "types": ["Kitchen"], "floor": 0, "polygon": square(6.0), "walls": []},
             {"id": "b", "name": "Bedroom", "types": ["Bedroom"], "floor": 0, "polygon": square(4.0), "walls": []}]
    candidates = [{"views": ["s"], "x": 0.5, "y": 0.5}, {"views": ["k"], "x": 5.5, "y": 0.5},
                  {"views": ["w"], "x": 3.0, "y": 5.5}, {"views": ["b"], "x": 9.0, "y": 2.0}]
    building = Building(
        id="k3", name="k3", address="3 C St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms, candidates=candidates, candidate_room=np.array([0, 0, 0, 1]),
        has_views=np.array([True, True, True, True]), view_embeddings=np.eye(4, dtype=np.float32),
        view_candidate=np.array([0, 1, 2, 3]),
    )
    scores = {p: np.zeros(4) for p in followups.clip_prompts()}
    scores[followups.near_prompt("stove")] = np.array([0.30, 0.20, 0.20, 0.20])
    scores[followups.near_prompt("sink")] = np.array([0.20, 0.30, 0.20, 0.20])
    scores[followups.near_prompt("window")] = np.array([0.20, 0.20, 0.30, 0.20])
    model = followups.build_model(building, scores)
    model.p_yes["window"] = np.full(4, 0.8)   # the kitchen has windows (no CLIP signal in this toy model)
    return building, model


def test_nearest_question_narrows_the_position_inside_the_room():
    building, model = make_kitchen_with_three_scans()
    xy = np.array([[c["x"], c["y"]] for c in building.candidates])
    posterior = np.array([0.33, 0.33, 0.33, 0.01])                     # kitchen known, spot unknown
    plan = followups.refinement(model, posterior, 0, xy, set())
    assert plan.kind == "nearest" and {"stove", "sink"} <= set(plan.items)
    reading = followups.read_message("I'm right by the sink", model, plan, "free")
    assert reading.likelihood is None                                   # nothing about which room
    post = locator.apply_within_rooms(posterior, reading.position, building.candidate_room)
    assert post[1] > 0.6 and np.isclose(post[3], posterior[3] / posterior.sum())   # the bedroom keeps its share
    none = followups.read_message("none of those", model, plan, "no")
    assert none.position is not None                                    # "none" is evidence too


def test_no_nearest_question_when_position_is_already_tight_or_room_has_one_scan():
    building, model = make_kitchen_with_three_scans()
    xy = np.array([[c["x"], c["y"]] for c in building.candidates])
    assert followups.refinement(model, np.array([0.98, 0.01, 0.0, 0.01]), 0, xy, set()) is None   # tight
    assert followups.refinement(model, np.array([0.0, 0.0, 0.0, 1.0]), 1, xy, set()) is None      # one scan


def make_lookalike_bedrooms():
    """Two bedrooms that look the same in photos, a hallway and a bathroom. Only the layout tells the bedrooms
    apart: Bedroom A has a window and doors to the hallway and to Bedroom B; Bedroom B has no window and opens
    only onto Bedroom A."""
    rooms = [
        {"id": "a", "name": "Bedroom A", "types": ["Bedroom"], "polygon": square(3.8), "walls": []},
        {"id": "b", "name": "Bedroom B", "types": ["Bedroom"], "polygon": square(3.8), "walls": []},
        {"id": "h", "name": "Hallway", "types": ["Hallway"], "polygon": [[0, 0], [6, 0], [6, 1], [0, 1]], "walls": []},
        {"id": "w", "name": "Bathroom", "types": ["Bathroom"], "polygon": square(2.3), "walls": []},
    ]
    building = Building(
        id="l", name="l", address="4 D St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms, candidates=[{"views": ["a"]}, {"views": ["b"]}, {"views": ["h"]}, {"views": []}],
        candidate_room=np.array([0, 1, 2, 3]), has_views=np.array([True, True, True, False]),
        view_embeddings=np.eye(3, dtype=np.float32), view_candidate=np.array([0, 1, 2]),
    )

    def door(to, kind):
        return {"to": to, "to_kind": kind, "x": 0, "y": 0, "width": 0.8}

    layout = {"rooms": {
        "a": {"kind": "bedroom", "doors": [door("h", "hallway"), door("b", "bedroom")],
              "windows": [{"x": 0, "y": 0, "normal": [0, 1], "width": 1.5}]},
        "b": {"kind": "bedroom", "doors": [door("a", "bedroom")], "windows": []},
        "h": {"kind": "hallway", "doors": [door("a", "bedroom"), door("w", "bathroom")], "windows": []},
        "w": {"kind": "bathroom", "doors": [door("h", "hallway")], "windows": []},
    }}
    scores = {p: np.zeros(4) for p in followups.clip_prompts()}
    return building, followups.build_model(building, scores, layout)


def test_layout_questions_tell_lookalike_rooms_apart():
    building, model = make_lookalike_bedrooms()
    bedrooms = np.array([0.5, 0.5, 0.0, 0.0])                          # "I'm in a bedroom": which one?
    plan = followups.choose(model, bedrooms, building.candidate_room, 4, set(), None)
    assert plan.key in ("windows", "window", "door_to")                 # photos can't help; the layout can
    for key, said, expected in [("windows", "none", 1), ("windows", "just one", 0), ("door_to", "a hallway", 0)]:
        choice = followups.Plan("choice", key, "?", options=model.choices[key].options)
        reading = followups.read_message(said, model, choice, "free")
        post = locator.apply_likelihood(bedrooms, reading.likelihood)
        assert post.argmax() == expected and post[expected] > 0.75, (key, said)


def test_more_than_one_window_is_not_also_one():
    building, model = make_lookalike_bedrooms()
    choice = followups.Plan("choice", "windows", "?", options=model.choices["windows"].options)
    assert followups.rule_cues("more than one", choice, "free").options == ("several",)
    reading = followups.read_message("more than one", model, choice, "free")
    assert "window" in reading.covered                                  # don't then ask "is there a window?"


def test_door_answer_is_about_the_next_room_not_this_one():
    building, model = make_lookalike_bedrooms()
    choice = followups.Plan("choice", "door_to", "?", options=model.choices["door_to"].options)
    cues = followups.rule_cues("a hallway", choice, "free")
    assert cues.options == ("hallway",) and "narrow" not in cues.mentioned   # "hallway" is not a narrow-room cue


def test_nearby_room_is_adjacency_evidence_not_a_claim():
    building, model = make_lookalike_bedrooms()
    reading = followups.read_message("I'm in a room next to the hallway", model, None, "free")
    post = locator.apply_likelihood(np.full(4, 0.25), reading.likelihood)
    assert post[0] > post[1]                                            # Bedroom A opens onto the hallway


def test_verification_picks_a_question_that_separates_the_leading_room():
    building, model = make_lookalike_bedrooms()
    posterior = np.array([0.8, 0.15, 0.03, 0.02])                       # Bedroom A leads, B is the alternative
    plan = followups.verification(model, posterior, 0, set())
    assert plan is not None and plan.key in ("windows", "window", "door_to", "bath_door")
    # Once everything that could separate them was asked, there is nothing left to check with.
    everything = {f.key for f in followups.FEATURES} | {c.key for c in followups.CHOICES} | set(model.choices)
    assert followups.verification(model, posterior, 0, everything) is None


def test_front_middle_back_follows_the_street_side():
    # Three same-size rooms in a row; the street is toward +x (orientation: yaw 0, street due east).
    rooms = [{"id": str(i), "floor": 0, "polygon": [[4 * i, 0], [4 * i + 4, 0], [4 * i + 4, 4], [4 * i, 4]]}
             for i in range(3)]
    toward = followups._toward_street({"yaw_deg": 0.0, "street": {"bearing_deg": 90.0}})
    probs = followups._position_likelihoods(rooms, toward)             # rows: front, middle, back
    assert list(probs.argmax(axis=0)) == [2, 1, 0]                     # back room, middle room, street-side room
    assert np.allclose(probs.sum(axis=0), 1)
