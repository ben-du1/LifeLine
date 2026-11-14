"""Floor-material answers on a synthetic house whose garage photos out-score every room on every flooring prompt and
on the caller's own words (the reported failure: "carpeted floor" put the dot in the garage)."""

import numpy as np

from app import followups, llm, locator
from app.building import Building

ROOMS = ["Garage", "Bedroom 1", "Bedroom 2", "Kitchen", "Bathroom"]


def square(side: float) -> list[list[float]]:
    return [[0, 0], [side, 0], [side, side], [0, side]]


def make_house():
    rooms = [
        {"id": "g", "name": "Garage", "types": ["Garage"], "polygon": square(6.0), "walls": []},
        {"id": "b1", "name": "Bedroom 1", "types": ["Bedroom"], "polygon": square(3.8), "walls": []},
        {"id": "b2", "name": "Bedroom 2", "types": ["Bedroom"], "polygon": square(3.6), "walls": []},
        {"id": "k", "name": "Kitchen", "types": ["Kitchen"], "polygon": square(4.5), "walls": []},
        {"id": "w", "name": "Bathroom", "types": ["Bathroom"], "polygon": square(2.3), "walls": []},
    ]
    building = Building(
        id="h", name="h", address="3 C St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms, candidates=[{"views": ["g"]}, {"views": ["b1"]}, {"views": ["b2"]}, {"views": ["k"]},
                                 {"views": ["w"]}],
        candidate_room=np.arange(5), has_views=np.ones(5, dtype=bool),
        view_embeddings=np.eye(5, dtype=np.float32), view_candidate=np.arange(5),
    )
    # The garage wins every CLIP prompt, flooring prompts included; the other rooms tie.
    garage_wins = np.array([0.30, 0.22, 0.22, 0.22, 0.22])
    return building, followups.build_model(building, {p: garage_wins for p in followups.clip_prompts()}), garage_wins


def rooms_after(text, pending=None, cues=None):
    """What sessions._process does with one message: free-text CLIP unless explained, then the cue likelihood."""
    building, model, clip_text = make_house()
    from app import questions
    reply = questions.parse_reply(text)
    reading = followups.read_message(text, model, pending, reply.kind, cues)
    post = locator.uniform_room_prior(building.candidate_room, len(ROOMS))
    if reply.kind == "free" and not reading.explained:
        post = locator.update(post, clip_text)
    if reading.likelihood is not None:
        post = locator.apply_likelihood(post, reading.likelihood)
    return locator.room_probabilities(post, building.candidate_room, len(ROOMS)), reading


def flooring_question():
    choice = followups.CHOICES_BY_KEY["flooring"]
    return followups.Plan("choice", "flooring", choice.question, options=choice.options)


def test_carpeted_floor_in_free_text_never_points_at_the_garage():
    for text in ["carpeted floor", "it's carpeted", "there's carpet on the floor", "grey carpet"]:
        probs, reading = rooms_after(text)
        assert probs[0] < 0.05, text
        assert probs[[1, 2]].min() > probs[[0, 3, 4]].max(), text
        assert "flooring" in reading.covered and reading.explained, text


def test_carpet_answer_to_the_flooring_question():
    for answer in ["carpet", "it's carpeted, soft grey carpet"]:
        probs, reading = rooms_after(answer, flooring_question())
        assert probs[0] < 0.05 and probs[[1, 2]].min() > probs[[0, 3, 4]].max(), answer
        assert reading.explained, answer


def test_concrete_points_at_the_garage():
    for text, pending in [("concrete", flooring_question()), ("bare concrete floor", None), ("cement", None)]:
        probs, _ = rooms_after(text, pending)
        assert probs.argmax() == 0, text


def test_flooring_question_offers_concrete():
    question = flooring_question().text
    assert "concrete" in question and "carpet" in question


def test_free_text_flooring_words_are_strict():
    assert followups.flooring_mentioned("a wooden table and chairs") == ()
    assert followups.flooring_mentioned("stone fireplace") == ()
    assert followups.flooring_mentioned("there's no carpet") == ()
    assert followups.flooring_mentioned("not carpet, it's tile") == ("tile",)
    assert followups.flooring_mentioned("hardwood floors") == ("wood",)
    assert not followups.read_message("dark carpet", make_house()[1], None, "free").covered & {"window"}


def test_floor_description_with_other_details_still_goes_to_clip():
    _, reading = rooms_after("carpet and a big bed")
    assert "flooring" in reading.covered and not reading.explained


def test_claude_flooring_cue_and_explained(monkeypatch):
    payload = {"answer": "not_answered", "options": [], "mentioned": [], "flooring": ["carpet", "lava"],
               "size": "unknown", "dark": False, "room_types": [], "floors": [], "acknowledgement": "",
               "explained": False}
    monkeypatch.setattr(llm, "_call", lambda prompt, schema: payload)
    cues = llm.read_message("carpeted floor", None, [])
    assert cues.flooring == ("carpet",) and cues.explained
    probs, _ = rooms_after("carpeted floor", cues=cues)
    assert probs[0] < 0.05
    assert "flooring" in llm.READ_SCHEMA["required"]
