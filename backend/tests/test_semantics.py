"""Paraphrase understanding with the real local model (downloads all-MiniLM-L6-v2 once, ~90 MB)."""

import numpy as np

from app import followups, semantics
from tests.test_followups import make_model


def cues_for(text, pending=None, kind="free"):
    return followups.rule_cues(text, pending, kind, semantics.scores(text))


def test_paraphrases_the_keywords_miss():
    assert cues_for("I can touch both walls with my arms").size == "tight"
    assert cues_for("there's the thing you wash clothes in").mentioned.get("washer") == "yes"
    assert cues_for("I'm next to where you cook").mentioned.get("stove") == "yes"
    assert cues_for("my coats are hanging right next to me").mentioned.get("shelves") == "yes"
    assert cues_for("no lights, can't see a thing").dark
    # The keyword rules alone see none of these.
    assert followups.rule_cues("I can touch both walls with my arms", None, "free").size is None
    assert not followups.rule_cues("there's the thing you wash clothes in", None, "free").mentioned


def test_no_false_cues_from_neutral_or_negated_messages():
    for text in ("please help me", "I'm scared, hurry", "I don't know"):
        cues = cues_for(text)
        assert not cues.mentioned and cues.size is None and not cues.dark, text
    assert cues_for("there's no toilet in here").mentioned.get("toilet") == "no"   # keyword negation wins


def test_paraphrase_drives_the_locator():
    building, model = make_model()
    reading = followups.read_message("I can touch both walls", model, None, "free",
                                     semantic=semantics.scores("I can touch both walls"))
    assert reading.acknowledgement == "You said it's a tight space."
    assert np.argmax(reading.likelihood) == 0   # the closet (smallest room)
