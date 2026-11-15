"""Caller-facing wording and reply reading for caller-initiated calls (workstream C): voice versions of the messages,
out_signal, emphatic no, heat as danger, and looks_like_address."""

import re

import pytest

from app import followups, questions, semantics


def caller_constants():
    """Every caller-facing message constant in questions.py (module-level uppercase strings)."""
    return {name: value for name, value in vars(questions).items()
            if name.isupper() and isinstance(value, str) and not name.startswith("_")}


# ---- wording

def test_first_message_and_out_of_area_wording():
    assert questions.ADDRESS_QUESTION == (
        "911: If you can get out safely, get out now and text OUT. "
        "If you can't, what is the address of the building you're in?")
    assert questions.OUT_OF_AREA_MESSAGE == (
        "That address isn't in this system's service area. "
        "I'm connecting you to standard emergency services now. Stay on the line.")
    assert questions.ADDRESS_CONFIRM.format(address="13122 Chipman Glen Dr") == "Did you mean 13122 Chipman Glen Dr?"
    assert questions.ADDRESS_CHOOSE.format(a="unit 1013", b="unit 107") == "Is it unit 1013 or unit 107?"
    assert "yes or no" in questions.OUT_CONFIRM.lower()


def test_no_demo_wording_anywhere():
    for name, text in caller_constants().items():
        assert "demo" not in text.lower(), name
    for text in questions.VOICE_VERSIONS.values():
        assert "demo" not in text.lower()


# ---- for_channel

def test_every_constant_mentioning_text_has_a_spoken_version():
    for name, text in caller_constants().items():
        if "text" in text.lower():
            spoken = questions.for_channel(text, "voice")
            assert spoken != text, name
            assert "text" not in spoken.lower(), name


def test_voice_versions_keep_the_911_framing_and_placeholders():
    for text, spoken in questions.VOICE_VERSIONS.items():
        assert text.startswith("911:") == spoken.startswith("911:")
        assert re.findall(r"\{(\w+)\}", text) == re.findall(r"\{(\w+)\}", spoken)
    assert "say you're out" in questions.for_channel(questions.ADDRESS_QUESTION, "voice")
    assert "say you're out" in questions.for_channel(questions.OPENING_MESSAGE, "voice")
    assert "on this call" in questions.for_channel(questions.OPENING_MESSAGE, "voice")
    assert "tell me" in questions.for_channel(questions.ADDRESS_RETRY, "voice")


def test_for_channel_finds_formatted_and_joined_messages():
    located = questions.LOCATED_MESSAGE.format(room="primary bedroom")
    spoken = questions.for_channel(located, "voice")
    assert spoken == questions.VOICE_VERSIONS[questions.LOCATED_MESSAGE].format(room="primary bedroom")
    # Joined with other text (an acknowledgement before it) the known part is still replaced.
    joined = questions.for_channel("Okay, a bed. " + questions.BEST_GUESS_MESSAGE, "voice")
    assert joined == "Okay, a bed. " + questions.VOICE_VERSIONS[questions.BEST_GUESS_MESSAGE]
    confirm = questions.ADDRESS_CONFIRM.format(address="3012 Houston Ave")
    assert questions.for_channel(confirm, "voice") == confirm   # reads the same aloud


def test_for_channel_leaves_text_and_unknown_messages_alone():
    for text in caller_constants().values():
        assert questions.for_channel(text, "messaging") == text
    assert questions.for_channel("Is there a window?", "voice") == "Is there a window?"
    assert questions.for_channel("Text me {weird} braces", "voice") == "Text me {weird} braces"
    assert questions.for_channel("", "voice") == ""


# ---- out_signal

@pytest.mark.parametrize("text", ["OUT", "out", "out.", "Out!", "I'm out", "I got out", "we're out", "we are all out",
                                  "out of the building", "outside now", "we got outside", "made it out of the house",
                                  "I left the building", "I’m out", "I'm safe outside", "we're in the front yard"])
def test_out_signal_out(text):
    assert questions.out_signal(text) == "out"
    assert questions.is_out(text)


@pytest.mark.parametrize("text", ["I'm outside the bathroom door", "outside the bathroom", "outside the door",
                                  "I'm outside the room", "outside of my room", "I got out of the bedroom",
                                  "out of the bedroom, in the hallway"])
def test_out_signal_inside_places_are_not_out(text):
    assert questions.out_signal(text) in ("maybe", None)
    assert not questions.is_out(text)   # a trapped caller's call must not end


@pytest.mark.parametrize("text", ["I can't get out", "get me out", "help me get out", "how do I get out",
                                  "we're not out yet", "trying to get out of the house", "I'm out of breath",
                                  "the door to outside is locked", "I can see outside through the window",
                                  "I'm in the bathroom", "i cant get out the bed is blocking the door"])
def test_out_signal_none(text):
    assert questions.out_signal(text) is None
    assert not questions.is_out(text)


@pytest.mark.parametrize("text", ["I think I'm out", "almost out", "not sure if I'm out", "I climbed out the window",
                                  "on the balcony", "outside the front door"])
def test_out_signal_maybe(text):
    assert questions.out_signal(text) == "maybe"
    assert not questions.is_out(text)


# ---- parse_reply

@pytest.mark.parametrize("text", ["absolutely not", "Absolutely not", "definitely not", "of course not", "not at all",
                                  "certainly not", "no way"])
def test_emphatic_denials_are_no(text):
    assert questions.parse_reply(text).kind == "no"


def test_emphatic_yes_still_yes():
    for text in ("absolutely", "definitely", "yes", "yeah"):
        assert questions.parse_reply(text).kind == "yes", text


# ---- danger

@pytest.mark.parametrize("text", ["it's getting hot", "it's so hot in here", "the door is warm", "the door is hot",
                                  "the door feels hot", "there's an orange light under the door",
                                  "something is glowing", "no smoke but the door is hot", "smoke is coming in"])
def test_heat_and_glow_are_danger(text):
    assert questions.mentions_danger(text)


@pytest.mark.parametrize("text", ["there's a fireplace next to me", "the fireplace", "no smoke here",
                                  "I'm in the kitchen", "the door isn't hot", "I'm next to the hot tub",
                                  "there's a fire extinguisher"])
def test_not_danger(text):
    assert not questions.mentions_danger(text)


def test_danger_examples_are_not_location_cues():
    assert semantics.EXAMPLES["danger"]
    assert "danger" not in followups.FEATURES_BY_KEY   # _add_semantic skips it, so no room evidence comes from it


# ---- looks_like_address

@pytest.mark.parametrize("text", ["13122 Chipman Glen Dr", "3012 Houston Ave, Nashville", "123 Main St",
                                  "I'm at 221B Baker Street", "thirty one twelve Houston Avenue",
                                  "one two three main street", "45 W 5th St apt 2", "45 Route 1"])
def test_looks_like_address(text):
    assert questions.looks_like_address(text)


@pytest.mark.parametrize("text", ["I'm in the bathroom", "2 people are in the back bedroom by the door",
                                  "I'm on the 3rd floor", "there's one way out", "three people in the green room",
                                  "5 minutes", "OUT", "help"])
def test_does_not_look_like_address(text):
    assert not questions.looks_like_address(text)
