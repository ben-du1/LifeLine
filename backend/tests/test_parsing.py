from app import questions
from app.sessions import normalize_phone


def test_parse_reply_kinds():
    assert questions.parse_reply("Yes").kind == "yes"
    assert questions.parse_reply("yeah, next to the big bed").extra == "next to the big bed"
    assert questions.parse_reply("No").kind == "no"
    assert questions.parse_reply("nope there isn't").kind == "no"
    assert questions.parse_reply("I don't know").kind == "unsure"
    free = questions.parse_reply("There's a couch and a fireplace")
    assert free.kind == "free" and free.extra.startswith("There's")
    assert questions.parse_reply("nothing like that").kind == "free"  # "no" must be a whole word


def test_room_types_mentioned():
    assert questions.room_types_mentioned("I'm hiding in the bathroom") == {"bathroom"}
    assert questions.room_types_mentioned("there is a bed") == set()


def test_normalize_phone():
    assert normalize_phone("+1 (609) 555-1234") == normalize_phone("6095551234")


def test_is_correction():
    assert questions.is_correction("sorry, I'm actually upstairs")
    assert questions.is_correction("no wait, it's the other bathroom")
    assert not questions.is_correction("there's a sorry looking couch")   # "sorry" only counts at the start
    assert not questions.is_correction("I'm in the kitchen")
    assert questions.is_correction("I moved to the bedroom")         # moving also invalidates the old description
    assert questions.is_correction("now I'm in the garage")


def test_mentions_danger():
    assert questions.mentions_danger("the hall is full of smoke")
    assert questions.mentions_danger("I can't breathe")
    assert not questions.mentions_danger("there's a fireplace next to me")
    assert not questions.mentions_danger("I'm in the kitchen")


def test_is_out():
    for text in ("OUT", "out.", "I'm out", "we got outside", "made it out of the house", "I left the building"):
        assert questions.is_out(text), text
    for text in ("I can't get out", "we're not out yet", "trying to get out of the house", "I'm out of breath",
                 "the door to outside is locked", "I'm in the bathroom", "I got out of the bedroom"):
        assert not questions.is_out(text), text


def test_room_mentions_separate_claims_from_comparisons_and_neighbours():
    assert not questions.room_mentions("big like a living room").claimed   # a size answer, not a location
    assert questions.room_mentions("I'm in the living room").claimed == {"living room"}
    m = questions.room_mentions("I'm in the bedroom closet")
    assert m.claimed == {"closet"} and m.nearby == {"bedroom"}
    assert questions.room_mentions("next to the kitchen").nearby == {"kitchen"}
    m = questions.room_mentions("I'm not in the bathroom, I'm in a bedroom")
    assert m.claimed == {"bedroom"} and m.denied == {"bathroom"}
    assert not questions.room_mentions("bedroom-sized").claimed
    assert questions.room_mentions("I'm by the toilet").claimed == {"bathroom"}   # a fixture places you there


def test_smart_punctuation_and_non_answers():
    # iOS turns "don't" into "don’t"; every reply pattern is written with a plain apostrophe.
    assert questions.parse_reply(questions.normalize("Don’t know")).kind == "unsure"
    assert questions.is_out(questions.normalize("I’m out"))
    assert questions.parse_reply("dunno").kind == "unsure"
    # Only text that could describe a room is scored against the photos.
    for text in ("3", "3rd", "ok", "what?", "Medium"):
        assert not questions.describes_something(text), text
    assert questions.describes_something("big bed and a desk")
