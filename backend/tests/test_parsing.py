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
