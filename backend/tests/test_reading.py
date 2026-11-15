"""Table of realistic caller texts (typos, no punctuation, panic, several clauses) and how the keyword rules must read
them: reply kind (questions.parse_reply), room types named / denied, and the cues followups.rule_cues finds."""

import sys
import types

import numpy as np
import pytest

from app import followups, questions

LIST = "list:toilet,stove,bed"


def plan(spec: str | None) -> followups.Plan | None:
    if spec is None:
        return None
    kind, key = spec.split(":")
    if kind == "yesno":
        return followups.Plan("yesno", key, followups.FEATURES_BY_KEY[key].question)
    if kind == "choice":
        choice = followups.CHOICES_BY_KEY[key]
        return followups.Plan("choice", key, choice.question, options=choice.options)
    return followups.Plan("list", "list", "What can you see?", tuple(key.split(",")))


def case(text, kind="free", rooms=(), denied=(), mentioned=None, size=None, dark=False, pending=None, answer=None,
         options=()):
    return pytest.param(text, kind, set(rooms), set(denied), mentioned or {}, size, dark, pending, answer,
                        tuple(options), id=text[:40])


CASES = [
    # ---- plain answers, as texted ----
    case("Yes", "yes"),
    case("yes!!", "yes"),
    case("yess", "yes"),
    case("YES THERE IS", "yes"),
    case("ya", "yes"),
    case("y", "yes"),
    case("yeah", "yes"),
    case("yep", "yes"),
    case("um yes", "yes"),
    case("i think so", "yes"),
    case("No", "no"),
    case("noo", "no"),
    case("nope", "no"),
    case("nah", "no"),
    case("nope there isn't", "no"),
    case("i don't think so", "no"),
    case("no, im in the kitchen", "no", rooms={"kitchen"}),
    case("idk", "unsure"),
    case("not sure", "unsure"),
    case("maybe?", "unsure"),
    case("I don't know", "unsure"),
    case("yes and no", "unsure"),
    case("no idea", "unsure"),
    # ---- answers to the pending question ----
    case("yes", "yes", pending="yesno:toilet", answer="yes"),
    case("no", "no", pending="yesno:window", answer="no"),
    case("yes, next to the big bed", "yes", mentioned={"bed": "yes"}, pending="yesno:sink", answer="yes"),
    case("no window", "free", pending="yesno:window", answer="no"),
    case("there's a tub", "free", rooms={"bathroom"}, pending="yesno:toilet", answer="yes"),
    case("i dont see any windows", "no", pending="yesno:window", answer="no"),
    case("there is no toilet", "free", pending="yesno:toilet", answer="no"),
    case("idk", "unsure", pending="yesno:fan", answer="unsure"),
    case("the second one", "free", mentioned={"stove": "yes"}, pending=LIST),
    case("last one", "free", mentioned={"bed": "yes"}, pending=LIST),
    case("sink", "free", mentioned={"sink": "yes"}, pending=LIST),
    case("none of those", "no", pending=LIST, answer="none"),
    case("all of them", "free", mentioned={"toilet": "yes", "stove": "yes", "bed": "yes"}, pending=LIST),
    case("the first one", "free", pending="choice:size", options=("small",)),
    case("bigger", "free", pending="choice:size", options=("large",)),
    # ---- room names ----
    case("I'm hiding in the bathroom", rooms={"bathroom"}),
    case("im in the bathrom", rooms={"bathroom"}),
    case("kitchn", rooms={"kitchen"}),
    case("upstairs bathroom", rooms={"bathroom"}),
    case("im in my room", rooms={"bedroom"}),
    case("hiding in the hall closet", rooms={"closet"}),
    case("in the bedroom closet", rooms={"closet"}),
    case("I'm NOT in the garage", denied={"garage"}),
    case("not the kitchen", denied={"kitchen"}),
    case("not in the kitchen im in the bathroom", rooms={"bathroom"}, denied={"kitchen"}),
    case("I'm in the bedroom, not the kitchen", rooms={"bedroom"}, denied={"kitchen"}),
    case("no not the kitchen the bathroom", "no", rooms={"bathroom"}, denied={"kitchen"}),
    case("the room next to the kitchen", rooms=()),
    case("i ran from the garage into a closet", rooms={"closet"}),
    case("maybe the kitchen", "unsure"),
    case("there is a bed", mentioned={"bed": "yes"}),
    case("i can see the front door", mentioned={"front_door": "yes"}),
    case("I can barely move, I'm squeezed in by the tub", rooms={"bathroom"}, size="tight",
         mentioned={"toilet": "yes"}),
    case("I'm downstairs in a room with wood paneling and a bookcase"),
    case("i can hear footsteps upstairs"),
    case("theres a laundry basket and a bed", mentioned={"bed": "yes"}),
    # ---- negation scope ----
    case("no window but there's a toilet", rooms={"bathroom"}, mentioned={"window": "no", "toilet": "yes"}),
    case("no window theres a toilet", rooms={"bathroom"}, mentioned={"window": "no", "toilet": "yes"}),
    case("there isn't a toilet", mentioned={"toilet": "no"}),
    case("there's no toilet here", mentioned={"toilet": "no"}),
    case("no toilet or sink", mentioned={"toilet": "no", "sink": "no"}),
    case("i don't think there's a window", mentioned={"window": "no"}),
    case("windowless room", mentioned={"window": "no"}),
    case("idk if theres a window", "unsure"),
    case("i cant get out the bed is blocking the door", mentioned={"bed": "yes"}),
    case("there's a shower but no toilet", rooms={"bathroom"}, mentioned={"toilet": "yes"}),
    # ---- size and darkness ----
    case("the door is open"),
    case("its wide open in here"),
    case("I'm in a cramped space", size="tight"),
    case("its really small and dark", size="small", dark=True),
    case("small room no windows", size="small", mentioned={"window": "no"}),
    case("there's a big bed", mentioned={"bed": "yes"}),
    case("a little scared"),
    case("im in a little room", size="small"),
    case("its not big", size=None),
    case("huge room with a couch and tv", size="large", mentioned={"sofa": "yes", "tv": "yes"}),
    case("lights are off", dark=True),
    case("dark wood floor and a table", mentioned={"table": "yes"}),
    case("hold tight"),
    # ---- object words that used to point at the wrong room ----
    case("cabinets and a mirror", mentioned={"mirror": "yes"}),
    case("theres a counter and a sink", mentioned={"sink": "yes"}),
    case("theres a stove and cabinets", mentioned={"stove": "yes"}),
    case("a dishwasher", mentioned={"stove": "yes"}),
    case("hair dryer on the counter", mentioned={}),
    case("bookshelf and a desk", mentioned={}),
    case("exhaust fan", mentioned={}),
    case("ceiling fan", mentioned={"fan": "yes"}),
    case("two steps away from the stairs", mentioned={"stairs": "yes"}),
    case("sliding wardrobe doors", mentioned={}),
    case("sliding glass door", mentioned={"sliding_door": "yes"}),
    case("a nightstand and bedside table", mentioned={}),
    case("im in a wheelchair", mentioned={}),
    # ---- messy, multi-clause ----
    case("HELP theres a toilet and a sink im in the bathroom pls", rooms={"bathroom"},
         mentioned={"toilet": "yes", "sink": "yes"}),
    case("yes theres a fridge and stove im in the kitchen", "yes", rooms={"kitchen"},
         mentioned={"fridge": "yes", "stove": "yes"}),
    case("its small and theres no window, i think im in a closet", rooms={"closet"}, size="small",
         mentioned={"window": "no"}),
]


@pytest.mark.parametrize("text,kind,rooms,denied,mentioned,size,dark,pending,answer,options", CASES)
def test_reading(text, kind, rooms, denied, mentioned, size, dark, pending, answer, options):
    reply = questions.parse_reply(text)
    assert reply.kind == kind
    assert questions.room_types_mentioned(text) == rooms
    assert questions.room_types_denied(text) == denied
    cues = followups.rule_cues(text, plan(pending), reply.kind)
    assert cues.mentioned == mentioned
    assert cues.size == size
    assert cues.dark == dark
    assert cues.answer == answer
    assert cues.options == options


def test_denied_room_types_are_down_weighted_in_evidence():
    # sessions imports the CLIP stack at load; stub it, the hook under test is pure numpy.
    for name in ("open_clip", "torch", "PIL", "PIL.Image"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["PIL"].Image = sys.modules["PIL.Image"]
    if not hasattr(sys.modules["torch"], "Tensor"):
        sys.modules["torch"].Tensor = object  # only used in an annotation at import
    from app import config, sessions
    from tests.test_followups import make_model

    building, _ = make_model()
    factor = sessions.room_denial(building, {"kitchen"})
    assert list(factor) == [1.0, 1.0, 1.0, config.ROOM_DENIAL_FACTOR]
    assert sessions.room_denial(building, set()) is None
    assert sessions.room_denial(building, {"garage"}) is None  # no such room: nothing to down-weight
    np.testing.assert_allclose(sessions.room_denial(building, {"Kitchen".lower()}), factor)
