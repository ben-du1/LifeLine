"""End-to-end regressions for the reported reliability bugs, run through the real SessionManager on synthetic
buildings (tests/fakes.py: fake CLIP, no network). Each reproduced a reported bug on the original code (they were
xfail there, with the observed numbers in A18's notes) and passes since the reliability fixes.
"""

import asyncio
import math

import pytest

import fakes
from fakes import Room, Spec, Sweep

# ---- buildings --------------------------------------------------------------------------------------------------

BATH = Spec("TWOBATH", "11 Two Bath Ln, Princeton, NJ", [
    # Only the first bathroom was photographed (common in Matterport scans: small rooms get no sweep).
    Room("Bathroom 1", ["Bathroom"], (0, 0, 2.4, 2.3), [Sweep(1.2, 1.1, [
        {"toilet": 1.0, "tile": 0.6}, {"tub": 1.0, "tile": 0.5}, {"sink": 1.0, "mirror": 0.8},
        {"door": 0.6, "white_wall": 0.6}])]),
    Room("Bathroom 2", ["Bathroom"], (6, 0, 8.4, 2.3)),
    Room("Bedroom", ["Bedroom"], (0, 3, 4, 7), [Sweep(2, 5, [
        {"bed": 1.0, "carpet": 0.5}, {"window": 1.0, "carpet": 0.4}, {"door": 0.6, "carpet": 0.5},
        {"shelves": 0.4, "white_wall": 0.6}])]),
    Room("Kitchen", ["Kitchen"], (4, 3, 9, 7), [Sweep(6.5, 5, [
        {"stove": 1.0, "counter": 0.8}, {"fridge": 1.0, "tile": 0.4}, {"sink": 0.8, "window": 0.6, "counter": 0.6},
        {"table": 1.0, "tile": 0.4}])]),
])

GARAGE = Spec("GARAGE", "22 Garage Ct, Princeton, NJ", [
    Room("Garage", ["Garage"], (0, 0, 6, 6), [
        Sweep(1.5, 3, [{"concrete": 1.0, "car": 0.8}, {"door": 0.8, "concrete": 0.6},
                       {"shelves": 0.8, "concrete": 0.5},
                       # Confuser: a grey floor mat that CLIP scored as carpet in the real model.
                       {"carpet": 0.9, "concrete": 0.4}]),
        Sweep(4.5, 3, [{"car": 1.0, "concrete": 0.6}, {"concrete": 1.0, "white_wall": 0.5},
                       {"carpet": 0.7, "shelves": 0.4, "concrete": 0.4}, {"door": 1.0}]),
    ]),
    Room("Bedroom 1", ["Bedroom"], (6, 0, 10, 4), [Sweep(8, 2, [
        {"bed": 1.0, "carpet": 0.6}, {"window": 1.0, "carpet": 0.4}, {"door": 0.6, "carpet": 0.4},
        {"tv": 0.6, "carpet": 0.4}])]),
    Room("Bedroom 2", ["Bedroom"], (6, 4, 10, 8), [Sweep(8, 6, [
        {"bed": 1.0, "carpet": 0.6}, {"window": 1.0, "fan": 0.5}, {"shelves": 0.6, "carpet": 0.4},
        {"door": 0.6, "white_wall": 0.5}])]),
    Room("Living Room", ["Living Room"], (0, 6, 6, 12), [Sweep(3, 9, [
        {"sofa": 1.0, "wood": 0.6}, {"tv": 1.0, "wood": 0.4}, {"fireplace": 1.0}, {"window": 1.0, "wood": 0.5}])]),
])

# One 12 x 8 m great room: kitchen at the west end, dining in the middle, living at the east end.
OPEN_PLAN = Spec("OPENPLAN", "33 Open Plan Way, Princeton, NJ", [
    Room("Dining Room / Kitchen / Living Room", ["Kitchen", "Dining Room", "Living Room"], (0, 0, 12, 8), [
        Sweep(1.0, 1.0, [{"stove": 1.0, "counter": 0.8}, {"fridge": 1.0, "counter": 0.5},
                         {"counter": 1.0, "sink": 0.6}, {"window": 0.8, "tile": 0.5}]),
        Sweep(3.0, 1.5, [{"counter": 1.0, "sink": 0.7}, {"stove": 0.5, "counter": 0.6}, {"table": 0.8},
                         {"window": 0.8, "tile": 0.4}]),
        Sweep(6.0, 4.0, [{"table": 1.0, "wood": 0.5}, {"window": 0.8}, {"counter": 0.5, "table": 0.4},
                         {"sofa": 0.5, "wood": 0.4}]),
        Sweep(9.0, 2.0, [{"sofa": 1.0, "wood": 0.5}, {"tv": 1.0}, {"window": 1.0}, {"table": 0.4, "wood": 0.5}]),
        Sweep(11.0, 6.0, [{"fireplace": 1.0}, {"sofa": 0.8, "wood": 0.5}, {"window": 0.8}, {"door": 0.6}]),
        Sweep(6.0, 7.0, [{"door": 1.0, "wood": 0.5}, {"window": 0.8}, {"stairs": 0.6}, {"table": 0.5}]),
    ]),
    Room("Bedroom", ["Bedroom"], (0, 8, 4, 12), [Sweep(2, 10, [
        {"bed": 1.0, "carpet": 0.6}, {"window": 1.0}, {"door": 0.6, "carpet": 0.4}, {"fan": 0.8}])]),
    Room("Bathroom", ["Bathroom"], (4, 8, 6.4, 10.3), [Sweep(5.2, 9.1, [
        {"toilet": 1.0, "tile": 0.6}, {"sink": 1.0, "mirror": 0.8}, {"tub": 1.0}, {"door": 0.6}])]),
])

# A normal furnished single-floor house, for the "still works" sanity check.
HOUSE = Spec("HOUSE", "44 Furnished St, Princeton, NJ", [
    Room("Kitchen", ["Kitchen"], (0, 0, 5, 4), [
        Sweep(1.5, 2, [{"stove": 1.0, "counter": 0.8}, {"fridge": 1.0, "tile": 0.4},
                       {"sink": 0.8, "window": 0.6, "counter": 0.6}, {"counter": 0.8, "tile": 0.6}]),
        Sweep(3.5, 2, [{"table": 1.0, "tile": 0.4}, {"window": 1.0}, {"fridge": 0.6, "counter": 0.5},
                       {"door": 0.6}])]),
    Room("Living Room", ["Living Room"], (5, 0, 11, 5), [
        Sweep(7, 2, [{"sofa": 1.0, "wood": 0.6}, {"tv": 1.0, "wood": 0.4}, {"window": 1.0}, {"door": 0.8}]),
        Sweep(9.5, 3, [{"fireplace": 1.0}, {"sofa": 0.6, "wood": 0.5}, {"window": 0.8, "fan": 0.5},
                       {"stairs": 0.3, "wood": 0.5}])]),
    Room("Bedroom 1", ["Bedroom"], (0, 4, 4, 8), [Sweep(2, 6, [
        {"bed": 1.0, "carpet": 0.6}, {"window": 1.0, "fan": 0.6}, {"door": 0.6, "carpet": 0.4},
        {"shelves": 0.4, "white_wall": 0.6}])]),
    Room("Bedroom 2", ["Bedroom"], (4, 5, 8, 9), [Sweep(6, 7, [
        {"bed": 1.0, "carpet": 0.6}, {"tv": 0.8, "carpet": 0.4}, {"door": 0.6, "white_wall": 0.5},
        {"shelves": 0.6, "carpet": 0.4}])]),
    Room("Bathroom", ["Bathroom"], (8, 5, 10.4, 7.3), [Sweep(9.2, 6.1, [
        {"toilet": 1.0, "tile": 0.6}, {"tub": 1.0, "tile": 0.5}, {"sink": 1.0, "mirror": 0.8},
        {"door": 0.6, "white_wall": 0.6}])]),
    Room("Closet", ["Closet"], (10.4, 5, 11.6, 6.2), [Sweep(11, 5.6, [
        {"shelves": 1.0, "carpet": 0.3}, {"shelves": 0.8, "white_wall": 0.6}, {"door": 0.8}, {"white_wall": 1.0}])]),
    Room("Hallway", ["Hallway"], (8, 7.3, 11.6, 8.5), [Sweep(9.8, 7.9, [
        {"door": 1.0, "wood": 0.5}, {"white_wall": 1.0, "wood": 0.4}, {"door": 0.8}, {"white_wall": 0.8}])]),
    Room("Garage", ["Garage"], (11, 0, 17, 5), [Sweep(14, 2.5, [
        {"concrete": 1.0, "car": 0.8}, {"door": 0.8, "concrete": 0.6}, {"shelves": 0.8, "concrete": 0.5},
        {"car": 1.0, "concrete": 0.6}])]),
])

ALL = [BATH, GARAGE, OPEN_PLAN, HOUSE]


@pytest.fixture(autouse=True)
def world(monkeypatch, tmp_path):
    fakes.install(monkeypatch, tmp_path, ALL)


def run(coroutine):
    return asyncio.run(coroutine)


async def call(spec: Spec, *messages: str) -> fakes.Call:
    c = await fakes.start(spec.address)
    for text in messages:
        await c.say(text)
    return c


async def truthful(spec: Spec, opening: str, facts: dict) -> fakes.Call:
    c = await fakes.start(spec.address)
    await c.say(opening)
    while c.session.status == "listening":
        await c.say(fakes.answer(c.session.pending, facts))
    return c


def bathroom_split(c: fakes.Call) -> tuple[float, float, float]:
    rooms = c.rooms()
    one, two = rooms["Bathroom 1"], rooms["Bathroom 2"]
    return one, two, one + two


# ---- (1) photographed vs photo-less bathroom --------------------------------------------------------------------

BATHROOM_MESSAGES = ["there's a toilet", "I'm in the bathroom", "I'm hiding in the bathroom next to the tub"]


@pytest.mark.parametrize("text", BATHROOM_MESSAGES)
def test_bathroom_words_point_at_the_bathrooms(text):
    """Sanity (passes now): whichever bathroom it is, the bathrooms together should hold the probability."""
    assert bathroom_split(run(call(BATH, text)))[2] >= 0.8


@pytest.mark.parametrize("text", BATHROOM_MESSAGES)
def test_photoless_bathroom_keeps_its_share(text):
    """Nothing in "there's a toilet" says which bathroom: the one without photos must not be ruled out."""
    c = run(call(BATH, text))
    one, two, total = bathroom_split(c)
    assert max(one, two) <= 0.7
    assert min(one, two) >= 0.25 * total
    assert c.session.status != "located"


# ---- (2) "carpeted floor" must never flag the garage ------------------------------------------------------------


@pytest.mark.parametrize("text", ["carpeted floor", "the floor is carpeted", "I'm sitting on the carpet",
                                  "carpeted floor, please hurry", "im scared theres carpet everywhere"])
def test_carpet_never_flags_the_garage(text):
    c = run(call(GARAGE, text))
    assert c.rooms()["Garage"] < 0.05
    assert all(s["estimate"] is None or s["estimate"]["room_name"] != "Garage" for s in c.snapshots[1:]
               if s["status"] != "loading")


# ---- (3) open-plan rooms need a region, not a single dot ---------------------------------------------------------

STOVE = (1.0, 1.0)   # the kitchen-end sweep of the great room


def test_open_plan_dot_goes_to_the_kitchen_end():
    """Sanity (passes now): the dot lands on the stove sweep."""
    est = run(call(OPEN_PLAN, "I'm in the kitchen area by the stove")).last["estimate"]
    assert est["room_name"] == "Dining Room / Kitchen / Living Room"
    assert math.dist((est["x"], est["y"]), STOVE) < 1.0


def test_open_plan_estimate_has_a_tight_region():
    est = run(call(OPEN_PLAN, "I'm in the kitchen area by the stove")).last["estimate"]
    region = est.get("region")
    assert region is not None
    assert {"x", "y", "radius_m", "sweeps"} <= set(region)
    assert region["radius_m"] < 3
    assert math.dist((region["x"], region["y"]), STOVE) < 3


# ---- (4) one vague message must not decide the room --------------------------------------------------------------

# Messages that name nothing distinctive (doors and windows are in most rooms). On the original code the ones marked
# originally reported the caller "located" after this single message.
VAGUE = [
    (BATH, "I'm in a room, please help"), (GARAGE, "help me please"), (HOUSE, "I'm in a room, please help"),
    (HOUSE, "there's a door"), (GARAGE, "it's a big room"),
    (BATH, "I'm near a window"),
    (OPEN_PLAN, "there's a door"),
    (BATH, "it's a big room"),
    (GARAGE, "white walls"),
]


@pytest.mark.parametrize("spec,text", VAGUE, ids=lambda v: v.model_id if isinstance(v, Spec) else v)
def test_one_vague_message_is_not_enough(spec, text):
    c = run(call(spec, text))
    # The key requirement is not declaring "located" from one vague message. The probability can still lead clearly
    # in these 4-room test houses (in TWOBATH only the bedroom has a window, and the photo match agrees).
    assert max(c.rooms().values()) <= 0.92
    assert c.session.status != "located"


# ---- (4) posterior floor: no room shown as certain or impossible --------------------------------------------------

CONVERSATIONS = [(BATH, ("there's a toilet",)), (GARAGE, ("the floor is carpeted",)),
                 (OPEN_PLAN, ("I'm in the kitchen area by the stove",)), (HOUSE, ("there's a toilet",)),
                 (HOUSE, ("I'm in a bedroom", "yes", "no", "no", "carpet", "bedroom-sized"))]


def all_room_probabilities() -> list[float]:
    async def go():
        out = []
        for spec, messages in CONVERSATIONS:
            c = await call(spec, *messages)
            out += [r["probability"] for s in c.snapshots + [c.last] for r in s["rooms"]]
        return out
    return run(go())


def test_no_room_is_exactly_zero_or_one():
    """Passes now (the smallest room probability seen is ~1e-4), kept as a guard for the posterior floor."""
    assert all(0.0 < p < 1.0 for p in all_room_probabilities())


def test_no_room_is_displayed_as_0_or_100_percent():
    """No room is ever certain or impossible (the posterior floor), so a later answer can still change the result.
    The dashboard writes anything under 0.5% as "<1%" and over 99.5% as ">99%" (confidence.js pct)."""
    probabilities = all_room_probabilities()
    assert all(0.0 < p < 0.995 for p in probabilities)


# ---- sanity: truthful callers in a normal furnished house are still located ---------------------------------------

TRUTHFUL = [
    ("Kitchen", "I can see a stove and a fridge",
     {"stove": "yes", "fridge": "yes", "sink": "yes", "window": "yes", "table": "yes", "flooring": "tile",
      "size": "bedroom-sized"}),
    ("Bathroom", "it's tiny in here",
     {"toilet": "yes", "sink": "yes", "mirror": "yes", "size": "closet-sized", "flooring": "tile"}),
    ("Bedroom 1", "there's a bed and a ceiling fan",
     {"bed": "yes", "fan": "yes", "window": "yes", "flooring": "carpet", "size": "bedroom-sized"}),
    ("Living Room", "I'm next to the fireplace",
     {"fireplace": "yes", "sofa": "yes", "tv": "yes", "window": "yes", "flooring": "wood", "size": "bigger"}),
]


@pytest.mark.parametrize("truth,opening,facts", TRUTHFUL, ids=[t[0] for t in TRUTHFUL])
def test_truthful_caller_is_located(truth, opening, facts):
    from app import config

    c = run(truthful(HOUSE, opening, facts))
    rooms = c.rooms()
    assert max(rooms, key=rooms.get) == truth
    assert c.session.status == "located"
    assert len(c.session.asked) <= config.MAX_QUESTIONS
