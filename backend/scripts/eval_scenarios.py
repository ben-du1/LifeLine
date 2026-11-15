"""Truthful scripted callers run through the real SessionManager (simulated messaging).

Usage (from backend/): .venv/Scripts/python scripts/eval_scenarios.py [-v]
Each caller opens with a description (often vague, like "I'm in a cramped space") and then answers the planned
follow-ups from a fact sheet of what the room really contains. -v prints every question asked.
The sample house (13122 Chipman Glen) is vacant: empty bedrooms, no furniture. Its bathrooms have no photos.
"""
import asyncio
import sys

sys.path.insert(0, ".")
from app import followups
from app.sessions import SessionManager

CHIPMAN = "13122 Chipman Glen Dr, Houston, TX"
POINTER = "7921 Pointer St, Houston, TX"
HOUSTON_AVE = "3012 Houston Ave, Houston, TX"   # 4 floors, 60 rooms (multi-floor scenarios)

# (address, true room name, opening message, facts). Facts: feature key -> "yes"; choice key -> answer text.
# Features not listed are answered "no"; choices not listed are answered "I don't know".
SCENARIOS = [
    (CHIPMAN, "Bedroom 1", "I'm in an empty room with a window and a ceiling fan",
     {"window": "yes", "fan": "yes", "size": "bedroom-sized", "flooring": "carpet"}),
    (CHIPMAN, "Bedroom 2", "empty room with two windows with blinds", {"window": "yes", "size": "bedroom-sized"}),
    (CHIPMAN, "Dining Room / Kitchen / Living Room", "I can see a stove and a fridge",
     {"stove": "yes", "fridge": "yes", "sink": "yes", "window": "yes", "size": "bigger"}),
    (CHIPMAN, "Dining Room / Kitchen / Living Room", "there's a brick fireplace",
     {"fireplace": "yes", "window": "yes", "fan": "yes", "size": "bigger", "sliding_door": "yes"}),
    (CHIPMAN, "Bathroom 1", "I'm in a cramped space",
     {"toilet": "yes", "sink": "yes", "mirror": "yes", "size": "closet-sized", "flooring": "tile"}),
    (CHIPMAN, "Hallway", "I'm in a narrow space", {"narrow": "yes", "size": "closet-sized"}),
    # Facts below were checked against the sweeps' photos (washer hook-ups but no machine; the kitchen has no fridge).
    (POINTER, "Pantry/Laundry Room", "I'm in a small room with lots of shelves",
     {"shelves": "yes", "window": "yes", "flooring": "wood", "size": "closet-sized"}),
    (POINTER, "Closet 4", "I'm in a cramped space", {"shelves": "yes", "flooring": "wood", "size": "closet-sized"}),
    (POINTER, "Bathroom", "it's tiny in here", {"toilet": "yes", "sink": "yes", "mirror": "yes", "size": "closet-sized"}),
    (POINTER, "Kitchen", "I'm next to the stove",
     {"stove": "yes", "sink": "yes", "window": "yes", "flooring": "wood", "size": "bedroom-sized"}),
    (POINTER, "Bedroom 3", "it's a big room with a bed", {"bed": "yes", "window": "yes", "fan": "yes", "size": "bigger"}),
    # Open-ended paraphrases that use none of the feature keywords (understood through semantics.py).
    (POINTER, "Closet 4", "I can touch both walls and my coats are hanging next to me",
     {"shelves": "yes", "flooring": "wood", "size": "closet-sized"}),
    (POINTER, "Kitchen", "I'm right next to where you cook",
     {"stove": "yes", "sink": "yes", "window": "yes", "flooring": "wood", "size": "bedroom-sized"}),
    (POINTER, "Bathroom", "I can barely move, I'm squeezed in by the tub",
     {"toilet": "yes", "sink": "yes", "mirror": "yes", "size": "closet-sized"}),
    (CHIPMAN, "Dining Room / Kitchen / Living Room", "the room is massive, there's a brick mantel",
     {"fireplace": "yes", "window": "yes", "fan": "yes", "size": "bigger", "sliding_door": "yes"}),
    # Multi-floor: 3012 Houston Ave. Facts checked against the photos. "floor" answers the floor question.
    (HOUSTON_AVE, "Office", "I'm downstairs in a room with wood paneling and a bookcase",
     {"window": "yes", "shelves": "yes", "flooring": "wood", "size": "bedroom-sized", "floor": "ground floor"}),
    (HOUSTON_AVE, "Game Room", "I'm in an empty room with built-in shelves",
     {"window": "yes", "shelves": "yes", "flooring": "wood", "size": "bedroom-sized", "floor": "3rd floor"}),
    (HOUSTON_AVE, "Loft", "I'm on the top floor near the stairs, there's a couch",
     {"stairs": "yes", "sofa": "yes", "window": "yes", "sliding_door": "yes", "flooring": "tile", "size": "bigger",
      "floor": "4th floor"}),
    (HOUSTON_AVE, "Kitchen", "I'm in the kitchen",   # there is a kitchen on floor 1 and another on floor 2
     {"stove": "yes", "fridge": "yes", "sink": "yes", "window": "yes", "flooring": "tile", "size": "bigger",
      "floor": "ground floor"}),
]


def answer(plan: followups.Plan, facts: dict) -> str:
    if plan.kind == "yesno":
        return facts.get(plan.key, "no")
    if plan.kind == "choice":
        return facts.get(plan.key, "I don't know")
    if plan.kind in ("position", "near"):
        # The fact sheets don't say which rooms share walls or which landmark is closest: a truthful caller who
        # doesn't know says so (answering "none of those" would read as a confident "no").
        return "I'm not sure"
    seen = [followups.FEATURES_BY_KEY[k].noun.split(" or ")[0] for k in plan.items if facts.get(k) == "yes"]
    return f"I see {' and '.join(seen)}" if seen else "none of those"


async def run(address, truth, opening, facts, verbose):
    async def notify(_):
        pass

    manager = SessionManager(notify)
    session = await manager.start_call("+10000000000", address)
    while session.status == "loading":
        await asyncio.sleep(0.1)
    await manager.handle_reply(session, opening)
    while session.status == "listening":
        await manager.handle_reply(session, answer(session.pending, facts))
    snap = manager.snapshot(session)
    top = snap["rooms"][0]
    ok = "OK " if top["name"] == truth else "BAD"
    floor_note = ""
    if len(snap["floors"]) > 1:  # also report whether the floor is right, even when the room isn't
        true_floor = next(r["floor_label"] for r in snap["rooms"] if r["name"] == truth)
        best_floor = max(snap["floors"], key=lambda f: f["probability"])
        floor_note = (f"  floor {'OK' if best_floor['label'] == true_floor else 'BAD'} "
                      f"({best_floor['label']} {best_floor['probability']:.2f})")
    rest = ", ".join(f"{r['name']}={r['probability']:.2f}" for r in snap["rooms"][1:3])
    print(f"{ok} {truth[:22]:22} <- {opening!r:55} -> {session.status:10} {top['name']}={top['probability']:.2f} "
          f"({rest})  {len(session.asked)} follow-ups{floor_note}")
    if verbose:
        for m in session.messages:
            if m["role"] != "system":
                print(f"      {m['role']:8}: {m['text']}")
    return ok == "OK "


async def main():
    verbose = "-v" in sys.argv
    results = [await run(*scenario, verbose) for scenario in SCENARIOS]
    print(f"\n{sum(results)}/{len(results)} correct")


asyncio.run(main())
