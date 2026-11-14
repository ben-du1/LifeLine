"""A commonsense guard on the feature model: what each kind of room can plausibly contain.

CLIP sometimes contradicts plain common sense. A white garage wall scores as "a bathroom with a toilet and bathtub",
a kitchen island as "a bedroom with a bed", and one "yes, there's a toilet" then puts the dot in the garage. The room
label already pulls against this (followups._combine counts it double), but a strong enough CLIP score still wins.

This module gives, per room type, each feature's category:
  - "typical": nearly always there (a toilet in a bathroom). P(yes | room) is at least COMMONSENSE_TYPICAL_MIN.
  - "possible": may or may not be there (a window anywhere). No bound; the other sources decide.
  - "implausible": would be absurd (a toilet in a garage, a bed in a kitchen). P(yes | room) is at most
    COMMONSENSE_IMPLAUSIBLE_MAX.

Why bounds rather than replacing CLIP with this table:
  - Labels can be wrong or loose (an open-plan "Living Room" also shows the stove; a "Bathroom" may be a laundry
    with a sink). So only the absurd combinations are capped, and the cap is small but not 0: a caller who insists on
    a toilet in a room labelled "Garage" can still move the dot there over several answers.
  - Within the allowed range CLIP still ranks candidates. Two sweeps in one bathroom, or two bedrooms, are told apart
    by their photos exactly as before; the guard only stops CLIP from leaving the range common sense allows.
  - Rooms with no recognisable type (unlabelled "Room 1", "Game Room" if unmapped, "Storage") get no bounds at all.

A room with several types (Matterport classifications ["Kitchen", "Dining Room"], or "Hall Bathroom") gets the most
permissive category among them: if any of its types makes a feature possible, it is not bounded.
"""

import re

import numpy as np

from . import config

# Room types. Each pattern is matched (case-insensitive) against the room's Matterport classifications, label and
# name joined together (followups._room_kinds). A room can match several.
KIND_PATTERNS = {
    "bathroom": r"bath|powder|restroom|lavatory|\bwc\b|toilet|shower",
    "bedroom": r"bed ?room|nursery|guest ?room|\bbed\b",
    "kitchen": r"kitchen",
    "living": r"living|family|great ?room|\bden\b|lounge|game ?room|media ?room|sun ?room|rec(reation)? ?room|bonus",
    "dining": r"dining|breakfast|\bnook\b",
    "hallway": r"hall|corridor|landing",
    "closet": r"closet|wardrobe",
    "laundry": r"laundry|utility",
    "garage": r"garage|carport",
    "office": r"office|study|library",
    "entry": r"entry|entrance|foyer|vestibule|mud ?room",
    "stairs": r"stair",
    "pantry": r"pantry|larder",
    "outdoor": r"patio|balcony|\bdeck\b|porch|terrace|\byard\b|outdoor|exterior|garden",
}

# Feature category per room type; anything not listed is "possible". Keys are followups.FEATURES keys, plus the
# options of the flooring and size choices ("carpet", "tile", "wood"; "small", "medium", "large").
# Deliberately conservative: only things a dispatcher would call absurd are "implausible". Stoves and fridges stay
# possible in living and dining rooms (open plans), washers in kitchens, bathrooms and hallways (laundry closets),
# beds in living rooms and offices (sofa beds, guest rooms). Size: only a closet or pantry can't be "bigger" and a
# garage can't be "closet-sized"; primary bathrooms and laundry rooms can be large.
T, I = "typical", "implausible"
TABLE = {
    "bathroom": {"toilet": T, "sink": T, "mirror": T, "stove": I, "fridge": I, "bed": I, "sofa": I, "table": I,
                 "fireplace": I, "stairs": I, "front_door": I},
    "bedroom": {"bed": T, "toilet": I, "stove": I, "fridge": I, "washer": I, "front_door": I},
    "kitchen": {"sink": T, "stove": T, "fridge": T, "toilet": I, "bed": I, "carpet": I},
    "living": {"sofa": T, "toilet": I, "washer": I},
    "dining": {"table": T, "toilet": I, "bed": I, "washer": I},
    "hallway": {"narrow": T, "toilet": I, "stove": I, "fridge": I, "bed": I, "sofa": I, "fireplace": I, "sink": I},
    "closet": {"shelves": T, "toilet": I, "sink": I, "stove": I, "fridge": I, "bed": I, "sofa": I, "table": I,
               "tv": I, "fireplace": I, "stairs": I, "front_door": I, "sliding_door": I, "large": I},
    "laundry": {"washer": T, "toilet": I, "stove": I, "bed": I, "sofa": I, "fireplace": I},
    "garage": {"toilet": I, "stove": I, "bed": I, "sofa": I, "fireplace": I, "front_door": I, "carpet": I,
               "small": I},
    "office": {"toilet": I, "stove": I, "washer": I},
    "entry": {"front_door": T, "toilet": I, "stove": I, "bed": I, "washer": I},
    "stairs": {"stairs": T, "toilet": I, "sink": I, "stove": I, "fridge": I, "bed": I, "sofa": I, "table": I,
               "washer": I, "tv": I, "fireplace": I},
    "pantry": {"shelves": T, "toilet": I, "bed": I, "sofa": I, "tv": I, "fireplace": I, "stairs": I,
               "front_door": I, "large": I},
    "outdoor": {"toilet": I, "bed": I, "washer": I, "mirror": I, "carpet": I},
}
del T, I


def room_kinds(text: str) -> frozenset[str]:
    """Normalized room types for a room's joined classifications/label/name (followups._room_kinds). Empty if none
    is recognised ("Room 1", "Storage")."""
    return frozenset(kind for kind, pattern in KIND_PATTERNS.items() if re.search(pattern, text or "", re.I))


def bounds(kinds) -> dict[str, tuple[float, float]]:
    """Allowed range of P(feature present | room) for a room of these kinds (a set from room_kinds(), or the raw
    text). Only bounded features are returned; an unrecognised room gets {}."""
    if isinstance(kinds, str):
        kinds = room_kinds(kinds)
    if not kinds:
        return {}
    keys = {key for kind in kinds for key in TABLE[kind]}
    out = {}
    for key in keys:
        # Most permissive over the room's kinds: "possible" in any of them means no bound.
        categories = {TABLE[kind].get(key, "possible") for kind in kinds}
        if categories == {"implausible"}:
            out[key] = (0.0, config.COMMONSENSE_IMPLAUSIBLE_MAX)
        elif categories == {"typical"}:
            out[key] = (config.COMMONSENSE_TYPICAL_MIN, 1.0)
    return out


def _cap_options(probs: np.ndarray, keys: list[str], cand_bounds: list[dict]) -> np.ndarray:
    """Cap implausible options of a choice (carpet in a garage) per candidate and give the excess to the other
    options in proportion, so each column still sums to 1. Lower bounds aren't applied to choices: their options
    share one unit of probability, so "typical" has no fixed meaning there."""
    probs = probs.copy()
    for c, room_bounds in enumerate(cand_bounds):
        capped = [i for i, key in enumerate(keys) if key in room_bounds and room_bounds[key][1] < 1.0]
        if not capped or len(capped) == len(keys):
            continue
        column = probs[:, c]
        excess = 0.0
        for i in capped:
            excess += max(column[i] - room_bounds[keys[i]][1], 0.0)
            column[i] = min(column[i], room_bounds[keys[i]][1])
        free = [i for i in range(len(keys)) if i not in capped]
        column[free] += excess * column[free] / max(column[free].sum(), 1e-9)
    return probs


def guard(p_yes: dict[str, np.ndarray], p_option: dict[str, np.ndarray], option_keys: dict[str, list[str]],
          kinds_text: list[str], cand_room: np.ndarray) -> None:
    """Clamp the feature model in place: each candidate's P(yes) into its room's bounds, and implausible choice
    options capped. `kinds_text` is followups._room_kinds() per room; `option_keys` maps a choice key to its option
    keys in row order. Candidates in unrecognised rooms are untouched."""
    room_bounds = [bounds(text) for text in kinds_text]
    cand_bounds = [room_bounds[r] for r in cand_room]
    for key, p in p_yes.items():
        lo = np.array([b.get(key, (0.0, 1.0))[0] for b in cand_bounds])
        hi = np.array([b.get(key, (0.0, 1.0))[1] for b in cand_bounds])
        p_yes[key] = np.clip(p, lo, hi)
    for choice_key, keys in option_keys.items():
        if choice_key in p_option:
            p_option[choice_key] = _cap_options(p_option[choice_key], keys, cand_bounds)
