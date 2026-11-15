"""Rooms the system can't tell apart yet (ties), so the dispatcher hears "Bathroom 1 or Bathroom 2" instead of a pick.

Two rooms are tied when both conditions hold:
  - Probability: the room is at least config.TIE_PROBABILITY_RATIO as likely as the leader.
  - Look-alike: same room type as the leader, or a near-identical profile over the content features (P(yes) for
    "toilet", "bed", ... and the flooring options), i.e. no question about what is in the room would separate them.
Two rooms that are close in probability but look different are not a tie: a follow-up can still split them.
"""

import numpy as np

from . import config, followups, locator


def _room_profiles(features: followups.FeatureModel, n_rooms: int) -> np.ndarray:
    """(n_rooms, n_features) mean P(yes) over each room's candidates, content features only (not position questions,
    which exist precisely to separate rooms that look alike)."""
    columns = [features.p_yes[f.key] for f in followups.FEATURES if f.key in features.p_yes]
    columns += list(features.p_option.get("flooring", np.empty((0, len(features.candidate_room)))))
    table = np.stack(columns, axis=1)
    counts = np.maximum(np.bincount(features.candidate_room, minlength=n_rooms), 1)
    sums = np.zeros((n_rooms, table.shape[1]))
    np.add.at(sums, features.candidate_room, table)
    return sums / counts[:, None]


def _room_type(room: dict) -> str:
    return " / ".join(sorted(t.lower() for t in room.get("types") or [])) or (room.get("label") or "").lower()


def tie_group(posterior: np.ndarray, building, features: followups.FeatureModel,
              ratio: float = config.TIE_PROBABILITY_RATIO, rms: float = config.TIE_PROFILE_RMS) -> list[int]:
    """Room indices tied with the leading room, most likely first (leader included); [] when nothing is tied."""
    n_rooms = len(building.rooms)
    probs = locator.room_probabilities(posterior, building.candidate_room, n_rooms)
    leader = int(np.argmax(probs))
    profiles = _room_profiles(features, n_rooms)
    group = [leader]
    for i in np.argsort(probs)[::-1]:
        i = int(i)
        if i == leader or probs[i] < ratio * probs[leader] or probs[i] <= 0:
            continue
        same_type = bool(_room_type(building.rooms[i])) and _room_type(building.rooms[i]) == _room_type(
            building.rooms[leader])
        alike = float(np.sqrt(np.mean((profiles[i] - profiles[leader]) ** 2))) <= rms
        if same_type or alike:
            group.append(i)
    # A look-alike pair sharing 20% of the belief is not yet "the answer is one of these" (e.g. a uniform prior).
    if len(group) < 2 or probs[group].sum() < config.TIE_MIN_GROUP_PROBABILITY:
        return []
    return group


def describe(building, posterior: np.ndarray, group: list[int]) -> str:
    """'Bathroom 1 or Bathroom 2 (58% / 42%), can't tell apart yet'."""
    probs = locator.room_probabilities(posterior, building.candidate_room, len(building.rooms))
    names = [building.rooms[i]["name"] for i in group]
    joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"
    percents = " / ".join(f"{round(probs[i] * 100)}%" for i in group)
    return f"{joined} ({percents}), can't tell apart yet"


def snapshot(building, posterior: np.ndarray, group: list[int]) -> list[dict]:
    """The snapshot's "tie_group" field: [{index, name, probability}], most likely first."""
    probs = locator.room_probabilities(posterior, building.candidate_room, len(building.rooms))
    return [{"index": i, "name": building.rooms[i]["name"], "probability": float(probs[i])} for i in group]
