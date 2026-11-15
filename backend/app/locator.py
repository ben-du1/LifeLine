"""Belief over caller location: prior, update, confidence rule, next-question choice. Pure numpy, no I/O."""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from . import config


def uniform_room_prior(candidate_room: np.ndarray, n_rooms: int) -> np.ndarray:
    """Uniform over rooms, split evenly across each room's candidates, so rooms with many sweeps aren't favored."""
    counts = np.bincount(candidate_room, minlength=n_rooms)
    prior = 1.0 / counts[candidate_room]
    return prior / prior.sum()


def update(posterior: np.ndarray, scores: np.ndarray, boost: np.ndarray | None = None,
           logit_scale: float = config.LOGIT_SCALE, candidate_room: np.ndarray | None = None,
           max_ratio: float = config.CLIP_MAX_LIKELIHOOD_RATIO) -> np.ndarray:
    """posterior ∝ prior × photo likelihood (see clip_likelihood), optionally × a per-candidate boost (room-type
    words). The boost is discrete evidence and is not capped."""
    likelihood = clip_likelihood(posterior, scores, candidate_room, logit_scale, max_ratio)
    if boost is not None:
        likelihood = likelihood * boost
    new = posterior * likelihood
    total = new.sum()
    return new / total if total > 0 else posterior


def _expected_max_z(n: np.ndarray) -> np.ndarray:
    """Expected maximum of n standard normal draws (Blom's approximation; 0 for n = 1)."""
    return np.array([NormalDist().inv_cdf((k - 0.375) / (k + 0.25)) if k > 1 else 0.0 for k in n])


def clip_likelihood(posterior: np.ndarray, scores: np.ndarray, candidate_room: np.ndarray | None = None,
                    logit_scale: float = config.LOGIT_SCALE,
                    max_ratio: float = config.CLIP_MAX_LIKELIHOOD_RATIO) -> np.ndarray:
    """Per-candidate likelihood for one free-text CLIP match, bounded so photos alone can't decide a room.

    Room level: a room's evidence is its best sweep's score minus the max that many sweeps would reach by chance
    (robust spread × expected max of N normals), so a kitchen with 11 sweeps isn't favored for having one lucky view.
    That is scaled by logit_scale (per cosine unit) and clipped so no room gets more than max_ratio times another's
    likelihood. Within a room, sweeps are reweighted by exp(logit_scale × score) normalized to the room's current
    mass, which picks the sweep without moving the room's probability. Without candidate_room every candidate is its
    own room."""
    scores = np.asarray(scores, dtype=float)
    if candidate_room is None:
        candidate_room = np.arange(len(scores))
    _, room, n = np.unique(candidate_room, return_inverse=True, return_counts=True)
    spread = 1.4826 * float(np.median(np.abs(scores - np.median(scores))))  # MAD: the true room can't inflate it
    best = np.full(len(n), -np.inf)
    np.maximum.at(best, room, scores)
    room_log = logit_scale * (best - spread * _expected_max_z(n))
    room_log = np.maximum(room_log - room_log.max(), -np.log(max_ratio))

    within = np.exp(logit_scale * (scores - best[room]))
    weight = posterior if posterior.sum() > 0 else np.ones(len(scores))
    mass = np.bincount(room, weights=weight, minlength=len(n))
    weighted = np.bincount(room, weights=weight * within, minlength=len(n))
    # Rescale so the prior-weighted mean of `within` is 1 in each room; rooms with no mass keep uniform weights.
    norm = np.divide(mass, weighted, out=np.ones(len(n)), where=weighted > 0)
    return np.exp(room_log)[room] * within * norm[room]


def room_probabilities(posterior: np.ndarray, candidate_room: np.ndarray, n_rooms: int) -> np.ndarray:
    return np.bincount(candidate_room, weights=posterior, minlength=n_rooms)


@dataclass
class Estimate:
    room: int          # index into rooms
    candidate: int     # most likely candidate inside that room
    room_probability: float
    runner_up_probability: float
    confident: bool


def estimate(posterior: np.ndarray, candidate_room: np.ndarray, n_rooms: int,
             threshold: float = config.CONFIDENCE_THRESHOLD, margin: float = config.CONFIDENCE_MARGIN) -> Estimate:
    rooms = room_probabilities(posterior, candidate_room, n_rooms)
    order = np.argsort(rooms)[::-1]
    top = int(order[0])
    runner_up = float(rooms[order[1]]) if n_rooms > 1 else 0.0
    in_room = np.where(candidate_room == top, posterior, -1.0)
    return Estimate(
        room=top, candidate=int(np.argmax(in_room)), room_probability=float(rooms[top]),
        runner_up_probability=runner_up,
        confident=bool(rooms[top] >= threshold and rooms[top] - runner_up >= margin),
    )


def apply_floor(posterior: np.ndarray, candidate_room: np.ndarray, n_rooms: int,
                floor: float = config.POSTERIOR_FLOOR) -> np.ndarray:
    """Mix a small share of the uniform-room prior back in, so no room is ever exactly 0% or 100%: callers and our
    readings are sometimes wrong, and a later contradicting answer must still be able to win."""
    return (1.0 - floor) * posterior + floor * uniform_room_prior(candidate_room, n_rooms)


def indistinguishable_rooms(room_probs: np.ndarray, room_types: list[set[str]], top: int,
                            ratio: float = config.SIBLING_RATIO) -> list[int]:
    """Other rooms of the leader's type (sharing any lowercase type label) holding at least `ratio` × the leader's probability:
    siblings like two identical bedrooms that the evidence hasn't really told apart. Rooms with no types (unlabeled
    regions) never count as siblings, since "no label" says nothing about what the room is."""
    leader = room_probs[top]
    return [i for i, types in enumerate(room_types)
            if i != top and types & room_types[top] and room_probs[i] >= ratio * leader]


def names_only_room(room_types: list[set[str]], top: int, hints) -> bool:
    """True when some hint picks out the leading room alone. Each hint is a group of words, and a room matches it if
    one of its lowercase types contains any of the words: ("kitchen",) for "I'm in the kitchen", or a feature's
    room-type hints ("bath", "kitchen", "laundry") for "there's a sink", which is unique only if just one room is any
    of those."""
    for words in hints:
        hits = [i for i, types in enumerate(room_types) if any(w in t for w in words for t in types)]
        if hits == [top]:
            return True
    return False


def is_located(est: Estimate, room_probs: np.ndarray, room_types: list[set[str]], evidence_messages: int,
               decisive: bool = False, min_evidence: int = config.MIN_EVIDENCE_MESSAGES) -> bool:
    """The full "located" rule: threshold and margin (est.confident), no same-type sibling still close, and at least
    `min_evidence` caller messages that carried evidence, unless one message `decisive`ly named the only room of
    its kind."""
    if not est.confident or indistinguishable_rooms(room_probs, room_types, est.room):
        return False
    return evidence_messages >= min_evidence or (decisive and evidence_messages >= 1)


def apply_likelihood(posterior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """posterior ∝ prior × P(observation | candidate), for answers with an explicit probability model."""
    new = posterior * likelihood
    total = new.sum()
    return new / total if total > 0 else posterior


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def expected_information_gain(posterior: np.ndarray, answer_probs: np.ndarray, candidate_room: np.ndarray,
                              n_rooms: int) -> float:
    """Expected reduction (bits) in uncertainty about the *room* from asking a question whose answers have
    probabilities answer_probs[a, c] = P(answer a | candidate c). Rooms, not candidates, because responders need the
    room; a question that only splits sweeps inside one room is worthless."""
    before = _entropy(room_probabilities(posterior, candidate_room, n_rooms))
    after = 0.0
    for row in answer_probs:
        joint = posterior * row
        p_answer = joint.sum()
        if p_answer > 0:
            after += p_answer * _entropy(room_probabilities(joint / p_answer, candidate_room, n_rooms))
    return before - after


def apply_within_rooms(posterior: np.ndarray, likelihood: np.ndarray, candidate_room: np.ndarray) -> np.ndarray:
    """Reweight positions while preserving each room’s probability."""
    weighted = posterior * likelihood
    before = np.bincount(candidate_room, weights=posterior)
    after = np.bincount(candidate_room, weights=weighted, minlength=len(before))
    scale = np.divide(before, after, out=np.zeros_like(before), where=after > 0)
    return np.where(after[candidate_room] > 0, weighted * scale[candidate_room], posterior)
