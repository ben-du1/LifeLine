"""Belief over caller location: prior, update, confidence rule, next-question choice. Pure numpy, no I/O."""

from dataclasses import dataclass

import numpy as np

from . import config


def uniform_room_prior(candidate_room: np.ndarray, n_rooms: int) -> np.ndarray:
    """Uniform over rooms, split evenly across each room's candidates, so rooms with many sweeps aren't favored."""
    counts = np.bincount(candidate_room, minlength=n_rooms)
    prior = 1.0 / counts[candidate_room]
    return prior / prior.sum()


def update(posterior: np.ndarray, scores: np.ndarray, boost: np.ndarray | None = None,
           logit_scale: float = config.LOGIT_SCALE) -> np.ndarray:
    """posterior ∝ prior × exp(scale × (score − max)), optionally × a per-candidate boost (room-type words)."""
    likelihood = np.exp(logit_scale * (scores - scores.max()))
    if boost is not None:
        likelihood = likelihood * boost
    new = posterior * likelihood
    total = new.sum()
    return new / total if total > 0 else posterior


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
