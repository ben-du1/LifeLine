"""The free-text photo update: room-level, room-size-aware, and capped (locator.clip_likelihood)."""
import numpy as np

from app import config, locator


def _room_mass(post, candidate_room, n_rooms):
    return locator.room_probabilities(post, candidate_room, n_rooms)


def test_many_sweep_room_is_not_favored_for_equal_relevance():
    # Room 0 has 11 sweeps, rooms 1-7 have one each; every sweep's score is drawn from the same distribution.
    candidate_room = np.array([0] * 11 + list(range(1, 8)))
    prior = locator.uniform_room_prior(candidate_room, 8)
    rng = np.random.default_rng(0)
    big = []
    for _ in range(2000):
        scores = 0.25 + 0.02 * rng.standard_normal(len(candidate_room))
        big.append(_room_mass(locator.update(prior, scores, candidate_room=candidate_room), candidate_room, 8)[0])
    # Neutral is 1/8 (the old uncapped update gave 0.148 here; this one about 0.12).
    assert abs(np.mean(big) - 1 / 8) < 0.04


def test_one_noisy_match_cannot_exceed_the_cap():
    candidate_room = np.array([0, 0, 0, 1, 2, 3, 4])
    prior = locator.uniform_room_prior(candidate_room, 5)
    scores = np.array([0.24, 0.25, 0.24, 0.25, 0.24, 0.25, 0.24])
    scores[2] = 0.40  # one view in room 0 happens to match very well
    rooms = _room_mass(locator.update(prior, scores, candidate_room=candidate_room), candidate_room, 5)
    assert rooms[0] / rooms.min() <= config.CLIP_MAX_LIKELIHOOD_RATIO + 1e-9
    assert rooms[0] < config.CONFIDENCE_THRESHOLD
    # Inside the room the lucky sweep still gets the mass, so the dot goes there.
    post = locator.update(prior, scores, candidate_room=candidate_room)
    assert post[:3].argmax() == 2


def test_clear_match_still_ranks_first_and_boost_is_not_capped():
    candidate_room = np.array([0, 0, 1, 1, 1, 1, 2, 3])
    prior = locator.uniform_room_prior(candidate_room, 4)
    scores = np.array([0.30, 0.31, 0.25, 0.26, 0.24, 0.25, 0.25, 0.26])
    rooms = _room_mass(locator.update(prior, scores, candidate_room=candidate_room), candidate_room, 4)
    assert rooms.argmax() == 0
    boost = np.where(candidate_room == 0, config.ROOM_KEYWORD_BOOST, 1.0)
    boosted = _room_mass(locator.update(prior, scores, boost, candidate_room=candidate_room), candidate_room, 4)
    assert boosted[0] > 0.9  # room-type words are discrete evidence and still decide


def test_room_mass_moves_only_by_the_room_factor():
    candidate_room = np.array([0, 0, 1])
    prior = np.array([0.1, 0.4, 0.5])
    scores = np.array([0.45, 0.20, 0.25])  # a lead far past the cap
    lik = locator.clip_likelihood(prior, scores, candidate_room)
    rooms = _room_mass(prior * lik, candidate_room, 2) / _room_mass(prior, candidate_room, 2)
    assert np.isclose(rooms.max() / rooms.min(), config.CLIP_MAX_LIKELIHOOD_RATIO)


def test_uninformative_scores_leave_posterior_unchanged():
    candidate_room = np.array([0, 0, 1, 2])
    prior = locator.uniform_room_prior(candidate_room, 3)
    assert np.allclose(locator.update(prior, np.zeros(4), candidate_room=candidate_room), prior)
