import numpy as np

from app import locator


def test_prior_is_uniform_over_rooms_not_sweeps():
    candidate_room = np.array([0, 0, 0, 1])  # room 0 has three sweeps, room 1 has one
    prior = locator.uniform_room_prior(candidate_room, 2)
    assert np.allclose(locator.room_probabilities(prior, candidate_room, 2), [0.5, 0.5])


def test_update_favors_higher_scores_and_boost():
    prior = np.full(3, 1 / 3)
    post = locator.update(prior, np.array([0.30, 0.25, 0.25]), logit_scale=60)
    assert post.argmax() == 0 and np.isclose(post.sum(), 1)
    boosted = locator.update(prior, np.zeros(3), boost=np.array([1.0, 5.0, 1.0]))
    assert boosted.argmax() == 1


def test_confidence_needs_threshold_and_margin():
    candidate_room = np.array([0, 1, 2])
    assert locator.estimate(np.array([0.7, 0.2, 0.1]), candidate_room, 3).confident
    assert not locator.estimate(np.array([0.55, 0.4, 0.05]), candidate_room, 3).confident   # below threshold
    assert not locator.estimate(np.array([0.62, 0.38, 0.0]), candidate_room, 3, margin=0.3).confident  # margin


def test_estimate_picks_best_candidate_inside_top_room():
    candidate_room = np.array([0, 0, 1])
    est = locator.estimate(np.array([0.3, 0.4, 0.3]), candidate_room, 2)
    assert est.room == 0 and est.candidate == 1


def test_information_gain_prefers_question_that_splits_plausible_rooms():
    candidate_room = np.array([0, 1, 2])
    posterior = np.array([0.5, 0.5, 0.0])
    splits_plausible = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]])     # P(yes|c), P(no|c)
    splits_implausible = np.array([[0.5, 0.5, 0.9], [0.5, 0.5, 0.1]])
    gain = lambda m: locator.expected_information_gain(posterior, m, candidate_room, 3)
    assert gain(splits_plausible) > 0.4 and abs(gain(splits_implausible)) < 1e-9


def test_information_gain_ignores_splits_within_one_room():
    candidate_room = np.array([0, 0, 1])
    posterior = np.array([0.25, 0.25, 0.5])
    within_room = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]])
    assert abs(locator.expected_information_gain(posterior, within_room, candidate_room, 2)) < 0.05


def test_apply_likelihood_is_bayes_rule():
    post = locator.apply_likelihood(np.array([0.5, 0.5]), np.array([0.9, 0.3]))
    assert np.allclose(post, [0.75, 0.25])
