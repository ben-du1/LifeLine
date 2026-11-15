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


def test_floor_keeps_every_room_off_zero_and_one():
    candidate_room = np.array([0, 0, 1, 2])
    post = locator.apply_floor(np.array([1.0, 0.0, 0.0, 0.0]), candidate_room, 3, floor=0.02)
    rooms = locator.room_probabilities(post, candidate_room, 3)
    assert np.isclose(post.sum(), 1) and rooms.max() < 1 and rooms.min() >= 0.02 / 3 - 1e-12


def test_floor_lets_a_contradicting_answer_recover():
    candidate_room = np.array([0, 1])
    post = np.array([0.5, 0.5])
    for _ in range(3):  # three answers that rule room 1 out almost entirely
        post = locator.apply_floor(locator.apply_likelihood(post, np.array([1.0, 1e-9])), candidate_room, 2)
    for _ in range(4):  # then four strong answers the other way
        post = locator.apply_floor(locator.apply_likelihood(post, np.array([0.05, 0.95])), candidate_room, 2)
    assert post.argmax() == 1


def test_information_gain_still_ignores_within_room_splits_after_floor():
    candidate_room = np.array([0, 0, 1])
    post = locator.apply_floor(np.array([0.25, 0.25, 0.5]), candidate_room, 2)
    within_room = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]])
    assert abs(locator.expected_information_gain(post, within_room, candidate_room, 2)) < 0.05


def test_siblings_of_the_same_type_block_located():
    types = [{"bedroom"}, {"bedroom"}, {"kitchen"}, set(), set()]
    probs = np.array([0.7, 0.2, 0.06, 0.02, 0.02])
    assert locator.indistinguishable_rooms(probs, types, 0) == [1]          # 0.2 >= 0.25 x 0.7
    assert locator.indistinguishable_rooms(np.array([0.85, 0.1, 0.03, 0.01, 0.01]), types, 0) == []
    assert locator.indistinguishable_rooms(np.array([0.02, 0.02, 0.06, 0.7, 0.2]), types, 3) == []  # unlabeled
    est = locator.estimate(probs, np.arange(5), 5)
    assert est.confident and not locator.is_located(est, probs, types, evidence_messages=3)


def test_located_needs_two_evidence_messages_unless_decisive():
    types = [{"bedroom"}, {"kitchen", "dining"}, {"bathroom"}]
    probs = np.array([0.1, 0.85, 0.05])
    est = locator.estimate(probs, np.arange(3), 3)
    assert not locator.is_located(est, probs, types, evidence_messages=1)
    assert locator.is_located(est, probs, types, evidence_messages=2)
    assert locator.is_located(est, probs, types, evidence_messages=1, decisive=True)
    assert not locator.is_located(est, probs, types, evidence_messages=0, decisive=True)


def test_names_only_room():
    types = [{"bedroom", "bedroom 1"}, {"bedroom", "bedroom 2"}, {"kitchen"}, {"bathroom"}, {"laundry room"}]
    assert locator.names_only_room(types, 2, [("kitchen",)])
    assert locator.names_only_room(types, 3, [("bath",)])                       # the toilet feature's hint
    assert not locator.names_only_room(types, 3, [("bath", "kitchen", "laundry")])  # a sink is in three rooms
    assert not locator.names_only_room(types, 0, [("bedroom",)])                # two bedrooms
    assert not locator.names_only_room(types, 2, [])
