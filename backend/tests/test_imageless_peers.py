"""Rooms of the same type where only some have photos: a photo-less bathroom must not lose to the photographed one
just because only the photographed one has CLIP evidence (a caller saying "toilet" is as likely in either)."""

import numpy as np

from app import config, followups, locator
from app.building import Building


def square(side: float) -> list[list[float]]:
    return [[0, 0], [side, 0], [side, side], [0, side]]


def unit(cos: float, axis: int, d: int = 4) -> np.ndarray:
    """A view embedding whose cosine with the text along `axis` is `cos` (the rest goes into the last dimension)."""
    v = np.zeros(d, dtype=np.float32)
    v[axis], v[-1] = cos, np.sqrt(1 - cos ** 2)
    return v


def make_house(bath_b_photos: bool = False) -> Building:
    """Bathroom A (2 sweeps), bathroom B (photo-less unless asked), bedroom (2 sweeps), kitchen (2 sweeps). Axis 0 is
    "toilet". Cosines are CLIP-like: the right room ~0.30, the others ~0.20-0.22, sweeps differing by ~0.01."""
    rooms = [
        {"id": "ba", "name": "Bathroom", "types": ["Bathroom"], "polygon": square(2.4), "walls": []},
        {"id": "bb", "name": "Bathroom", "types": ["Bathroom"], "polygon": square(2.3), "walls": []},
        {"id": "r", "name": "Bedroom", "types": ["Bedroom"], "polygon": square(3.8), "walls": []},
        {"id": "k", "name": "Kitchen", "types": ["Kitchen"], "polygon": square(4.5), "walls": []},
    ]
    toilet_cos = {0: [0.31, 0.29], 1: [0.30], 2: [0.21, 0.20], 3: [0.22, 0.21]}
    candidates, candidate_room, views, view_candidate = [], [], [], []
    for room, sweeps in toilet_cos.items():
        if room == 1 and not bath_b_photos:
            candidates.append({"views": []})
            candidate_room.append(room)
            continue
        for cos in sweeps:
            view_candidate.append(len(candidates))
            views.append(unit(cos, 0))
            candidates.append({"views": ["v"]})
            candidate_room.append(room)
    return Building(
        id="m", name="m", address="1 A St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms, candidates=candidates, candidate_room=np.array(candidate_room),
        has_views=np.array([bool(c["views"]) for c in candidates]),
        view_embeddings=np.stack(views), view_candidate=np.array(view_candidate),
    )


def feature_model(building: Building) -> followups.FeatureModel:
    # Fake prompt embeddings: the toilet prompt lines up with axis 0, every other prompt is flat noise.
    toilet = next(f.prompt for f in followups.FEATURES if f.key == "toilet")
    toilets = set(followups._ensemble(toilet))   # every phrasing of the toilet prompt (A08 ensembles)
    vectors = {p: (unit(1.0, 0) if p in toilets else unit(0.0, 1)) for p in followups.clip_prompts()}
    return followups.build_model(building, {p: building.candidate_scores(v) for p, v in vectors.items()})


def bathroom_shares(building: Building, posterior: np.ndarray) -> np.ndarray:
    probs = locator.room_probabilities(posterior, building.candidate_room, len(building.rooms))
    return probs[:2] / probs[:2].sum(), probs


def test_imageless_bathroom_borrows_the_photographed_bathrooms_score():
    building = make_house()
    scores = building.candidate_scores(unit(1.0, 0))
    peers = building.imageless_peers()
    assert list(peers) == [2] and [g.tolist() for g in peers[2]] == [[0, 1]]   # bathroom A's two sweeps
    assert 0.30 <= scores[2] <= 0.31                    # between bathroom A's mean and max, not the median (~0.22)


def test_toilet_free_text_splits_both_bathrooms():
    """The session path for "there's a toilet": CLIP free-text scores with the bathroom keyword boost
    (sessions._evidence), then the toilet feature cue (followups.read_message)."""
    building = make_house()
    model = feature_model(building)
    prior = locator.uniform_room_prior(building.candidate_room, len(building.rooms))

    scores = building.candidate_scores(unit(1.0, 0))
    is_bath = np.array([bool({"bathroom"} & {t.lower() for t in r["types"]}) for r in building.rooms])
    boost = np.where(is_bath[building.candidate_room], config.ROOM_KEYWORD_BOOST, 1.0)
    posterior = locator.update(prior, scores, boost)
    shares, probs = bathroom_shares(building, posterior)
    assert np.all((0.4 <= shares) & (shares <= 0.6)) and probs.max() <= 0.7

    reading = followups.read_message("there's a toilet", model, None, "free")
    assert "toilet" in reading.covered
    posterior = locator.apply_likelihood(posterior, reading.likelihood)
    shares, probs = bathroom_shares(building, posterior)
    assert np.all((0.4 <= shares) & (shares <= 0.6)) and probs.max() <= 0.7

    # And the feature cue on its own (no CLIP free text) is symmetric too.
    only_cue = locator.apply_likelihood(prior, reading.likelihood)
    shares, probs = bathroom_shares(building, only_cue)
    assert np.all((0.4 <= shares) & (shares <= 0.6)) and probs.max() <= 0.7


def test_no_same_type_photos_falls_back_to_median():
    building = make_house()
    building.rooms[0]["types"] = ["Laundry"]           # no other bathroom has photos now
    scores = building.candidate_scores(unit(1.0, 0))
    assert building.imageless_peers() == {2: []}
    assert np.isclose(scores[2], np.quantile(scores[building.has_views], config.IMAGELESS_SCORE_QUANTILE))


def test_photographed_only_house_is_unchanged(monkeypatch):
    # This checks the photo-less imputation only, so the CLIP calibration (followups, A08) is switched off here.
    monkeypatch.setattr(config, "CLIP_CALIBRATED", False)
    building = make_house(bath_b_photos=True)
    assert building.imageless_peers() == {}
    text = unit(1.0, 0)
    assert np.allclose(building.candidate_scores(text), building.view_embeddings @ text)   # one view per sweep
    model = feature_model(building)
    raw = building.candidate_scores(text)
    z = (raw - raw.mean()) / (raw.std() + 1e-6)
    by_type = np.where(np.isin(building.candidate_room, [0, 1]), 0.85, 0.15)
    expected = np.clip((followups._sigmoid(1.5 * z) + 2 * by_type) / 3, 0.05, 0.95)
    # The commonsense guard (commonsense.py) then clamps each room into its type's range.
    from app import commonsense
    ranges = [commonsense.bounds(followups._room_kinds(r)).get("toilet", (0.0, 1.0)) for r in building.rooms]
    lo, hi = np.array(ranges)[building.candidate_room].T
    assert np.allclose(model.p_yes["toilet"], np.clip(expected, lo, hi))
