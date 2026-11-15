"""Snapshot fields that let the dashboard be honest about confidence: per-room photo/type/area facts and the evidence
log. Uses a tiny hand-made building, so no Matterport data or CLIP model is needed."""

import importlib.util
import sys
import types
from unittest import mock

import numpy as np

# sessions imports embeddings, which imports torch/open_clip/PIL at module level. Snapshots need none of them, so
# stand-ins are used where they aren't installed (real installs are left alone).
for name in ("torch", "open_clip", "PIL"):
    if name not in sys.modules and importlib.util.find_spec(name) is None:   # conftest may have stubbed it already
        sys.modules.setdefault(name, mock.MagicMock())

from app import locator, sessions  # noqa: E402


def _manager():
    from app import followups
    from app.building import Building

    def square(side):
        return [[0, 0], [side, 0], [side, side], [0, side]]

    rooms = [{"id": "k", "name": "Kitchen", "types": ["Kitchen"], "floor": 0, "polygon": square(3.513), "walls": []},
             {"id": "b1", "name": "Bathroom 1", "types": ["Bathroom"], "floor": 0, "polygon": square(2.236),
              "walls": []},
             {"id": "b2", "name": "Bathroom 2", "types": ["Bathroom"], "floor": 0, "polygon": None, "walls": []}]
    # Bathroom 2 has a scan point but no photos (and no outline, so no area).
    building = Building(
        id="test", name="test", address="1 Test St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}],
        floorplans={}, rooms=rooms,
        candidates=[{"x": 0, "y": 0, "views": ["a.jpg"]}, {"x": 1, "y": 0, "views": ["b.jpg"]},
                    {"x": 2, "y": 0, "views": []}],
        candidate_room=np.array([0, 1, 2]), has_views=np.array([True, True, False]),
        view_embeddings=np.eye(2, dtype=np.float32), view_candidate=np.array([0, 1]),
    )
    features = followups.build_model(building, {p: np.zeros(3) for p in followups.clip_prompts()})
    manager = sessions.SessionManager(notify=None)
    manager.buildings["test"] = sessions.LoadedBuilding(building, features)
    session = sessions.Session(id="s1", phone="+1 555 0100", address="1 Test St", created=0.0, model_id="test",
                               status="listening")
    session.posterior = locator.uniform_room_prior(building.candidate_room, 3)
    session.estimate = manager._estimate(session, manager.buildings["test"])
    return manager, session


def test_rooms_carry_photos_types_and_area():
    manager, session = _manager()
    rooms = {r["name"]: r for r in manager.snapshot(session)["rooms"]}
    assert rooms["Kitchen"]["has_photos"] and rooms["Bathroom 1"]["has_photos"]
    assert not rooms["Bathroom 2"]["has_photos"]
    assert rooms["Bathroom 1"]["types"] == ["Bathroom"]
    assert rooms["Kitchen"]["area_m2"] == 12.3 and rooms["Bathroom 2"]["area_m2"] is None


def test_evidence_log_records_biggest_changes_and_caps_length():
    manager, session = _manager()
    loaded = manager.buildings["test"]
    assert manager.snapshot(session)["evidence"] == []

    session.messages.append({"role": "caller", "text": "I'm in the bathroom", "ts": 123.0})
    before = locator.room_probabilities(session.posterior, loaded.building.candidate_room, 3)
    session.posterior = np.array([0.1, 0.6, 0.3])
    manager._log_evidence(session, loaded, "I'm in the bathroom", before)

    entry = manager.snapshot(session)["evidence"][-1]
    assert entry["ts"] == 123.0 and entry["text"] == "I'm in the bathroom"
    assert entry["top_before"] == "Kitchen"
    assert [c["name"] for c in entry["changes"]] == ["Bathroom 1", "Kitchen", "Bathroom 2"]
    assert entry["changes"][0]["delta"] == round(0.6 - 1 / 3, 3)

    # A message that changes nothing logs no changes; the log keeps only the newest entries.
    for i in range(sessions.EVIDENCE_LOG_LENGTH + 5):
        manager._log_evidence(session, loaded, f"msg {i}", session.posterior.copy())
    evidence = manager.snapshot(session)["evidence"]
    assert len(evidence) == sessions.EVIDENCE_LOG_LENGTH
    assert evidence[-1]["changes"] == [] and evidence[-1]["text"] == f"msg {sessions.EVIDENCE_LOG_LENGTH + 4}"
