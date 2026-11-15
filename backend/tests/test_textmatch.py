"""Only the visual part of a caller's message reaches CLIP (textmatch.visual_query), and sessions skip the CLIP update
when nothing visual is left."""

import asyncio
import sys
import types

import numpy as np
import pytest

# sessions imports embeddings, which imports torch/open_clip/PIL at load; none of them run in these tests.
for name in ("torch", "open_clip", "PIL", "PIL.Image"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["PIL"].Image = sys.modules["PIL.Image"]
if not hasattr(sys.modules["torch"], "Tensor"):  # only annotations use it
    sys.modules["torch"].Tensor = object

from app import config, embeddings, followups, llm, locator, semantics, sessions, textmatch  # noqa: E402
from app.building import Building  # noqa: E402


@pytest.mark.parametrize("text, cues, expected", [
    # Panic, people and actions carry nothing a photo shows.
    ("please hurry I'm scared", (), None),
    ("help me, someone is in the house", (), None),
    ("I think someone is downstairs", (), None),       # "downstairs" is a floor, not a staircase in view
    # The example from the task: only the floor survives.
    ("I'm so scared please hurry, carpeted floor, I think someone is downstairs", (),
     "a photo of a room with carpet floor"),
    ("there's a desk, blue curtains and a rug", (), "a photo of a room with a desk, blue curtains and a rug"),
    ("a wooden desk and stone counters", (), "a photo of a room with a wooden desk and stone counters"),
    ("hardwood floors and a red armchair", (), "a photo of a room with wood floor and a red armchair"),
    ("the walls are blue", (), "a photo of a room with blue walls"),
    ("it's bright, there's a crib", (), "a photo of a bright room with a crib"),
    ("my son is in the nursery with his toys", (), "a photo of a bedroom with toys"),
    # Negated phrases never reach CLIP; the negation ends at the clause.
    ("there's no window here", (), None),
    ("the window isn't there", (), None),
    ("I can't see a window or a door but there's a piano", (), "a photo of a room with a piano"),
    ("no window, just boxes and tools", (), "a photo of a room with boxes and tools"),
    # Already read as cues: nothing new for CLIP.
    ("there's a toilet", ("toilet", "bathroom"), None),
    ("I'm in the bedroom", ("bedroom",), None),
    ("next to the big bed", ("bed",), None),
    ("tile", ("tile", "flooring"), None),
    ("a rug on the floor", ("flooring",), None),        # the flooring answer already counted it
    # Captured words drop out, the rest stays.
    ("there's a toilet and a blue towel", ("toilet", "bathroom"), "a photo of a room with blue towels"),
    ("I'm in the bedroom", (), "a photo of a bedroom"),
])
def test_visual_query(text, cues, expected):
    assert textmatch.visual_query(text, cues) == expected


# ---- sessions: what reaches CLIP ------------------------------------------------------------------------------

# A tiny 3-d "CLIP": axis 0 is windows, axis 1 is desks, axis 2 is everything else.
AXES = {"window": 0, "desk": 1}


def fake_embed(calls):
    def embed_texts(texts):
        calls.extend(texts)
        out = []
        for text in texts:
            v = np.zeros(3, dtype=np.float32)
            for word, axis in AXES.items():
                v[axis] += word in text
            v[2] += 0.2
            out.append(v / np.linalg.norm(v))
        return np.array(out)
    return embed_texts


def make_manager(monkeypatch):
    """Two photographed rooms with no type labels: room 0's view shows a window, room 1's a desk."""
    rooms = [{"id": "a", "name": "Room A", "types": [], "polygon": None, "walls": []},
             {"id": "b", "name": "Room B", "types": [], "polygon": None, "walls": []}]
    building = Building(
        id="m", name="m", address="1 A St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms, candidates=[{"views": ["a"], "x": 0, "y": 0}, {"views": ["b"], "x": 1, "y": 0}],
        candidate_room=np.array([0, 1]), has_views=np.array([True, True]),
        view_embeddings=np.array([[0.98, 0, 0.2], [0, 0.98, 0.2]], dtype=np.float32), view_candidate=np.array([0, 1]),
    )
    features = followups.build_model(building, {p: np.zeros(2) for p in followups.clip_prompts()})
    calls = []
    monkeypatch.setattr(embeddings, "embed_texts", fake_embed(calls))
    monkeypatch.setattr(semantics, "scores", lambda text: {})
    monkeypatch.setattr(llm, "enabled", lambda: False)
    monkeypatch.setattr(config, "MESSAGING_MODE", "simulated")

    async def notify(session):
        pass

    manager = sessions.SessionManager(notify)
    manager.buildings["m"] = sessions.LoadedBuilding(building, features)
    session = sessions.Session(id="s", phone="+1 555 0100", address="1 A St", created=0.0, status="listening",
                               model_id="m", posterior=locator.uniform_room_prior(building.candidate_room, 2))
    return manager, session, calls


def test_panic_words_get_no_clip_update(monkeypatch):
    manager, session, calls = make_manager(monkeypatch)
    before = session.posterior.copy()
    asyncio.run(manager._process(session, "please hurry I'm scared"))
    assert calls == []
    assert np.allclose(session.posterior, before)


def test_denied_window_does_not_favour_windowed_rooms(monkeypatch):
    manager, session, calls = make_manager(monkeypatch)
    asyncio.run(manager._process(session, "there's no window here"))
    assert not any("window" in c for c in calls)
    assert session.posterior[0] <= session.posterior[1] + 1e-9


def test_visual_words_reach_clip_as_a_prompt(monkeypatch):
    manager, session, calls = make_manager(monkeypatch)
    asyncio.run(manager._process(session, "I'm scared, I'm hiding next to a desk"))
    assert calls == ["a photo of a room with a desk"]
    assert session.posterior[1] > 0.75   # capped at CLIP_MAX_LIKELIHOOD_RATIO (4x) for one photo match
