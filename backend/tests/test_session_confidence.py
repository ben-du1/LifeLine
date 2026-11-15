"""Session-level "located" rule on a synthetic building: evidence count, unique room types, sibling rooms, floor.

app.sessions imports embeddings (open_clip, torch, PIL) at load, so those are stubbed; CLIP text embeddings are
replaced by fixed vectors and the paraphrase model by "no matches"."""

import asyncio
import sys
import types

import numpy as np

for name in ("open_clip", "torch", "PIL", "PIL.Image"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["PIL"].Image = sys.modules["PIL.Image"]
if not hasattr(sys.modules["torch"], "Tensor"):
    sys.modules["torch"].Tensor = object   # only used in a type annotation at import

from app import followups, llm, locator, messaging, semantics, sessions  # noqa: E402
from app.building import Building  # noqa: E402

KITCHEN_VIEW = np.array([0, 0, 0, 1], dtype=np.float32)


def square(side: float) -> list[list[float]]:
    return [[0, 0], [side, 0], [side, side], [0, side]]


def make_manager(monkeypatch, embedding=KITCHEN_VIEW):
    rooms = [
        {"id": "c", "name": "Closet", "label": "Closet", "types": ["Closet"], "polygon": square(1.2), "walls": []},
        {"id": "b", "name": "Bathroom", "label": "Bathroom", "types": ["Bathroom"], "polygon": square(2.3),
         "walls": []},
        {"id": "r1", "name": "Bedroom 1", "label": "Bedroom 1", "types": ["Bedroom"], "polygon": square(3.8),
         "walls": []},
        {"id": "r2", "name": "Bedroom 2", "label": "Bedroom 2", "types": ["Bedroom"], "polygon": square(3.8),
         "walls": []},
        {"id": "k", "name": "Kitchen", "label": "Kitchen", "types": ["Kitchen"], "polygon": square(4.5), "walls": []},
    ]
    building = Building(
        id="m", name="m", address="1 A St", floors=[{"index": 0, "id": "f0", "label": "Floor 1"}], floorplans={},
        rooms=rooms, candidates=[{"views": [v] if v else [], "x": 0.0, "y": 0.0} for v in ("a", "", "b", "c", "d")],
        candidate_room=np.arange(5), has_views=np.array([True, False, True, True, True]),
        view_embeddings=np.eye(4, dtype=np.float32), view_candidate=np.array([0, 2, 3, 4]),
    )
    model = followups.build_model(building, {p: np.zeros(5) for p in followups.clip_prompts()})

    async def send(phone, text):
        pass

    monkeypatch.setattr(sessions.embeddings, "embed_texts", lambda texts: np.stack([embedding] * len(texts)),
                        raising=False)
    monkeypatch.setattr(semantics, "scores", lambda text: {})
    monkeypatch.setattr(llm, "enabled", lambda: False)
    monkeypatch.setattr(messaging, "send", send)

    async def notify(_):
        pass

    manager = sessions.SessionManager(notify)
    manager.buildings["m"] = sessions.LoadedBuilding(building, model)
    session = sessions.Session(id="s", phone="+15555550100", address="1 A St", created=0.0, status="listening",
                               model_id="m", posterior=locator.uniform_room_prior(building.candidate_room, 5))
    manager.sessions[session.id] = session
    return manager, session


def room_probs(session):
    return locator.room_probabilities(session.posterior, np.arange(5), 5)


def test_naming_the_only_kitchen_leads_to_verification(monkeypatch):
    manager, session = make_manager(monkeypatch, embedding=np.zeros(4, dtype=np.float32))
    asyncio.run(manager.handle_reply(session, "I'm in the kitchen"))
    assert session.status == "listening" and session.estimate.room == 4
    assert session.pending.purpose == "verify"
    assert session.evidence_messages == 1 and 0 < room_probs(session).min() and room_probs(session).max() < 1


def test_one_strong_photo_match_needs_a_second_message(monkeypatch):
    manager, session = make_manager(monkeypatch)           # every text "looks like" the kitchen photo
    # Only visual words reach the photo match (textmatch), and one photo match moves a room at most
    # CLIP_MAX_LIKELIHOOD_RATIO-fold (locator), so the kitchen leads without being certain.
    asyncio.run(manager.handle_reply(session, "there is a desk and a lamp"))
    assert room_probs(session).argmax() == 4 and room_probs(session).max() > 0.4
    assert session.status == "listening" and session.pending is not None   # asked instead of declaring
    assert not session.estimate.confident and manager.snapshot(session)["estimate"]["evidence_messages"] == 1
    plan = session.pending   # answer truthfully for the kitchen
    p_kitchen = followups.answer_matrix(plan, manager.buildings["m"].features)[:, 4]
    if plan.kind == "yesno":
        answer = "yes" if p_kitchen[0] > 0.5 else "no"
    elif plan.kind == "choice":
        answer = plan.options[int(np.argmax(p_kitchen))].label
    else:
        answer = followups.FEATURES_BY_KEY[plan.items[int(np.argmax(p_kitchen[:-1]))]].noun
    asyncio.run(manager.handle_reply(session, answer))
    assert session.estimate.room == 4 and session.evidence_messages == 2
    assert session.status == "located" or session.pending.purpose == "verify"

def test_two_bedrooms_are_not_located_on_a_bed(monkeypatch):
    manager, session = make_manager(monkeypatch, embedding=np.zeros(4, dtype=np.float32))
    asyncio.run(manager.handle_reply(session, "I'm in the bedroom, there's a bed"))
    probs = room_probs(session)
    assert session.status != "located" and probs[2] > 0.3 and probs[3] > 0.3
