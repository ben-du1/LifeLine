"""Caller-initiated calls end to end, through the real FastAPI app (text) and the real VoiceBridge (voice), with the
real address matcher (resolve.py) and the offline harness (tests/fakes.py: synthetic building, fake CLIP, no
network). Regression test for the merged workstreams A to E.

Text: "help" -> address question -> the building's address -> loading -> listening -> room description and answers
-> located; a second number with an address outside coverage -> retry -> second miss -> transferred (stub).
Voice: start with the address -> room description gets a follow-up; an address outside coverage -> ended.
"""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

import fakes
from app import embeddings, main, messaging, questions, semantics, sessions
from app.voice import END_INSTRUCTION, VoiceBridge
from test_reliability_regressions import HOUSE as FURNISHED

# Matterport's own comma format, as a cached building.json stores it. The caller's text below adds the ZIP in the
# usual "TX 77082" form.
BUILDING = fakes.Spec("E2EHOUSE", "13122 Chipman Glen Dr, Houston, TX, 77082, US", FURNISHED.rooms)
CALLER_ADDRESS = "13122 Chipman Glen Dr, Houston, TX 77082"
OUTSIDE = ("500 Nowhere Rd, Fargo, ND", "77 Elsewhere Ave, Bismarck, ND")
KITCHEN_FACTS = {"stove": "yes", "fridge": "yes", "sink": "yes", "window": "yes", "table": "yes",
                 "flooring": "tile", "size": "bedroom-sized"}


@pytest.fixture
def world(monkeypatch, tmp_path):
    fakes.install(monkeypatch, tmp_path, [BUILDING])
    outbox: list[tuple[str, str]] = []

    async def send(phone, text):
        outbox.append((phone, text))
    monkeypatch.setattr(messaging, "send", send)
    # main's lifespan warms CLIP and MiniLM; the fakes replace both, so there is nothing to load.
    monkeypatch.setattr(embeddings, "warm_up", lambda: None)
    monkeypatch.setattr(semantics, "warm_up", lambda: None)
    manager = sessions.SessionManager(main.broadcast)
    monkeypatch.setattr(main, "manager", manager)
    return manager, outbox


def wait_loaded(client: TestClient, session_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while True:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] != "loading":
            return snap
        assert time.time() < deadline, snap["status_detail"]
        time.sleep(0.02)


def text(client: TestClient, sender: str, body: str) -> dict:
    response = client.post("/api/inbound", json={"from": sender, "text": body})
    assert response.status_code == 200, response.text
    return response.json()


def test_text_call_from_help_to_located(world):
    manager, outbox = world
    with TestClient(main.app) as client:
        first = text(client, "+15555550142", "help")
        assert first["ok"] and first["created"] and isinstance(first["session"], str)
        sid = first["session"]
        snap = client.get(f"/api/sessions/{sid}").json()
        assert snap["status"] == "awaiting_address" and snap["address"] == "" and snap["phone"] == "+15555550142"
        assert outbox == [("+15555550142", questions.ADDRESS_QUESTION)]

        second = text(client, "+15555550142", CALLER_ADDRESS)
        assert second == {"ok": True, "session": sid, "created": False}
        snap = wait_loaded(client, sid)
        assert snap["status"] == "listening" and snap["model_id"] == BUILDING.model_id, snap["status_detail"]
        assert snap["match"]["exact"]
        assert outbox[-1] == ("+15555550142", questions.OPENING_MESSAGE)

        text(client, "+15555550142", "I can see a stove and a fridge")
        session = manager.sessions[sid]
        for _ in range(20):
            if session.status != "listening":
                break
            text(client, "+15555550142", fakes.answer(session.pending, KITCHEN_FACTS))
        snap = client.get(f"/api/sessions/{sid}").json()
        assert snap["status"] == "located", snap["status_detail"]
        assert snap["estimate"]["room_name"] == "Kitchen"
        assert [m for m in client.get("/api/sessions").json() if m["id"] == sid]


def test_text_call_outside_coverage_is_transferred(world):
    manager, outbox = world
    caller = "+15555550143"
    with TestClient(main.app) as client:
        sid = text(client, caller, OUTSIDE[0])["session"]   # an address as the first text skips the question
        snap = client.get(f"/api/sessions/{sid}").json()
        assert snap["status"] == "awaiting_address" and snap["transfer"] is None
        assert outbox[-1] == (caller, questions.ADDRESS_RETRY)

        assert text(client, caller, OUTSIDE[1])["session"] == sid
        snap = client.get(f"/api/sessions/{sid}").json()
        assert snap["status"] == "transferred"
        assert snap["transfer"]["stub"] is True and snap["transfer"]["target"]
        assert outbox[-1] == (caller, questions.OUT_OF_AREA_MESSAGE)

        sent = len(outbox)
        text(client, caller, "hello?")   # recorded only: the call is with standard 911 now
        assert len(outbox) == sent and manager.sessions[sid].status == "transferred"


def history(*messages: str) -> str:
    return json.dumps({"entries": [{"role": "user", "message": m} for m in messages]})


def test_voice_call_with_address_then_room_description(world):
    manager, _ = world

    async def run():
        bridge = VoiceBridge(manager, wait_seconds=1)
        opening = await bridge.start("conv-1", CALLER_ADDRESS, "+15555550144")
        session = bridge.calls["conv-1"].session
        assert session.channel == "voice" and session.phone == "+15555550144" and not opening["ended"]
        await asyncio.wait_for(session.ready.wait(), 5)
        assert session.status == "listening" and session.model_id == BUILDING.model_id
        reply = await bridge.reply("conv-1", history("I can see a stove and a fridge"))
        assert not reply["ended"] and reply["say"]
        assert session.evidence and session.pending is not None
        assert manager.find_by_phone("+15555550144") is None   # texts never join a voice call
    asyncio.run(run())


def test_voice_call_outside_coverage_ends(world):
    manager, outbox = world

    async def run():
        bridge = VoiceBridge(manager, wait_seconds=1)
        first = await bridge.start("conv-2", OUTSIDE[0], "anonymous")
        assert first["status"] == "awaiting_address" and not first["ended"]
        assert questions.for_channel(questions.ADDRESS_RETRY, "voice") in first["say"]
        second = await bridge.reply("conv-2", history(OUTSIDE[0], OUTSIDE[1]))
        assert second["ended"] is True and second["status"] == "transferred"
        assert second["instruction"] == END_INSTRUCTION
        assert questions.for_channel(questions.OUT_OF_AREA_MESSAGE, "voice") in second["say"]
        session = bridge.calls["conv-2"].session
        assert session.phone is None and manager.snapshot(session)["transfer"]["stub"] is True
        assert outbox == []   # voice never texts
    asyncio.run(run())
