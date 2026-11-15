"""Voice calls through the call intake: the address goes to sessions.open_call, retries and "Did you mean...?" come
back as text to speak, and an address outside the coverage ends the call. Offline (tests/fakes.py, simulated
messaging, no ElevenLabs).

Tests marked xfail need workstream B's open_call; the integrator removes the marks in Phase 2. The rest use a small
fake manager and pass now.
"""

import asyncio
import functools
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import fakes
from app import config, questions, resolve, sessions
from app.voice import END_INSTRUCTION, INTAKE_INSTRUCTION, VoiceBridge, VoiceError

NEEDS_B = pytest.mark.xfail(strict=False, reason="needs workstream B intake")

ADDRESS = "13122 Chipman Glen Dr, Houston, TX"
BUILDING = fakes.Spec("voice-e2e", ADDRESS, [
    fakes.Room("Kitchen", ["kitchen"], (0, 0, 4, 4), [fakes.Sweep(2, 2, [{"stove": 0.8}] * 4)]),
    fakes.Room("Bathroom", ["bathroom"], (4, 0, 7, 3), [fakes.Sweep(5.5, 1.5, [{"toilet": 0.8}] * 4)]),
])


def run(test):
    """Run an async test in its own event loop (the suite has no pytest-asyncio; see test_reliability_regressions)."""
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))
    return wrapper


def history(*messages):
    return json.dumps({"entries": [{"role": "user", "message": text} for text in messages]})


def voice(text: str) -> str:
    return questions.for_channel(text, "voice")


async def settle(session, timeout: float = 5.0):
    """Wait for a background building load to finish."""
    waited = 0.0
    while session.status == "loading" and waited < timeout:
        await asyncio.sleep(0.01)
        waited += 0.01


@pytest.fixture
def real(monkeypatch, tmp_path):
    fakes.install(monkeypatch, tmp_path, [BUILDING])
    manager = sessions.SessionManager(AsyncMock())
    return manager, VoiceBridge(manager, wait_seconds=1)


def fake_resolution(monkeypatch, bands: dict[str, str]):
    """resolve.resolve answers by the address text, so these tests don't depend on workstream A's thresholds."""
    candidate = {"id": BUILDING.model_id, "name": ADDRESS, "address": ADDRESS, "score": 0.75, "exact": False}

    def fake(text, exclude=frozenset()):
        band = next((b for key, b in bands.items() if key.lower() in text.lower()), "none")
        if band == "match":
            return resolve.Resolution("match", [dict(candidate, score=1.0, exact=True)], text)
        if band == "clarify" and candidate["id"] not in exclude:
            return resolve.Resolution("clarify", [candidate], text)
        return resolve.Resolution("none", [], text)
    monkeypatch.setattr(resolve, "resolve", fake)


# ---- with workstream B's intake (sessions.open_call) -----------------------------------------------------------


@NEEDS_B
@run
async def test_unknown_address_retries_once_then_ends_with_out_of_area(real, monkeypatch):
    manager, bridge = real
    fake_resolution(monkeypatch, {})
    first = await bridge.start("c1", "999 Nowhere Rd, Springfield, IL", "+15555550100")
    assert first["status"] == "awaiting_address" and not first["ended"]
    assert voice(questions.ADDRESS_RETRY) in first["say"]
    assert first["instruction"] == INTAKE_INSTRUCTION and first["awaiting_answer"]

    second = await bridge.reply("c1", history("999 Nowhere Rd, Springfield, IL", "1 Fake Street, Nowhere"))
    session = bridge.calls["c1"].session
    assert second["ended"] is True and second["status"] == "transferred"
    assert voice(questions.OUT_OF_AREA_MESSAGE) in second["say"]
    assert second["instruction"] == END_INSTRUCTION
    assert manager.snapshot(session)["transfer"]["stub"] is True
    # The agent hangs up; a late status poll still says the call is over.
    assert (await bridge.status("c1"))["ended"] is True


@NEEDS_B
@run
async def test_clarify_over_voice_then_yes_loads_the_building(real, monkeypatch):
    manager, bridge = real
    fake_resolution(monkeypatch, {"Chipmen": "clarify"})
    first = await bridge.start("c1", "13122 Chipmen Glen, Houston", "+15555550100")
    assert first["status"] == "awaiting_address" and not first["ended"]
    assert "13122 Chipman Glen" in first["say"]           # "Did you mean 13122 Chipman Glen Dr ...?"
    second = await bridge.reply("c1", history("13122 Chipmen Glen, Houston", "yes"))
    session = bridge.calls["c1"].session
    await settle(session)
    assert session.model_id == BUILDING.model_id and session.status == "listening"
    assert not second["ended"]


@NEEDS_B
@run
async def test_intake_session_is_voice_and_keyed_by_conversation(real, monkeypatch):
    manager, bridge = real
    fake_resolution(monkeypatch, {"Chipman": "match"})
    await bridge.start("c1", ADDRESS, "+15555550100")
    await bridge.start("c2", ADDRESS, "+15555550100")      # the same phone calling twice: two calls
    one, two = bridge.calls["c1"].session, bridge.calls["c2"].session
    assert one is not two and one.channel == two.channel == "voice"
    assert manager.snapshot(one)["phone"] == "+15555550100"
    await settle(one)
    assert one.status == "listening"


@NEEDS_B
@run
async def test_voice_messages_are_worded_for_speech(real, monkeypatch):
    manager, bridge = real
    fake_resolution(monkeypatch, {})
    await bridge.start("c1", "999 Nowhere Rd, Springfield, IL")
    session = bridge.calls["c1"].session
    spoken = [m["text"] for m in session.messages if m["role"] == "dispatch"]
    assert spoken and all(text == voice(text) for text in spoken)     # sessions._say applies for_channel


# ---- passes before the merge ---------------------------------------------------------------------------------


@run
async def test_withheld_caller_id_gives_an_unknown_caller(real):
    manager, bridge = real
    result = await bridge.start("c1", ADDRESS, "anonymous")
    session = bridge.calls["c1"].session
    assert manager.snapshot(session)["phone"] is None and session.channel == "voice"
    assert not result["ended"]
    await bridge.end("c1", "test over")


def intake_manager(status="awaiting_address", says=("I couldn't find that address.",)):
    session = SimpleNamespace(id="s1", address="", status=status, pending=None, ended=False, ready=asyncio.Event(),
                              messages=[{"role": "dispatch", "text": questions.ADDRESS_QUESTION},
                                        {"role": "caller", "text": "1 Nowhere Rd"}]
                              + [{"role": "dispatch", "text": text} for text in says])

    async def handle(session, text):
        session.messages += [{"role": "caller", "text": text}, {"role": "dispatch", "text": "Did you mean 1 Main St?"}]
    manager = SimpleNamespace(open_call=AsyncMock(return_value=session), start_call=AsyncMock(),
                              handle_reply=AsyncMock(side_effect=handle), end_call=AsyncMock())
    return manager, session


@run
async def test_start_goes_through_open_call_with_parsed_caller_id():
    manager, session = intake_manager()
    bridge = VoiceBridge(manager, wait_seconds=30)
    # Intake statuses answer at once: nothing is loading, so `ready` is never awaited.
    result = await asyncio.wait_for(bridge.start("c1", " 1 Nowhere Rd ", "(609) 555-0100"), 1)
    manager.open_call.assert_awaited_once_with("+16095550100", "voice", "1 Nowhere Rd")
    manager.start_call.assert_not_awaited()
    # The agent already asked for the address, so only the retry is spoken.
    assert result["say"] == "I couldn't find that address."
    assert result["ended"] is False and result["awaiting_answer"] is True
    assert result["instruction"] == INTAKE_INSTRUCTION
    assert await bridge.start("c1", "1 Nowhere Rd") == result     # a retried request
    assert manager.open_call.await_count == 1

    reply = await asyncio.wait_for(bridge.reply("c1", history("1 Nowhere Rd", "1 Main Street")), 1)
    manager.handle_reply.assert_awaited_once_with(session, "1 Main Street")
    assert reply["say"] == "Did you mean 1 Main St?" and reply["status"] == "awaiting_address"


@run
async def test_withheld_caller_id_reaches_open_call_as_none():
    manager, _ = intake_manager()
    await VoiceBridge(manager).start("c1", "1 Nowhere Rd", "restricted")
    manager.open_call.assert_awaited_once_with(None, "voice", "1 Nowhere Rd")


@run
async def test_new_address_during_intake_is_an_answer_not_a_conflict():
    manager, session = intake_manager()
    bridge = VoiceBridge(manager, wait_seconds=30)
    await bridge.start("c1", "1 Nowhere Rd")
    result = await asyncio.wait_for(bridge.start("c1", "1 Main Street"), 1)
    manager.handle_reply.assert_awaited_once_with(session, "1 Main Street")
    assert result["say"] == "Did you mean 1 Main St?"
    session.status = "listening"
    with pytest.raises(VoiceError) as error:
        await bridge.start("c1", "2 Other St")
    assert error.value.status == 409


@run
async def test_transferred_ends_the_call_with_the_handoff_text():
    manager, session = intake_manager(status="transferred", says=(questions.OUT_OF_AREA_MESSAGE,))
    bridge = VoiceBridge(manager, wait_seconds=30)
    result = await asyncio.wait_for(bridge.start("c1", "1 Nowhere Rd"), 1)
    assert result["ended"] is True and result["status"] == "transferred"
    assert result["say"] == voice(questions.OUT_OF_AREA_MESSAGE)
    assert result["instruction"] == END_INSTRUCTION and result["awaiting_answer"] is False
    assert (await bridge.status("c1"))["say"] == "The location interview has ended."
    # Even if the intake also marked the session ended, the hand-off text is what the caller hears.
    manager, session = intake_manager(status="transferred", says=(questions.OUT_OF_AREA_MESSAGE,))
    session.ended = True
    result = await VoiceBridge(manager).start("c1", "1 Nowhere Rd")
    assert result["say"] == voice(questions.OUT_OF_AREA_MESSAGE) and result["ended"] is True


@run
async def test_out_ends_the_call():
    manager, session = intake_manager(status="out", says=("Stay outside.",))
    result = await VoiceBridge(manager).start("c1", "1 Main St")
    assert result["ended"] is True and result["instruction"] == END_INSTRUCTION


@run
async def test_manager_without_intake_falls_back_to_start_call():
    session = SimpleNamespace(id="s", address="1 Main St", status="loading", pending=None, ended=False,
                              ready=asyncio.Event(), messages=[{"role": "dispatch", "text": "Describe the room."}])
    manager = SimpleNamespace(open_call=AsyncMock(side_effect=NotImplementedError), start_call=AsyncMock(
        return_value=session))
    result = await VoiceBridge(manager).start("c1", "1 Main St", "anonymous")
    manager.start_call.assert_awaited_once_with("", "1 Main St", channel="voice")
    assert result["say"] == "Describe the room." and result["ended"] is False


def test_for_channel_is_idempotent_for_intake_text():
    # The bridge applies for_channel to text sessions._say may already have worded for voice.
    for name in ("ADDRESS_QUESTION", "ADDRESS_RETRY", "OUT_OF_AREA_MESSAGE", "OUT_CONFIRM", "OPENING_MESSAGE"):
        text = getattr(questions, name)
        assert voice(voice(text)) == voice(text)


# ---- .env loading ----------------------------------------------------------------------------------------------


def test_dotenv_last_value_wins_and_real_environment_still_wins(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# setup runs left a blank line above the real one\n"
                   "LL_TEST_TOKEN=\n"
                   "LL_TEST_TOKEN=abc$123\n"
                   "LL_TEST_QUOTED=\"two words\"\n"
                   "export LL_TEST_EXPORTED=yes\n"
                   "LL_TEST_REAL=from-file\n"
                   "LL_TEST_BARE\n", encoding="utf-8")
    for name in ("LL_TEST_TOKEN", "LL_TEST_QUOTED", "LL_TEST_EXPORTED", "LL_TEST_BARE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LL_TEST_REAL", "from-environment")
    config._load_dotenv(env)
    assert os.environ["LL_TEST_TOKEN"] == "abc$123"
    assert os.environ["LL_TEST_QUOTED"] == "two words"
    assert os.environ["LL_TEST_EXPORTED"] == "yes"
    assert os.environ["LL_TEST_REAL"] == "from-environment"
    assert "LL_TEST_BARE" not in os.environ
    config._load_dotenv(tmp_path / "missing.env")     # no file: nothing happens
