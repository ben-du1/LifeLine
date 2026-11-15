"""Caller-initiated calls (workstream B): address intake, the out-of-area handoff stub and inbound texts.

Runs the real SessionManager on a synthetic building (tests/fakes.py). resolve.resolve is replaced by a fake that
returns the band each test needs, so these tests don't depend on the address matcher (workstream A).
"""

import asyncio
import time

import httpx
import pytest

import fakes
from fakes import Room, Spec, Sweep
from app import config, main, messaging, questions, resolve, sessions

HOUSE = Spec("HOUSE", "13122 Chipman Glen Dr, Houston, TX 77082", [
    Room("Kitchen", ["Kitchen"], (0, 0, 5, 4), [Sweep(2.5, 2, [
        {"stove": 1.0, "counter": 0.8}, {"fridge": 1.0, "tile": 0.4}, {"sink": 0.8, "counter": 0.6},
        {"table": 1.0, "tile": 0.4}])]),
    Room("Bedroom", ["Bedroom"], (6, 0, 10, 4), [Sweep(8, 2, [
        {"bed": 1.0, "carpet": 0.5}, {"window": 1.0, "carpet": 0.4}, {"door": 0.6, "carpet": 0.5},
        {"shelves": 0.4, "white_wall": 0.6}])]),
])
NEIGHBOUR = Spec("NEIGHBOUR", "13120 Chipman Glen Dr, Houston, TX 77082", HOUSE.rooms)

PHONE = "+1 (609) 555-0142"
E164 = "+16095550142"


def candidate(spec: Spec, score: float = 1.0, exact: bool = True) -> dict:
    return {"id": spec.model_id, "name": spec.address, "address": spec.address, "score": score, "exact": exact}


def fake_resolve(text: str, exclude=frozenset()) -> resolve.Resolution:
    """Bands by keyword: "chipman" with 13122 -> match, "chipmen" -> clarify, "both" -> choose, else none."""
    t = text.casefold()
    if "both" in t:
        options = [c for c in (candidate(HOUSE, 0.9, False), candidate(NEIGHBOUR, 0.88, False))
                   if c["id"] not in exclude]
        return resolve.Resolution("choose", options, text.strip(), ["13122", "13120"])
    if "chipmen" in t and HOUSE.model_id not in exclude:
        return resolve.Resolution("clarify", [candidate(HOUSE, 0.85, False)], text.strip())
    if "13122 chipman" in t and HOUSE.model_id not in exclude:
        return resolve.Resolution("match", [candidate(HOUSE)], text.strip())
    return resolve.Resolution("none", [], text.strip())


@pytest.fixture
def sent(monkeypatch, tmp_path):
    """Install the fakes; record every text "sent" to a caller instead of sending it."""
    fakes.install(monkeypatch, tmp_path, [HOUSE, NEIGHBOUR])
    monkeypatch.setattr(resolve, "resolve", fake_resolve)
    outbox: list[tuple[str, str]] = []

    async def send(phone, text):
        outbox.append((phone, text))
    monkeypatch.setattr(messaging, "send", send)
    return outbox


def manager() -> sessions.SessionManager:
    async def notify(session):
        return None
    return sessions.SessionManager(notify)


async def settle(session, timeout: float = 5.0) -> None:
    waited = 0.0
    while session.status == "loading":
        assert waited < timeout, session.status_detail
        await asyncio.sleep(0.01)
        waited += 0.01


def dispatched(session) -> list[str]:
    return [m["text"] for m in session.messages if m["role"] == "dispatch"]


def test_unknown_number_opens_awaiting_address(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "help")
        assert s.status == "awaiting_address" and s.phone == E164
        assert dispatched(s) == [questions.ADDRESS_QUESTION]
        assert sent == [(E164, questions.ADDRESS_QUESTION)]
        snap = m.snapshot(s)
        assert snap["address"] == "" and snap["phone"] == E164 and snap["transfer"] is None
    asyncio.run(run())


def test_no_first_text_asks_for_the_address(sent):
    async def run():
        s = await manager().open_call(PHONE, "messaging")
        assert s.status == "awaiting_address" and dispatched(s) == [questions.ADDRESS_QUESTION]
    asyncio.run(run())


def test_address_first_text_skips_the_question_and_loads(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipman Glen Dr Houston")
        assert questions.ADDRESS_QUESTION not in dispatched(s)
        await settle(s)
        assert s.status == "listening" and s.model_id == HOUSE.model_id and s.match["exact"]
        assert s.address == "13122 Chipman Glen Dr Houston"
        before = len(s.evidence)
        await m.handle_reply(s, "I'm in the kitchen next to the stove")
        assert len(s.evidence) == before + 1   # read as a room description, not as an address
        assert m.snapshot(s)["rooms"][0]["name"] == "Kitchen"
    asyncio.run(run())


def test_unresolved_address_twice_hands_off_without_network(sent, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("the handoff must not make a network call")
    monkeypatch.setattr(httpx, "AsyncClient", no_network)
    monkeypatch.setattr(httpx, "Client", no_network)

    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "hi")
        await m.handle_reply(s, "45 Elm Street Springfield")
        assert s.status == "awaiting_address" and dispatched(s)[-1] == questions.ADDRESS_RETRY
        await m.handle_reply(s, "45 Elm St, Springfield IL")
        assert s.status == "transferred"
        assert s.transfer["stub"] is True and s.transfer["target"] == "standard 911 (PSAP)"
        assert dispatched(s)[-1] == questions.OUT_OF_AREA_MESSAGE
        assert any("Stub: no real transfer" in msg["text"] for msg in s.messages if msg["role"] == "system")
        snap = m.snapshot(s)
        assert snap["status"] == "transferred" and snap["transfer"]["stub"] is True
        # Later texts are recorded only: no more questions.
        count = len(dispatched(s))
        await m.handle_reply(s, "hello? are you there")
        await m.handle_reply(s, "13122 Chipman Glen Dr")
        assert len(dispatched(s)) == count and s.status == "transferred"
        assert s.messages[-1]["role"] == "caller"
    asyncio.run(run())


def test_non_address_reply_is_asked_again_without_using_the_retry(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging")
        await m.handle_reply(s, "please hurry")
        assert s.status == "awaiting_address" and s.address_attempts == 0
        assert dispatched(s)[-1] == questions.ADDRESS_QUESTION
    asyncio.run(run())


def test_clarify_yes_loads_the_suggestion(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipmen Glen Dr")
        assert s.status == "awaiting_address" and s.waiting_for_match
        assert dispatched(s)[-1] == questions.ADDRESS_CONFIRM.format(address=HOUSE.address)
        await m.handle_reply(s, "yes")
        await settle(s)
        assert s.status == "listening" and s.model_id == HOUSE.model_id and s.address_confirmed
        assert not s.waiting_for_match
    asyncio.run(run())


def test_clarify_no_asks_again_and_two_rejections_hand_off(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipmen Glen Dr")
        await m.handle_reply(s, "no")
        assert s.status == "awaiting_address" and HOUSE.model_id in s.rejected_models
        assert dispatched(s)[-1] == questions.ADDRESS_RETRY
        await m.handle_reply(s, "both of them on chipman glen")   # choose between the two houses
        assert s.waiting_for_match and [o["id"] for o in s.address_options] == [NEIGHBOUR.model_id]
        await m.handle_reply(s, "no")
        assert s.status == "transferred" and s.transfer["stub"] is True
    asyncio.run(run())


def test_unclear_answer_to_a_suggestion_is_asked_once_more(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipmen Glen Dr")
        await m.handle_reply(s, "what?")
        assert s.waiting_for_match
        assert dispatched(s)[-1] == f"{questions.NOT_UNDERSTOOD} {questions.ADDRESS_CONFIRM.format(address=HOUSE.address)}"
    asyncio.run(run())


@pytest.mark.parametrize("answer, chosen", [
    ("13120", NEIGHBOUR.model_id),
    ("it's thirteen one twenty two", HOUSE.model_id),
    ("the second one", NEIGHBOUR.model_id),
])
def test_choose_picks_one_of_two(sent, answer, chosen):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "both 1312 chipman dr")
        assert dispatched(s)[-1] == questions.ADDRESS_CHOOSE.format(a=HOUSE.address, b=NEIGHBOUR.address)
        await m.handle_reply(s, answer)
        await settle(s)
        assert s.status == "listening" and s.model_id == chosen
    asyncio.run(run())


def test_choose_neither_asks_again(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "both 1312 chipman dr")
        await m.handle_reply(s, "neither")
        assert s.status == "awaiting_address" and dispatched(s)[-1] == questions.ADDRESS_RETRY
        assert s.rejected_models == {HOUSE.model_id, NEIGHBOUR.model_id}
    asyncio.run(run())


def test_out_first_text_never_asks_for_an_address(sent):
    async def run():
        s = await manager().open_call(PHONE, "messaging", "OUT")
        assert s.status == "out"
        assert dispatched(s) == [questions.OUT_MESSAGE]
    asyncio.run(run())


def test_smoke_before_the_address_sends_advice_and_still_asks(sent):
    async def run():
        s = await manager().open_call(PHONE, "messaging", "there's smoke everywhere")
        assert s.danger and s.status == "awaiting_address"
        assert dispatched(s) == [questions.SAFETY_MESSAGE, questions.ADDRESS_QUESTION]
    asyncio.run(run())


def test_maybe_out_asks_to_confirm(sent, monkeypatch):
    original = questions.out_signal
    monkeypatch.setattr(questions, "out_signal",
                        lambda text: "maybe" if "mailbox" in text else original(text))

    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging")
        await m.handle_reply(s, "I'm by the mailbox")
        assert s.out_pending and dispatched(s)[-1] == questions.OUT_CONFIRM
        await m.handle_reply(s, "no")
        assert not s.out_pending and s.status == "awaiting_address"
        assert dispatched(s)[-1] == questions.ADDRESS_QUESTION   # back to the question they were answering
        await m.handle_reply(s, "near the mailbox now")
        await m.handle_reply(s, "yes")
        assert s.status == "out" and dispatched(s)[-1] == questions.OUT_MESSAGE
    asyncio.run(run())


def test_bare_out_acts_at_once_while_listening(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipman Glen Dr")
        await settle(s)
        await m.handle_reply(s, "out")
        assert s.status == "out" and not s.out_pending
    asyncio.run(run())


def test_inbound_opens_reuses_and_reopens(sent, monkeypatch):
    m = manager()
    monkeypatch.setattr(main, "manager", m)

    def inbound(text, sender=PHONE, event_id=None):
        return main.inbound(main.InboundIn(**{"from": sender, "text": text, "event_id": event_id}))

    async def run():
        first = await inbound("help", event_id="e1")
        assert first["ok"] and first["created"]
        s = m.sessions[first["session"]]
        assert s.status == "awaiting_address"
        again = await inbound("help", event_id="e1")   # the sidecar retried the same event
        assert not again["created"] and again["session"] == s.id
        assert len([msg for msg in s.messages if msg["role"] == "caller"]) == 1
        # Same number in another format: same call.
        other_format = await inbound("13122 Chipman Glen Dr", sender="6095550142")
        assert not other_format["created"] and other_format["session"] == s.id
        await settle(s)
        await m.end_call(s, "test over")
        fresh = await inbound("hello")
        assert fresh["created"] and fresh["session"] != s.id
        # A call quiet for longer than SESSION_REUSE_MINUTES is not continued.
        stale = m.sessions[fresh["session"]]
        old = time.time() - (config.SESSION_REUSE_MINUTES + 1) * 60
        stale.created = old
        for msg in stale.messages:
            msg["ts"] = old
        newer = await inbound("hello again")
        assert newer["created"] and newer["session"] != stale.id
    asyncio.run(run())


def test_find_by_phone_never_matches_an_unknown_number(sent):
    async def run():
        m = manager()
        s = await m.open_call(None, "messaging", "help")
        assert s.phone is None and m.snapshot(s)["phone"] is None
        assert m.find_by_phone(None) is None and m.find_by_phone("") is None
        assert sent == []   # nobody to text
    asyncio.run(run())


def test_imessage_email_handle_is_kept_for_replies(sent):
    async def run():
        m = manager()
        s = await m.open_call("caller@example.com", "messaging", "help")
        assert s.phone is None and sent == [("caller@example.com", questions.ADDRESS_QUESTION)]
        assert m.find_by_phone("Caller@example.com") is s
    asyncio.run(run())


def test_voice_session_never_texts_and_uses_spoken_wording(sent, monkeypatch):
    monkeypatch.setattr(questions, "for_channel",
                        lambda text, channel: f"[spoken] {text}" if channel == "voice" else text)

    async def run():
        m = manager()
        s = await m.open_call(PHONE, "voice")
        assert s.ready.is_set()   # the voice bridge can answer at once
        assert dispatched(s) == [f"[spoken] {questions.ADDRESS_QUESTION}"]
        await m.handle_reply(s, "13122 Chipman Glen Dr")
        await settle(s)
        assert s.status == "listening"
        assert all(text.startswith("[spoken] ") for text in dispatched(s))
        assert sent == []
        assert m.find_by_phone(PHONE) is None   # texts never join a voice call
    asyncio.run(run())


def test_start_call_without_a_usable_model_hands_off(sent, monkeypatch):
    monkeypatch.setattr(sessions, "find_matches", lambda address: [])

    async def run():
        s = await manager().start_call("+15555550100", "1 Nowhere Rd, Nowhere, ZZ")
        await settle(s)
        assert s.status == "transferred" and s.transfer["stub"] is True
        assert dispatched(s) == [questions.OPENING_MESSAGE, questions.OUT_OF_AREA_MESSAGE]
        assert s.ready.is_set()
    asyncio.run(run())


def test_voice_rejecting_the_loaded_building_asks_for_the_address(sent, monkeypatch):
    near = {**candidate(HOUSE, 0.9, False)}
    monkeypatch.setattr(sessions, "find_matches", lambda address: [near])

    async def run():
        m = manager()
        s = await m.start_call("voice:conv", "13122 Chipman Glenn", channel="voice")
        await settle(s)
        assert s.waiting_for_match
        await m.handle_reply(s, "no")
        assert s.status == "awaiting_address" and s.model_id is None and s.posterior is None
        assert HOUSE.model_id in s.rejected_models
        assert dispatched(s)[-1] == questions.for_channel(questions.ADDRESS_RETRY, "voice")
        await m.handle_reply(s, "13122 Chipman Glen Dr")   # excluded now, so it doesn't resolve
        assert s.status == "transferred"
    asyncio.run(run())
