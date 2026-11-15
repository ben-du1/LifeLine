"""Review findings: third-person "outside" read as the caller being out, a silent "no" to OUT_CONFIRM, OUT during
loading overwritten by the transfer, and address lookups that stall or fail silently. Offline (tests/fakes.py)."""

import asyncio

import httpx
import pytest

import fakes
from app import catalog, config, ingest, matterport, messaging, questions, resolve
from test_intake import HOUSE, NEIGHBOUR, PHONE, dispatched, fake_resolve, manager, settle


@pytest.fixture
def sent(monkeypatch, tmp_path):
    fakes.install(monkeypatch, tmp_path, [HOUSE, NEIGHBOUR])
    monkeypatch.setattr(resolve, "resolve", fake_resolve)
    outbox: list[tuple[str, str]] = []

    async def send(phone, text):
        outbox.append((phone, text))
    monkeypatch.setattr(messaging, "send", send)
    return outbox


# ---- 1. only the caller's own "out" counts

@pytest.mark.parametrize("text", ["my kids are outside, im in the bedroom", "smoke is outside my window",
                                  "the fire is outside", "someone is outside banging", "power is out",
                                  "lights are out", "the lights went out"])
def test_third_person_out_is_not_the_caller(text):
    assert questions.out_signal(text) is None


def test_out_on_the_balcony_is_only_maybe():
    assert questions.out_signal("im out on the balcony") == "maybe"
    assert questions.out_signal("I'm out on the roof") == "maybe"


@pytest.mark.parametrize("text", ["I'm out", "we got out", "OUT", "I'm outside now", "I got out", "we are all out",
                                  "made it out"])
def test_first_person_out_still_counts(text):
    assert questions.out_signal(text) == "out"


def test_smoke_outside_the_window_keeps_a_trapped_caller_listening(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipman Glen Dr")
        await settle(s)
        await m.handle_reply(s, "smoke is outside my window, I'm in the bedroom")
        assert s.status != "out" and s.danger
        assert questions.SAFETY_MESSAGE in dispatched(s)
        assert questions.OUT_MESSAGE not in dispatched(s)
    asyncio.run(run())


def test_out_text_that_mentions_fire_still_flags_danger(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipman Glen Dr")
        await settle(s)
        await m.handle_reply(s, "I'm out, the house is on fire")
        assert s.status == "out" and s.danger
        assert dispatched(s)[-1] == questions.OUT_MESSAGE
    asyncio.run(run())


# ---- 2. "no" to OUT_CONFIRM before any question was asked

def test_no_to_out_confirm_as_first_text_asks_for_the_address(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "I think I'm outside")
        assert s.out_pending and dispatched(s) == [questions.OUT_CONFIRM]
        await m.handle_reply(s, "no")
        assert s.status == "awaiting_address" and s.address_asked
        assert dispatched(s)[-1] == questions.ADDRESS_QUESTION
    asyncio.run(run())


# ---- 3. OUT while the building loads is not overwritten by the transfer

def test_out_during_loading_is_not_overwritten_by_the_transfer(sent, monkeypatch):
    async def run():
        m = manager()
        release = asyncio.Event()

        async def unusable(model_id):
            await release.wait()
            raise ingest.UnusableModel("no interior photos")
        monkeypatch.setattr(m, "_get_building", unusable)
        s = await m.open_call(PHONE, "messaging", "13122 Chipman Glen Dr")
        assert s.status == "loading"
        await asyncio.sleep(0.01)   # _load is waiting in _get_building
        await m.handle_reply(s, "OUT")
        assert s.status == "out"
        release.set()
        await asyncio.wait_for(s.ready.wait(), 5)
        assert s.status == "out" and s.transfer is None
        assert questions.OUT_OF_AREA_MESSAGE not in dispatched(s)
        assert dispatched(s)[-1] == questions.OUT_MESSAGE
    asyncio.run(run())


# ---- 4. address lookups: no search for non-addresses, no silent failures, capped time

def test_non_address_text_is_nudged_without_a_lookup(sent, monkeypatch):
    def no_lookup(text, exclude=frozenset()):
        raise AssertionError(f"looked up {text!r}")
    monkeypatch.setattr(resolve, "resolve", no_lookup)

    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging")
        await m.handle_reply(s, "there's smoke, please hurry")
        assert s.status == "awaiting_address" and s.address_attempts == 0
        assert dispatched(s)[-1] == questions.ADDRESS_QUESTION
    asyncio.run(run())


def test_failed_lookup_still_gets_the_address_question(sent, monkeypatch):
    def broken(text, exclude=frozenset()):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")
    monkeypatch.setattr(resolve, "resolve", broken)

    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging")
        count = len(dispatched(s))
        await m.handle_reply(s, "45 Elm Street Springfield")
        assert s.status == "awaiting_address"
        assert len(dispatched(s)) == count + 1 and dispatched(s)[-1] == questions.ADDRESS_QUESTION
    asyncio.run(run())


@pytest.fixture
def no_local(monkeypatch):
    monkeypatch.setattr(resolve, "cached_models", lambda: [])
    monkeypatch.setattr(catalog, "candidates", lambda address, limit=None: [])


def test_resolve_survives_a_captive_portal_response(no_local, monkeypatch):
    def portal(address):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")   # HTML login page instead of JSON
    monkeypatch.setattr(matterport, "search_models", portal)
    found = resolve.resolve("7921 Pointer St")
    assert isinstance(found, resolve.Resolution) and found.band == "none"


def test_search_uses_the_short_address_timeout(monkeypatch):
    timeouts = []

    def fake_graphql(query, variables=None, timeout=60):
        timeouts.append(timeout)
        return {"models": {"results": []}}
    monkeypatch.setattr(matterport, "graphql", fake_graphql)
    matterport.search_models("7921 Pointer St, Houston")
    assert timeouts and set(timeouts) == {config.ADDRESS_SEARCH_TIMEOUT_S}


def test_graphql_timeout_defaults_to_60(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(kwargs)
        return httpx.Response(200, json={"data": {}}, request=httpx.Request("POST", url))
    monkeypatch.setattr(config, "MATTERPORT_TOKEN_ID", "id")
    monkeypatch.setattr(config, "MATTERPORT_TOKEN_SECRET", "secret")
    monkeypatch.setattr(httpx, "post", post)
    matterport.graphql("query { x }")
    assert seen["timeout"] == 60


# ---- 5. "yes, I'm in the bedroom" to "Did you mean ...?"

def test_confirmation_with_a_description_is_read_after_loading(sent):
    async def run():
        m = manager()
        s = await m.open_call(PHONE, "messaging", "13122 Chipmen Glen Dr")
        assert s.waiting_for_match
        await m.handle_reply(s, "yes, I'm in the bedroom by the bed")
        await settle(s)
        assert s.status in ("listening", "located", "best_guess") and s.model_id == HOUSE.model_id
        assert s.evidence and m.snapshot(s)["rooms"][0]["name"] == "Bedroom"
    asyncio.run(run())
