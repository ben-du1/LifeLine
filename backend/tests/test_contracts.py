"""Shape of the caller-initiated-call contracts (Phase 0): names and shapes the workstreams agreed on."""

import asyncio

import pytest

import fakes  # noqa: F401  (stubs torch / open_clip / PIL when missing)
from app import handoff, phones, questions, resolve, sessions


def test_resolution_shape(monkeypatch):
    # resolve's real inputs, emptied: no cached buildings, no catalog rows, and no Matterport search (network).
    monkeypatch.setattr(resolve, "cached_models", lambda: [])
    monkeypatch.setattr(resolve.catalog, "candidates", lambda text: [])
    monkeypatch.setattr(resolve.matterport, "search_models", lambda text: [])
    r = resolve.resolve("1 Nowhere Rd")
    assert r.band == "none" and r.candidates == [] and r.cleaned == "1 Nowhere Rd" and r.distinguishing == []


def test_phone_contract():
    assert phones.to_e164("+1 (609) 555-0100") == "+16095550100"
    assert phones.to_e164(None) is None and phones.to_e164("") is None


def test_handoff_contract():
    s = sessions.Session(id="s", phone="+15555550100", address="", created=0.0)
    record = handoff.transfer_to_emergency_services(s, "address not in coverage")
    assert s.status == "transferred" and s.transfer == record and record["stub"] is True


def test_question_contract():
    for name in ("ADDRESS_QUESTION", "ADDRESS_RETRY", "ADDRESS_CONFIRM", "ADDRESS_CHOOSE", "OUT_OF_AREA_MESSAGE",
                 "OUT_CONFIRM"):
        assert isinstance(getattr(questions, name), str)
    assert "{address}" in questions.ADDRESS_CONFIRM and "{a}" in questions.ADDRESS_CHOOSE
    assert isinstance(questions.for_channel("hi", "voice"), str)
    assert questions.out_signal("OUT") == "out"
    assert questions.looks_like_address("13122 Chipman Glen Dr")


def test_snapshot_has_new_fields():
    manager = sessions.SessionManager(notify=lambda s: asyncio.sleep(0))
    s = sessions.Session(id="s", phone="", address="", created=0.0)
    snap = manager.snapshot(s)
    assert snap["transfer"] is None and snap["phone"] is None and "channel" in snap


def test_open_call_exists():
    manager = sessions.SessionManager(notify=lambda s: asyncio.sleep(0))
    s = asyncio.run(manager.open_call("+15555550100", "messaging"))
    assert s.status == "awaiting_address"
