"""Fixes from the pre-demo audit: size words aren't room names, one reply at a time, no search for a cached exact
address."""

import asyncio
import json

import fakes  # noqa: F401  (stubs torch / open_clip / PIL when missing)
from app import config, matterport, questions, sessions


def test_size_comparisons_do_not_name_a_room():
    for text in ("closet-sized", "bedroom-sized", "bigger, like a living room", "about the size of a closet",
                 "smaller than a bedroom", "bathroom sized"):
        assert questions.room_types_mentioned(text) == set(), text
    assert questions.room_types_mentioned("it looks like a bedroom") == {"bedroom"}
    assert questions.room_types_mentioned("I'm in the closet") == {"closet"}


def test_cached_exact_address_skips_the_search(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path)
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "building.json").write_text(json.dumps(
        {"id": "m1", "name": "13122 Chipman Glen Dr", "address": "13122 Chipman Glen Dr, Houston, TX, 77082, US"}))

    def no_search(address):
        raise AssertionError("searched although the exact address is cached")

    monkeypatch.setattr(matterport, "search_models", no_search)
    matches = sessions.find_matches("13122 Chipman Glen Dr, Houston, TX")
    assert matches[0]["id"] == "m1" and matches[0]["exact"]


def test_replies_are_processed_one_at_a_time(monkeypatch):
    manager = sessions.SessionManager(notify=lambda s: asyncio.sleep(0))
    session = sessions.Session(id="s", phone="+15555550100", address="x", created=0.0, status="listening")
    active, seen = [0], []

    async def fake_process(sess, text):
        active[0] += 1
        seen.append(active[0])
        await asyncio.sleep(0.01)   # _process awaits CLIP / Claude in the middle
        active[0] -= 1

    monkeypatch.setattr(manager, "_process", fake_process)

    async def both():
        await asyncio.gather(manager.handle_reply(session, "a"), manager.handle_reply(session, "b"))

    asyncio.run(both())
    assert seen == [1, 1]
