"""GET /api/demo logic (demo.statuses) on a temporary cache: one cached synthetic building and one missing."""

import json
import sys
import types

import numpy as np
import pytest

import app
from app import config, demo, ingest


@pytest.fixture
def demo_cache(tmp_path, monkeypatch):
    buildings = tmp_path / "buildings"
    cached = buildings / "CACHED01"
    cached.mkdir(parents=True)
    (cached / "building.json").write_text(json.dumps({
        "schema": ingest.SCHEMA_VERSION, "id": "CACHED01", "name": "12 Test Ln",
        "address": "12 Test Ln, Houston, TX", "floors": [{"id": "f0"}], "floorplans": {}, "rooms": [{"id": "r0"}],
        "candidates": [{"room_id": "r0", "views": []}],
    }), encoding="utf-8")
    np.savez(cached / "embeddings.npz", view_embeddings=np.zeros((0, 4)), view_candidate=np.zeros(0, dtype=np.int32))
    monkeypatch.setattr(config, "BUILDINGS_DIR", buildings)
    entries = tmp_path / "demo_buildings.json"
    entries.write_text(json.dumps([
        {"address": "99 Missing Rd, Houston, TX", "label": "Missing", "model_id": "MISSING01"},
        {"address": "12 Test Lane, Houston, TX", "label": "Cached", "note": "synthetic"},
    ]), encoding="utf-8")
    return cached, entries


def test_cached_building_is_ready_and_listed_first(demo_cache):
    cached, entries = demo_cache
    ready, missing = demo.statuses(entries)
    assert ready["label"] == "Cached" and ready["model_id"] == "CACHED01"   # resolved from the cache by address
    assert ready["ready"] and ready["cached"] and ready["match_score"] >= config.ADDRESS_MATCH_THRESHOLD
    assert not ready["orientation"]
    assert any("orientation" in p for p in ready["problems"])               # reported, but doesn't block

    assert missing["label"] == "Missing" and missing["model_id"] == "MISSING01"
    assert not missing["ready"] and not missing["cached"] and missing["match_score"] is None
    assert any("not cached" in p for p in missing["problems"])

    (cached / "orientation.json").write_text("{}", encoding="utf-8")
    ready = demo.statuses(entries)[0]
    assert ready["orientation"] and not any("orientation" in p for p in ready["problems"])


def test_stale_schema_is_not_ready(demo_cache):
    cached, entries = demo_cache
    data = json.loads((cached / "building.json").read_text(encoding="utf-8"))
    (cached / "building.json").write_text(json.dumps({**data, "schema": ingest.SCHEMA_VERSION - 1}), encoding="utf-8")
    assert not any(s["ready"] for s in demo.statuses(entries))


def test_missing_demo_file_gives_empty_list(tmp_path):
    assert demo.statuses(tmp_path / "nope.json") == []


def test_repo_demo_file_lists_the_readme_buildings():
    addresses = [e["address"] for e in demo.load_entries()]
    assert any("Chipman Glen" in a for a in addresses) and any("3012 Houston Ave" in a for a in addresses)


def test_compat_score_is_added_when_the_module_exists(demo_cache, monkeypatch):
    _, entries = demo_cache
    stub = types.ModuleType("app.compat")
    stub.score_building = lambda building, orientation=None: {"score": 0.5, "checks": [], "problems": ["no floorplan"]}
    monkeypatch.setitem(sys.modules, "app.compat", stub)
    monkeypatch.setattr(app, "compat", stub, raising=False)
    ready, missing = demo.statuses(entries)
    assert ready["score"] == 0.5 and "no floorplan" in ready["problems"]
    assert "score" not in missing                                             # only scored once cached
