"""Map view geometry and Vexcel aerial caching (no network: Matterport and Vexcel are faked)."""

import json
import math

import httpx
import numpy as np
import pytest

from app import config, geo, orientation

GEO = {"lat": 29.8, "lon": -95.3, "yaw": 2 * math.atan2(-0.5729831074645213, 0.8195671775762502),
       "tx": 1.2245, "ty": 0.8141}


@pytest.fixture
def building_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path)
    out = tmp_path / "M1"
    out.mkdir()
    walls_only = [[10, 0, 14, 0], [14, 0, 14, 3], [14, 3, 10, 3]]  # an open room: no polygon
    building = {
        "rooms": [
            {"id": "r0", "name": "Kitchen", "label": "Kitchen", "types": ["kitchen"], "floor": 0,
             "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]], "walls": []},
            {"id": "r1", "name": "Bedroom", "label": None, "types": ["bedroom"], "floor": 1,
             "polygon": None, "walls": walls_only},
        ],
        "candidates": [{"id": "s0", "room_id": "r0", "x": 5, "y": 2, "views": ["v.jpg"]},
                       {"id": "s1", "room_id": "r1", "x": 12, "y": 1.5, "views": []}],
    }
    (out / "building.json").write_text(json.dumps(building))
    (out / "geo.json").write_text(json.dumps(GEO))  # cached, so Matterport is never called
    monkeypatch.setattr(geo, "_aerial_unavailable", {})
    monkeypatch.setattr(geo, "_outside_coverage", set())
    return out


def test_model_to_lonlat_matches_model_to_enu():
    xy = np.array([[10.0, 0.0], [-3.0, 7.5]])
    en = orientation.model_to_enu(GEO, xy)
    lonlat = geo.model_to_lonlat(GEO, xy)
    mx = orientation.METERS_PER_DEG_LAT * math.cos(math.radians(GEO["lat"]))
    assert np.allclose(lonlat[:, 0], GEO["lon"] + en[:, 0] / mx)
    assert np.allclose(lonlat[:, 1], GEO["lat"] + en[:, 1] / orientation.METERS_PER_DEG_LAT)
    # Sample model check from test_orientation: (10, 0) is 4.66 m east, 8.59 m south of the anchor.
    assert abs((lonlat[0, 0] - GEO["lon"]) * mx - 4.664) < 0.05
    assert abs((lonlat[0, 1] - GEO["lat"]) * orientation.METERS_PER_DEG_LAT + 8.588) < 0.05


def test_building_geojson(building_dir):
    out = geo.building_geojson("M1")
    assert out["type"] == "FeatureCollection"
    rooms = [f for f in out["features"] if f["properties"]["kind"] == "room"]
    assert [r["properties"]["name"] for r in rooms] == ["Kitchen", "Bedroom"]
    assert rooms[1]["properties"]["floor"] == 1 and rooms[1]["properties"]["types"] == ["bedroom"]
    ring = rooms[0]["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    assert np.allclose(ring[1], geo.model_to_lonlat(GEO, [[10, 0]])[0], atol=1e-7)
    assert len(rooms[1]["geometry"]["coordinates"][0]) == 5   # hull of the 4 wall endpoints, closed
    assert any(f["properties"]["kind"] == "building" for f in out["features"])
    assert out["candidates"][1] == {"candidate": 1, "room": 1, "floor": 1,
                                    **dict(zip(["lon", "lat"], np.round(geo.model_to_lonlat(GEO, [[12, 1.5]])[0], 8)))}
    (s, w), (n, e) = out["bounds"]
    assert s < out["center"]["lat"] < n and w < out["center"]["lon"] < e
    assert out["yaw_deg"] == round(math.degrees(GEO["yaw"]), 2)


def test_building_geojson_missing_building(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path)
    assert geo.building_geojson("nope") is None


def fake_post(status, content=b"\xff\xd8jpegdata", calls=None):
    def post(url, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return httpx.Response(status, content=content, request=httpx.Request("POST", url))
    return post


def test_aerial_success_is_cached(building_dir, monkeypatch):
    monkeypatch.setattr(config, "VEXCEL_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(geo.httpx, "post", fake_post(200, calls=calls))
    first = geo.aerial("M1", size_m=100)
    assert first["source"] == "Vexcel" and (building_dir / "aerial.jpg").read_bytes().startswith(b"\xff\xd8")
    (s, w), (n, e) = first["bounds"]
    assert abs((n - s) * orientation.METERS_PER_DEG_LAT - 100) < 1e-6
    assert abs((e - w) * orientation.METERS_PER_DEG_LAT * math.cos(math.radians((n + s) / 2)) - 100) < 0.01
    body = calls[0]["json"]
    assert body["srid"] == 4326 and body["wkt"].startswith("POLYGON((") and calls[0]["headers"]["X-API-KEY"]
    assert geo.aerial("M1") == first and len(calls) == 1   # second call served from disk


def test_aerial_403_outside_coverage(building_dir, monkeypatch):
    monkeypatch.setattr(config, "VEXCEL_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(geo.httpx, "post", fake_post(403, content=b"{}", calls=calls))
    assert geo.aerial("M1") is None and "coverage" in geo.aerial_reason("M1")
    assert geo.aerial("M1") is None and len(calls) == 1   # not retried
    assert not (building_dir / "aerial.jpg").exists()


def test_aerial_without_key_or_with_non_jpeg(building_dir, monkeypatch):
    monkeypatch.setattr(config, "VEXCEL_API_KEY", "")
    monkeypatch.setattr(geo.httpx, "post", lambda *a, **k: pytest.fail("no request without a key"))
    assert geo.aerial("M1") is None and "VEXCEL_API_KEY" in geo.aerial_reason("M1")
    monkeypatch.setattr(config, "VEXCEL_API_KEY", "test-key")
    monkeypatch.setattr(geo.httpx, "post", fake_post(200, content=b'{"error": "bad"}'))
    assert geo.aerial("M1") is None and not (building_dir / "aerial.json").exists()


def test_cached_geo_fetches_once(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path)
    (tmp_path / "M2").mkdir()
    calls = []
    monkeypatch.setattr(orientation.matterport, "fetch_geo", lambda model_id: calls.append(model_id) or GEO)
    assert orientation.cached_geo("M2") == GEO and orientation.cached_geo("M2") == GEO
    assert calls == ["M2"]
