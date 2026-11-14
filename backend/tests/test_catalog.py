"""Address suggestions from the hackathon's model index (catalog.py)."""

import fakes  # noqa: F401  (stubs torch / open_clip / PIL when missing)
from app import catalog

INDEX = """sid,name,segment,lat,lon,latlon_source,address,city,state,postal_code,internal_id,costar_property_id,homes_listing_id,splat,drone_photos,pano_count,room_count,floor_count,created,original_created,defurnished,ai_summary
aaa,5317 Saturn St,residential,34,-118,model,"5317 Saturn St, Los Angeles, CA, 90019, US",Los Angeles,CA,90019,,,1,no,,37,10,2,,,yes,yes
bbb,"1 Br, 1 Ba",commercial,29,-95,model,"1 Br, 1 Ba",Houston,TX,77001,,,,no,,9,4,1,,,no,yes
ccc,Lobby,commercial,29,-95,model,US,,,,,,,no,,3,1,1,,,no,no
ddd,7921 Pointer St,residential,29,-95,model,"7921 Pointer St, Houston, TX, 77016, US",Houston,TX,77016,,,2,no,,20,14,1,,,yes,yes
eee,900 Main St,commercial,29,-95,model,"900 Main St, Houston, TX, 77002, US",Houston,TX,77002,,,,no,,80,30,3,,,no,yes
"""


def setup_index(tmp_path, monkeypatch):
    path = tmp_path / "models_index.csv"
    path.write_text(INDEX, encoding="utf-8")
    monkeypatch.setattr(catalog, "INDEX_PATH", path)
    monkeypatch.setattr(catalog, "_entries", None)


def test_only_real_street_addresses_are_suggested(tmp_path, monkeypatch):
    setup_index(tmp_path, monkeypatch)
    assert [e["model_id"] for e in catalog.search("st")] == ["aaa", "ddd", "eee"]   # houses first, then the rest
    assert catalog.search("br") == []   # listing titles aren't addresses


def test_every_word_must_match_in_any_order(tmp_path, monkeypatch):
    setup_index(tmp_path, monkeypatch)
    assert [e["address"] for e in catalog.search("pointer houston")] == ["7921 Pointer St, Houston, TX 77016"]
    assert [e["model_id"] for e in catalog.search("houston 7921")] == ["ddd"]
    assert catalog.search("") == []
