"""Caller address -> band (resolve.py) and caller-side parsing (addresses.parse_caller). Offline: the cache, the catalog
and Matterport's search are all replaced."""

import pytest

import fakes  # noqa: F401  (stubs torch / open_clip / PIL when missing)
from app import addresses, catalog, matterport, resolve

CHIPMAN = {"id": "chip", "name": "13122 Chipman Glen Dr", "address": "13122 Chipman Glen Dr, Houston, TX, 77082, US"}
POINTER = {"id": "ptr", "name": "7921 Pointer St", "address": "7921 Pointer St, Houston, TX, 77016, US"}
ELDRIDGE = {"id": "eld", "name": "x", "address": "5819 Eldridge Glen Dr, Houston, TX, 77041, US"}
NEIGHBOUR = {"id": "c120", "name": "x", "address": "13120 Chipman Glen Dr, Houston, TX, 77082, US"}
UNIT_1013 = {"id": "u1013", "name": "x", "address": "700 Dunson Glen Drive, Unit 1013, Houston, TX, 77090, US"}
UNIT_107 = {"id": "u107", "name": "x", "address": "700 Dunson Glen Drive, Unit 107, Houston, TX, 77090, US"}


@pytest.fixture
def models(monkeypatch):
    """Set the cached models; the catalog is empty and the search returns nothing unless a test says otherwise."""
    cached: list[dict] = []
    monkeypatch.setattr(resolve, "cached_models", lambda: list(cached))
    monkeypatch.setattr(catalog, "candidates", lambda address, limit=None: [])
    monkeypatch.setattr(matterport, "search_models", lambda address: [])
    return cached


# The messy inputs from the plan (docs/plans/2026-11-15-caller-initiated-calls.md, Agent A), with one model per street.
@pytest.mark.parametrize("text, band, model", [
    ("13122 Chapman Glen Dr", "clarify", "chip"),                                # street typo: ask
    ("7921 Poynter St Houston", "clarify", "ptr"),
    ("Chipman Glen Drive, Houston", "clarify", "chip"),                          # no house number
    ("13212 Chipman Glen Dr Houston", "clarify", "chip"),                        # digits swapped (0.79)
    ("thirteen one twenty two chipman glen drive houston", "match", "chip"),     # spoken digit groups
    ("seventy nine twenty one pointer street", "match", "ptr"),
    ("um it's 13122 chipman glen in houston texas", "match", "chip"),            # filler and "in houston"
])
def test_messy_inputs_get_the_right_band(models, text, band, model):
    models += [CHIPMAN, POINTER, ELDRIDGE]
    r = resolve.resolve(text)
    assert r.band == band
    assert [c["id"] for c in r.candidates] == [model]


@pytest.mark.parametrize("text", ["1600 Pennsylvania Ave, Washington DC", "help", "13122 Chipman Glen Dr, Dallas TX",
                                  "5819 Eldridge Glen Dr"])
def test_unrelated_addresses_are_none(models, text):
    models += [CHIPMAN, POINTER]
    r = resolve.resolve(text)
    assert r.band == "none" and r.candidates == [] and r.distinguishing == []


def test_neighbouring_houses_are_a_choice_unless_one_is_exact(models):
    models += [CHIPMAN, NEIGHBOUR]
    r = resolve.resolve("13121 Chipman Glen Dr Houston")
    assert r.band == "choose" and {c["id"] for c in r.candidates} == {"chip", "c120"}
    assert sorted(r.distinguishing) == ["13120", "13122"]
    exact = resolve.resolve("13122 Chipman Glen Dr Houston")
    assert exact.band == "match" and [c["id"] for c in exact.candidates] == ["chip"]


def test_units_of_one_building(models):
    models += [UNIT_1013, UNIT_107]
    r = resolve.resolve("700 Dunson Glen Dr, Houston")
    assert r.band == "choose" and sorted(r.distinguishing) == ["unit 1013", "unit 107"]
    r = resolve.resolve("700 Dunson Glen Dr Unit 107 Houston")
    assert r.band == "match" and r.candidates[0]["id"] == "u107"


def test_same_address_twice_is_not_a_choice(models):
    models += [CHIPMAN, {**CHIPMAN, "id": "chip-rescan"}]
    r = resolve.resolve("13122 Chipman Glen Dr")
    assert r.band == "match" and r.candidates[0]["id"] == "chip"       # cache order breaks the tie


def test_exclude_drops_rejected_models(models):
    models += [CHIPMAN, NEIGHBOUR]
    r = resolve.resolve("13122 Chipman Glen Dr", exclude={"chip"})
    assert r.band == "clarify" and r.candidates[0]["id"] == "c120"
    assert resolve.resolve("13122 Chipman Glen Dr", exclude={"chip", "c120"}).band == "none"


def test_cleaned_text(models):
    assert resolve.resolve("um it's thirteen one twenty two chipman glen").cleaned == "13122 chipman glen"
    assert resolve.resolve("1 Nowhere Rd").cleaned == "1 Nowhere Rd"


def test_local_exact_match_skips_the_search(models, monkeypatch):
    models.append(CHIPMAN)

    def no_search(address):
        raise AssertionError("searched although the exact address is cached")
    monkeypatch.setattr(matterport, "search_models", no_search)
    assert resolve.resolve("13122 Chipman Glen Dr").band == "match"


def test_search_results_and_failures(models, monkeypatch):
    monkeypatch.setattr(matterport, "search_models", lambda address: [POINTER])
    assert resolve.resolve("7921 Pointer St").band == "match"

    def down(address):
        raise matterport.MatterportError("offline")
    monkeypatch.setattr(matterport, "search_models", down)
    assert resolve.resolve("7921 Pointer St").band == "none"


def test_catalog_candidates_are_used(models, monkeypatch):
    monkeypatch.setattr(catalog, "candidates", lambda address, limit=None: [POINTER])
    assert resolve.resolve("seventy nine twenty one pointer street").band == "match"


# ---- catalog.candidates --------------------------------------------------------------------------------------------

INDEX = """sid,name,segment,lat,lon,latlon_source,address,city,state,postal_code,internal_id,costar_property_id,homes_listing_id,splat,drone_photos,pano_count,room_count,floor_count,created,original_created,defurnished,ai_summary
aaa,5317 Saturn St,residential,34,-118,model,"5317 Saturn St, Los Angeles, CA, 90019, US",Los Angeles,CA,90019,,,1,no,,37,10,2,,,yes,yes
ddd,7921 Pointer St,residential,29,-95,model,"7921 Pointer St, Houston, TX, 77016, US",Houston,TX,77016,,,2,no,,20,14,1,,,yes,yes
fff,13122 Chipman Glen Dr,residential,29,-95,model,"13122 Chipman Glen Dr, Houston, TX, 77082, US",Houston,TX,77082,,,2,no,,20,14,1,,,yes,yes
"""


def test_catalog_candidates_from_the_index(tmp_path, monkeypatch):
    path = tmp_path / "models_index.csv"
    path.write_text(INDEX, encoding="utf-8")
    monkeypatch.setattr(catalog, "INDEX_PATH", path)
    monkeypatch.setattr(catalog, "_entries", None)
    found = catalog.candidates("13122 Chapman Glen Dr")                 # a typo still finds the street
    assert [c["id"] for c in found] == ["fff"]
    assert found[0]["address"] == "13122 Chipman Glen Dr, Houston, TX, 77082, US"   # Matterport's comma format
    assert catalog.candidates("um it's 13122 chipman glen") and catalog.candidates("help") == []
    assert matterport.rank_matches("13122 Chipman Glen Dr", found)[0]["exact"]


def test_catalog_candidates_never_download(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "INDEX_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(catalog, "_entries", None)

    def no_download(*args, **kwargs):
        raise AssertionError("resolve must not download the index")
    monkeypatch.setattr(catalog.httpx, "get", no_download)
    assert catalog.candidates("13122 Chipman Glen Dr") == []


# ---- addresses.parse_caller -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, number, street, unit, state", [
    ("thirteen one twenty two chipman glen drive houston", "13122", ["chipman", "glen"], None, None),
    ("seventy nine twenty one pointer street", "7921", ["pointer"], None, None),
    ("um it's 13122 chipman glen in houston texas", "13122", ["chipman", "glen"], None, "tx"),
    ("five oh five oak lane", "505", ["oak"], None, None),
    ("4010 West Bellfort Avenue #227, Houston", "4010", ["w", "bellfort"], "227", None),
    ("700 Dunson Glen Dr Unit 1013 Houston", "700", ["dunson", "glen"], "1013", None),
    ("I'm on Chipman Glen Drive", None, ["chipman", "glen"], None, None),
    ("233 W 27th St Houston", "233", ["w", "27th"], None, None),
])
def test_parse_caller(text, number, street, unit, state):
    parsed = addresses.parse_caller(text)
    assert (parsed.number, parsed.street, parsed.unit, parsed.state) == (number, street, unit, state)


def test_is_exact_respects_units():
    assert addresses.is_exact("700 Dunson Glen Dr Unit 1013", UNIT_1013["address"])
    assert not addresses.is_exact("700 Dunson Glen Dr Unit 1013", UNIT_107["address"])
    assert addresses.is_exact("700 Dunson Glen Dr", UNIT_107["address"])    # no unit given: every unit is exact
