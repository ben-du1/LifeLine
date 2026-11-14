from app import addresses, config, matterport

SAMPLE = "13122 Chipman Glen Dr, Houston, TX, 77082, US"
THRESHOLD = config.ADDRESS_MATCH_THRESHOLD


def accepted(query: str, candidate: str = SAMPLE) -> bool:
    return addresses.similarity(query, candidate) >= THRESHOLD


def test_accepts_formatting_differences_and_typos():
    assert accepted("13122 Chipman Glen Dr, Houston, TX")
    assert accepted("13122 chipman glen drive")                       # suffix spelled out, no city
    assert accepted("13122 Chipmen Glen Drive, Houston TX 77082")      # typo in the street name
    assert accepted("13122 chipman glen houston tx")                   # no suffix, no commas
    assert accepted("4010 West Bellfort Avenue #227, Houston", "4010 W Bellfort Ave, 227, Houston, TX, 77025, US")
    assert accepted("4010 Bellfort, Houston", "4010 W Bellfort Ave, 227, Houston, TX, 77025, US")


def test_accepts_nearby_house_on_same_street():
    assert accepted("13120 Chipman Glen Dr Houston")
    assert addresses.similarity("13120 Chipman Glen Dr", SAMPLE) < addresses.similarity("13122 Chipman Glen Dr", SAMPLE)


def test_rejects_different_street_or_city():
    assert not accepted("5819 Eldridge Glen Dr")                       # shares only the word "Glen"
    assert not accepted("13122 Chipman Glen Dr, Dallas, TX 75201")     # same street name, other city
    assert not accepted("233 W 27th St Houston", "233 W 26th St, Houston, TX, 77008, US")
    assert not accepted("1600 Pennsylvania Ave, Washington DC", "2520 Washington Ave, Houston, TX, 77007, US")
    assert not accepted("123 Main St, Princeton NJ", "2810 Princeton St, Houston, TX, 77009, US")


def test_unit_breaks_ties_within_a_building():
    query = "700 Dunson Glen Dr Unit 1013 Houston"
    same = addresses.similarity(query, "700 Dunson Glen Drive, Unit 1013, Houston, TX, 77090, US")
    other = addresses.similarity(query, "700 Dunson Glen Drive, Unit 107, Houston, TX, 77090, US")
    assert same > other >= THRESHOLD


def test_is_exact():
    assert addresses.is_exact("13122 chipman glen drive, houston", SAMPLE)
    assert not addresses.is_exact("13120 Chipman Glen Dr", SAMPLE)


def test_rank_matches_filters_and_orders():
    models = [
        {"id": "far", "name": "x", "address": "5819 Eldridge Glen Dr, Houston, TX, 77041, US"},
        {"id": "near", "name": "x", "address": "13120 Chipman Glen Dr, Houston, TX, 77082, US"},
        {"id": "exact", "name": "13122 Chipman Glen Dr", "address": ""},   # address only in the name
    ]
    ranked = matterport.rank_matches("13122 Chipman Glen Dr, Houston", models)
    assert [m["id"] for m in ranked] == ["exact", "near"]
    assert ranked[0]["exact"] and not ranked[1]["exact"]
