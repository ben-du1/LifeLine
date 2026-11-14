"""North and the street out front (no network)."""

import math

import numpy as np

from app import orientation

LAT, LON = 29.8, -95.3
MX = orientation.METERS_PER_DEG_LAT * math.cos(math.radians(LAT))


def lonlat(east_m: float, north_m: float) -> list[float]:
    return [LON + east_m / MX, LAT + north_m / orientation.METERS_PER_DEG_LAT]


def test_street_to_the_east_running_north_south():
    result = {"name": "Pointer Street", "geojson": {"type": "LineString",
              "coordinates": [lonlat(20, -50), lonlat(20, 0), lonlat(20, 50)]}}
    street = orientation.street_from_nominatim(result, (LON, LAT))
    assert street["side"] == "east" and abs(street["distance_m"] - 20) < 0.2
    assert abs(street["direction_deg"] % 180) < 1 or abs(street["direction_deg"] - 180) < 1


def test_street_to_the_south_running_east_northeast():
    # A road 25 m south, rising toward the east-northeast (like Chapman Glen Drive).
    line = [lonlat(-40, -25 - 40 * math.tan(math.radians(11))), lonlat(40, -25 + 40 * math.tan(math.radians(11)))]
    street = orientation.street_from_nominatim({"name": "Chapman Glen Drive",
                                                "geojson": {"type": "LineString", "coordinates": line}}, (LON, LAT))
    assert street["side"] == "south" and abs(street["direction_deg"] - 79) < 1


def test_missing_geometry_or_name_gives_no_street():
    assert orientation.street_from_nominatim({"name": "X", "geojson": {"type": "Point"}}, (LON, LAT)) is None
    assert orientation.street_from_nominatim({"geojson": {"type": "LineString",
                                              "coordinates": [lonlat(0, 5), lonlat(5, 5)]}}, (LON, LAT)) is None


def test_model_heading_matches_matterport_geolocation():
    # Sample model: Matterport's geoLocationOf put model (10, 0) at 4.66 m east, 8.59 m south of its anchor.
    geo = {"yaw": 2 * math.atan2(-0.5729831074645213, 0.8195671775762502), "tx": 1.2245, "ty": 0.8141}
    en = orientation.model_to_enu(geo, np.array([[10.0, 0.0]]))[0]
    assert np.allclose(en, [4.664, -8.588], atol=0.05)
