"""Minimal client for the Matterport Model API (GraphQL over HTTP Basic auth)."""

import math

import httpx

from . import addresses, config


class MatterportError(RuntimeError):
    pass


def graphql(query: str, variables: dict | None = None) -> dict:
    if not config.MATTERPORT_TOKEN_ID or not config.MATTERPORT_TOKEN_SECRET:
        raise MatterportError("MATTERPORT_TOKEN_ID / MATTERPORT_TOKEN_SECRET are not set in .env")
    response = httpx.post(
        config.MATTERPORT_API_URL,
        json={"query": query, "variables": variables or {}},
        auth=(config.MATTERPORT_TOKEN_ID, config.MATTERPORT_TOKEN_SECRET),
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise MatterportError("; ".join(e.get("message", "?") for e in body["errors"]))
    return body["data"]


SEARCH_QUERY = """
query Search($q: String!, $n: Int!) {
  models(query: $q, pageSize: $n) { results { id name address { address } } }
}
"""

MODEL_QUERY = """
query Model($id: ID!, $res: String!) {
  model(id: $id) {
    id
    name
    address { address }
    floors { id label sequence }
    rooms {
      id
      label
      classifications { label }
      floor { id }
      boundary { edges { vertices { position { x y } } } }
    }
    panoLocations {
      id
      position { x y z }
      anchor { room { id } floor { id } }
      skybox(resolution: $res) { children }
    }
    assets {
      floorplans { format flags width height resolution origin { x y } floor { id } url }
    }
  }
}
"""


def _search_variants(address: str) -> list[str]:
    """Query strings to search with. Matterport ranks by word overlap, so the full address can be crowded out by
    city words ("Houston" matches thousands of models); the bare street line and street name are searched too."""
    parsed = addresses.parse_freeform(address)
    street = " ".join(parsed.street)
    variants = [address, f"{parsed.number or ''} {street}".strip(), street]
    return list(dict.fromkeys(v for v in variants if v))


def search_models(address: str) -> list[dict]:
    """Candidate models for an address as [{id, name, address}], in Matterport's rank order, de-duplicated."""
    found: dict[str, dict] = {}
    for query in _search_variants(address):
        results = graphql(SEARCH_QUERY, {"q": query, "n": config.ADDRESS_SEARCH_PAGE_SIZE})["models"]["results"]
        for result in results or []:
            model_address = (result.get("address") or {}).get("address") or result.get("name") or ""
            found.setdefault(result["id"], {"id": result["id"], "name": result["name"], "address": model_address})
    return list(found.values())


def rank_matches(address: str, models: list[dict]) -> list[dict]:
    """Models whose address is sufficiently similar (config.ADDRESS_MATCH_THRESHOLD), best first, each with
    `score` and `exact` added. Ties keep the input order (local cache first, then Matterport's rank)."""
    scored = []
    for model in models:
        # Some models carry the address only in the name ("7921 Pointer St"), so the better of the two counts. The
        # full address field is still what the dispatcher sees, since it includes the city.
        texts = [t for t in (model["address"], model["name"]) if t]
        score = max((addresses.similarity(address, t) for t in texts), default=0.0)
        if score >= config.ADDRESS_MATCH_THRESHOLD:
            scored.append({**model, "address": model["address"] or model["name"], "score": round(score, 3),
                           "exact": any(addresses.is_exact(address, t) for t in texts)})
    return sorted(scored, key=lambda m: -m["score"])


GEO_QUERY = """
query Geo($id: ID!) {
  model(id: $id) { geocoordinates { latitude longitude rotation { z w } translation { x y } } }
}
"""


def fetch_geo(model_id: str) -> dict | None:
    """Where the model sits on Earth: {lat, lon, yaw, tx, ty}. Model meters map to east/north meters around
    (lat, lon) by rotating by `yaw` (radians, counter-clockwise) and adding (tx, ty). Checked against Matterport's own
    geoLocationOf on two models: exact (residuals ~1e-11 m). The heading matched the roofs in aerial imagery; the
    position can be a few meters off, which doesn't matter for which side of the house a street is on."""
    coords = graphql(GEO_QUERY, {"id": model_id})["model"]["geocoordinates"]
    if not coords or coords.get("latitude") is None:
        return None
    rotation, translation = coords["rotation"], coords["translation"]
    return {
        "lat": coords["latitude"], "lon": coords["longitude"],
        # Rotation is about the vertical axis only (x = y = 0 in every model seen), so yaw = 2·atan2(z, w).
        "yaw": 2 * math.atan2(rotation["z"], rotation["w"]),
        "tx": translation["x"], "ty": translation["y"],
    }


def fetch_model(model_id: str) -> dict:
    return graphql(MODEL_QUERY, {"id": model_id, "res": config.SKYBOX_RESOLUTION})["model"]
