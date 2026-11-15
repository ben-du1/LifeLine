"""An offline stand-in for CLIP and Matterport, so the real SessionManager can run a whole call in a test.

No torch, open_clip, PIL, network or API key is needed:
  - `embed_texts` maps text to a unit vector over a fixed vocabulary of visual concepts (toilet, stove, carpet, ...)
    by word matching, plus a little deterministic noise.
  - Synthetic buildings are written as building.json + embeddings.npz into a temporary BUILDINGS_DIR. Each view's
    embedding is built from the same concepts ({"concrete": 0.7, "car": 0.7}), so a test controls exactly what the
    "photos" show, including deliberate confusers (a garage view with some carpet weight, as CLIP really scored it).

Similarities are kept in CLIP's narrow range: every vector shares a large common "a photo of a room" component, so a
good match beats an unrelated one by ~0.06 cosine, like ViT-B-32 does (see config.LOGIT_SCALE's comment).
"""

import asyncio
import hashlib
import json
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _stub_missing(name: str) -> None:
    """Register an empty module for a heavy dependency that isn't installed, so `app.embeddings` imports. Installed
    packages are left alone: the fake embed_texts replaces every call into them anyway."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = types.ModuleType(name)


for _name in ("torch", "open_clip", "PIL"):
    _stub_missing(_name)
if not hasattr(sys.modules["torch"], "Tensor"):   # embeddings.py uses torch.Tensor in an annotation
    sys.modules["torch"].Tensor = object
if not hasattr(sys.modules["PIL"], "Image"):
    _image = types.ModuleType("PIL.Image")
    sys.modules["PIL.Image"] = _image
    sys.modules["PIL"].Image = _image

from app import config, embeddings, ingest, llm, matterport, orientation, semantics  # noqa: E402
from app.building import Building  # noqa: E402
from app.sessions import SessionManager  # noqa: E402

# ---- the fake embedding space ---------------------------------------------------------------------------------

# Concept -> words that name it in a caller message or a CLIP prompt.
CONCEPTS = {
    "toilet": r"toilet",
    "tub": r"bath ?tubs?|\btubs?\b|shower",
    "sink": r"\bsinks?\b|faucet",
    "mirror": r"mirror|reflection",
    "stove": r"stove|oven|cook|microwave",
    "fridge": r"fridge|refrigerator|freezer",
    "counter": r"counters?|countertop|cabinets?",
    "washer": r"washer|dryer|washing machine|laundry",
    "shelves": r"shel(f|ves|ving)|hangers?|\brod\b|coats?",
    "bed": r"\bbeds?\b|mattress|sleep",
    "sofa": r"couch|sofa",
    "table": r"\btables?\b|chairs?",
    "tv": r"\btv\b|television",
    "fireplace": r"fireplace|mantel|hearth|brick",
    "window": r"windows?",
    "fan": r"\bfan\b",
    "stairs": r"stairs?|staircase|steps",
    "door": r"\bdoors?\b|entrance",
    "carpet": r"carpet|\brugs?\b",
    "tile": r"\btiles?\b|tiled|linoleum|ceramic",
    "wood": r"wood|hardwood|laminate|planks?",
    "concrete": r"concrete|cement",
    "car": r"\bcars?\b|vehicle",
    "empty": r"empty|vacant|\bbare\b",
    "white_wall": r"white walls?|plain walls?|blank walls?",
}
CONCEPT_INDEX = {name: i for i, name in enumerate(CONCEPTS)}
# Room words carry the look of the room, the way CLIP's "a photo of a kitchen" lands near stoves and cabinets.
ROOM_WORDS = {
    r"bath ?room|restroom": {"toilet": 0.6, "tub": 0.5, "sink": 0.5, "mirror": 0.4, "tile": 0.4},
    r"kitchen": {"stove": 0.7, "fridge": 0.6, "counter": 0.7, "sink": 0.3},
    r"bed ?room": {"bed": 1.0, "carpet": 0.3},
    r"living room|family room|lounge": {"sofa": 0.8, "tv": 0.5, "fireplace": 0.3},
    r"dining": {"table": 1.0},
    r"garage": {"concrete": 0.7, "car": 0.7, "shelves": 0.3},
    r"closet|pantry": {"shelves": 1.0},
    r"hallway|corridor": {"door": 0.4, "white_wall": 0.5},
}
NOISE_DIMS = 16
DIM = 1 + len(CONCEPTS) + NOISE_DIMS
# Weights of the shared "photo of a room" axis, the concept part and the noise part (each part unit length). With
# 1 / 0.3 / 0.15, a matching concept adds ~0.07 cosine and noise moves scores by ~0.005 (CLIP-like; see module doc).
COMMON, CONCEPT_WEIGHT, NOISE_WEIGHT = 1.0, 0.3, 0.15


def _noise(key: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(NOISE_DIMS)
    return v / np.linalg.norm(v)


def vector(concepts: dict[str, float], noise_key: str) -> np.ndarray:
    """Unit vector: common axis + normalized concept weights + deterministic noise keyed by `noise_key`."""
    c = np.zeros(len(CONCEPTS))
    for name, weight in concepts.items():
        c[CONCEPT_INDEX[name]] += weight
    if c.any():
        c = c / np.linalg.norm(c)
    v = np.concatenate([[COMMON], CONCEPT_WEIGHT * c, NOISE_WEIGHT * _noise(noise_key)])
    return (v / np.linalg.norm(v)).astype(np.float32)


def text_concepts(text: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for name, words in CONCEPTS.items():
        if re.search(words, text, re.I):
            found[name] = found.get(name, 0.0) + 1.0
    for words, concepts in ROOM_WORDS.items():
        if re.search(words, text, re.I):
            for name, weight in concepts.items():
                found[name] = found.get(name, 0.0) + weight
    return found


def embed_texts(texts: list[str]) -> np.ndarray:
    """Drop-in for embeddings.embed_texts (no negation handling, like CLIP)."""
    return np.stack([vector(text_concepts(t), "text:" + t.lower().strip()) for t in texts])


# ---- synthetic buildings --------------------------------------------------------------------------------------


@dataclass
class Sweep:
    x: float
    y: float
    views: list[dict[str, float]]     # one concept dict per horizontal skybox face (usually 4)


@dataclass
class Room:
    name: str
    types: list[str]
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in meters
    sweeps: list[Sweep] = field(default_factory=list)   # empty: no photos (one image-less candidate at the centroid)
    floor: int = 0


@dataclass
class Spec:
    model_id: str
    address: str
    rooms: list[Room]
    floors: int = 1


def write_building(spec: Spec, base: Path) -> Path:
    """building.json + embeddings.npz in the shape ingest.py writes them."""
    out = base / spec.model_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "views").mkdir(exist_ok=True)
    rooms, candidates, vectors, owners = [], [], [], []
    for r_index, room in enumerate(spec.rooms):
        x0, y0, x1, y1 = room.box
        corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        walls = [[*a, *b] for a, b in zip(corners, corners[1:] + corners[:1])]
        rid = f"r{r_index}"
        rooms.append({"id": rid, "label": room.name, "types": room.types, "floor": room.floor, "walls": walls,
                      "polygon": corners, "centroid": [(x0 + x1) / 2, (y0 + y1) / 2], "name": room.name})
        for s_index, sweep in enumerate(room.sweeps):
            pid = f"{rid}s{s_index}"
            views = [f"views/{pid}_{face}.jpg" for face in range(1, len(sweep.views) + 1)]
            for view, concepts in zip(views, sweep.views):
                (out / view).touch()
                vectors.append(vector(concepts, f"{spec.model_id}/{view}"))
                owners.append(len(candidates))
            candidates.append({"id": pid, "room_id": rid, "x": sweep.x, "y": sweep.y, "views": views})
        if not room.sweeps:
            candidates.append({"id": f"room-{rid}", "room_id": rid, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                               "views": []})
    building = {
        "schema": ingest.SCHEMA_VERSION, "id": spec.model_id, "name": spec.address, "address": spec.address,
        "floors": [{"index": i, "id": f"f{i}", "label": f"Floor {i + 1}"} for i in range(spec.floors)],
        "floorplans": {}, "rooms": rooms, "candidates": candidates,
    }
    (out / "building.json").write_text(json.dumps(building, indent=1), encoding="utf-8")
    np.savez(out / "embeddings.npz", view_embeddings=np.stack(vectors),
             view_candidate=np.array(owners, dtype=np.int32))
    return out


def install(monkeypatch, base: Path, specs: list[Spec]) -> None:
    """Point the app at the synthetic buildings and cut every external dependency (CLIP, MiniLM, Claude, Matterport,
    Nominatim, Photon)."""
    for spec in specs:
        write_building(spec, base)
    monkeypatch.setattr(config, "BUILDINGS_DIR", base)
    original_load = Building.load.__func__   # its `base` default was bound to the real data dir at import time
    monkeypatch.setattr(Building, "load", classmethod(lambda cls, model_id, base=base: original_load(cls, model_id,
                                                                                                      base)))
    monkeypatch.setattr(embeddings, "embed_texts", embed_texts)
    monkeypatch.setattr(semantics, "scores", lambda text: {})
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(llm, "enabled", lambda: False)
    monkeypatch.setattr(config, "MESSAGING_MODE", "simulated")
    monkeypatch.setattr(orientation, "ensure", lambda model_id: None)
    monkeypatch.setattr(matterport, "search_models",
                        lambda address: [{"id": s.model_id, "name": s.address, "address": s.address} for s in specs])

    def no_ingest(model_id, embed=True):
        raise AssertionError(f"tests must not download {model_id}: synthetic building cache is stale")
    monkeypatch.setattr(ingest, "ingest", no_ingest)


# ---- running a call -------------------------------------------------------------------------------------------


@dataclass
class Call:
    manager: SessionManager
    session: object
    snapshots: list[dict]     # every snapshot the dashboard would have received, in order

    @property
    def last(self) -> dict:
        return self.manager.snapshot(self.session)

    def rooms(self) -> dict[str, float]:
        """Room name -> probability, from the current snapshot."""
        return {r["name"]: r["probability"] for r in self.last["rooms"]}

    async def say(self, text: str) -> dict:
        await self.manager.handle_reply(self.session, text)
        return self.last


async def start(address: str, timeout: float = 5.0) -> Call:
    snapshots: list[dict] = []
    manager: SessionManager | None = None

    async def notify(session):
        snapshots.append(manager.snapshot(session))

    manager = SessionManager(notify)
    session = await manager.start_call("+15555550100", address)
    waited = 0.0
    while session.status == "loading":
        if waited > timeout:
            raise TimeoutError(f"building never loaded: {session.status_detail}")
        await asyncio.sleep(0.01)
        waited += 0.01
    assert session.status == "listening", session.status_detail
    return Call(manager, session, snapshots)


def answer(plan, facts: dict) -> str:
    """A truthful caller's reply to a planned follow-up (same rules as scripts/eval_scenarios.py)."""
    from app import followups

    if plan.kind == "yesno":
        return facts.get(plan.key, "no")
    if plan.kind == "choice":
        return facts.get(plan.key, "I don't know")
    if plan.kind != "list":   # a question kind added after this harness (e.g. within-room): an honest "not sure"
        return "I don't know"
    seen = [followups.FEATURES_BY_KEY[k].noun.split(" or ")[0] for k in plan.items if facts.get(k) == "yes"]
    return f"I see {' and '.join(seen)}" if seen else "none of those"
