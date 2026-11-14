"""A cached building (output of ingest.py) loaded into arrays the locator can use."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from . import config

# Room types too vague to say two rooms look alike.
GENERIC_ROOM_TYPES = {"other", "room", "unknown"}


@dataclass
class Building:
    id: str
    name: str
    address: str
    floors: list[dict]            # lowest first: {index, id, label}
    floorplans: dict              # floor index (string) -> aerial raster metadata, for floors that have one
    rooms: list[dict]             # each has "floor" (index into floors)
    candidates: list[dict]
    candidate_room: np.ndarray    # (n_candidates,) index into rooms
    has_views: np.ndarray         # (n_candidates,) bool
    view_embeddings: np.ndarray   # (n_views, d)
    view_candidate: np.ndarray    # (n_views,) index into candidates

    @property
    def room_floor(self) -> np.ndarray:
        """(n_rooms,) floor index of each room."""
        return np.array([r.get("floor", 0) for r in self.rooms], dtype=np.int32)

    @property
    def candidate_floor(self) -> np.ndarray:
        """(n_candidates,) floor index of each candidate."""
        return self.room_floor[self.candidate_room]

    @classmethod
    def load(cls, model_id: str, base: Path = config.BUILDINGS_DIR) -> "Building":
        d = base / model_id
        data = json.loads((d / "building.json").read_text(encoding="utf-8"))
        emb = np.load(d / "embeddings.npz")
        room_index = {r["id"]: i for i, r in enumerate(data["rooms"])}
        return cls(
            id=data["id"], name=data["name"], address=data["address"], floors=data["floors"],
            floorplans=data["floorplans"],
            rooms=data["rooms"], candidates=data["candidates"],
            candidate_room=np.array([room_index[c["room_id"]] for c in data["candidates"]], dtype=np.int32),
            has_views=np.array([bool(c["views"]) for c in data["candidates"]]),
            view_embeddings=emb["view_embeddings"], view_candidate=emb["view_candidate"],
        )

    def to_pixels(self, x: float, y: float, floor: int = 0) -> tuple[float, float]:
        """Model coordinates (meters) to that floor's floorplan image pixels. `resolution` is px per meter and
        `origin` is the model coordinate at the image's bottom-left; image rows grow downward, model y upward."""
        fp = self.floorplans[str(floor)]
        px = (x - fp["origin"]["x"]) * fp["resolution"]
        py = fp["height"] - (y - fp["origin"]["y"]) * fp["resolution"]
        return px, py

    @cached_property
    def same_type_peers(self) -> list[np.ndarray]:
        """For each image-less candidate, the photographed candidates in rooms of the same type (e.g. the one
        photographed bathroom for an unphotographed upstairs bathroom); empty arrays for imaged candidates."""
        types = [{t.lower() for t in r.get("types", [])} - GENERIC_ROOM_TYPES for r in self.rooms]
        imaged = np.flatnonzero(self.has_views)
        peers = []
        for c in range(len(self.candidates)):
            mine = types[self.candidate_room[c]]
            if self.has_views[c] or not mine:
                peers.append(np.empty(0, dtype=np.int64))
            else:
                peers.append(imaged[[bool(mine & types[self.candidate_room[i]]) for i in imaged]])
        return peers

    def candidate_scores(self, text_embedding: np.ndarray) -> np.ndarray:
        """Per-candidate match score for one text: the best cosine similarity over that candidate's views (the caller
        may be facing any direction).

        Image-less candidates (rooms that were never photographed) start from a neutral score, a fixed quantile of
        the imaged scores (config.IMAGELESS_SCORE_QUANTILE), plus how far photographed rooms of the same type stand
        out for this text (mean peer score minus mean imaged score): an unphotographed bathroom is assumed to look
        like the photographed bathrooms.
        - Without the lift, the single photographed bathroom won every bathroom description by ~20x (LOGIT_SCALE),
          so at 12331 Basilwood Dr, whose whole upper floor has no photos, "I'm in the bathroom" was "located" in
          the downstairs bathroom.
        - The peer *mean* is used, not the max: the best of several peers scores high for almost any text, and in
          the scripted eval that let unphotographed closets and a 2 m² "bedroom" beat the photographed room the
          caller was in.
        Rooms with no photographed peer (or no label) keep the neutral score."""
        sims = self.view_embeddings @ text_embedding
        scores = np.full(len(self.candidates), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.view_candidate, sims)
        if not self.has_views.any():
            return np.zeros(len(self.candidates), dtype=np.float32)
        imaged = scores[self.has_views]
        neutral = np.quantile(imaged, config.IMAGELESS_SCORE_QUANTILE)
        for c in np.flatnonzero(~self.has_views):
            peers = self.same_type_peers[c]
            scores[c] = neutral + (scores[peers].mean() - imaged.mean()) if len(peers) else neutral
        return scores
