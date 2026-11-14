"""A cached building (output of ingest.py) loaded into arrays the locator can use."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


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

    def candidate_scores(self, text_embedding: np.ndarray) -> np.ndarray:
        """Per-candidate match score for one text: the best cosine similarity over that candidate's views (the caller
        may be facing any direction). Image-less candidates borrow the score of photographed rooms of the same type (see
        imageless_peers), or else a fixed quantile of the imaged scores (config.IMAGELESS_SCORE_QUANTILE), roughly
        neutral."""
        sims = self.view_embeddings @ text_embedding
        scores = np.full(len(self.candidates), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.view_candidate, sims)
        if self.has_views.any():
            fallback = np.quantile(scores[self.has_views], config.IMAGELESS_SCORE_QUANTILE)
            for c, groups in self.imageless_peers().items():
                scores[c] = np.mean([_room_score(scores[g]) for g in groups]) if groups else fallback
        else:
            scores[:] = 0.0
        return scores

    def imageless_peers(self) -> dict[int, list[np.ndarray]]:
        """For each image-less candidate: the imaged candidates of every *other* room that shares a room type with its
        room, one index array per room. A caller who says "toilet" is as likely in a photo-less bathroom as in the
        photographed one, so the photo-less one borrows that room's evidence instead of the house-wide median (which
        let the photographed bathroom take ~100%). Empty when no room of the same type has photos (or the room has
        no types), in which case the median is used as before."""
        room_types = [{t.lower() for t in r.get("types") or []} for r in self.rooms]
        imaged_rooms = {int(r): np.flatnonzero(self.has_views & (self.candidate_room == r))
                        for r in np.unique(self.candidate_room[self.has_views])}
        peers = {}
        for c in np.flatnonzero(~self.has_views):
            room = int(self.candidate_room[c])
            peers[int(c)] = [g for r, g in imaged_rooms.items() if r != room and room_types[room] & room_types[r]]
        return peers


def _room_score(scores: np.ndarray, scale: float = config.LOGIT_SCALE) -> float:
    """One score standing for a whole room's candidates: the log-mean-exp at the update's logit scale, so that
    exp(scale × this) equals the room's average likelihood under the uniform-per-room prior (locator.update). A plain
    mean would understate a room with many sweeps (one of them usually matches well); the max would overstate it."""
    top = float(scores.max())
    return top + float(np.log(np.mean(np.exp(scale * (scores - top))))) / scale
