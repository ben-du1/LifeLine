"""Call sessions: building lookup/loading, the question loop, and snapshots for the dashboard.

Status flow:  loading -> listening -> located | best_guess
              loading -> no_model | error
Replies that arrive while the building is still loading are queued and processed once it is ready.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx
import numpy as np

from . import config, embeddings, followups, ingest, llm, locator, matterport, messaging, orientation, questions, semantics
from .building import Building

log = logging.getLogger(__name__)


@dataclass
class LoadedBuilding:
    building: Building
    features: followups.FeatureModel   # P(answer | candidate) for every follow-up question


@dataclass
class Session:
    id: str
    phone: str
    address: str
    created: float
    status: str = "loading"
    status_detail: str = "Looking up the address..."
    model_id: str | None = None
    match: dict | None = None     # the Matterport property used: {id, name, address, score, exact}
    messages: list[dict] = field(default_factory=list)
    posterior: np.ndarray | None = None
    asked: list[followups.Plan] = field(default_factory=list)   # follow-ups, in the order asked
    pending: followups.Plan | None = None                        # the follow-up awaiting an answer
    covered: set[str] = field(default_factory=set)               # topics settled; never asked about again
    refined: list[followups.Plan] = field(default_factory=list)  # "what's closest to you?" questions asked
    queued: list[str] = field(default_factory=list)
    estimate: locator.Estimate | None = None


def normalize_phone(phone: str) -> str:
    """Compare phones by their last 10 digits so '+1 (609) 555-1234' matches '6095551234'."""
    digits = "".join(ch for ch in phone if ch.isdigit())
    return digits[-10:]


def cached_models() -> list[dict]:
    """Already-ingested buildings as [{id, name, address}], so they can match without (or before) a search."""
    if not config.BUILDINGS_DIR.exists():
        return []
    models = []
    for path in config.BUILDINGS_DIR.glob("*/building.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        models.append({"id": data["id"], "name": data["name"], "address": data["address"]})
    return models


def find_matches(address: str) -> list[dict]:
    """Matterport models whose address is sufficiently similar to `address`, best first (see addresses.py).
    Cached buildings come first on ties since they load instantly; if the search API is unreachable, the cache alone
    is used so the demo still works offline."""
    models = cached_models()
    try:
        models += matterport.search_models(address)
    except (matterport.MatterportError, httpx.HTTPError) as exc:
        log.warning("Matterport search failed, matching against cached buildings only: %s", exc)
    unique, seen = [], set()
    for model in models:
        if model["id"] not in seen:
            seen.add(model["id"])
            unique.append(model)
    return matterport.rank_matches(address, unique)


def describe_match(match: dict) -> str:
    if match["exact"]:
        return f"Exact address match: {match['address']}."
    return (f"No exact match for this address. Using the closest Matterport property: {match['address']} "
            f"({round(match['score'] * 100)}% similar). Confirm with the caller.")


class SessionManager:
    def __init__(self, notify: Callable[[Session], Awaitable[None]]):
        self.notify = notify
        self.sessions: dict[str, Session] = {}
        self.buildings: dict[str, LoadedBuilding] = {}
        self._building_locks: dict[str, asyncio.Lock] = {}

    # ---- call lifecycle -------------------------------------------------------------------------------------

    async def start_call(self, phone: str, address: str) -> Session:
        session = Session(id=uuid.uuid4().hex[:8], phone=phone.strip(), address=address.strip(), created=time.time())
        self.sessions[session.id] = session
        # Texting the caller and loading the building happen in parallel (README step 2).
        await self._say(session, questions.OPENING_MESSAGE)
        asyncio.create_task(self._load(session))
        return session

    async def _load(self, session: Session) -> None:
        try:
            matches = await asyncio.to_thread(find_matches, session.address)
            loaded, tried = None, []
            for match in matches[:config.ADDRESS_MATCH_ATTEMPTS]:
                session.model_id, session.match = match["id"], match
                session.status_detail = f"Loading interior model: {match['address']}..."
                await self.notify(session)
                try:
                    loaded = await self._get_building(match["id"])
                    break
                except ingest.UnusableModel as exc:  # e.g. a model with no interior photos: try the next match
                    log.info("Skipping model %s: %s", match["id"], exc)
                    tried.append(match["address"])
            if loaded is None:
                session.model_id = session.match = None
                detail = "No Matterport property with a sufficiently similar address."
                if tried:
                    detail = f"Similar Matterport properties had no usable interior: {'; '.join(tried)}."
                session.status, session.status_detail = "no_model", detail
                await self._say(session, questions.NO_MODEL_MESSAGE)
                return

            session.messages.append({"role": "system", "text": describe_match(session.match), "ts": time.time()})
            session.posterior = locator.uniform_room_prior(loaded.building.candidate_room, len(loaded.building.rooms))
            session.estimate = self._estimate(session, loaded)
            session.status, session.status_detail = "listening", "Waiting for the caller's description."
            await self.notify(session)
            asyncio.create_task(self._load_orientation(session))  # extra context; never delays the questions
            while session.queued and session.status == "listening":
                await self._process(session, session.queued.pop(0))
        except Exception as exc:  # surface any failure on the dashboard instead of leaving the call stuck
            session.status, session.status_detail = "error", f"{type(exc).__name__}: {exc}"
            await self.notify(session)

    async def _load_orientation(self, session: Session) -> None:
        """Fetch north and the street out front once per building (cached on disk), then tell the dashboard."""
        model_id = session.model_id
        lock = self._building_locks.setdefault(f"orientation:{model_id}", asyncio.Lock())
        async with lock:
            found = await asyncio.to_thread(orientation.ensure, model_id)
        if found is not None and session.model_id == model_id:
            await self.notify(session)

    async def _get_building(self, model_id: str) -> LoadedBuilding:
        lock = self._building_locks.setdefault(model_id, asyncio.Lock())
        async with lock:  # two calls for the same address must not ingest it twice
            if model_id not in self.buildings:
                if not ingest.is_ingested(model_id):
                    await asyncio.to_thread(ingest.ingest, model_id)
                self.buildings[model_id] = await asyncio.to_thread(self._load_building, model_id)
            return self.buildings[model_id]

    @staticmethod
    def _load_building(model_id: str) -> LoadedBuilding:
        building = Building.load(model_id)
        return LoadedBuilding(building, followups.model_from_building(building, embeddings.embed_texts))

    # ---- conversation -----------------------------------------------------------------------------------------

    async def handle_reply(self, session: Session, text: str) -> None:
        text = text.strip()
        if not text:
            return
        session.messages.append({"role": "caller", "text": text, "ts": time.time()})
        if session.status == "loading":
            session.queued.append(text)
            await self.notify(session)
            return
        if session.status not in ("listening", "located", "best_guess"):
            await self.notify(session)  # no_model / error: recorded only
            return
        # A located call keeps listening: "actually I'm upstairs" must still move the estimate.
        await self._process(session, text)

    async def _process(self, session: Session, text: str) -> None:
        loaded = self.buildings[session.model_id]
        building = loaded.building
        previous = (session.status, session.estimate.room if session.estimate else None)
        reply = questions.parse_reply(text)
        if questions.is_correction(text):
            # "Sorry, I'm actually upstairs": start over from the prior instead of multiplying the correction into
            # the earlier description. One correction after a confident wrong room (98%) otherwise only got it
            # down to ~50%. Questions already answered stay counted toward MAX_QUESTIONS.
            session.posterior = locator.uniform_room_prior(building.candidate_room, len(building.rooms))
            session.covered.clear()
        cues, semantic = None, None
        if llm.enabled():  # Claude reads paraphrase; None (failure/timeout) falls back to the local reading
            cues = await asyncio.to_thread(llm.read_message, text, session.pending, session.messages,
                                           len(building.floors))
        if cues is None:
            semantic = await asyncio.to_thread(semantics.scores, text)
        reading = followups.read_message(text, loaded.features, session.pending, reply.kind, cues, semantic)
        room_types = questions.room_types_mentioned(text) | set(cues.room_types if cues else ())
        scores, boost = await asyncio.to_thread(self._evidence, loaded, text, reply, reading.explained, room_types)
        if scores is not None:
            session.posterior = locator.update(session.posterior, scores, boost,
                                               candidate_room=building.candidate_room)
        if reading.likelihood is not None:
            session.posterior = locator.apply_likelihood(session.posterior, reading.likelihood)
        if reading.position is not None:
            session.posterior = locator.apply_within_rooms(session.posterior, reading.position, building.candidate_room)
        session.covered |= reading.covered
        last_kind = session.pending.kind if session.pending else None
        session.pending = None
        session.estimate = self._estimate(session, loaded)

        if session.estimate.confident and self._floor_settled(session, loaded):
            await self._located(session, loaded, previous, last_kind)
            return
        plan = None
        if len(session.asked) < config.MAX_QUESTIONS:
            plan = followups.choose(loaded.features, session.posterior, building.candidate_room, len(building.rooms),
                                    session.covered, last_kind)
        if plan is None:
            reason = ("Question limit reached" if len(session.asked) >= config.MAX_QUESTIONS
                      else "No follow-up would tell the remaining rooms apart")
            session.status_detail = f"{reason}. Most likely: {self._leading_rooms(session, loaded)}."
            if previous[0] == "best_guess":
                await self.notify(session)  # estimate updated; the caller already got the closing text
                return
            session.status = "best_guess"
            await self._say(session, questions.BEST_GUESS_MESSAGE)
            return
        session.status = "listening"  # also after a correction reopened a located call
        session.asked.append(plan)
        session.pending = plan
        session.status_detail = f"Asked follow-up {len(session.asked)} of {config.MAX_QUESTIONS}."
        question = None
        if llm.enabled():
            question = await asyncio.to_thread(llm.phrase_question, plan, session.messages, reading.acknowledgement)
        await self._say(session, question or followups.phrase(plan, reading.acknowledgement))

    @staticmethod
    def _evidence(loaded: LoadedBuilding, text: str, reply: questions.Reply, explained: bool,
                  mentioned: set[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Photo evidence for one reply: per-candidate CLIP scores of the caller's own words (None when the reply was
        just an answer to our question, e.g. "yes" or "tile"), plus an optional room-type boost."""
        building = loaded.building
        parts = []
        if reply.kind in ("yes", "no"):
            if questions.has_detail(reply.extra):  # "yes, next to the big bed"
                parts.append(building.candidate_scores(embeddings.embed_texts([reply.extra])[0]))
        elif reply.kind == "free" and not explained:
            parts.append(building.candidate_scores(embeddings.embed_texts([text])[0]))

        boost = None
        if mentioned:
            room_match = np.array([bool(mentioned & {t.lower() for t in r["types"]}) for r in building.rooms])
            if room_match.any():
                boost = np.where(room_match[building.candidate_room], config.ROOM_KEYWORD_BOOST, 1.0)

        if parts:
            return np.mean(parts, axis=0), boost
        if boost is not None:
            return np.zeros(len(building.candidates)), boost
        return None, None

    async def _located(self, session: Session, loaded: LoadedBuilding, previous: tuple, last_kind: str | None) -> None:
        """The room is known. Tell the caller (once), then narrow down where in the room with up to MAX_REFINEMENTS
        "what's closest to you?" questions while the room has several scan points and the position is spread out."""
        building = loaded.building
        room = session.estimate.room
        refine = None
        if len(session.refined) < config.MAX_REFINEMENTS:
            asked = {k for plan in session.refined for k in plan.items}
            refine = followups.refinement(loaded.features, session.posterior, room, self._xy(building), asked)
        newly_located = previous != ("located", room)
        session.status = "located"
        session.status_detail = "Caller located." if refine is None else "Caller located; narrowing down where in the room."
        if refine is not None:
            session.refined.append(refine)
            session.pending = refine
        if newly_located:
            name = building.rooms[room]["name"]
            if len(building.floors) > 1:  # "Kitchen, Floor 2": the floor matters as much as the room upstairs
                name = f"{name}, {building.floors[int(building.room_floor[room])]['label']}"
            text = questions.LOCATED_MESSAGE.format(room=name)
            await self._say(session, f"{text} {refine.text}" if refine else text)
        elif refine is not None:
            await self._say(session, refine.text)
        elif last_kind == "nearest":
            await self._say(session, questions.REFINED_MESSAGE)
        else:
            await self.notify(session)  # same room as before: don't repeat the "located" text

    @staticmethod
    def _xy(building: Building) -> np.ndarray:
        return np.array([[c["x"], c["y"]] for c in building.candidates], dtype=float)

    @staticmethod
    def _floor_settled(session: Session, loaded: LoadedBuilding) -> bool:
        """In a multi-storey building a room is only "located" once its floor is likely too: "I'm in the kitchen"
        with a kitchen on two floors must not be reported as found before the floor is asked."""
        building = loaded.building
        if len(building.floors) < 2:
            return True
        floors = np.bincount(building.candidate_floor, weights=session.posterior, minlength=len(building.floors))
        return floors[int(building.room_floor[session.estimate.room])] >= config.FLOOR_FIRST_UNTIL

    @staticmethod
    def _leading_rooms(session: Session, loaded: LoadedBuilding, count: int = 3) -> str:
        """'Bathroom 2 (60%), Bathroom 1 (40%)': the rooms responders should check, in order."""
        building = loaded.building
        probs = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        order = [i for i in np.argsort(probs)[::-1][:count] if probs[i] >= 0.05]
        return ", ".join(f"{building.rooms[i]['name']} ({round(probs[i] * 100)}%)" for i in order)

    @staticmethod
    def _estimate(session: Session, loaded: LoadedBuilding) -> locator.Estimate:
        return locator.estimate(session.posterior, loaded.building.candidate_room, len(loaded.building.rooms))

    async def _say(self, session: Session, text: str) -> None:
        session.messages.append({"role": "dispatch", "text": text, "ts": time.time()})
        try:
            await messaging.send(session.phone, text)
        except messaging.MessagingError as exc:
            session.messages.append({"role": "system", "text": str(exc), "ts": time.time()})
        await self.notify(session)

    def find_by_phone(self, phone: str) -> Session | None:
        key = normalize_phone(phone)
        matches = [s for s in self.sessions.values() if normalize_phone(s.phone) == key]
        return max(matches, key=lambda s: s.created) if matches else None

    def room_shots(self, session: Session) -> list[dict]:
        """Every room of the call's building, most likely first, with all of its photos grouped by scan point.
        Kept out of the snapshot because the photo lists are large and the snapshot is broadcast on every change."""
        loaded = self.buildings.get(session.model_id) if session.model_id else None
        if loaded is None or session.posterior is None:
            return []
        building = loaded.building
        base = f"/buildings/{building.id}/"
        room_probs = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        rooms = []
        for index, room in enumerate(building.rooms):
            sweeps = [{"candidate": c, "x": cand["x"], "y": cand["y"], "probability": float(session.posterior[c]),
                       "views": [base + v for v in cand["views"]]}
                      for c, cand in enumerate(building.candidates)
                      if building.candidate_room[c] == index and cand["views"]]
            area = loaded.features.room_area[index]
            rooms.append({
                "index": index, "name": room["name"], "types": room["types"], "floor": room.get("floor", 0),
                "area_m2": None if np.isnan(area) else round(float(area), 1),
                "probability": float(room_probs[index]),
                "sweeps": sorted(sweeps, key=lambda s: -s["probability"]),
            })
        return sorted(rooms, key=lambda r: -r["probability"])

    # ---- snapshots ------------------------------------------------------------------------------------------

    def snapshot(self, session: Session) -> dict:
        data = {
            "id": session.id, "phone": session.phone, "address": session.address, "created": session.created,
            "status": session.status, "status_detail": session.status_detail, "model_id": session.model_id,
            "match": session.match,
            "orientation": bool(session.model_id) and orientation.path(session.model_id).exists(),
            "messages": session.messages, "questions_asked": len(session.asked),
            "max_questions": config.MAX_QUESTIONS, "rooms": [], "floors": [], "candidate_probabilities": [],
            "estimate": None,
        }
        loaded = self.buildings.get(session.model_id) if session.model_id else None
        if loaded is None or session.posterior is None:
            return data
        building = loaded.building
        room_probs = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        floor_labels = [f["label"] for f in building.floors]
        data["rooms"] = sorted(({"index": i, "name": r["name"], "floor": r.get("floor", 0),
                                 "floor_label": floor_labels[r.get("floor", 0)], "probability": float(room_probs[i])}
                                for i, r in enumerate(building.rooms)), key=lambda r: -r["probability"])
        floor_probs = np.bincount(building.room_floor, weights=room_probs, minlength=len(building.floors))
        data["floors"] = [{"index": i, "label": label, "probability": float(floor_probs[i])}
                          for i, label in enumerate(floor_labels)]
        data["candidate_probabilities"] = [float(p) for p in session.posterior]
        est = session.estimate
        if est is not None:
            # Where in the room: probability-weighted position of the room's candidates, with its RMS spread as an
            # uncertainty radius (at least half a sweep spacing, since positions are only known at scan points).
            members = np.flatnonzero(building.candidate_room == est.room)
            (x, y), spread = followups.position_spread(session.posterior, self._xy(building), members)
            data["estimate"] = {
                "room": est.room, "room_name": building.rooms[est.room]["name"], "candidate": est.candidate,
                "floor": int(building.room_floor[est.room]),
                "floor_label": floor_labels[int(building.room_floor[est.room])],
                "x": float(x), "y": float(y), "radius_m": round(max(spread, config.REFINE_MIN_SPREAD_M / 2), 2),
                "room_probability": est.room_probability,
                "runner_up_probability": est.runner_up_probability, "confident": est.confident,
            }
        return data
