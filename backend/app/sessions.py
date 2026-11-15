"""Call sessions: building lookup/loading, the question loop, and snapshots for the dashboard.

Status flow:  loading -> listening -> located | best_guess
              loading -> no_model | error
Replies that arrive while the building is still loading are queued and processed once it is ready.
"""

import asyncio
import re
import dataclasses
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx
import numpy as np

from . import approach, config, embeddings, followups, geo, ingest, llm, locator, matterport, messaging, orientation, questions, semantics
from . import adjacency, subroom, textmatch, ties
from .building import Building
from .addresses import is_exact, similarity
from .voice import spoken_message

log = logging.getLogger(__name__)

EVIDENCE_LOG_LENGTH = 20   # caller messages kept in snapshot["evidence"]; a call rarely has more than 8


@dataclass
class LoadedBuilding:
    building: Building
    features: followups.FeatureModel   # P(answer | candidate) for every follow-up question


@dataclass
class Evidence:
    topics: frozenset[str]
    key: str
    scores: np.ndarray | None
    boost: np.ndarray | None
    room: np.ndarray | None
    position: np.ndarray | None


@dataclass
class Session:
    id: str
    phone: str
    address: str
    created: float
    channel: str = "messaging"
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    revision: int = 0
    ended: bool = False
    waiting_for_match: bool = False
    ledger: list[Evidence] = field(default_factory=list)
    floor_unresolved: bool = False
    floor_retry_used: bool = False
    confirmed_floor: int | None = None
    evidence_texts: set[str] = field(default_factory=set)
    provider_events: set[str] = field(default_factory=set)
    status: str = "loading"
    status_detail: str = "Looking up the address..."
    model_id: str | None = None
    # A model the dispatcher picked from the address suggestions: used as-is, since one address can have several
    # models (apartment units), and matching by address alone would pick any of them.
    requested_model: str | None = None
    match: dict | None = None     # the Matterport property used: {id, name, address, score, exact}
    messages: list[dict] = field(default_factory=list)
    posterior: np.ndarray | None = None
    asked: list[followups.Plan] = field(default_factory=list)   # follow-ups, in the order asked
    pending: followups.Plan | None = None                        # the follow-up awaiting an answer
    covered: set[str] = field(default_factory=set)               # topics settled; never asked about again
    queued: list[str] = field(default_factory=list)
    # One caller message at a time: _process awaits (CLIP, Claude), and two replies processed at once both read the
    # same pending question and both can send "located" (audit #4).
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    estimate: locator.Estimate | None = None
    evidence_messages: int = 0    # caller messages that moved the posterior; "located" needs enough (is_located)
    within_room_asked: int = 0   # within-room follow-ups asked after the room was found (subroom.py)
    evidence: list[dict] = field(default_factory=list)   # how each caller message moved the room probabilities


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
    cached = matterport.rank_matches(address, models)
    if cached and cached[0]["exact"]:
        # A downloaded building with this exact address: no need to search. The search is 3 GraphQL calls with long
        # timeouts, and on venue Wi-Fi it kept a demo call on "Looking up the address..." for minutes (audit #5).
        return cached
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


def room_denial(building: Building, denied: set[str]) -> np.ndarray | None:
    """Per-candidate factor for room types the caller said they are not in (questions.room_types_denied), or None
    when no room has a denied type."""
    if not denied:
        return None
    room_match = np.array([bool(denied & {t.lower() for t in r["types"]}) for r in building.rooms])
    if not room_match.any():
        return None
    return np.where(room_match[building.candidate_room], config.ROOM_DENIAL_FACTOR, 1.0)


class SessionManager:
    def __init__(self, notify: Callable[[Session], Awaitable[None]]):
        self._notify = notify
        self.sessions: dict[str, Session] = {}
        self.buildings: dict[str, LoadedBuilding] = {}
        self._building_locks: dict[str, asyncio.Lock] = {}

    async def notify(self, session: Session) -> None:
        session.revision += 1
        await self._notify(session)

    # ---- call lifecycle -------------------------------------------------------------------------------------

    async def start_call(self, phone: str, address: str, model_id: str | None = None, *, channel: str = "messaging") -> Session:
        session = Session(id=uuid.uuid4().hex[:8], phone=phone.strip(), address=address.strip(), created=time.time())
        session.channel = channel
        session.requested_model = model_id
        self.sessions[session.id] = session
        # Texting the caller and loading the building happen in parallel (README step 2).
        await self._say(session, questions.OPENING_MESSAGE)
        asyncio.create_task(self._load(session))
        return session

    async def _load(self, session: Session) -> None:
        try:
            if session.requested_model:
                matches = [{"id": session.requested_model, "name": session.address, "address": session.address,
                            "score": 1.0, "exact": True}]
            else:
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

            async with session.lock:
                if session.ended:
                    return
                if session.requested_model:
                    actual = loaded.building.address
                    session.match = {
                        "id": loaded.building.id, "name": loaded.building.name, "address": actual,
                        "score": similarity(session.address, actual),
                        "exact": is_exact(session.address, actual),
                    }
                session.messages.append({"role": "system", "text": describe_match(session.match), "ts": time.time()})
                session.posterior = locator.uniform_room_prior(loaded.building.candidate_room, len(loaded.building.rooms))
                session.estimate = self._estimate(session, loaded)
                session.status, session.status_detail = "listening", "Waiting for the caller's description."
                await self.notify(session)
                asyncio.create_task(self._load_orientation(session))
                if session.channel == "voice" and not session.match["exact"]:
                    session.waiting_for_match = True
                    await self._say(session, f"The available floor plan is for {session.match['address']}. Is that the correct building?")
                elif session.queued:
                    text, session.queued = " ".join(session.queued), []
                    await self._process(session, text)
        except Exception as exc:  # surface any failure on the dashboard instead of leaving the call stuck
            session.status, session.status_detail = "error", f"{type(exc).__name__}: {exc}"
            await self.notify(session)

        finally:
            session.ready.set()

    async def _load_orientation(self, session: Session) -> None:
        """Fetch north and the street out front once per building (cached on disk), then tell the dashboard."""
        model_id = session.model_id
        lock = self._building_locks.setdefault(f"orientation:{model_id}", asyncio.Lock())
        async with lock:
            found = await asyncio.to_thread(orientation.ensure, model_id)
            if found is not None:  # map position for the approach card's Street View link (cached, never raises)
                await asyncio.to_thread(approach.ensure_geo, model_id)
        if found is not None and model_id in self.buildings:
            # The street is known now: add the "front of the house" question (adjacency.py).
            loaded = self.buildings[model_id]
            adjacency.register(loaded.features, loaded.building, found)
        if found is not None and session.model_id == model_id:
            await self.notify(session)
        if config.VEXCEL_API_KEY:
            # Warm the aerial photo cache (one Vexcel call per building) after orientation has cached the
            # geocoordinates it needs. Background only: geo.aerial never raises and nothing waits on it.
            asyncio.create_task(asyncio.to_thread(geo.aerial, model_id))

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
        features = followups.model_from_building(building, embeddings.embed_texts)
        # Position questions from the floor plan; the street side only if orientation is already cached.
        adjacency.register(features, building, orientation.load(model_id))
        return LoadedBuilding(building, features)

    # ---- conversation -----------------------------------------------------------------------------------------

    async def handle_reply(self, session: Session, text: str, *, event_id: str | None = None) -> None:
        text = text.strip()
        if not text:
            return
        async with session.lock:
            if session.ended:
                return
            if event_id is not None:
                if event_id in session.provider_events:
                    return
                session.provider_events.add(event_id)
            message = {"id": uuid.uuid4().hex, "role": "caller", "text": text, "ts": time.time()}
            session.messages.append(message)
            if session.waiting_for_match:
                reply = questions.parse_reply(text)
                if reply.kind == "yes":
                    session.waiting_for_match = False
                    if session.queued:
                        description, session.queued = " ".join(session.queued), []
                        await self._process(session, description)
                    else:
                        await self._say(session, "Describe the room you are in.")
                elif reply.kind == "no":
                    session.waiting_for_match = False
                    session.queued.clear()
                    session.status = "no_model"
                    session.model_id = None
                    session.status_detail = "Caller rejected the suggested building."
                    await self._say(session, "I could not confirm the floor plan. Please start again with the complete address.")
                else:
                    await self._say(session, "Please say yes or no to confirm the building address.")
                return
            if session.status == "loading":
                session.queued.append(text)
            elif session.status in ("listening", "located", "best_guess"):
                try:
                    await self._process(session, text)
                except Exception:
                    log.exception("Could not process caller reply")
                    session.messages.append({"role": "system", "text": "Could not process this reply. Please try again.", "ts": time.time()})
            await self.notify(session)

    async def end_call(self, session: Session, reason: str) -> None:
        async with session.lock:
            if session.ended:
                return
            session.ended = True
            session.ready.set()
            session.queued.clear()
            if session.status in ("loading", "listening"):
                session.status = "best_guess" if session.estimate is not None else "error"
            session.status_detail = reason
            await self.notify(session)

    async def _process(self, session: Session, text: str) -> None:
        loaded = self.buildings[session.model_id]
        building = loaded.building
        normalized = " ".join(text.casefold().split())
        if session.pending is None and normalized in session.evidence_texts and not questions.is_correction(text):
            session.messages.append({"role": "system", "text": "Repeated description. Confidence was not increased.", "ts": time.time()})
            return
        reply = questions.parse_reply(text)
        cues, semantic = None, None
        if llm.enabled():  # Claude reads paraphrase; None (failure/timeout) falls back to the local reading
            cues = await asyncio.to_thread(llm.read_message, text, session.pending, session.messages,
                                           len(building.floors))
        if cues is None:
            semantic = await asyncio.to_thread(semantics.scores, text)
        reading = followups.read_message(text, loaded.features, session.pending, reply.kind, cues, semantic)
        before = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        stated_floors = followups.stated_floors(text, loaded.features.floor_options)
        explicit_numbers = re.findall(r"\b(?:(\d+)(?:st|nd|rd|th)?\s+(?:floor|level)|(?:floor|level)\s+(\d+))\b", text, re.I)
        unsupported = any(not 1 <= int(a or b) <= len(building.floors) for a, b in explicit_numbers)
        if unsupported:
            session.floor_unresolved = True
            session.messages.append({"role": "system", "text": "The caller named a floor absent from the building model. The estimate is tentative.", "ts": time.time()})
        elif len(stated_floors) == 1:
            session.floor_unresolved = False
        floor_changed = len(stated_floors) == 1 and session.confirmed_floor not in (None, stated_floors[0])
        floor_selected = len(stated_floors) == 1 and session.confirmed_floor != stated_floors[0]
        if len(stated_floors) == 1:
            session.confirmed_floor = stated_floors[0]
        room_types = questions.room_types_mentioned(text) | set(cues.room_types if cues else ())
        denied = questions.room_types_denied(text) - room_types  # "I'm not in the garage"
        scores, boost = await asyncio.to_thread(self._evidence, loaded, text, reply, reading.explained, room_types,
                                                reading.covered, denied)
        topics = (reading.covered - {"floor"}) | {f"room:{room}" for room in room_types | denied}
        key = f"{session.pending.key if session.pending else ''}:{normalized}"
        entries = session.ledger
        if floor_changed:
            entries = [entry for entry in entries if entry.topics]
        elif questions.is_correction(text) and topics:
            entries = [entry for entry in entries if not (entry.topics & topics)]
        duplicate = any(entry.key == key for entry in entries)
        if not duplicate and any(value is not None for value in (scores, reading.likelihood, reading.position)):
            entries = [*entries, Evidence(frozenset(topics), key, scores, boost, reading.likelihood, reading.position)]
        if duplicate:
            session.messages.append({"role": "system", "text": "This evidence was already counted.", "ts": time.time()})
        if entries is not session.ledger or floor_selected:
            prior = locator.uniform_room_prior(building.candidate_room, len(building.rooms))
            allowed = (building.candidate_floor == session.confirmed_floor
                       if session.confirmed_floor is not None else np.ones(len(prior), dtype=bool))
            prior = np.where(allowed, prior, 0)
            prior /= prior.sum()
            posterior = prior.copy()
            for entry in entries:
                if entry.scores is not None:
                    posterior = locator.update(posterior, entry.scores, entry.boost, candidate_room=building.candidate_room)
                if entry.room is not None:
                    posterior = locator.apply_likelihood(posterior, entry.room)
                if entry.position is not None:
                    posterior = locator.apply_within_rooms(posterior, entry.position, building.candidate_room)
                if entry.scores is not None or entry.room is not None:
                    posterior = (1 - config.POSTERIOR_FLOOR) * posterior + config.POSTERIOR_FLOOR * prior
            session.posterior = posterior
            session.ledger = entries
            session.evidence_messages = sum(entry.scores is not None or entry.room is not None for entry in entries)
            session.evidence_texts.add(normalized)
        hints = self._room_hints(session, loaded, text, reply, cues, semantic, room_types)
        session.covered |= reading.covered
        last_kind = session.pending.kind if session.pending else None
        session.pending = None
        session.estimate = self._estimate(session, loaded)
        self._log_evidence(session, loaded, text, before)
        located, _ = self._located(session, loaded, hints)

        if located and not session.floor_unresolved and self._floor_settled(session, loaded):
            within = self._within_room_plan(session, loaded)
            if within is not None:  # room found, but it is big: ask which part of it before "located"
                session.within_room_asked += 1
                name = building.rooms[session.estimate.room]["name"]
                await self._ask(session, within, reading, f"Room found; narrowing down within the {name}.",
                                use_llm=False)
                return
            session.status = "located"
            session.status_detail = "Caller located."
            tied = ties.tie_group(session.posterior, building, loaded.features)
            if tied:  # located by the confidence rule, but a look-alike room is still close: say so
                session.status_detail = f"Caller located: {ties.describe(building, session.posterior, tied)}."
            card = self._approach(session, loaded)
            if card and card.get("summary"):  # dispatcher only; the caller's message below is unchanged
                session.status_detail = f"{session.status_detail.rstrip('.')}. Approach: {card['summary']}."
            room = building.rooms[session.estimate.room]["name"]
            zone = self._zone(session, loaded)[1]
            if zone:  # "Living Room, near the fireplace": the part of a big room responders should head for
                room = f"{room}, {zone}"
            if len(building.floors) > 1:  # "Kitchen, Floor 2": the floor matters as much as the room upstairs
                room = f"{room}, {building.floors[int(building.room_floor[session.estimate.room])]['label']}"
            await self._say(session, questions.LOCATED_MESSAGE.format(room=room))
            return
        if session.floor_unresolved and not session.floor_retry_used and loaded.features.floor_options:
            session.floor_retry_used = True
            options = loaded.features.floor_options
            plan = followups.Plan("choice", "floor", "Please confirm your current floor. " + followups.floor_question(options), options=options)
            await self._ask(session, plan, reading, "Confirming the caller’s floor.", use_llm=False)
            return
        plan = None
        if len(session.asked) < config.MAX_QUESTIONS:
            min_gain = config.MIN_INFORMATION_GAIN
            plan = followups.choose(loaded.features, session.posterior, building.candidate_room, len(building.rooms),
                                    session.covered | {p.key for p in session.asked}, last_kind, min_gain)
        if plan is None:
            reason = ("Question limit reached" if len(session.asked) >= config.MAX_QUESTIONS
                      else "No follow-up would tell the remaining rooms apart")
            session.status = "best_guess"
            tied = ties.tie_group(session.posterior, building, loaded.features)
            leading = (ties.describe(building, session.posterior, tied) if tied
                       else self._leading_rooms(session, loaded))
            session.status_detail = f"{reason}. Most likely: {leading}."
            if tied:  # for the dispatcher only; the caller still gets the plain best-guess text
                session.messages.append({"role": "system", "ts": time.time(), "text": (
                    f"Best guess: {ties.describe(building, session.posterior, tied)}. Send responders to check each.")})
            await self._say(session, questions.BEST_GUESS_MESSAGE)
            return
        await self._ask(session, plan, reading, "Waiting for the caller’s answer.")

    async def _ask(self, session: Session, plan: followups.Plan, reading: followups.Reading, detail: str,
                   use_llm: bool = True) -> None:
        session.status = "listening"
        session.asked.append(plan)
        session.pending = plan
        session.status_detail = detail
        question = None
        if use_llm and llm.enabled():
            question = await asyncio.to_thread(llm.phrase_question, plan, session.messages, reading.acknowledgement)
        await self._say(session, question or followups.phrase(plan, reading.acknowledgement))

    @staticmethod
    def _within_room_plan(session: Session, loaded: LoadedBuilding) -> followups.Plan | None:
        """A question about where inside the (confident) room, or None to report "located" now. getattr: the
        within-room planner (followups.choose_within_room) may not exist yet, and then the flow is unchanged.
        Within-room questions are phrased from the template only: llm.phrase_question knows the room-level kinds."""
        choose_within_room = getattr(followups, "choose_within_room", None)
        if (choose_within_room is None or session.within_room_asked >= config.MAX_WITHIN_ROOM_QUESTIONS
                or len(session.asked) >= config.MAX_QUESTIONS):
            return None
        building, room = loaded.building, session.estimate.room
        area = float(loaded.features.room_area[room])
        if not subroom.needs_narrowing(area, subroom.region(session.posterior, building, room)):
            return None
        return choose_within_room(loaded.features, session.posterior, building.candidate_room, room, session.covered)

    @staticmethod
    def _zone(session: Session, loaded: LoadedBuilding) -> tuple[dict, str | None]:
        """The estimate's region in its room, and a zone label for big rooms only: in a bathroom, "northwest
        corner" is noise that a responder would read as precision."""
        building, room = loaded.building, session.estimate.room
        region = subroom.region(session.posterior, building, room)
        area = float(loaded.features.room_area[room])
        area = region["area_m2"] if np.isnan(area) else area
        if area is None or area < config.BIG_ROOM_M2:
            return region, None
        return region, subroom.zone_label(building, room, (region["x"], region["y"]), loaded.features,
                                          session.posterior)

    @staticmethod
    def _evidence(loaded: LoadedBuilding, text: str, reply: questions.Reply, explained: bool,
                  mentioned: set[str], covered: set[str] = frozenset(),
                  denied: set[str] = frozenset()) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Photo evidence for one reply: per-candidate CLIP scores of the visual part of the caller's words (None when
        the reply was just an answer to our question, e.g. "yes" or "tile", or had nothing visual left that the cues
        didn't already read), plus an optional room-type boost (down for room types the caller denied). `covered` is
        Reading.covered."""
        building = loaded.building
        parts = []
        # Only what a photo could show goes to CLIP ("please hurry" and "someone is downstairs" match no view); see
        # textmatch.py. Words the cues already counted ("stove") still go too: in theory that counts them twice, but
        # on the real eval houses dropping them cost 5 of 19 scenarios (15/19 -> 10/19), because the photo match is
        # what separates the kitchen from rooms whose label merely allows a stove. CLIP_MAX_LIKELIHOOD_RATIO bounds
        # the extra weight. Flooring words are the exception: CLIP reads grey concrete as carpet (the garage bug).
        captured = {key for key in covered if key == "flooring"}
        query = None
        if reply.kind in ("yes", "no"):
            if questions.has_detail(reply.extra):  # "yes, next to the blue dresser"
                query = textmatch.visual_query(reply.extra, captured)
        elif reply.kind == "free" and not explained:
            query = textmatch.visual_query(text, captured)
        if query is not None:
            parts.append(building.candidate_scores(embeddings.embed_texts([query])[0]))

        boost = None
        if mentioned:
            room_match = np.array([bool(mentioned & {t.lower() for t in r["types"]}) for r in building.rooms])
            if room_match.any():
                boost = np.where(room_match[building.candidate_room], config.ROOM_KEYWORD_BOOST, 1.0)
        denial = room_denial(building, denied)
        if denial is not None:
            boost = denial if boost is None else boost * denial

        if parts:
            return np.mean(parts, axis=0), boost
        if boost is not None:
            return np.zeros(len(building.candidates)), boost
        return None, None

    @staticmethod
    def _room_hints(session: Session, loaded: LoadedBuilding, text: str, reply: questions.Reply,
                    cues: followups.Cues | None, semantic: dict | None, room_types: set[str]) -> list[tuple[str, ...]]:
        """What this message said about the room's kind, for locator.names_only_room: each room type the caller named,
        plus the room-type hints of each feature they said is there ("stove" -> kitchen). Call before clearing
        session.pending. Re-reads the keyword cues (cheap regexes) when Claude didn't supply them."""
        if cues is None:
            cues = followups.rule_cues(text, session.pending, reply.kind, semantic, loaded.features.floor_options)
        keys = {key for key, said in cues.mentioned.items() if said == "yes"}
        pending = session.pending
        if pending is not None and pending.kind == "yesno" and cues.answer == "yes":
            keys.add(pending.key)
        hints = [(t.lower(),) for t in room_types]
        hints += [followups.FEATURES_BY_KEY[k].room_types for k in keys
                  if k in followups.FEATURES_BY_KEY and followups.FEATURES_BY_KEY[k].room_types]
        return hints

    @staticmethod
    def _located(session: Session, loaded: LoadedBuilding, hints: list[tuple[str, ...]]) -> tuple[bool, bool]:
        """(located, short only of evidence). Also rewrites estimate.confident to the full rule, so the dashboard
        never shows "confident" while we are still asking."""
        building = loaded.building
        est = session.estimate
        # Types plus the room's own label, as the feature model reads them (followups._room_kinds): a "Pantry/Laundry
        # Room" label counts for a washer even if its only type is Pantry.
        room_types = [{t.lower() for t in r["types"]} | ({r["label"].lower()} if r.get("label") else set())
                      for r in building.rooms]
        probs = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        decisive = locator.names_only_room(room_types, est.room, hints)
        located = locator.is_located(est, probs, room_types, session.evidence_messages, decisive)
        short = (est.confident and not located and not locator.indistinguishable_rooms(probs, room_types, est.room))
        session.estimate = dataclasses.replace(est, confident=located)
        return located, short

    @staticmethod
    def _log_evidence(session: Session, loaded: LoadedBuilding, text: str, before: np.ndarray) -> None:
        """Record what one caller message did to the room probabilities, so the dashboard can show why a room leads
        ("-> Bathroom 1 +32%, Kitchen -20%") instead of a bare number. `ts` is the caller message's own timestamp, so
        the dashboard can pair the entry with the message even when queued replies are processed later."""
        building = loaded.building
        after = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        delta = after - before
        order = [i for i in np.argsort(-np.abs(delta))[:3] if abs(delta[i]) >= 0.005]  # hide sub-1% noise
        ts = next((m["ts"] for m in reversed(session.messages) if m["role"] == "caller" and m["text"] == text),
                  time.time())
        session.evidence.append({
            "ts": ts, "text": text,
            "message_id": next((m.get("id") for m in reversed(session.messages) if m["role"] == "caller" and m["ts"] == ts), None),
            "note": "This reply did not change the leading room probabilities." if not len(order) else None, "top_before": building.rooms[int(np.argmax(before))]["name"],
            "changes": [{"name": building.rooms[i]["name"], "delta": round(float(delta[i]), 3)} for i in order],
        })
        del session.evidence[:-EVIDENCE_LOG_LENGTH]

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
    def _approach(session: Session, loaded: LoadedBuilding) -> dict | None:
        """Approach card (side, floor, outside wall) for the current estimate; see approach.py. Cached there."""
        est = session.estimate
        if est is None:
            return None
        window = loaded.features.p_yes.get("window")
        return approach.for_estimate(loaded.building, est.room, est.candidate,
                                     None if window is None else float(window[est.candidate]))

    @staticmethod
    def _estimate(session: Session, loaded: LoadedBuilding) -> locator.Estimate:
        return locator.estimate(session.posterior, loaded.building.candidate_room, len(loaded.building.rooms))

    async def _say(self, session: Session, text: str) -> None:
        text = spoken_message(text)
        session.messages.append({"role": "dispatch", "text": text, "ts": time.time()})
        try:
            if session.channel != "voice":
                await messaging.send(session.phone, text)
        except messaging.MessagingError as exc:
            session.messages.append({"role": "system", "text": str(exc), "ts": time.time()})
        await self.notify(session)

    def find_by_phone(self, phone: str) -> Session | None:
        key = normalize_phone(phone)
        matches = [s for s in self.sessions.values() if s.channel != "voice" and normalize_phone(s.phone) == key]
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
            "id": session.id, "channel": session.channel, "revision": session.revision, "ended": session.ended, "phone": session.phone, "address": session.address, "created": session.created,
            "status": session.status, "status_detail": session.status_detail, "model_id": session.model_id,
            "match": session.match,
            "orientation": bool(session.model_id) and orientation.path(session.model_id).exists(),
            "messages": session.messages, "questions_asked": len(session.asked),
            "max_questions": config.MAX_QUESTIONS, "rooms": [], "floors": [], "candidate_probabilities": [],
            "estimate": None, "tie_group": [], "approach": None,
            "evidence": session.evidence[-EVIDENCE_LOG_LENGTH:],
        }
        loaded = self.buildings.get(session.model_id) if session.model_id else None
        if loaded is None or session.posterior is None:
            return data
        building = loaded.building
        room_probs = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        # Before the caller's words carry any evidence, every room is equally likely (the prior) and the "top room" is
        # just whichever comes first in Matterport's list: report no estimate, no tie and no dot until then.
        informed = session.evidence_messages > 0
        floor_labels = [f["label"] for f in building.floors]
        # has_photos/types/area_m2 let the dashboard flag thin evidence: a photo-less room is found from words alone.
        photographed = {int(building.candidate_room[c]) for c, cand in enumerate(building.candidates) if cand["views"]}
        areas = loaded.features.room_area
        data["rooms"] = sorted(({"index": i, "name": r["name"], "floor": r.get("floor", 0),
                                 "floor_label": floor_labels[r.get("floor", 0)], "probability": float(room_probs[i]),
                                 "has_photos": i in photographed, "types": list(r.get("types", [])),
                                 "area_m2": None if np.isnan(areas[i]) else round(float(areas[i]), 1)}
                                for i, r in enumerate(building.rooms)),
                               key=lambda r: -r["probability"] if informed else 0)

        floor_probs = np.bincount(building.room_floor, weights=room_probs, minlength=len(building.floors))
        data["floors"] = [{"index": i, "label": label, "probability": float(floor_probs[i])}
                          for i, label in enumerate(floor_labels)]
        data["candidate_probabilities"] = [float(p) for p in session.posterior]
        if not informed:
            return data
        data["tie_group"] = ties.snapshot(building, session.posterior,
                                          ties.tie_group(session.posterior, building, loaded.features))
        est = session.estimate
        if est is not None:
            candidate = building.candidates[est.candidate]
            data["estimate"] = {
                "room": est.room, "room_name": building.rooms[est.room]["name"], "candidate": est.candidate,
                "floor": int(building.room_floor[est.room]),
                "floor_label": floor_labels[int(building.room_floor[est.room])],
                "x": candidate["x"], "y": candidate["y"], "room_probability": est.room_probability,
                "runner_up_probability": est.runner_up_probability, "confident": est.confident,
                "evidence_messages": session.evidence_messages,
            }
            # Where inside the room (subroom.py): a circle for the dashboard, and words like "near the stove".
            region, zone = self._zone(session, loaded)
            data["estimate"]["region"] = {k: region[k] for k in ("x", "y", "radius_m", "sweeps")}
            data["estimate"]["zone_label"] = zone
            data["approach"] = self._approach(session, loaded)  # which side, floor and outside wall (approach.py)
        return data
