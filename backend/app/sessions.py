"""Call sessions: building lookup/loading, the question loop, and snapshots for the dashboard.

Status flow:  loading -> listening -> located | best_guess
              loading -> no_model | error
              any -> out, when the caller texts that they got out of the building
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

from . import (config, embeddings, followups, ingest, layout, llm, locator, matterport, messaging, orientation,
               questions, semantics)
from .building import Building
from .addresses import is_exact, similarity
from . import approach, geo, subroom, ties, adjacency, textmatch

log = logging.getLogger(__name__)


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
    confirmed_floor: int | None = None
    provider_events: set[str] = field(default_factory=set)
    requested_model: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    evidence_messages: int = 0    # caller messages that moved the posterior; "located" needs enough (is_located)
    within_room_asked: int = 0   # within-room follow-ups asked after the room was found (subroom.py)
    evidence: list[dict] = field(default_factory=list)   # how each caller message moved the room probabilities
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
    extra_asked: list[followups.Plan] = field(default_factory=list)  # asked at the dispatcher's request
    misses: dict[str, int] = field(default_factory=dict)  # per question: answers we couldn't read
    last_key: str | None = None        # key of the last question sent (a re-ask says "Sorry, I didn't catch that")
    verified_room: int | None = None   # the leading room once an independent answer corroborated it
    unverified: bool = False           # located without a check (nothing in the question bank could check it)
    safety_sent: bool = False                                    # smoke/fire advice goes out once per call
    danger: bool = False                                         # caller mentioned smoke, fire, or trouble breathing
    estimate: locator.Estimate | None = None


def _odds(p: float) -> float:
    return p / max(1.0 - p, 1e-9)


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


class SessionManager:
    def __init__(self, notify: Callable[[Session], Awaitable[None]]):
        self._notify = notify
        self.sessions: dict[str, Session] = {}
        self.buildings: dict[str, LoadedBuilding] = {}
        self._building_locks: dict[str, asyncio.Lock] = {}

    # ---- call lifecycle -------------------------------------------------------------------------------------

    async def start_call(self, phone: str, address: str, model_id: str | None = None, *, channel: str = "messaging") -> Session:
        session = Session(id=uuid.uuid4().hex[:8], phone=phone.strip(), address=address.strip(), created=time.time())
        session.channel = channel
        session.requested_model = model_id
        self.sessions[session.id] = session
        # Texting the caller and loading the building happen in parallel (AGENTS.md, "How it works" step 2).
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
                if session.ended or session.status == "out":
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
        """Building plus its question model. Doors and windows (layout.py, one GraphQL call the first time) feed the
        layout questions; the street direction (orientation.py) adds "can you see the street?" when already cached
        (it is fetched in the background on a building's first call, so that first call goes without it)."""
        building = Building.load(model_id)
        features = followups.model_from_building(building, embeddings.embed_texts, layout.ensure(model_id),
                                                 orientation.load(model_id))
        adjacency.register(features, building, orientation.load(model_id))
        return LoadedBuilding(building, features)

    # ---- conversation -----------------------------------------------------------------------------------------

    async def handle_reply(self, session: Session, text: str, *, event_id: str | None = None) -> None:
        async with session.lock:
            if session.ended or (event_id is not None and event_id in session.provider_events):
                return
            if event_id is not None:
                session.provider_events.add(event_id)
            try:
                await self._handle_reply(session, text)
            except Exception:
                log.exception("Could not interpret caller reply")
                session.messages.append({"role": "system", "text": "Could not process this reply. Please try again.", "ts": time.time()})
                await self.notify(session)

    async def _handle_reply(self, session: Session, text: str) -> None:
        shown = text.strip()   # logged as the caller typed it; read with plain quotes (questions.normalize)
        text = questions.normalize(text)
        if not text:
            return
        session.messages.append({"id": uuid.uuid4().hex, "role": "caller", "text": shown, "ts": time.time()})
        # "Out" and danger are checked first, even while the building is still loading: neither should wait.
        if session.status != "out" and questions.is_out(text):
            # The caller got out: stop asking about rooms, so they don't keep texting next to a burning building.
            session.status = "out"
            session.status_detail = "Caller says they are out of the building."
            session.pending, session.queued = None, []
            await self._say(session, questions.OUT_MESSAGE)
            return
        if session.status != "out" and questions.mentions_danger(text):
            # Smoke or fire where the caller is: the dispatcher sees it on the call, and the caller gets the smoke
            # advice now instead of after being located.
            session.danger = True
            if not session.safety_sent:
                session.safety_sent = True
                await self._say(session, questions.SAFETY_MESSAGE)
        if session.waiting_for_match:
            answer = questions.parse_reply(text).kind
            if answer == "yes":
                session.waiting_for_match = False
                if session.queued:
                    description, session.queued = " ".join(session.queued), []
                    await self._process(session, description)
                else:
                    await self._say(session, "Tell us what room you’re in and what you can see around you.")
            elif answer == "no":
                session.waiting_for_match = False
                session.queued.clear()
                session.status, session.model_id = "no_model", None
                session.status_detail = "Caller rejected the suggested building."
                await self._say(session, "Please start again with the complete address.")
            else:
                await self._say(session, "Please say yes or no to confirm the building address.")
            return
        if session.status == "loading":
            session.queued.append(text)
            await self.notify(session)
            return
        if session.status == "out":
            # Not parsed for rooms, but shown up front: it is often about someone still inside.
            session.status_detail = f"Caller is out of the building. Latest text: “{shown}”"
        if session.status not in ("listening", "located", "best_guess"):
            await self.notify(session)  # no_model / error / out: recorded only
            return
        # A located call keeps listening: "actually I'm upstairs" must still move the estimate.
        await self._process(session, text)

    async def _process(self, session: Session, text: str) -> None:
        loaded = self.buildings[session.model_id]
        building = loaded.building
        previous = (session.status, session.estimate.room if session.estimate else None)
        normalized = " ".join(text.casefold().split())
        reply = questions.parse_reply(text)
        if questions.is_correction(text):
            # "Sorry, I'm actually upstairs": start over from the prior instead of multiplying the correction into
            # the earlier description. One correction after a confident wrong room (98%) otherwise only got it
            # down to ~50%. Questions already answered stay counted toward MAX_QUESTIONS.
            session.covered.clear()
            session.misses.clear()
            session.verified_room, session.unverified = None, False
        cues, semantic = None, None
        if llm.enabled():  # Claude reads paraphrase; None (failure/timeout) falls back to the local reading
            cues = await asyncio.to_thread(llm.read_message, text, session.pending, session.messages,
                                           len(building.floors))
        if cues is None:
            semantic = await asyncio.to_thread(semantics.scores, text)
        reading = followups.read_message(text, loaded.features, session.pending, reply.kind, cues, semantic)
        before = locator.room_probabilities(session.posterior, building.candidate_room, len(building.rooms))
        pending = session.pending
        if pending is not None and pending.key == "door_to":
            mentions = questions.RoomMentions(set(), set(), set())  # the answer names the room outside the door
        else:
            mentions = questions.room_mentions(text)
        claimed = mentions.claimed | set(cues.room_types if cues else ())
        scores, boost = await asyncio.to_thread(self._evidence, loaded, text, reply, reading.explained, claimed,
                                                reading.covered, mentions.denied)
        floors = followups.stated_floors(text, loaded.features.floor_options)
        if pending and pending.key in ("floor", "flights") and not floors:
            parsed = followups.rule_cues(text, pending, reply.kind, floors=loaded.features.floor_options)
            floors = tuple(int(key.removeprefix("floor")) for key in parsed.options if key.startswith("floor"))
        floor_selected = len(floors) == 1 and session.confirmed_floor != floors[0]
        floor_changed = floor_selected and session.confirmed_floor is not None
        if floor_selected:
            session.confirmed_floor = floors[0]
        topics = (reading.covered - {"floor"}) | {f"room:{room}" for room in claimed | mentions.denied}
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
        self._log_evidence(session, loaded, text, before)
        session.covered |= reading.covered
        if pending is not None and pending.kind in ("yesno", "choice") and pending.key not in reading.covered:
            # No usable answer ("what?", or a description that ignores the question). Asked once more, with a hint
            # (see _ask); after a second miss the question is dropped rather than sent a third time.
            session.misses[pending.key] = session.misses.get(pending.key, 0) + 1
            if session.misses[pending.key] >= 2:
                session.covered.add(pending.key)
        last_kind = pending.kind if pending else None
        session.pending = None
        session.estimate = self._estimate(session, loaded)
        room = session.estimate.room
        confident = session.estimate.confident and self._floor_settled(session, loaded)

        # Corroboration: an answer to a check of this room counts only if it made the room clearly likelier (odds up
        # VERIFY_MIN_ODDS times). "Didn't go down" wasn't enough: "Is there a closet?" answered yes "confirmed"
        # Bedroom 3 although the runner-up, Bedroom 2, has a closet too. Otherwise another check is asked.
        if pending is not None and pending.purpose == "verify" and pending.room == room and confident:
            if _odds(session.estimate.room_probability) >= config.VERIFY_MIN_ODDS * _odds(pending.before):
                session.verified_room, session.unverified = room, False
        if confident and session.verified_room == room:
            await self._located(session, loaded, previous, last_kind, reading.acknowledgement)
            return
        if confident:
            check = None
            if self._auto_left(session):
                check = followups.verification(loaded.features, session.posterior, room, session.covered)
            if check is None:
                # Nothing in the question bank can check this room (or no questions left): report it, flagged.
                session.verified_room, session.unverified = room, True
                await self._located(session, loaded, previous, last_kind, reading.acknowledgement)
                return
            check.purpose, check.room, check.before = "verify", room, session.estimate.room_probability
            session.status = "listening"
            session.status_detail = (f"Likely {self._room_label(building, room)} "
                                     f"({round(session.estimate.room_probability * 100)}%); checking it with one "
                                     f"more question before telling the caller.")
            await self._ask(session, check, reading.acknowledgement, questions.VERIFY_INTRO)
            return

        plan = None
        if self._auto_left(session):
            plan = followups.choose(loaded.features, session.posterior, building.candidate_room, len(building.rooms),
                                    session.covered, last_kind)
        if plan is None:
            reason = ("Question limit reached" if not self._auto_left(session)
                      else "No follow-up would tell the remaining rooms apart")
            tied = ties.tie_group(session.posterior, building, loaded.features)
            leading = ties.describe(building, session.posterior, tied) if tied else self._leading_rooms(session, loaded)
            session.status_detail = f"{reason}. Most likely: {leading}."
            if tied:
                session.messages.append({"role": "system", "text": leading, "ts": time.time()})
            if previous[0] == "best_guess":
                await self.notify(session)  # estimate updated; the caller already got the closing text
                return
            session.status = "best_guess"
            await self._say(session, questions.BEST_GUESS_MESSAGE)
            return
        session.status = "listening"  # also after a correction reopened a located call
        session.status_detail = "Waiting for the caller’s answer."
        await self._ask(session, plan, reading.acknowledgement)

    @staticmethod
    def _auto_left(session: Session) -> bool:
        return len(session.asked) < config.MAX_QUESTIONS

    @staticmethod
    def _room_label(building: Building, room: int) -> str:
        """"Kitchen", or "Kitchen, 2nd floor" in a multi-storey building (the floor matters as much as the room).
        Floors are named as the floor question names them ("ground floor", "3rd floor"), not by Matterport's "Floor 1"
        labels: a caller asked "ground floor, 2nd floor, or 3rd floor?" and then told "Floor 1" can't tell which."""
        name = building.rooms[room]["name"]
        if len(building.floors) > 1:
            name = f"{name}, {followups.floor_options(len(building.floors))[int(building.room_floor[room])].label}"
        return name

    async def _ask(self, session: Session, plan: followups.Plan, acknowledgement: str = "", intro: str = "",
                   counted: bool = True) -> None:
        """Send one planned question and wait for its answer. `counted` questions use up MAX_QUESTIONS; the
        dispatcher's own requests don't."""
        (session.asked if counted else session.extra_asked).append(plan)
        repeat = plan.key == session.last_key and session.misses.get(plan.key, 0) > 0
        session.pending, session.last_key = plan, plan.key
        question = None
        if repeat:
            # Same question again because its answer couldn't be read: say so, and say how to answer, instead of
            # sending the identical text (a live caller got "Which floor are you on?" four times in a row).
            lead = acknowledgement or questions.NOT_UNDERSTOOD
            question = f"{lead} {plan.text} {followups.answer_hint(plan)}".strip()
        elif llm.enabled() and plan.kind != "nearest":
            question = await asyncio.to_thread(llm.phrase_question, plan, session.messages, acknowledgement)
        if question is None:  # template: an intro ("To make sure:") takes the place of the acknowledgement
            question = f"{intro} {plan.text}" if intro else followups.phrase(plan, acknowledgement)
        await self._say(session, question)

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

    async def _located(self, session: Session, loaded: LoadedBuilding, previous: tuple, last_kind: str | None,
                       acknowledgement: str = "") -> None:
        """The room is known (and checked, see _process). Tell the caller (once per room), then keep asking:
          1. while the room is under CONFIRM_UNTIL, the best room question (an answer can still move the estimate,
             and the caller is told again if the room changes);
          2. then up to MAX_REFINEMENTS "what's closest to you?" questions while the position in the room is spread
             out over several scan points."""
        building = loaded.building
        room = session.estimate.room
        more = refine = None
        if session.estimate.room_probability < config.CONFIRM_UNTIL and self._auto_left(session):
            more = followups.choose(loaded.features, session.posterior, building.candidate_room, len(building.rooms),
                                    session.covered, last_kind)
        if more is None and len(session.refined) < config.MAX_REFINEMENTS:
            asked = {k for plan in session.refined for k in plan.items}
            refine = followups.refinement(loaded.features, session.posterior, room, self._xy(building), asked)
        newly_located = previous != ("located", room)
        session.status = "located"
        note = " Could not be cross-checked: nothing left to ask would tell it from the others." \
            if session.unverified else ""
        session.status_detail = ("Caller located; asking more to be sure." if more else
                                 "Caller located; narrowing down where in the room." if refine else
                                 "Caller located.") + note
        if newly_located:
            # Separate texts, safety first: further questions are optional and come last.
            label = self._room_label(building, room)
            _, zone = self._zone(session, loaded)
            if zone:
                label = f"{label}, {zone}"
            await self._say(session, questions.LOCATED_MESSAGE.format(room=label))
            if not session.safety_sent:
                session.safety_sent = True
                await self._say(session, questions.SAFETY_MESSAGE)
            acknowledgement = ""
        if more is not None:
            await self._ask(session, more, acknowledgement, questions.CONFIRM_INTRO if newly_located else "")
        elif refine is not None:
            session.refined.append(refine)
            session.pending = refine
            await self._say(session, f"{questions.REFINE_INTRO} {refine.text}" if newly_located else refine.text)
        elif last_kind == "nearest":
            await self._say(session, questions.REFINED_MESSAGE)
        elif not newly_located:
            await self.notify(session)  # same room as before: don't repeat the "located" text

    # ---- questions the dispatcher asks for -------------------------------------------------------------------

    async def ask_next(self, session: Session) -> str | None:
        """The dispatcher wants another question now, even after the call is located or out of automatic
        questions. Picks, in order: a check of an unchecked leading room, the best room question, then "what's
        closest to you?". Returns a reason if nothing useful is left (or the call can't take questions)."""
        if session.ended or session.status not in ("listening", "located", "best_guess") or session.posterior is None:
            return "This call isn't taking questions."
        loaded = self.buildings[session.model_id]
        building, room = loaded.building, session.estimate.room
        plan = intro = None
        if session.verified_room != room:
            plan = followups.verification(loaded.features, session.posterior, room, session.covered)
            if plan is not None:
                plan.purpose, plan.room, plan.before = "verify", room, session.estimate.room_probability
                intro = questions.VERIFY_INTRO
        # The dispatcher asked, so the automatic "worth a caller's text" bars don't apply: any question that still
        # tells rooms apart will do, or else one more independent check of the leading room.
        if plan is None:
            plan = followups.choose(loaded.features, session.posterior, building.candidate_room, len(building.rooms),
                                    session.covered, session.pending.kind if session.pending else None,
                                    min_gain=config.DISPATCHER_MIN_GAIN)
        if plan is None:
            plan = followups.verification(loaded.features, session.posterior, room, session.covered,
                                          min_gain=config.DISPATCHER_MIN_GAIN)
            if plan is not None:
                plan.purpose, plan.room, plan.before = "verify", room, session.estimate.room_probability
                intro = questions.VERIFY_INTRO
        if plan is None and session.status == "located":
            asked = {k for p in session.refined for k in p.items}
            plan = followups.refinement(loaded.features, session.posterior, room, self._xy(building), asked)
            if plan is not None:
                session.refined.append(plan)
        if plan is None:
            return "No question left that would tell the likely rooms apart. Try asking your own."
        await self._ask(session, plan, intro=intro or "", counted=False)
        return None

    async def ask_custom(self, session: Session, text: str) -> str | None:
        """Send the dispatcher's own question. The reply is read as free text (no planned answer to match)."""
        text = text.strip()
        if not text:
            return "The question is empty."
        if session.ended or session.status in ("out", "no_model", "error"):
            return "This call isn't taking questions."
        session.pending = followups.Plan("custom", "custom", text, purpose="dispatcher") \
            if session.posterior is not None else None
        await self._say(session, text, source="dispatcher")
        return None

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
        if "floor" not in session.covered:
            # The caller hasn't given the floor (nor said they don't know): photo matches alone once put a 3rd-floor
            # caller in a ground-floor bedroom at 97%. followups.choose asks the floor first, so this only waits for it.
            return False
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

    async def _say(self, session: Session, text: str, source: str = "system") -> None:
        """Text the caller. `source` is "dispatcher" for questions the dispatcher typed (shown as such in the log)."""
        session.messages.append({"role": "dispatch", "text": text, "ts": time.time(), "source": source})
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
            "match": session.match, "danger": session.danger,
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
                "checked": session.verified_room == est.room and not session.unverified,
            }
            # Where inside the room (subroom.py): a circle for the dashboard, and words like "near the stove".
            region, zone = self._zone(session, loaded)
            data["estimate"]["region"] = {k: region[k] for k in ("x", "y", "radius_m", "sweeps")}
            data["estimate"]["zone_label"] = zone
            data["approach"] = self._approach(session, loaded)  # which side, floor and outside wall (approach.py)
        return data


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
    def _approach(session: Session, loaded: LoadedBuilding) -> dict | None:
        """Approach card (side, floor, outside wall) for the current estimate; see approach.py. Cached there."""
        est = session.estimate
        if est is None:
            return None
        window = loaded.features.p_yes.get("window")
        return approach.for_estimate(loaded.building, est.room, est.candidate,
                                     None if window is None else float(window[est.candidate]))


    async def notify(self, session: Session) -> None:
        session.revision += 1
        await self._notify(session)

EVIDENCE_LOG_LENGTH = 20


def room_denial(building: Building, denied: set[str]) -> np.ndarray | None:
    """Per-candidate factor for room types the caller said they are not in (questions.room_types_denied), or None
    when no room has a denied type."""
    if not denied:
        return None
    room_match = np.array([bool(denied & {t.lower() for t in r["types"]}) for r in building.rooms])
    if not room_match.any():
        return None
    return np.where(room_match[building.candidate_room], config.ROOM_DENIAL_FACTOR, 1.0)
