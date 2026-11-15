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
            if session.status == "out":  # the caller got out while the building was loading
                return
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
        """Building plus its question model. Doors and windows (layout.py, one GraphQL call the first time) feed the
        layout questions; the street direction (orientation.py) adds "can you see the street?" when already cached
        (it is fetched in the background on a building's first call, so that first call goes without it)."""
        building = Building.load(model_id)
        features = followups.model_from_building(building, embeddings.embed_texts, layout.ensure(model_id),
                                                 orientation.load(model_id))
        return LoadedBuilding(building, features)

    # ---- conversation -----------------------------------------------------------------------------------------

    async def handle_reply(self, session: Session, text: str) -> None:
        shown = text.strip()   # logged as the caller typed it; read with plain quotes (questions.normalize)
        text = questions.normalize(text)
        if not text:
            return
        session.messages.append({"role": "caller", "text": shown, "ts": time.time()})
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
        reply = questions.parse_reply(text)
        if questions.is_correction(text):
            # "Sorry, I'm actually upstairs": start over from the prior instead of multiplying the correction into
            # the earlier description. One correction after a confident wrong room (98%) otherwise only got it
            # down to ~50%. Questions already answered stay counted toward MAX_QUESTIONS.
            session.posterior = locator.uniform_room_prior(building.candidate_room, len(building.rooms))
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
        pending = session.pending
        if pending is not None and pending.key == "door_to":
            mentions = questions.RoomMentions(set(), set(), set())  # the answer names the room outside the door
        else:
            mentions = questions.room_mentions(text)
        claimed = mentions.claimed | set(cues.room_types if cues else ())
        scores, boost = await asyncio.to_thread(self._evidence, loaded, text, reply, reading.explained, claimed,
                                                mentions.denied)
        if scores is not None:
            session.posterior = locator.update(session.posterior, scores, boost,
                                               candidate_room=building.candidate_room)
        if reading.likelihood is not None:
            session.posterior = locator.apply_likelihood(session.posterior, reading.likelihood)
        if reading.position is not None:
            session.posterior = locator.apply_within_rooms(session.posterior, reading.position, building.candidate_room)
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
            session.status_detail = f"{reason}. Most likely: {self._leading_rooms(session, loaded)}."
            if previous[0] == "best_guess":
                await self.notify(session)  # estimate updated; the caller already got the closing text
                return
            session.status = "best_guess"
            await self._say(session, questions.BEST_GUESS_MESSAGE)
            return
        session.status = "listening"  # also after a correction reopened a located call
        session.status_detail = f"Asked follow-up {len(session.asked) + 1} of {config.MAX_QUESTIONS}."
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
                  mentioned: set[str], denied: set[str] = frozenset()) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Photo evidence for one reply: per-candidate CLIP scores of the caller's own words (None when the reply was
        just an answer to our question, e.g. "yes" or "tile"), plus a room-type boost for rooms the caller says they
        are in (`mentioned`) and a penalty for rooms they say they are not in (`denied`)."""
        building = loaded.building
        parts = []
        if reply.kind in ("yes", "no"):
            if questions.has_detail(reply.extra):  # "yes, next to the big bed"
                parts.append(building.candidate_scores(embeddings.embed_texts([reply.extra])[0]))
        elif reply.kind == "free" and not explained and questions.describes_something(text):
            parts.append(building.candidate_scores(embeddings.embed_texts([text])[0]))

        boost = None
        for names, factor in ((mentioned, config.ROOM_KEYWORD_BOOST), (denied, config.ROOM_DENIAL_FACTOR)):
            if not names:
                continue
            room_match = np.array([bool(names & {t.lower() for t in r["types"]}) for r in building.rooms])
            if room_match.any():
                factors = np.where(room_match[building.candidate_room], factor, 1.0)
                boost = factors if boost is None else boost * factors

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
            await self._say(session, questions.LOCATED_MESSAGE.format(room=self._room_label(building, room)))
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
        if session.status not in ("listening", "located", "best_guess") or session.posterior is None:
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
        if session.status in ("no_model", "error"):
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
            "match": session.match, "danger": session.danger,
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
                # True once a caller answer corroborated this room (False: not yet checked, or nothing could check it)
                "checked": session.verified_room == est.room and not session.unverified,
            }
        return data
