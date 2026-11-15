"""Voice transport for the existing session manager; no separate location or question logic."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import config, phones, questions

if TYPE_CHECKING:
    from .sessions import Session, SessionManager


class VoiceError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# Statuses in which the backend is talking to the caller about the address (sessions.open_call): nothing is loading,
# so the bridge answers at once instead of waiting for `session.ready`.
INTAKE_STATUSES = ("awaiting_address",)
# The call is over for the agent once the caller was handed off (address outside the coverage) or is out.
FINAL_STATUSES = ("transferred", "out")

ASK_INSTRUCTION = "Say the returned text exactly. Accept corrections; do not invent another question."
INTAKE_INSTRUCTION = ("Say the returned text exactly, then send the caller's answer with submit_location_answer. "
                      "Do not call start_location again.")
END_INSTRUCTION = "Say the text exactly, then end the call."


@dataclass
class VoiceCall:
    session: "Session"
    address: str = ""                  # the address start_location last sent (a retried request repeats it)
    opening: dict | None = None        # start's response, returned again to a retried start_location
    delivered: int = 0
    offered: int = 0
    expiry: asyncio.TimerHandle | None = None
    loading_deadline: float = field(default_factory=lambda: time.monotonic() + config.VOICE_LOADING_SECONDS)
    replies: dict[int, tuple[str, dict]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def caller_messages(history: str) -> list[str]:
    """Use the platform transcript rather than an LLM paraphrase of a caller's answer."""
    try:
        entries = json.loads(history)["entries"]
        users = [entry["message"] for entry in entries
                 if entry.get("role") == "user" and isinstance(entry.get("message"), str)]
    except (ValueError, KeyError, TypeError, AttributeError):
        raise VoiceError("A valid ElevenLabs conversation history is required.") from None
    if not users or not users[-1].strip():
        raise VoiceError("The conversation has no caller answer.")
    if len(users[-1]) > 8000:
        raise VoiceError("Caller answer is too long.")
    return users


def latest_reply(history: str) -> tuple[int, str]:
    users = caller_messages(history)
    return len(users), users[-1]




def finished(session: "Session") -> bool:
    """True once the agent should say its last words and hang up."""
    return bool(getattr(session, "ended", False)) or session.status in FINAL_STATUSES


class VoiceBridge:
    def __init__(self, manager: "SessionManager", wait_seconds: float = 10):
        self.manager = manager
        self.wait_seconds = wait_seconds
        self.calls: dict[str, VoiceCall] = {}
        # Per conversation: opening a call can take seconds (the address lookup), and other callers must not wait.
        self._start_locks: dict[str, asyncio.Lock] = {}

    def _call(self, conversation_id: str) -> VoiceCall:
        call = self.calls.get(conversation_id)
        if call is None:
            raise VoiceError("Start a location session for this conversation first.", 404)
        return call

    async def _open(self, address: str, phone: str | None) -> "Session":
        """Send the address through the call intake (sessions.open_call: resolve, retry, "Did you mean", hand-off).
        Managers without a working open_call (before workstream B is merged, and the fakes in test_voice_bridge.py)
        get the older direct start_call."""
        open_call = getattr(self.manager, "open_call", None)
        if open_call is not None:
            try:
                return await open_call(phone, "voice", address)
            except NotImplementedError:
                pass
        return await self.manager.start_call(phone or "", address, channel="voice")

    async def start(self, conversation_id: str, address: str, phone: str | None = None) -> dict:
        address = address.strip()
        if not conversation_id.strip() or not address:
            raise VoiceError("conversation_id and address are required.")
        async with self._start_locks.setdefault(conversation_id, asyncio.Lock()):
            call = self.calls.get(conversation_id)
            if call is None:
                # The conversation id stays the key: the caller ID may be withheld (phone None), and one phone can
                # call twice.
                session = await self._open(address, phones.to_e164(phone))
                call = VoiceCall(session, address=address)
                self.calls[conversation_id] = call
                loop = asyncio.get_running_loop()
                def expire():
                    asyncio.create_task(self.end(conversation_id, "Voice interview reached its time limit."))
                call.expiry = loop.call_later(config.VOICE_INTERVIEW_SECONDS, expire)
                async with call.lock:
                    call.opening = self._speak(call, opening=True)
                return call.opening
            if address == call.address:
                return call.opening   # a retried tool request never starts another session
            session = call.session
            if session.status in INTAKE_STATUSES or finished(session):
                # The agent sent a new address instead of submitting it as an answer: the intake reads it as one.
                async with call.lock:
                    if session.status in INTAKE_STATUSES and not finished(session):
                        await self.manager.handle_reply(session, address)
                    call.address = address
                    call.opening = await self._result(call)
                return call.opening
        raise VoiceError("This conversation already has a different building address.", 409)

    async def reply(self, conversation_id: str, history: str, acknowledge: int | None = None) -> dict:
        users = caller_messages(history)
        turn, text = len(users), users[-1]
        call = self._call(conversation_id)
        async with call.lock:
            if any(users[n - 1] != old for n, (old, _) in call.replies.items() if n <= turn):
                raise VoiceError("A processed transcript turn cannot be changed.", 409)
            if turn in call.replies:
                previous_text, response = call.replies[turn]
                if previous_text != text:
                    raise VoiceError("A processed transcript turn cannot be changed.", 409)
                if response["pending"] and call.session.ready.is_set():
                    response = await self._result(call, acknowledge)
                    call.replies[turn] = (text, response)
                return response
            if call.replies and turn < max(call.replies):
                raise VoiceError("This transcript is older than the last processed answer.", 409)
            # If loading is unfinished, handle_reply queues the exact answer for _load.
            # Skip all earlier messages, including the opening already returned by start.
            call.delivered = max(call.delivered, 1)
            previous = max(call.replies, default=turn - 1)
            await self.manager.handle_reply(call.session, " ".join(users[previous:]))
            response = await self._result(call, acknowledge)
            call.replies[turn] = (text, response)
            return response

    async def status(self, conversation_id: str, acknowledge: int | None = None) -> dict:
        call = self._call(conversation_id)
        async with call.lock:
            return await self._result(call, acknowledge)

    async def end(self, conversation_id: str, reason: str) -> None:
        call = self.calls.get(conversation_id)
        if call is not None:
            if call.expiry is not None:
                call.expiry.cancel()
            await self.manager.end_call(call.session, reason)

    async def _result(self, call: VoiceCall, acknowledge: int | None = None) -> dict:
        session = call.session
        if acknowledge is not None:
            if acknowledge > call.offered:
                raise VoiceError("Unknown response cursor.", 409)
            call.delivered = max(call.delivered, acknowledge)
        if session.status in INTAKE_STATUSES:
            # Nothing loads while the address is being settled; the loading time limit starts once it is.
            call.loading_deadline = time.monotonic() + config.VOICE_LOADING_SECONDS
        elif not finished(session):
            try:
                await asyncio.wait_for(session.ready.wait(), timeout=self.wait_seconds)
            except TimeoutError:
                if time.monotonic() >= call.loading_deadline:
                    if call.expiry is not None:
                        call.expiry.cancel()
                    await self.manager.end_call(session, "Voice interview timed out while loading the building.")
                    return {"session_id": session.id, "status": "error", "pending": False, "ended": True,
                            "say": "The building took too long to load.", "awaiting_answer": False,
                            "instruction": END_INSTRUCTION}
                return {"session_id": session.id, "status": "loading", "pending": True, "ended": False,
                        "say": "I'm still loading the building's interior.",
                        "instruction": "Call location_status again. Do not resubmit the caller's answer."}
        return self._speak(call, acknowledge is None)

    def _speak(self, call: VoiceCall, advance: bool = True, opening: bool = False) -> dict:
        """The dispatch messages the agent hasn't said yet, worded for speech (questions.for_channel; sessions._say
        may already apply it, and for_channel leaves voice wording unchanged)."""
        session = call.session
        messages = [questions.for_channel(m["text"], "voice") for m in session.messages[call.delivered:]
                    if m["role"] == "dispatch"]
        if opening and len(messages) > 1:
            # The agent's first message already asked for the address: don't ask it again before the answer to it.
            asked = {questions.ADDRESS_QUESTION, questions.for_channel(questions.ADDRESS_QUESTION, "voice")}
            messages = [m for m in messages if m not in asked]
        call.offered = len(session.messages)
        if advance:
            call.delivered = call.offered
        ended = finished(session)
        intake = session.status in INTAKE_STATUSES and not ended
        if ended:
            say = " ".join(messages) or "The location interview has ended."
            instruction = END_INSTRUCTION
        elif session.status == "error":
            say, instruction = "The floor plan could not be loaded.", ASK_INSTRUCTION
        else:
            say = " ".join(messages) or "Thank you. Your answer has been recorded."
            instruction = INTAKE_INSTRUCTION if intake else ASK_INSTRUCTION
            if opening and not intake:
                instruction = "Ask this question. If the caller already described the room, submit that answer now."
        return {"session_id": session.id, "status": session.status, "pending": False, "ended": ended, "say": say,
                "awaiting_answer": not ended and (intake or getattr(session, "pending", None) is not None),
                "response_id": call.offered, "instruction": instruction}
