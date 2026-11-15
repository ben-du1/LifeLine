"""Voice transport for the existing session manager; no separate location or question logic."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sessions import Session, SessionManager


class VoiceError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class VoiceCall:
    session: "Session"
    delivered: int = 0
    offered: int = 0
    expiry: asyncio.TimerHandle | None = None
    loading_deadline: float = field(default_factory=lambda: time.monotonic() + 60)
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


class VoiceBridge:
    def __init__(self, manager: "SessionManager", wait_seconds: float = 10):
        self.manager = manager
        self.wait_seconds = wait_seconds
        self.calls: dict[str, VoiceCall] = {}
        self._start_lock = asyncio.Lock()

    def _call(self, conversation_id: str) -> VoiceCall:
        call = self.calls.get(conversation_id)
        if call is None:
            raise VoiceError("Start a location session for this conversation first.", 404)
        return call

    async def start(self, conversation_id: str, address: str, phone: str | None = None) -> dict:
        if not conversation_id.strip() or not address.strip():
            raise VoiceError("conversation_id and address are required.")
        async with self._start_lock:
            call = self.calls.get(conversation_id)
            if call is None:
                session = await self.manager.start_call(
                    phone or f"voice:{conversation_id}", address, channel="voice")
                call = VoiceCall(session)
                self.calls[conversation_id] = call
                loop = asyncio.get_running_loop()
                def expire():
                    asyncio.create_task(self.end(conversation_id, "Voice interview reached its time limit."))
                call.expiry = loop.call_later(240, expire)
            elif call.session.address != address.strip():
                raise VoiceError("This conversation already has a different building address.", 409)
        # A retry returns the opening again but never starts another session.
        return {"session_id": call.session.id, "status": call.session.status, "pending": False,
                "say": call.session.messages[0]["text"],
                "response_id": 0, "instruction": "Ask this question. If the caller already described the room, submit that answer now."}

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
        if getattr(session, "ended", False):
            return {"session_id": session.id, "status": session.status, "pending": False,
                    "say": "The location interview has ended.", "response_id": call.offered}
        try:
            await asyncio.wait_for(session.ready.wait(), timeout=self.wait_seconds)
        except TimeoutError:
            if time.monotonic() >= call.loading_deadline:
                if call.expiry is not None:
                    call.expiry.cancel()
                await self.manager.end_call(session, "Voice interview timed out while loading the building.")
                return {"session_id": session.id, "status": "error", "pending": False,
                        "say": "The building took too long to load.", "awaiting_answer": False}
            return {"session_id": session.id, "status": "loading", "pending": True,
                    "say": "I'm still loading the building's interior.",
                    "instruction": "Call location_status again. Do not resubmit the caller's answer."}
        messages = [m["text"] for m in session.messages[call.delivered:]
                    if m["role"] == "dispatch"]
        call.offered = len(session.messages)
        if acknowledge is None:
            call.delivered = call.offered
        if session.status == "error":
            messages = ["The floor plan could not be loaded."]
        return {"session_id": session.id, "status": session.status, "pending": False,
                "say": " ".join(messages) or "Thank you. Your answer has been recorded.",
                "awaiting_answer": session.pending is not None, "response_id": call.offered,
                "instruction": "Say the returned text exactly. Accept corrections; do not invent another question."}
