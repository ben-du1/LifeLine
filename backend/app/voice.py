"""Voice transport for the existing session manager; no separate location or question logic."""

import asyncio
import json
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
    replies: dict[int, tuple[str, dict]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def latest_reply(history: str) -> tuple[int, str]:
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
    return len(users), users[-1]


def spoken_message(text: str) -> str:
    """Keep the MVP's question wording without claiming a demo dispatched real responders."""
    return (text.replace(
        "This is 911 emergency services. We're working to find you inside the building.",
        "This is the HackPrinceton caller-location demo.")
        .replace("Responders now know which room you're in", "The demo estimates you're in this room")
        .replace("Stay where you are; help is on the way.", "You can correct this estimate if needed.")
        .replace("Responders have your approximate location.", "The demo has an approximate location estimate.")
        .replace("Stay where you are and keep your phone with you.", "You can correct this estimate if needed.")
        .replace("Thank you. Stay where you are if it is safe; responders are on the way to your address.",
                 "The demo could not find an interior model for this address.")
        .replace("Thank you, that helps responders find you. Stay where you are.",
                 "Thank you, that refines the demo's location estimate."))


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
            elif call.session.address != address.strip():
                raise VoiceError("This conversation already has a different building address.", 409)
        # A retry returns the opening again but never starts another session.
        return {"session_id": call.session.id, "status": call.session.status, "pending": False,
                "say": spoken_message(call.session.messages[0]["text"]),
                "instruction": "Ask this question. If the caller already described the room, submit that answer now."}

    async def reply(self, conversation_id: str, history: str) -> dict:
        turn, text = latest_reply(history)
        call = self._call(conversation_id)
        async with call.lock:
            if turn in call.replies:
                previous_text, response = call.replies[turn]
                if previous_text != text:
                    raise VoiceError("A processed transcript turn cannot be changed.", 409)
                return response
            if call.replies and turn < max(call.replies):
                raise VoiceError("This transcript is older than the last processed answer.", 409)
            # If loading is unfinished, handle_reply queues the exact answer for _load.
            # Skip all earlier messages, including the opening already returned by start.
            call.delivered = max(call.delivered, 1)
            await self.manager.handle_reply(call.session, text)
            response = await self._result(call)
            call.replies[turn] = (text, response)
            return response

    async def status(self, conversation_id: str) -> dict:
        call = self._call(conversation_id)
        async with call.lock:
            return await self._result(call)

    async def _result(self, call: VoiceCall) -> dict:
        session = call.session
        try:
            await asyncio.wait_for(session.ready.wait(), timeout=self.wait_seconds)
        except TimeoutError:
            return {"session_id": session.id, "status": "loading", "pending": True,
                    "say": "I'm still loading the building's interior.",
                    "instruction": "Call location_status again. Do not resubmit the caller's answer."}
        messages = [spoken_message(m["text"]) for m in session.messages[call.delivered:]
                    if m["role"] == "dispatch"]
        call.delivered = len(session.messages)
        if session.status == "error":
            messages = ["The demo could not process the building. Please check the dispatcher dashboard."]
        return {"session_id": session.id, "status": session.status, "pending": False,
                "say": " ".join(messages) or "Thank you. The demo has recorded your answer.",
                "awaiting_answer": session.pending is not None,
                "instruction": "Say the returned text exactly. Accept corrections; do not invent another question."}
