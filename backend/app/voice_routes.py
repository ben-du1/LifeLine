"""Authenticated ElevenLabs server tools. Only these routes are exposed by voice/gateway.mjs."""

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from . import config
from .voice import VoiceBridge, VoiceError


class ConversationIn(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=200)


class StartIn(ConversationIn):
    address: str = Field(min_length=1, max_length=500)
    phone: str | None = Field(default=None, max_length=100)


class ReplyIn(ConversationIn):
    history: str = Field(min_length=1, max_length=200000)


def authorize(authorization: str = Header(default="")) -> None:
    if not config.VOICE_TOOL_TOKEN:
        raise HTTPException(503, "Voice tools are not configured.")
    expected = f"Bearer {config.VOICE_TOOL_TOKEN}"
    if not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(401, "Invalid voice tool credential.")


def create_router(manager) -> APIRouter:
    bridge = VoiceBridge(manager)
    router = APIRouter(prefix="/api/voice", dependencies=[Depends(authorize)])

    async def invoke(method, *args):
        try:
            return await method(*args)
        except VoiceError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    @router.post("/start")
    async def start(body: StartIn):
        return await invoke(bridge.start, body.conversation_id, body.address, body.phone)

    @router.post("/reply")
    async def reply(body: ReplyIn):
        return await invoke(bridge.reply, body.conversation_id, body.history)

    @router.post("/status")
    async def status(body: ConversationIn):
        return await invoke(bridge.status, body.conversation_id)

    return router
