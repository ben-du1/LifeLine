"""Authenticated ElevenLabs server tools. Only these routes are exposed by voice/gateway.mjs."""

import hmac
import json
import os
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import config
from .voice import VoiceBridge, VoiceError


class ConversationIn(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=200)
    acknowledge: int | None = Field(default=None, ge=0)


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
    router = APIRouter(prefix="/api/voice")

    async def invoke(method, *args):
        try:
            return await method(*args)
        except VoiceError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    @router.post("/start", dependencies=[Depends(authorize)])
    async def start(body: StartIn):
        return await invoke(bridge.start, body.conversation_id, body.address, body.phone)

    @router.post("/reply", dependencies=[Depends(authorize)])
    async def reply(body: ReplyIn):
        return await invoke(bridge.reply, body.conversation_id, body.history, body.acknowledge)

    @router.post("/status", dependencies=[Depends(authorize)])
    async def status(body: ConversationIn):
        return await invoke(bridge.status, body.conversation_id, body.acknowledge)

    @router.post("/post-call", dependencies=[])
    async def post_call(request: Request):
        secret = os.environ.get("ELEVENLABS_POST_CALL_SECRET", "")
        raw = await request.body()
        try:
            parts = dict(value.strip().split("=", 1) for value in request.headers.get("ElevenLabs-Signature", "").split(","))
            timestamp = parts["t"]
            expected = hmac.digest(secret.encode(), timestamp.encode() + b"." + raw, "sha256").hex()
            if not secret or abs(time.time() - int(timestamp)) > 1800 or not hmac.compare_digest(parts["v0"], expected):
                raise ValueError()
            event = json.loads(raw)
        except (ValueError, KeyError, TypeError):
            raise HTTPException(401, "Invalid ElevenLabs webhook signature") from None
        if isinstance(event, dict) and event.get("type") == "post_call_transcription":
            data = event.get("data")
            if isinstance(data, dict) and isinstance(data.get("conversation_id"), str):
                await bridge.end(data["conversation_id"], "Voice call ended.")
        return {"ok": True}

    return router
