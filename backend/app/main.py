"""FastAPI app: REST for calls and replies, a WebSocket that pushes session snapshots, and cached building assets."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, embeddings, llm, orientation, semantics
from .sessions import Session, SessionManager

logging.basicConfig(level=logging.INFO)
clients: set[WebSocket] = set()


async def broadcast(session: Session) -> None:
    payload = {"type": "session", "session": manager.snapshot(session)}
    for ws in list(clients):
        try:
            await ws.send_json(payload)
        except Exception:
            clients.discard(ws)


manager = SessionManager(broadcast)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load CLIP and the paraphrase model in the background at startup so the first caller reply doesn't wait
    # ~10 s for each.
    asyncio.create_task(asyncio.to_thread(embeddings.warm_up))
    asyncio.create_task(asyncio.to_thread(semantics.warm_up))
    yield


app = FastAPI(title="Caller locator", lifespan=lifespan)
config.BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/buildings", StaticFiles(directory=config.BUILDINGS_DIR), name="buildings")


class CallIn(BaseModel):
    phone: str
    address: str


class ReplyIn(BaseModel):
    text: str


class InboundIn(BaseModel):
    sender: str = Field(alias="from")  # "from" is a Python keyword
    text: str


def _session_or_404(session_id: str) -> Session:
    session = manager.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    return session


@app.get("/api/config")
def get_config() -> dict:
    return {"messaging_mode": config.MESSAGING_MODE, "dev_test_phone": config.DEV_TEST_PHONE,
            "max_questions": config.MAX_QUESTIONS, "followups": "claude" if llm.enabled() else "rules"}


@app.post("/api/calls")
async def create_call(body: CallIn) -> dict:
    if not body.phone.strip() or not body.address.strip():
        raise HTTPException(400, "phone and address are required")
    session = await manager.start_call(body.phone, body.address)
    return manager.snapshot(session)


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    return [manager.snapshot(s) for s in sorted(manager.sessions.values(), key=lambda s: -s.created)]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return manager.snapshot(_session_or_404(session_id))


@app.get("/api/sessions/{session_id}/rooms")
def get_session_rooms(session_id: str) -> dict:
    """All rooms of the call's building with their current probability and every photo, for responders."""
    session = _session_or_404(session_id)
    return {"session": session.id, "model_id": session.model_id, "status": session.status,
            "rooms": manager.room_shots(session)}


@app.post("/api/sessions/{session_id}/reply")
async def simulated_reply(session_id: str, body: ReplyIn) -> dict:
    session = _session_or_404(session_id)
    await manager.handle_reply(session, body.text)
    return manager.snapshot(session)


@app.post("/api/inbound")
async def inbound(body: InboundIn) -> dict:
    """Real caller replies, forwarded by the messaging sidecar."""
    session = manager.find_by_phone(body.sender)
    if session is None:
        raise HTTPException(404, "No session for this phone number")
    await manager.handle_reply(session, body.text)
    return {"ok": True, "session": session.id}


@app.get("/api/buildings/{model_id}")
def get_building(model_id: str) -> dict:
    path = config.BUILDINGS_DIR / model_id / "building.json"
    if not path.exists():
        raise HTTPException(404, "Building not loaded")
    data = json.loads(path.read_text(encoding="utf-8"))
    base = f"/buildings/{model_id}/"
    for floorplan in data["floorplans"].values():
        floorplan["url"] = base + floorplan["file"]
    data["orientation"] = orientation.load(model_id)  # north + the street out front, if fetched
    for candidate in data["candidates"]:
        candidate["views"] = [base + v for v in candidate["views"]]
    return data


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        for session in sorted(manager.sessions.values(), key=lambda s: s.created):
            await ws.send_json({"type": "session", "session": manager.snapshot(session)})
        while True:
            await ws.receive_text()  # keep the connection open; the dashboard doesn't send anything
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
