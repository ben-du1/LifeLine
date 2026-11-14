"""HTTP authentication and schema checks without starting the app or a real agent."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app import config
from app.voice_routes import create_router


class VoiceRouteTests(unittest.TestCase):
    def test_authentication_validation_and_idempotent_start(self):
        session = SimpleNamespace(id="s", address="1 Main St", status="loading",
                                  messages=[{"role": "dispatch", "text": "Describe the room."}])
        manager = SimpleNamespace(start_call=AsyncMock(return_value=session))
        app = FastAPI()
        app.include_router(create_router(manager))
        with patch.object(config, "VOICE_TOOL_TOKEN", "test-secret"), TestClient(app) as client:
            body = {"conversation_id": "c", "address": "1 Main St"}
            self.assertEqual(client.post("/api/voice/start", json=body).status_code, 401)
            headers = {"Authorization": "Bearer test-secret"}
            self.assertEqual(client.post("/api/voice/start", headers=headers, json={}).status_code, 422)
            for _ in range(2):
                response = client.post("/api/voice/start", headers=headers, json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["say"], "Describe the room.")
            self.assertEqual(manager.start_call.await_count, 1)
            self.assertEqual(client.post("/api/voice/reply", headers=headers, json={
                "conversation_id": "c", "history": "invalid"}).status_code, 400)
        with patch.object(config, "VOICE_TOOL_TOKEN", ""), TestClient(app) as client:
            self.assertEqual(client.post("/api/voice/start", json=body).status_code, 503)


if __name__ == "__main__":
    unittest.main()
