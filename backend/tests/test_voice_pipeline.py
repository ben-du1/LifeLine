"""Exercise the real session pipeline with deterministic model outputs and no external calls."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from app import config, semantics, sessions
from app.voice import VoiceBridge
from test_followups import make_model, make_two_floor_model


class VoicePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def prepare(self, factory=make_model):
        building, features = factory()
        for i, candidate in enumerate(building.candidates):
            candidate.update(x=float(i), y=0.0)
        manager = sessions.SessionManager(AsyncMock())
        manager.buildings[building.id] = sessions.LoadedBuilding(building, features)
        manager._get_building = AsyncMock(return_value=manager.buildings[building.id])
        manager._load_orientation = AsyncMock()
        match = {"id": building.id, "address": building.address, "exact": True, "score": 1.0}
        with patch.object(sessions, "find_matches", return_value=[match]):
            bridge = VoiceBridge(manager, wait_seconds=1)
            await bridge.start("conversation", building.address, "+15555550100")
            session = bridge.calls["conversation"].session
            await asyncio.wait_for(session.ready.wait(), 2)
        self.assertEqual(session.status, "listening")
        return bridge, session, manager

    async def test_semantic_answer_drives_existing_planner_and_retry_is_safe(self):
        with (patch.object(sessions.llm, "enabled", return_value=False),
              patch.object(semantics, "scores", return_value={"tight": 0.9}) as semantic,
              patch.object(sessions.embeddings, "embed_texts", return_value=np.zeros((1, 3))),
              patch.object(sessions.messaging, "send", new_callable=AsyncMock) as send):
            bridge, session, manager = await self.prepare()
            transcript = json.dumps({"entries": [
                {"role": "user", "message": "I can touch both walls with my arms"}]})
            result = await bridge.reply("conversation", transcript)
            semantic.assert_called_once_with("I can touch both walls with my arms")
            self.assertIn("size", session.covered)
            self.assertEqual(len(session.asked), 1)
            self.assertIsNotNone(session.pending)
            self.assertIn("tight space", result["say"])
            before = session.posterior.copy()
            await bridge.reply("conversation", transcript)
            np.testing.assert_array_equal(before, session.posterior)
            self.assertEqual(len(session.asked), 1)
            send.assert_not_awaited()
            self.assertEqual(manager.snapshot(session)["channel"], "voice")
            self.assertIsNone(manager.find_by_phone("+15555550100"))

    async def test_floor_question_and_spoken_correction_use_same_locator(self):
        with (patch.object(sessions.llm, "enabled", return_value=False),
              patch.object(semantics, "scores", return_value={}),
              patch.object(sessions.embeddings, "embed_texts", return_value=np.zeros((1, 3))),
              patch.object(sessions.messaging, "send", new_callable=AsyncMock) as send):
            bridge, session, manager = await self.prepare(make_two_floor_model)
            messages = []
            async def answer(text):
                messages.append({"role": "user", "message": text})
                return await bridge.reply("conversation", json.dumps({"entries": messages}))
            first = await answer("I can't tell what room this is")
            self.assertEqual(session.pending.key, "floor")
            self.assertIn("floor", first["say"].lower())
            await answer("upstairs")
            self.assertEqual(manager.snapshot(session)["floors"][1]["probability"] > 0.8, True)
            await answer("Actually I'm downstairs in the kitchen")
            estimate = manager.snapshot(session)["estimate"]
            self.assertEqual(estimate["floor"], 0)
            self.assertEqual(estimate["room_name"], "Kitchen")
            self.assertEqual(session.status, "located")
            self.assertNotIn("help is on the way", session.messages[-1]["text"])
            send.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
