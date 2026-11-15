"""Transport tests use a fake session manager: no model downloads, SMS, or real calls."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.voice import VoiceBridge, VoiceError, latest_reply


def history(*messages):
    return json.dumps({"entries": [{"role": "user", "message": text} for text in messages]})


class VoiceBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = SimpleNamespace(
            id="s1", address="1 Main St", status="listening", pending=None, ready=asyncio.Event(),
            messages=[{"role": "dispatch", "text": "Describe the room."}])
        self.session.ready.set()

        async def handle(session, text):
            session.messages += [{"role": "caller", "text": text},
                                 {"role": "dispatch", "text": "Are there shelves?"}]
        self.manager = SimpleNamespace(start_call=AsyncMock(return_value=self.session),
                                       handle_reply=AsyncMock(side_effect=handle))
        self.bridge = VoiceBridge(self.manager, wait_seconds=0.01)

    async def test_start_is_idempotent_and_uses_voice_transport(self):
        await self.bridge.start("c1", "1 Main St", "+15555550100")
        await self.bridge.start("c1", "1 Main St", "+15555550100")
        self.manager.start_call.assert_awaited_once_with("+15555550100", "1 Main St", channel="voice")
        with self.assertRaises(VoiceError):
            await self.bridge.start("c1", "2 Other St")

    async def test_exact_answers_retries_and_repeated_yes(self):
        await self.bridge.start("c1", "1 Main St")
        transcript = history("1 Main St", "yes, actually upstairs")
        result = await self.bridge.reply("c1", transcript)
        self.assertEqual(result["say"], "Are there shelves?")
        self.manager.handle_reply.assert_awaited_once_with(self.session, "yes, actually upstairs")
        self.assertEqual(await self.bridge.reply("c1", transcript), result)
        self.assertEqual(self.manager.handle_reply.await_count, 1)
        await self.bridge.reply("c1", history("1 Main St", "yes, actually upstairs", "yes"))
        self.assertEqual(self.manager.handle_reply.await_count, 2)

    async def test_loading_polls_without_resubmitting_answer(self):
        await self.bridge.start("c1", "1 Main St")
        self.session.ready.clear()
        self.session.status = "loading"
        result = await self.bridge.reply("c1", history("I see a bed"))
        self.assertTrue(result["pending"])
        self.session.ready.set()
        self.session.status = "listening"
        result = await self.bridge.status("c1")
        self.assertFalse(result["pending"])
        self.assertEqual(result["say"], "Are there shelves?")
        self.assertEqual(self.manager.handle_reply.await_count, 1)
        self.assertNotEqual((await self.bridge.status("c1"))["say"], "Are there shelves?")

    async def test_cursor_replays_output_until_agent_acknowledges(self):
        await self.bridge.start("c1", "1 Main St")
        response = await self.bridge.reply("c1", history("I see a bed"), acknowledge=0)
        self.assertEqual(response, await self.bridge.status("c1", acknowledge=0))
        after = await self.bridge.status("c1", acknowledge=response["response_id"])
        self.assertNotEqual(after["say"], response["say"])
        self.assertEqual(self.manager.handle_reply.await_count, 1)

    async def test_conversations_do_not_share_state_and_older_turns_rejected(self):
        await self.bridge.start("c1", "1 Main St")
        with self.assertRaises(VoiceError):
            await self.bridge.reply("other", history("yes"))
        await self.bridge.reply("c1", history("address", "yes"))
        with self.assertRaises(VoiceError):
            await self.bridge.reply("c1", history("changed"))
        with self.assertRaises(VoiceError):
            await self.bridge.reply("c1", history("address", "different"))

    def test_history_and_demo_wording(self):
        for invalid in ("not json", "{}", '{"entries": [4]}', history()):
            with self.assertRaises(VoiceError):
                latest_reply(invalid)
        self.assertEqual(latest_reply(history("yes", "no")), (2, "no"))


if __name__ == "__main__":
    unittest.main()
