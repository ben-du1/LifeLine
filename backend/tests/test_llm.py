"""The Claude layer with a stub client: request shape, parsing into cues, and every fallback path."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np

from app import followups, llm
from tests.test_followups import make_model, rooms_after


class StubMessages:
    def __init__(self, payload=None, stop_reason="end_turn", error=None):
        self.payload, self.stop_reason, self.error, self.requests = payload, stop_reason, error, []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                               content=[SimpleNamespace(type="text", text=text)])


def use_stub(monkeypatch, **kwargs) -> StubMessages:
    stub = StubMessages(**kwargs)
    monkeypatch.setattr(llm, "_client", SimpleNamespace(beta=SimpleNamespace(messages=stub)))
    return stub


READ_REPLY = {"answer": "not_answered", "options": [], "mentioned": [{"feature": "shelves", "present": True}],
              "size": "tight", "dark": False, "room_types": [], "floors": [], "acknowledgement": "You can touch both walls.",
              "explained": False}


def test_paraphrase_becomes_cues_and_evidence(monkeypatch):
    stub = use_stub(monkeypatch, payload=READ_REPLY)
    cues = llm.read_message("I can touch both walls, there are shelves", None, [])
    assert cues.size == "tight" and cues.mentioned == {"shelves": "yes"}
    assert cues.acknowledgement == "You can touch both walls."
    request = stub.requests[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in request["betas"]
    # The same cues drive the locator: a tight space with shelves is the closet.
    building, model = make_model()
    reading = followups.read_message("", model, None, "free", cues)
    assert reading.covered >= {"size", "shelves"}
    probs, _ = rooms_after(model, building, "I can touch both walls")  # keyword rules alone miss this...
    assert np.allclose(probs, 0.25)
    post = reading.likelihood / reading.likelihood.sum()
    assert post.argmax() == 0                                          # ...Claude's reading finds the closet


def test_choice_answers_are_limited_to_the_pending_options(monkeypatch):
    use_stub(monkeypatch, payload={**READ_REPLY, "options": ["tile", "large"], "mentioned": [], "size": "unknown"})
    plan = followups.Plan("choice", "flooring", "Floor?", options=followups.CHOICES_BY_KEY["flooring"].options)
    cues = llm.read_message("it's tiled", plan, [])
    assert cues.options == ("tile",) and cues.size is None


def test_failures_fall_back_to_none(monkeypatch):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    for kwargs in ({"error": anthropic.APIConnectionError(request=request)},
                   {"payload": READ_REPLY, "stop_reason": "refusal"},
                   {"payload": "not json"},
                   {"payload": {"answer": "yes"}}):                    # schema mismatch
        use_stub(monkeypatch, **kwargs)
        assert llm.read_message("hello", None, []) is None


def test_phrased_question_must_keep_the_options(monkeypatch):
    plan = followups.Plan("choice", "flooring", "What is the floor like: carpet, tile, or wood?",
                          options=followups.CHOICES_BY_KEY["flooring"].options)
    use_stub(monkeypatch, payload={"question": "Okay, it's tight. Is the floor carpet, tile, or wood?"})
    assert llm.phrase_question(plan, [], "") == "Okay, it's tight. Is the floor carpet, tile, or wood?"
    use_stub(monkeypatch, payload={"question": "Is the floor soft?"})   # dropped the options -> template
    assert llm.phrase_question(plan, [], "") is None
    use_stub(monkeypatch, payload={"question": "Tell me about the floor."})  # not a question
    assert llm.phrase_question(plan, [], "") is None


def test_disabled_without_key(monkeypatch):
    monkeypatch.setattr(llm.config, "ANTHROPIC_API_KEY", "")
    assert not llm.enabled()
