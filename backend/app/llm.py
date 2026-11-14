"""Optional Claude layer for the follow-up conversation. Enabled only when ANTHROPIC_API_KEY is set.

Two jobs, each one call to the Messages API with structured JSON output:
  read_message     the caller's message -> followups.Cues. Understands paraphrase the keyword rules miss
                   ("I can touch both walls" -> size "tight"; "the thing you wash clothes in" -> washer).
  phrase_question  the topic the planner chose -> one short question built around the caller's own words.

What to ask is still decided by followups.choose (expected information gain over the building's rooms); Claude
only reads and words. Every failure (no key, timeout, API error, refusal, malformed output) returns None, and the
caller falls back to the keyword rules and templates, so a call never stalls on the language model.

UNVERIFIED against the live API: no credentials were available while this was written. tests/test_llm.py covers
the request shape and the parsing with a stub client.
"""

import json
import logging

from . import config, followups

log = logging.getLogger(__name__)

ROOM_TYPES = ["bathroom", "bedroom", "kitchen", "living room", "dining room", "hallway", "garage", "office",
              "closet", "laundry room", "entryway"]

# Stable system prompt (kept byte-identical across calls so it can be cached).
SYSTEM = (
    "You assist a 911 dispatcher. A caller is trapped or hiding inside a building and texts short messages. "
    "Responders know the address and use your output to work out which room the caller is in. Report only what "
    "the caller actually says; never guess or add details. Messages may be terse, misspelled or panicked."
)

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no", "none", "unsure", "not_answered"],
                   "description": "The caller's reply to the pending yes/no or 'which of these' question."},
        "options": {"type": "array", "items": {"type": "string",
                    "enum": [o.key for c in followups.CHOICES for o in c.options]
                    + [f"floor{i}" for i in range(12)]},
                    "description": "Option keys the caller chose, if the pending question was a choice."},
        "mentioned": {"type": "array", "items": {
            "type": "object",
            "properties": {"feature": {"type": "string", "enum": [f.key for f in followups.FEATURES]},
                           "present": {"type": "boolean"}},
            "required": ["feature", "present"], "additionalProperties": False}},
        "size": {"type": "string", "enum": ["tight", "small", "medium", "large", "unknown"]},
        "dark": {"type": "boolean"},
        "room_types": {"type": "array", "items": {"type": "string", "enum": ROOM_TYPES}},
        "floors": {"type": "array", "items": {"type": "integer"},
                   "description": "Floor numbers the caller says they are on, 1 = ground floor. "
                                  "'Upstairs' in a two-storey building is 2. Empty if not said."},
        "acknowledgement": {"type": "string",
                            "description": "Up to 12 words restating the most useful thing the caller said, "
                                           "addressed to them, e.g. \"You can touch both walls.\" Empty if none."},
        "explained": {"type": "boolean",
                      "description": "True if the message only answers the pending question and adds nothing."},
    },
    "required": ["answer", "options", "mentioned", "size", "dark", "room_types", "floors", "acknowledgement",
                 "explained"],
    "additionalProperties": False,
}

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}

_client = None


def enabled() -> bool:
    return bool(config.ANTHROPIC_API_KEY)


def _get_client():
    global _client
    if _client is None:
        import anthropic  # imported lazily: only needed when the feature is on

        # One retry at most: a caller is waiting, and the keyword fallback is always available.
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=config.LLM_TIMEOUT_SECONDS,
                                      max_retries=1)
    return _client


def _call(prompt: str, schema: dict) -> dict | None:
    """One structured-output request; None on any failure."""
    import anthropic

    try:
        response = _get_client().beta.messages.create(
            model=config.LLM_MODEL,
            max_tokens=4000,  # covers low-effort thinking plus a small JSON object
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            # Low effort: these are small reading/wording tasks and latency matters more than depth here.
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            # A safety classifier can decline an emergency-themed request; fall back server-side instead of failing.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError as exc:
        log.warning("Claude rate limited; using keyword rules: %s", exc)
        return None
    except anthropic.APIStatusError as exc:
        log.warning("Claude API error %s; using keyword rules: %s", exc.status_code, exc)
        return None
    except anthropic.APIConnectionError as exc:  # includes timeouts
        log.warning("Claude unreachable; using keyword rules: %s", exc)
        return None
    if response.stop_reason == "refusal":
        log.warning("Claude declined (%s); using keyword rules", getattr(response.stop_details, "category", None))
        return None
    text = next((block.text for block in response.content if block.type == "text"), None)
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        log.warning("Claude returned invalid JSON; using keyword rules")
        return None


def _conversation(messages: list[dict], limit: int = 8) -> str:
    turns = [m for m in messages if m["role"] in ("dispatch", "caller")][-limit:]
    return "\n".join(f"{'Dispatcher' if m['role'] == 'dispatch' else 'Caller'}: {m['text']}" for m in turns)


def _pending_description(pending: followups.Plan | None) -> str:
    if pending is None:
        return "No question is pending; this is a free description."
    if pending.kind == "choice":
        options = ", ".join(f"{o.key} ({o.label})" for o in pending.options)
        return f"Pending choice question: \"{pending.text}\" Options: {options}."
    if pending.kind == "list":
        return (f"Pending 'which of these' question: \"{pending.text}\" "
                f"Items: {', '.join(pending.items)}. Report items the caller sees under 'mentioned'.")
    return f"Pending yes/no question about '{pending.key}': \"{pending.text}\""


def read_message(text: str, pending: followups.Plan | None, messages: list[dict],
                 n_floors: int = 1) -> followups.Cues | None:
    features = "\n".join(f"- {f.key}: {f.noun}" for f in followups.FEATURES)
    prompt = (
        f"Conversation so far:\n{_conversation(messages)}\n\n"
        f"{_pending_description(pending)}\n\n"
        f"Features you may report (key: meaning):\n{features}\n\n"
        "Size: 'tight' = cramped, barely room to move, can touch both walls; 'small' = a small room; "
        "'medium' = about bedroom-sized; 'large' = big or open; 'unknown' if not said.\n\n"
        f"The building has {n_floors} floor(s).\n\n"
        f"Read the caller's latest message:\n\"{text}\""
    )
    data = _call(prompt, READ_SCHEMA)
    if data is None:
        return None
    try:
        allowed = {o.key for o in pending.options} if pending is not None and pending.kind == "choice" else set()
        answer = data["answer"] if data["answer"] != "not_answered" else None
        return followups.Cues(
            answer=answer if pending is not None else None,
            options=tuple(k for k in data["options"] if k in allowed),
            mentioned={m["feature"]: "yes" if m["present"] else "no" for m in data["mentioned"]
                       if m["feature"] in followups.FEATURES_BY_KEY},
            size=None if data["size"] == "unknown" else data["size"],
            dark=bool(data["dark"]),
            room_types=tuple(t for t in data["room_types"] if t in ROOM_TYPES),
            floors=tuple(n - 1 for n in data["floors"] if isinstance(n, int) and 1 <= n <= n_floors)
            if n_floors > 1 else (),
            acknowledgement=data["acknowledgement"].strip()[:120],
            explained=bool(data["explained"]),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        log.warning("Claude output did not match the schema (%s); using keyword rules", exc)
        return None


def _required_words(plan: followups.Plan) -> list[str]:
    """Words the question must contain so the caller's answer still maps onto the planned topic."""
    if plan.kind == "choice":
        return [o.label.split("-")[0].split()[0] for o in plan.options]
    if plan.kind == "list":
        return [followups.FEATURES_BY_KEY[k].noun.removeprefix("a ").removeprefix("the ").split()[0].strip(",")
                for k in plan.items]
    return []


def phrase_question(plan: followups.Plan, messages: list[dict], acknowledgement: str) -> str | None:
    """Word the planned follow-up around what the caller said. None (use the template) if the result is unusable."""
    if plan.kind == "yesno":
        topic = f"a yes/no question: whether there is {followups.FEATURES_BY_KEY[plan.key].noun}"
    elif plan.kind == "choice":
        labels = ", ".join(o.label for o in plan.options)
        topic = f"which of these describes the room ({plan.key}): {labels}. Offer exactly these options"
    else:
        nouns = ", ".join(followups.FEATURES_BY_KEY[k].noun for k in plan.items)
        topic = f"which of these they can see: {nouns}. Name all of them"
    prompt = (
        f"Conversation so far:\n{_conversation(messages)}\n\n"
        f"Write the dispatcher's next text message. It must ask {topic}. Briefly acknowledge what the caller just "
        f"said if useful (a suggested acknowledgement: \"{acknowledgement}\"). One or two short sentences, at most "
        "30 words, calm and plain, no emoji, ending with a question mark. Do not ask about anything else."
    )
    data = _call(prompt, QUESTION_SCHEMA)
    question = (data or {}).get("question", "").strip()
    if not question.endswith("?") or len(question) > 240:
        return None
    if not all(word.lower() in question.lower() for word in _required_words(plan)):
        return None  # Claude dropped an option; the template keeps the answer parseable
    return question
