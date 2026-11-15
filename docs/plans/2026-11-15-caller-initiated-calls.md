# Plan: caller-initiated calls, built in parallel by subagents

Base: `lifeline-integration` at f94c194 ("Merge improved caller logic and unify LifeLine dashboard"), which contains
every other branch. Baseline, checked Nov 14 23:35: 309 backend tests pass offline (`test_semantics.py` skipped),
the voice gateway test passes, the dashboard builds.

Deadline: Devpost hard stop **Sun Nov 15, 8:00 AM ET**; code after 8:00 AM is outside the allowed build window.
Judging 9:30 AM to 2 PM.

## How to use this plan

1. **Phase 0 (one agent, about 20 min, sequential).** Write the shared contracts below as working stubs on a branch
   `e2e-contracts`, with the suite still green. Everything later builds on this commit.
2. **Phase 1 (five agents in parallel, about 1.5 h each).** Each agent gets one brief (A to E), works in its own git
   worktree on its own branch from `e2e-contracts`, and edits **only the files it owns**. Shared files have a single
   owner; other agents read them or call the contract, never edit them.
3. **Phase 2 (one agent, about 45 min).** Merge in the order A, C, D, B, E, run all checks, fix integration bugs.
4. **Phase 3 (people, about 1 h).** Live text and voice checks, backup video, Devpost.

Rules for every agent:
- Read `CLAUDE.MD` first. Never run with `MESSAGING_MODE=photon`; tests use `tests/fakes.py` (fake CLIP, synthetic
  buildings, no network). Use made-up `+1 555` numbers. Never print `.env` values.
- Do not change a contract's names or shapes. If one is wrong, stop and report it instead of working around it.
- Put tests in your own new test file (named in your brief) so test files don't conflict.
- Done means: your tests pass, the full backend suite passes, `git diff --stat` shows only your owned files, and a
  short report lists what you did and anything left undone.
- Leave committing to your branch's final commit; the integrator merges. Don't push.

## Goal

A call starts when the caller makes contact, not when someone fills in a form, and the screen serves the search and
rescue team that has to reach the caller.

1. The caller's number comes from the channel (caller ID for voice, sender handle for iMessage).
2. The system asks the caller for the building address, matches it to a Matterport model (clarifying once when it
   is close to one), and continues with the existing room-locating conversation.
3. An address outside the Matterport coverage gets: "That address isn't in this system's service area. I'm
   connecting you to standard emergency services now. Stay on the line." The transfer is a stub that records what
   would happen, so it can be shown to judges, and connects nothing.
4. The responder screen shows each new call by itself and leads with where to go. No operator form; a hidden test
   mode keeps a way to simulate calls.

Decisions already made (Nov 14 to 15): no "this is a demo" wording anywhere; `usaddress` (trained US parser) for
caller addresses; voice keeps collecting the address in the ElevenLabs agent, the backend handles retries and
clarification; one retry before the out-of-area handoff; OUT and danger checks stay first in the reply flow.

**Hanging up needs no new work.** It is already handled: the ElevenLabs post-call webhook (`/api/voice/post-call`,
signed with `ELEVENLABS_POST_CALL_SECRET`) ends the session; a voice interview also ends after 240 s
(`voice.py`, `call_later(240, ...)`); and the agent's `end_call` tool hangs up. Phase 3 only needs to confirm the
webhook is configured in ElevenLabs. Note: the 240 s cap ends the backend interview even if the caller is still
talking; raise it if live calls run long.

## Call flow after this change

```
caller texts the line / calls the number
  └─ "awaiting_address": "911: If you can get out safely, get out now and text OUT.
                         If you can't, what is the address of the building you're in?"
       (OUT and danger checks run first on every message, as today)
       caller gives an address → resolve.resolve(text)
         ├─ "match"   exact, score ≥ 0.80              → load building
         ├─ "clarify" one model ≥ 0.65, 0.10 ahead     → "Did you mean <address>?"  yes → load, no → ask again
         ├─ "choose"  two models within 0.10           → "Is it <a> or <b>?"         pick → load, neither → ask again
         └─ "none"                                      → ask again once, then out of service area
       out of service area → handoff stub → status "transferred" (voice then ends the call)
  └─ existing flow: loading → listening → located | best_guess  (untouched)
```

Voice: the agent asks for the address and calls `start_location` as today; that address goes through the same
`resolve` step, and any retry or clarification answer arrives through `submit_location_answer` into the intake.

## Phase 0: contracts (one agent, sequential)

Create these as **working stubs** that keep today's behaviour, so every Phase 1 agent can code and test against
them at once. Add tests only for the stubs' basic shape.

1. **`backend/app/resolve.py`** (owner later: A)
   ```python
   @dataclass
   class Resolution:
       band: Literal["match", "clarify", "choose", "none"]
       candidates: list[dict]   # rank_matches entries {id, name, address, score, exact}, best first; 1 for
                                # match/clarify, 2 for choose, [] for none
       cleaned: str             # the address text after cleaning, for logs and the responder screen
       distinguishing: list[str] = field(default_factory=list)  # for "choose": the parts that differ, e.g.
                                # ["13120", "13122"] or ["unit 1013", "unit 107"]
   def resolve(text: str, exclude: set[str] = frozenset()) -> Resolution  # sync; callers run it in a thread;
                                # exclude = model ids the caller already rejected
   ```
   Stub: call `sessions.find_matches` (move it here, or import lazily to avoid a cycle), "match" if the top one is
   exact, "clarify" if non-exact, "none" if empty.
2. **`backend/app/phones.py`** (owner: D): `def to_e164(raw: str | None) -> str | None`. Stub: today's
   `normalize_phone` logic, returning `None` for empty input.
3. **`backend/app/handoff.py`** (owner: B): `def transfer_to_emergency_services(session, reason: str) -> dict`.
   Stub: sets `session.status = "transferred"` and `session.transfer` (below), returns it.
4. **`backend/app/questions.py` additions** (owner: C). Draft wording is fine; C finalizes it:
   `ADDRESS_QUESTION`, `ADDRESS_RETRY`, `ADDRESS_CONFIRM` (`{address}`), `ADDRESS_CHOOSE` (`{a}`, `{b}`),
   `OUT_OF_AREA_MESSAGE`, `OUT_CONFIRM` ("Are you outside the building now? Yes or no.");
   `def for_channel(text: str, channel: str) -> str` (identity stub; C makes voice versions);
   `def out_signal(text: str) -> Literal["out", "maybe", None]` (stub: `"out" if is_out(text) else None`);
   `def looks_like_address(text: str) -> bool` (stub: a digit followed by a word).
5. **Session and snapshot fields** (owner: B), added to `Session` and `snapshot()` with defaults so nothing breaks:
   - statuses `awaiting_address` and `transferred` (existing: loading, listening, located, best_guess, out,
     no_model, error);
   - `address_attempts: int = 0`, `rejected_models: set[str]`, `out_pending: bool = False`;
   - snapshot adds `"transfer": None | {"reason": str, "ts": float, "target": "standard 911 (PSAP)", "stub": true}`,
     `"phone": str | None` (None means unknown caller), `"channel": "messaging" | "voice"` (exists),
     `"address": ""` while awaiting.
6. **`SessionManager.open_call(phone: str | None, channel: str, first_text: str | None = None) -> Session`**
   (owner: B). Stub: raises `NotImplementedError` behind a test that is marked xfail.
7. **API contract** (owner: B): `POST /api/inbound {from, text, event_id?}` returns `{ok, session, created}`;
   unknown numbers will open a call. Voice bridge responses gain `"ended": bool` (owner: D).
8. **`config.py`**: add one labelled block per workstream (`# ---- A: address matching`, `# ---- B: intake`, ...)
   with the constants named in the briefs, so agents only edit inside their own block and merges stay trivial.
9. **`backend/requirements.txt`**: add `usaddress`, `text2num`, `phonenumbers`, `python-dotenv` (and optionally
   `rapidfuzz`) now, so no Phase 1 agent edits it.

Commit as `e2e-contracts`; all Phase 1 branches start from it.

## Phase 1 briefs

### Agent A: address matching
**Owns:** `backend/app/addresses.py`, `resolve.py`, `matterport.py` (`rank_matches` only), `catalog.py`,
config block A, `backend/tests/test_resolve.py`. **Reads:** `sessions.find_matches`, `test_addresses.py`.

Today: hand-written parser and scorer (street 0.45, house number 0.30, city/state/ZIP 0.25), accept at 0.80.
Messy inputs checked on f94c194:

| Input | Score | Today | Should become |
|---|---|---|---|
| `13122 Chapman Glen Dr` | 0.96 | match (approximate) | clarify |
| `7921 Poynter St Houston` | 0.91 | match (approximate) | clarify |
| `Chipman Glen Drive, Houston` (no number) | 0.96 | match (approximate) | clarify |
| `13212 Chipman Glen Dr Houston` (digits swapped) | 0.79 | miss | clarify |
| `thirteen one twenty two chipman glen drive houston` | 0.75 | miss | match |
| `seventy nine twenty one pointer street` | 0.60 | miss | match |
| `um it's 13122 chipman glen in houston texas` | 0.21 | miss | match |

Work:
1. `addresses.parse_caller(text) -> Address`: `text2num.alpha2digit(text, "en")`; join digit groups said before the
   street into one house number ("13 1 22" → 13122, "79 21" → 7921); start at the first digit (drops "um it's",
   "I'm at"); `usaddress.tag`; map `AddressNumber`, `StreetNamePreDirectional`, `StreetName`, `StreetNamePostType`,
   `OccupancyIdentifier`, `PlaceName`, `StateName`, `ZipCode` onto the existing `Address`. On `RepeatedLabelError`
   or no street, fall back to `parse_freeform`. Use it in `similarity` for the query side; Matterport's side stays
   `parse_matterport`. (Checked: usaddress 0.5.16 is 156 KB, ~0.2 ms per address, tags units and directionals
   correctly, but alone fails on filler and grouped numbers, hence the first steps.)
2. `resolve()` bands with config `ADDRESS_CLARIFY_THRESHOLD = 0.65`, `ADDRESS_CLARIFY_MARGIN = 0.10` (existing
   `ADDRESS_MATCH_THRESHOLD = 0.80`); `distinguishing` for "choose" (house numbers or units that differ); honour
   `exclude`.
3. Candidates: the local catalog (`catalog.search`, ~15,000 models) plus the cache plus Matterport search, deduped.
4. Optional: `rapidfuzz.fuzz.ratio / 100` instead of `difflib.SequenceMatcher`; re-run `test_addresses.py`.
5. Tests: the table above, parametrized, asserting the "Should become" band; existing `test_addresses.py` unchanged.

Not in scope: the Census Geocoder stretch (leave a `# TODO(geocode)` where it would go), libpostal.

### Agent B: call intake, handoff and text inbound
**Owns:** `backend/app/sessions.py`, `handoff.py`, `main.py`, config block B, `backend/tests/test_intake.py`.
**Calls:** `resolve.resolve`, `questions.*` contract names, `phones.to_e164`. **Reads:** `voice.py`.

Work:
1. `open_call(phone, channel, first_text)`: session in `awaiting_address`, phone via `to_e164` (None = unknown);
   send `ADDRESS_QUESTION`, or if `looks_like_address(first_text)` treat it as the address.
2. `_handle_reply` in `awaiting_address` (after the existing OUT and danger checks, before `waiting_for_match`):
   `resolve` in a thread. match → set `session.address`, start loading through one helper shared with
   `start_call` (keep `start_call`'s signature: the test drawer, `preflight_demo.py`, `eval_scenarios.py` and
   tests use it). clarify/choose → `waiting_for_match` with `ADDRESS_CONFIRM` / `ADDRESS_CHOOSE`, once. none →
   `ADDRESS_RETRY` while `address_attempts < ADDRESS_RETRIES` (config B: `ADDRESS_RETRIES = 1`), then handoff.
3. `waiting_for_match` "no" (or neither option): add to `rejected_models`, back to `awaiting_address` while
   attempts remain, then handoff. Replace today's "Please start again with the complete address" dead end.
4. `_load`'s `no_model` exit (no match, or all models unusable) → handoff, with `OUT_OF_AREA_MESSAGE` instead of
   `NO_MODEL_MESSAGE`.
5. `handoff.transfer_to_emergency_services`: no network. Log it, append a system message ("Transferred to standard
   911: address not in coverage. Stub: no real transfer in this build."), set `transferred` and `transfer`, clear
   `pending` and `queued`. Docstring: where a real build connects (voice: ElevenLabs call transfer or carrier
   routing to the local 911 center; text: Text-to-911). Later texts are recorded only, like `out`.
6. OUT confirmation: when `out_signal` returns `"maybe"`, send `OUT_CONFIRM` and set `out_pending`; "yes" → today's
   out flow, "no" → continue. `"out"` (the bare keyword) acts at once as today.
7. `/api/inbound`: no open session for the sender (none, ended, or older than `SESSION_REUSE_MINUTES = 30`,
   config B) → `open_call(sender, "messaging", text)`; return `{ok, session, created}`. `find_by_phone` uses
   `to_e164` and skips ended sessions; never matches a `None` phone.
8. `_say` passes every message through `questions.for_channel(text, session.channel)` before sending, so voice
   sessions get the spoken versions everywhere.
9. Tests (`test_intake.py`, using `fakes.install` and a fake `resolve`): unknown number opens `awaiting_address`;
   address-first text skips the question; match loads and the next text is a room description; none twice →
   `transferred` with `stub: true`, no network; clarify yes/no; choose; two rejections → transferred; texts after
   transfer get no questions; same number reuses the session, ended one opens a new one; "OUT" first text never
   asks for an address; smoke before the address sends advice and still asks; "maybe" OUT asks to confirm.

### Agent C: caller language
**Owns:** `backend/app/questions.py`, `semantics.py`, config block C, `backend/tests/test_caller_language.py`.
**Reads:** `followups.py`, `sessions.py`.

Work:
1. Final wording for the Phase 0 constants, same tone as today's messages, no demo wording. Text first message:
   "911: If you can get out safely, get out now and text OUT. If you can't, what is the address of the building
   you're in?" `OUT_OF_AREA_MESSAGE` as in the Goal.
2. `for_channel`: a table of spoken versions for voice (opening, address messages, located, best guess, refined,
   out, out-of-area, safety): "text OUT" → "say you're out", "texting" → "on this call", "text us" → "tell me".
   Keyed by the constants, not by string replacement, with a test that every constant containing "text" has a
   voice version.
3. `out_signal`: bare "OUT" → "out"; "I'm out", "I got out", "we're out", "out of the building", "outside now" →
   "out"; "outside the bathroom/door/room", "outside of my room" → "maybe" or None (today `is_out("I'm outside the
   bathroom door")` is True, which would end a trapped caller's call); "can't get out", "get me out" → None. Keep
   `is_out` working for existing callers.
4. `parse_reply`: "absolutely not", "definitely not", "of course not", "not at all" → no (today "Absolutely not"
   is yes).
5. Danger: add heat ("it's getting hot", "the door is warm/hot", "orange light", "glowing") to `_DANGER` and a
   "danger" entry to `semantics.EXAMPLES`; keep "the fireplace" and "no smoke here" negative.
6. `looks_like_address`: a house number (digits, or spoken number words) followed by a street word; use
   `addresses.parse_caller` if Agent A's function exists at integration time, otherwise a regex (note it in your
   report).
7. Optional if time remains: a MiniLM second opinion for yes/no/unsure/out/danger (the model is already loaded,
   ~3 ms per message), used only when the regexes return "free".
8. Tests: every probe above, plus the existing `test_parsing.py` unchanged.

### Agent D: voice and platform
**Owns:** `backend/app/voice.py`, `voice_routes.py`, `phones.py`, `config.py` loader and block D, `voice/setup.mjs`,
`voice/gateway.mjs`, `voice/README.md`, `backend/tests/test_voice_e2e.py`, `backend/tests/test_phones.py`.
**Calls:** `SessionManager` methods, `questions.for_channel`.

Work:
1. `phones.to_e164` with `phonenumbers` (region "US"): E.164 or None for empty, "anonymous", "unknown",
   "restricted", invalid. Use it for `system__caller_id` in `VoiceBridge.start` (session phone None when withheld;
   the conversation id stays the key).
2. `VoiceBridge.start` passes the address through the intake instead of straight to `start_call`: call
   `open_call(phone, "voice", address)` (B's contract). No 409 when the session is `awaiting_address` or
   `transferred`.
3. Responses: `"ended": true` when the session is `transferred`, `out` and finished, or ended; instruction "Say the
   text exactly, then end the call." Voice text goes through `for_channel` (B's `_say` applies it; check it does).
4. Agent prompt in `setup.mjs`: pass house numbers as digits; if the backend asks for the address again or asks
   "Did you mean...", relay the caller's answer with `submit_location_answer`; after a response with `ended`, say it
   and use `end_call`. Replace `appendFileSync` with a helper that rewrites an existing `NAME=` line in place.
5. `config.py`: load `.env` with `python-dotenv` (`dotenv_values`, last value wins, real environment variables
   still win), so a blank `VOICE_TOOL_TOKEN=` above the real one no longer shadows it.
6. Tests: `test_phones.py` (formats, withheld, invalid); `test_voice_e2e.py` with `fakes.install`: start with an
   unknown address → retry asked → second miss → `ended: true` with the out-of-area text; clarify over voice;
   withheld caller ID; `.env` duplicate key. Tests that need B's intake (`open_call`) are marked
   `xfail(strict=False)` until Phase 2, where the integrator removes the marks. Keep `test_voice_bridge.py`, `test_voice_pipeline.py`,
   `test_voice_routes.py` and `voice/gateway.test.mjs` passing.

### Agent E: responder screen
**Owns:** everything under `dashboard/src/`, `dashboard/index.html`. **Reads:** the snapshot contract above,
`backend/app/main.py` routes. No backend edits.

The viewer is a search and rescue team member or incident commander on a laptop or tablet. Nobody types into it
during a real call.

```
+----------------------------------------------------------------------------+
| LifeLine                                          2 active calls    Live   |
+-------------------+--------------------------------------------------------+
| CALLS             |  GO TO  Floor 2 . Bedroom 2            74%  Likely      |
| > Text +1 678..08 |         rear (north) side, away from Pointer St         |
|   7921 Pointer St |         Search order: Bedroom 2, then Bedroom 3         |
|   Locating  2 min |  [!] Caller reported smoke                              |
|   Voice unknown   +-------------------------------+------------------------+
|   Getting address |  floor plan / map, approach   |  CALLER SAID (live,    |
|                   |  card, room photos            |  read-only)            |
+-------------------+-------------------------------+------------------------+
```

Work:
1. Remove the "New call" form and the reply box from the normal view. Behind `?test=1`, a "Test tools" drawer:
   "Simulate incoming text" (phone + message → `POST /api/inbound`), "Simulate a call with address" (the old
   form → `/api/calls`), and the reply box.
2. Call list: appears by itself; Voice / Text badge; number or "Unknown caller" (`phone: null`); address or
   "Address pending"; status; minutes since start. `out`, danger and `transferred` calls marked and sorted first.
3. Headline: "Go to" floor, room, probability, confidence word, approach side, search order on a best guess or tie,
   danger note. `awaiting_address`: "Getting the address from the caller". `out`: "Caller is out of the building"
   plus their latest text. `transferred`: "Outside service area: handed off to standard 911", with the stub note
   visible.
4. Conversation panel: read-only "Caller said" transcript, keeping each message's evidence line.
5. Auto-select a new call when nobody picked one in the last minute, with a short "New call" alert.
6. `STATUS_LABELS`: `awaiting_address: "Getting address"`, `transferred: "Transferred to 911"`. Remove
   "dispatcher", "operator", "simulated" and "demo" from visible text.
7. Keep the plan, map, floor tabs, approach card and photos; only order and headline change.
8. Check: `npm run build`; run against the Phase 0 backend with `?test=1` and simulate a text call; take a
   headless screenshot (command in `CLAUDE.MD`) for each status: awaiting_address, listening, located, out,
   transferred.

## Phase 2: integration (one agent)

1. Merge into `e2e-contracts`, in order: A (matching), C (language), D (voice and platform), B (intake), E (screen).
   B goes after A, C and D because it calls all three; conflicts should be limited to `config.py` blocks.
2. Replace stubs that are still in place, and remove the xfail on `open_call`.
3. Run: full backend suite; `voice/` `npm test`; `dashboard` `npm run build`; `scripts/eval_scenarios.py` (needs
   the three cached buildings; compare with 15/19 and no confidently wrong rooms); one simulated text call and one
   simulated voice conversation end to end through `?test=1` and `tests/fakes.py`.
4. Docs: README status row for caller-initiated calls, the new statuses, `/api/inbound` behaviour (note: it is
   unauthenticated, keep the backend on localhost and off the tunnel), the walkthrough ("text the line"), and a
   short "address matching" note (usaddress, clarification bands, best practice and what is future work). One line
   in `CLAUDE.MD` under "Decisions already made".
5. Merge into `lifeline-integration` only after everything passes.

## Phase 3: live checks (people)

- Text: a phone assigned to Photon's shared line texts "help", gets the address question, replies
  `13122 Chipman Glen Dr, Houston, TX`, and the room conversation follows (backend in `photon` mode, sidecar up).
  Also try an address outside coverage and see the handoff on the screen.
- Voice: needs an eligible Photon number with SIP to `sips:sip.rtc.elevenlabs.io:5061` (see `voice/README.md`), the
  gateway, the tunnel, `npm run setup` with the new prompt, and the post-call webhook configured. If no
  voice-capable number exists by about 3 AM, show voice through the ElevenLabs talk page and tell judges so.
- Record a 2-minute backup video of whichever channel works.
- Submit to Devpost before 8:00 AM.

## Future work (README, not tonight)

- Census Geocoder (free, no key) to standardize addresses and break ties by location; libpostal if non-US
  addresses matter.
- `shapely` for the hand-written geometry (polygon chaining, point-in-polygon, shared walls, hull).
- Calibrating the hand-tuned CLIP and confidence constants on a labelled set.
- Claude reader live (built, never run; adds 1 to 3 s per turn).
- A real handoff to emergency services; authentication on `/api/inbound`; Photon webhooks instead of polling.

## Timing (wall clock, starting about 00:30)

| Phase | Who | Time | Ends about |
|---|---|---|---|
| 0 Contracts | 1 agent | 20 min | 00:50 |
| 1 Workstreams A to E | 5 agents in parallel | 1.5 h | 02:20 |
| 2 Integration and docs | 1 agent | 45 min | 03:05 |
| 3 Live checks, video, Devpost | people | 1 h | 04:05 |

That leaves about 4 hours of slack before 8:00 AM for fixes. If something slips, cut in this order: Agent C item 7,
Agent A item 4, Agent E item 5, the "choose" band (treat it as clarify with the top model).
