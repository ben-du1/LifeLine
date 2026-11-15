- Before editing, read the nearest working code. Match its naming, formatting, and error handling.
- Keep the diff focused. Reuse existing functions and add dependencies only when the feature needs them.
- Use straightforward control flow. Extract a helper when it makes the caller easier to read.
- Comment on hardware assumptions, timing constraints, and reasons a decision was made.
- Run the relevant check and report what you verified.
- Leave staging, committing, and pushing to me.

# Project context (for agents)

HackPrinceton Fall 2026 MVP (CoStar × Matterport track, Devpost hard stop Nov 15 2026, 8:00 AM ET). A 911 dispatcher enters a
caller's phone and address. The system finds the building's Matterport digital twin, texts the caller (Photon
iMessage, or a simulated caller box), asks follow-up questions, and puts a dot on the room the caller is in.
**README.md is the full spec and status page:** read its "Status" table first, and keep it current when behaviour
changes.

## Layout
- `backend/app/`: FastAPI (Python 3.11, CPU only). Start at `sessions.py`, which holds the call flow.
  - `followups.py`: cue reading and question planning (expected information gain).
  - `locator.py`: pure-numpy Bayesian update.
  - `ingest.py`: Matterport download into the `data/` cache.
  - `config.py`: every tunable value, each with a comment explaining it. Read that comment before changing a value.
- `backend/tests/`: pytest. `backend/scripts/eval_scenarios.py`: scripted callers, the accuracy check (15/19; every miss is a best guess, none a confident wrong room).
- `dashboard/`: React + Vite; `src/App.jsx` and `src/FloorPlan.jsx` (SVG plan, north-up, floor-aware).
- `messaging/`: Node sidecar for Photon's Beta REST API (no npm dependencies).
- `data/buildings/<model_id>/`: downloaded interiors and embeddings (gitignored; Matterport content is licensed
  "hackathon use only"). `building.json` has a `schema` field; bump `ingest.SCHEMA_VERSION` when its shape or the
  room filtering changes, and old caches re-ingest automatically.

## Commands (Windows, Git Bash)
- **Use absolute paths:** the shell's working directory does not reliably persist between commands, so a relative
  `.venv/Scripts/python` can fail ("No module named app" or exit 127). Use
  `/c/Users/ben/desktop/coding/hackprinecton-mvp/backend/.venv/Scripts/python` and `cd` into `backend/` in the same
  command.
- **Tests:** `cd backend && .venv/Scripts/python -m pytest -q` (about 490 tests; `tests/fakes.py` runs the real SessionManager offline with a fake CLIP).
- **Eval:** `cd backend && MESSAGING_MODE=simulated .venv/Scripts/python scripts/eval_scenarios.py -v`. It is slow
  the first time, because it ingests 3 buildings.
- **Backend:** `cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000`.
- **Dashboard:** `cd dashboard && npm run dev` (:5173, proxies to `BACKEND_URL`).
- **Sidecar:** `cd messaging && npm start` (:8787).
- **Verify UI changes with a screenshot:** `"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
  --headless=new --window-size=1500,1000 --virtual-time-budget=10000 --screenshot=<file> http://localhost:5173/`.
  The dashboard shows the newest call.

## Safety and secrets
- Keys live only in the repo-root `.env` (gitignored; `.env.example` lists the names). Never print key values.
  Mask phone numbers in output.
- `.env` sets `MESSAGING_MODE=photon`, so calls started on a backend that reads it **send real iMessages**. For
  testing, run with `MESSAGING_MODE=simulated` or use made-up `+1 555 …` numbers. Photon's shared line refuses
  numbers that never texted it.
- **The user often has their own servers on :8000, :5173 and :8787.** Check a process's command line before
  stopping it. Run test servers on other ports (e.g. :8001, :5174 with `BACKEND_URL=http://localhost:8001`).

## Decisions already made (don't redo without a reason)
- **Room matching:** CLIP (open_clip ViT-B-32) over 4 skybox faces per sweep. Room labels and floor area also feed
  the feature model, so rooms without photos can still be found.
- **Paraphrase:** `semantics.py` uses all-MiniLM-L6-v2. CLIP's text encoder was tested for this and rejected: it
  confuses opposites like small and large.
- **Optional LLM layer:** `llm.py` (Claude, `claude-opus-5-5`) is used only if `ANTHROPIC_API_KEY` is set. It reads
  replies and words questions, but never chooses them. It has never run live (no key yet).
- **Orientation:** the plan is north-up from Matterport's geocoordinates yaw, plus the nearest street from
  OpenStreetMap Nominatim.
  - **Vexcel overlay on the plan:** removed by the user's request, because the floor plan didn't line up exactly
    with the imagery. Vexcel is now used in the separate Map view (`geo.py` `/aerial`, `MapView.jsx`), with manual
    nudge controls for the few-meter offset; `VEXCEL_API_KEY` in `.env` turns it on.
  - **Overpass:** too unreliable; it timed out or reported "at capacity" on every attempt.
  - **CoStar:** not used; its extract has no roads or imagery.
- **Photon:** the project is on Photon Beta (`pho_prj_…`, `pho_sk_…`). The `spectrum-ts` SDK rejects Beta ids,
  hence plain REST. Replies come in by polling, so no webhook or tunnel is needed.
- **Caller-initiated intake:** the caller texts or calls; `sessions.open_call` asks for the address
  (`awaiting_address`), clarifies once, allows one retry, then hands off to standard 911 (`handoff.py`, a stub).
  OUT and danger checks stay first. No "demo" wording in anything a caller sees.
- **Caller addresses:** `usaddress` (trained US parser) after `text2num` and filler stripping
  (`addresses.parse_caller`); bands in `resolve.py`. libpostal and the Census Geocoder are future work.
- **Multi-floor:**
  - all floors are ingested;
  - the floor question comes first while no floor has 80%;
  - "located" also requires the floor to be at least 80% likely;
  - unlabeled regions are kept only in models that label no rooms.

## Open work
- **Multi-floor:** done and verified: eval 4/4 on 3012 Houston Ave (model `M7BdXkpzrrE`), plus a browser check of
  the floor tabs and the per-floor plan. That building's "nearest street" is Woodland Street (Nominatim returns only
  the single nearest road, so a corner lot shows one street).
- **Claude layer:** verify live once a key is provided.
- **Known accuracy limits:** identical rooms, rooms without photos, and labels that contradict the photos (see the
  README's "Evaluation").
