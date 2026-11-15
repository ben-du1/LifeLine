![LifeLine Logo](https://raw.githubusercontent.com/ben-du1/hackprinceton-mvp/refs/heads/main/LifeLine%20logo.png)

# LifeLine: finding a 911 caller inside a building

**HackPrinceton Fall 2026 · CoStar × Matterport track ("Inside and Out: AI for the Built World")**

An address can bring responders to a building, but it does not tell them which room or floor to search.
LifeLine speaks with the caller over the phone or iMessage, matches their descriptions to a Matterport 3D scan,
and shows the estimated location on the building’s floor plan.

- **What the dispatcher sees:** the conversation, the most likely rooms with probabilities, a north-up floor plan
  with the caller's estimated position, and the room's photos. An approach card says which side of the building
  and which floor to go to.
- **What the caller does:** answers a few plain questions like "Can you see a sink?" or "Which floor are you on?".
  Each question is chosen by the system as the one that best narrows down where they are.

---

## Demo in one minute

1. **The caller calls LifeLine.** The Photon number connects to an ElevenLabs voice agent, which asks for the building address. A caller can also start by texting the assigned iMessage line.
2. **LifeLine finds the building.** It matches the caller’s address to an accessible Matterport model and asks for clarification when the match is ambiguous.
3. **The conversation locates the room.** The caller describes what they see. LifeLine selects the next question, and the voice agent asks it aloud. Each answer updates the room and floor estimates.
4. **Responders see the search unfold.** The live dashboard shows the transcript, candidate rooms, floor plan, location marker, and room photos. If the evidence remains ambiguous, LifeLine reports its best guess.

For a live presentation, use a building and caller responses your team has rehearsed.

## How it works

```
 caller (Photon iMessage / Photon + ElevenLabs voice)    responder dashboard (React)
          │  replies                                          ▲  live updates (WebSocket)
          ▼                                                   │
 ┌──────────────────────────── FastAPI backend (CPU only) ────────────────────────────┐
 │ 1. read the reply        keywords + local paraphrase model (+ optional Claude)      │
 │ 2. score every view      CLIP: reply text vs. 4 photos per Matterport scan point    │
 │ 3. update the belief     Bayesian update over scan points → rooms → floors          │
 │ 4. pick the next question  highest expected information gain about the room        │
 └─────────────────────────────────────────────────────────────────────────────────────┘
          ▲                                     ▲                            ▲
   Matterport Model API              OpenStreetMap (nearest street)   Vexcel aerial imagery
   rooms, walls, scan photos         north-up plan, approach side     map view
```

### 1. The building becomes a set of views
- Every Matterport scan point ("sweep") contributes the 4 horizontal faces of its 360° photo.
- These are embedded once with **CLIP** (open_clip ViT-B-32) and cached, so a building costs a few minutes the
  first time and nothing after.
- Room outlines, labels (Bedroom, Kitchen…) and floors come from Matterport's GraphQL API.

### 2. Each reply updates a probability over locations
- The caller's words are compared with every view. Only the parts a camera could see are used, so *"please hurry,
  there's a fridge"* is matched on "fridge".
- The result updates a probability distribution over scan points, which is summed per room and per floor.
- The update is deliberately cautious. Photos alone can never locate someone from one message, a room with many
  scan points isn't favoured for one lucky view, and a little probability always stays on every room, so a later
  answer can overturn an early guess.

### 3. Questions are planned, not scripted
- **No fixed question list.** For every candidate question, LifeLine computes how much the answer is expected to
  narrow down the room (expected information gain) and asks the best one.
- **Answers are predicted for every room** by a feature model that combines:
  - **Photos:** CLIP, contrasted against neutral prompts like "an empty room".
  - **Room labels:** a toilet implies a bathroom, which lets rooms *without photos* be found.
  - **Geometry:** floor area answers "how big is the room?"; the outline's shape answers "long and narrow?".
  - **Commonsense bounds:** no toilet in a garage, no bed in a kitchen, whatever a blurry photo suggests.
- **Position questions** come in when rooms look alike: *"Is the room next to the kitchen?"*, *"Is it at the front
  of the house, on the Pointer Street side?"*.
- **Floors first:** in a multi-storey building, *"Which floor are you on?"* comes first. The floor is the first
  thing responders act on.
- **Understanding replies** needs no cloud service:
  - keywords and a local paraphrase model (all-MiniLM-L6-v2) understand *"I can touch both walls"* as "cramped";
  - an optional Claude layer can read messier replies and word questions naturally, but it **never chooses what
    to ask**. That stays with the maths, and every Claude failure falls back to the local rules.

### 4. Responders get directions, not just a room name
- **North-up plan:** the floor plan is drawn north-up from Matterport's geolocation, with the nearest street
  (OpenStreetMap) drawn on the side where it really is.
- **Approach card:** turns the estimate into words for the crew outside: *"2nd floor, rear (north) side, away from
  Pointer Street; the caller's room has an outside wall on the north side."* It includes a Street View link to
  that wall.
- **Map view:** places the rooms on Vexcel aerial imagery, for the approach: street, driveway, neighbours.
- **Big rooms:** in a large open-plan room, a "where in the room" region and a zone label avoid claiming more
  precision than the data has.

### Caller safety
The building may be on fire, so no text tells the caller to stay put unconditionally. They are told to get out if
they safely can (texting **OUT** stops the questions). Smoke advice is sent once. Mentions of smoke or fire flag the
call for the dispatcher.

## Results

In the team’s latest evaluation, the updated location logic identified **18 of 19 rooms**. The team also tested the phone questioning flow end to end with Photon and ElevenLabs. The repository’s scripted scenarios include an earlier run, shown below for context.

| Check | Result |
|---|---|
| Updated room-location logic | **18/19** rooms in the team’s latest evaluation |
| Earlier scripted callers (`backend/scripts/eval_scenarios.py`) | **15/19** rooms, including **4/4** rooms and floors in a multi-floor building |
| Paraphrases with no feature keywords (*"I'm right next to where you cook"*) | Understood locally, no cloud model |
| iMessage via Photon | Questions delivered, replies read, and estimates updated on a real phone |
| Phone via Photon and ElevenLabs | Team-tested end-to-end address intake and room questioning |
| Automated tests | Backend and voice test suites are included in the repository |

The earlier scripted run highlights difficult cases such as identical empty bedrooms, bathrooms without photos, and misleading room labels.

## Design challenges

- **Rooms that look the same.** A vacant house has four identical empty bedrooms. No photo matcher can separate
  them, so LifeLine asks about *position* instead (neighbouring rooms, the street side) and reports a tie honestly
  ("Bedroom 2 or Bedroom 3, can't tell apart yet") rather than picking one confidently.
- **Rooms with no photos.** Bathrooms and hallways often have no scan point. They are found through room labels and
  geometry. "There's a toilet" splits evenly between a photographed bathroom and a photo-less one, instead of
  always picking the one with a picture.
- **Panicked, noisy language.** *"I'm so scared, someone is downstairs, carpeted floor"* contains one useful fact.
  Feeding the whole sentence to CLIP moved the estimate on noise, so only visual content is matched.
- **Choosing the right model for each job.** CLIP's text encoder was tried for understanding replies and rejected:
  it rated "a little space" as close to "large" as to "small". A small sentence-embedding model handles paraphrase,
  and negations stay with keyword rules, because embeddings can't tell "toilet" from "no toilet".
- **Addresses that don't match.** The 911 address and Matterport's address text rarely agree exactly ("Chipman Glen
  Drive" vs "Chipman Glen Dr", a neighbour's house number). A fuzzy matcher weighs street, house number and
  city/ZIP. It accepts typos and a missing suffix, rejects a different street or city, and flags anything inexact
  for the dispatcher to confirm.
- **Imagery that doesn't line up.** Laying the floor plan over aerial photos looked great until it sat a few metres
  off the roof. Matterport's geolocation isn't that precise. The plan now uses only the heading (which is reliable),
  and the aerial photo lives in a separate map view with manual nudge controls.
- **Confidence vs. speed.** Every extra question costs the caller time. Too few, and responders go to the wrong
  room. LifeLine stops as soon as one room clearly leads, or when no remaining question would help.
- **Messaging constraints.** Photon's shared iMessage line can only reply, so the caller must text it first. A
  dedicated line removes this. Replies are polled rather than pushed, so no public webhook is needed.

## Future improvements

- **Real 911 integration:** take the caller's number, address and device location from the 911 system (NG911 /
  dispatch software, alongside the current caller-initiated flow.
- **Use the phone's own sensors:** Wi-Fi and barometric pressure from the caller's phone would settle the floor and
  narrow the room before the first question.
- **Track the responders too:** a boot-mounted inertial sensor could show firefighters' positions on the same floor
  plan as the caller.
- **Calibrate on real conversations.** The thresholds were hand-tuned on a few buildings. Real call transcripts
  (and the optional Claude layer, still untested live) would let them be measured properly.
- **More of the building's data:** Matterport's object labels, door swing and window sizes would make better
  questions ("Is there a window big enough to climb out of?") and better rescue planning.
- **Basements and floor naming:** Matterport labels floors "Floor 1, Floor 2…", so a basement is currently taken
  as the ground floor.
- **Languages and accessibility:** questions in the caller's language, and a mode for callers who can only tap
  answers.

## Run it

Requirements: Python 3.11+, Node.js 20+, and a Matterport API token in a repo-root `.env` (copy `.env.example`).
The browser rehearsal needs no phone services. The live voice path needs Photon and ElevenLabs setup; iMessage needs Photon. Vexcel and Anthropic are optional.

```bash
# Backend (CPU only)
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # bin/python on macOS/Linux
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt
cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000

# Dashboard (new terminal)
cd dashboard && npm install && npm run dev        # http://localhost:5173
```

- **Browser rehearsal:** open the dashboard with `?test=1` to reveal caller simulation controls.
- **Phone and iMessage:** follow [voice setup](voice/README.md) for the Photon–ElevenLabs phone path and [messaging setup](AGENTS.md#messaging-photon) for iMessage. Use an address and room description the team has verified before a live demo.

## Repository

| Path | What's there |
|---|---|
| `backend/app/` | FastAPI backend: call flow (`sessions.py`), question planning (`followups.py`), belief update (`locator.py`), Matterport ingest, address matching, orientation, approach card, voice bridge |
| `dashboard/` | React + Vite dispatcher dashboard: floor plan, map view, conversation, room photos |
| `messaging/` | Node sidecar for Photon's iMessage API |
| `voice/` | Gateway and setup for the ElevenLabs voice agent |
| `backend/tests/`, `backend/scripts/` | Tests, the scripted-caller evaluation, demo preflight |
| [`AGENTS.md`](AGENTS.md) | Full technical spec: every module, the API, the matching maths, evaluation details, setup |

**Data:** Matterport scans are licensed for hackathon use only and are cached locally, never committed. Street data
© OpenStreetMap contributors.
