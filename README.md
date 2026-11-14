# HackPrinceton MVP: Locating Callers Inside Buildings

Built for the CoStar × Matterport track ("Inside and Out: AI for the Built World") at HackPrinceton Fall 2026
(Nov 13–15, 2026; submissions close **Nov 15, 4:00 PM EST**).

## Status (2026-11-14)

| Area | State |
|---|---|
| Core loop (call → texts → room estimate → dot on plan), simulated caller | Working, verified |
| Address → Matterport model (fuzzy matching) | Working, verified against the live API |
| Follow-up questions (cues, information gain, paraphrase matching) | Working; scripted eval 14/19 top room, 0 confidently wrong |
| Room photos per call (API + dashboard gallery) | Working, verified |
| Spot within the room ("what's closest to you?", uncertainty circle) | Working: mean error 3.1 m → 1.0 m on three photo-checked spots; verified in the browser |
| Photon iMessage (send + receive) | Working, verified with a real phone |
| Orientation: north-up plan + nearest street (OpenStreetMap) | Working, verified on two houses |
| Multi-floor buildings | Working, verified: eval 4/4 rooms and floors. In the browser, the floor tabs follow the estimate and the plan switches floors after the caller's answer. |
| Optional Claude layer (`llm.py`) | Written and unit-tested with a stub; **never run live** (no key) |
| Vexcel aerial overlay | Tried and removed (see "Orientation"). CoStar data: not used. |

## Problem

When people are trapped or lost inside a building during an emergency, they often cannot find the exit and must
rely on first responders to reach them. Responders usually know the building's address, but not where inside the
building the caller is. This tool helps a dispatcher find a caller's position on a floorplan, using only a short
text conversation with the caller.

## How it works

1. **911 call comes in.** The caller's phone number and address are forwarded to our program. For this MVP they are
   entered on the dispatcher dashboard (simulated 911 input).
2. **Two things start in parallel:**
   - The program texts the caller over iMessage (via Photon) and asks them to describe the room they are in.
   - The address is matched to the Matterport digital twin whose address is most similar. The match does not have
     to be exact (see "Address matching"), and approximate matches are flagged to the dispatcher. The building is
     downloaded and embedded the first time, and read from the local cache afterwards.
3. **The interior is represented as room views.** Every Matterport scan point ("sweep") contributes 4 images: the
   4 horizontal faces of its cube-map skybox. Rooms usually contain several sweeps.
4. **Each caller reply updates a belief over locations.** An image-text embedding model scores how well the reply
   matches every view; the scores update a probability distribution over sweeps, which is summed per room (and per
   floor in multi-storey buildings).
5. **Confident or ask again.**
   - If the top room passes the confidence rule, the dashboard shows the floorplan with a dot at the most likely
     sweep's position, and the caller is told help knows where they are.
   - Otherwise the program asks a follow-up that builds on what the caller said ("I'm in a cramped space" → "You
     said it's a tight space. Can you see a sink or faucet?"). It is chosen to best split the rooms still in play,
     and step 4 repeats. In a multi-storey building, "Which floor are you on?" comes first while the floor is
     uncertain. It stops with a best guess, marked low-confidence, after `MAX_QUESTIONS` rounds or when no question
     would help.
6. **Where in the room.** Once the room is known, the system asks up to two "what's closest to you?" questions, for
   example "the stove, the sink, or the fireplace?", built from that room's scan points. The dot moves toward the
   matching spot, with a circle showing how sure it is.
7. **Orientation for responders.** The plan is drawn north-up, with the nearest street on the side where it really
   is.

## MVP scope

**In scope**
- Single- and multi-floor buildings: every floor of the model, a floor estimate, and floor tabs on the dashboard.
- Any address that has a Matterport model reachable with our token (about 15,000, mostly Houston and Los Angeles).
- Matching on physical room properties (windows, furniture, fixtures, flooring, ...).
- Dispatcher web dashboard: start a call, watch the conversation, see candidate rooms, floors, photos and the
  predicted position on a north-up plan with the nearest street.
- A simulated caller box on the dashboard, so the full loop can be tested without a phone.
- Running everything locally on a laptop (CPU only; no GPU needed).

**Out of scope (non-goals)**
- Fire or smoke conditions (low visibility, panic-affected descriptions).
- Basements: Matterport floor labels are just "Floor 1", "Floor 2", ..., so the lowest floor is treated as the
  ground floor.
- Addresses with no Matterport model: the session is marked "no interior model" and no location is attempted.
- Real 911 integration (call data is typed into the dashboard).
- CoStar data (not used: the extract is property, lease and sale records, with no roads or imagery). Vexcel aerial
  imagery was integrated, then removed (see "Orientation").

## Architecture

```
                         iMessage
   caller's phone  <---------------->  Photon cloud (Beta REST API)
                                            ^
                                            | REST: sendMessage / polled listMessages (no webhook needed)
                                            v
                                  messaging/ sidecar (Node, :8787)
                                     POST /send  ^      | POST /api/inbound
                                                 |      v
   dispatcher dashboard  <--- WebSocket /ws ---  backend/ FastAPI (:8000)  ---> Matterport Model API (GraphQL)
   (React + Vite, :5173)  --- REST /api/*  --->      |  locator (open_clip, CPU)  ---> OpenStreetMap Nominatim
                                                     v                                  (nearest street)
                                          data/buildings/<model_id>/  (cache)
```

| Component  | Path         | Tech                              | Responsibility |
|------------|--------------|-----------------------------------|----------------|
| Backend    | `backend/`   | Python 3.11, FastAPI, open_clip   | Sessions, building ingest, matching, question choice, REST + WebSocket API |
| Messaging  | `messaging/` | Node 20+, `fetch` (Photon REST)   | Sends iMessages for the backend; forwards caller replies to the backend |
| Dashboard  | `dashboard/` | React + Vite                      | Simulated 911 form, live conversation, room probabilities, floor plan with dot |
| Cache      | `data/`      | Files on disk (gitignored)        | Building JSON, skybox images, floorplan image, embeddings |

Sessions are kept in memory in the backend (restarting the backend clears them). Buildings are cached on disk.

### Backend modules (`backend/app/`)

| Module          | Purpose |
|-----------------|---------|
| `config.py`     | Loads `.env`, paths, and tunable constants (thresholds, temperature, max questions). |
| `matterport.py` | Minimal GraphQL client (HTTP Basic auth), the queries we use, model search and match ranking. |
| `addresses.py`  | Fuzzy address similarity (street, house number, city/state/ZIP) for picking a model. |
| `ingest.py`     | Downloads every floor's rooms (with wall outlines), sweeps, skybox faces and floorplans; computes embeddings. Also a CLI. |
| `building.py`   | Loads a cached building (floors, rooms, candidates); per-candidate text scores; floorplan pixel mapping. |
| `embeddings.py` | Wraps the open_clip model (image and text encoders). |
| `locator.py`    | Pure-numpy belief update, confidence rule, and expected information gain. |
| `followups.py`  | Reads cues from the caller's words; feature model (CLIP + room labels + geometry + floors); plans follow-ups. |
| `semantics.py`  | Local paraphrase matching of caller messages (sentence embeddings). |
| `orientation.py`| North-up heading (Matterport) and the nearest street (OpenStreetMap Nominatim) for the floor plan. |
| `llm.py`        | Optional Claude reading of replies and wording of follow-ups (falls back to keyword rules). |
| `questions.py`  | Opening/closing messages and reply parsing (yes / no / free text / unsure). |
| `sessions.py`   | Session state machine and conversation flow. |
| `messaging.py`  | Outbound messages: `simulated` (dashboard only) or `photon` (via the sidecar). |
| `main.py`       | FastAPI routes, WebSocket broadcast, static files for cached building assets. |

### API

| Method | Path                           | Body / notes |
|--------|--------------------------------|--------------|
| GET    | `/api/config`                  | `{messaging_mode, dev_test_phone, max_questions, followups}` for the dashboard |
| POST   | `/api/calls`                   | `{phone, address}`: simulated 911 call; creates a session |
| GET    | `/api/sessions`                | All sessions (snapshots) |
| GET    | `/api/sessions/{id}`           | One session snapshot |
| GET    | `/api/sessions/{id}/rooms`     | Every room of the call's building, most likely first: `{index, name, types, floor, area_m2, probability, sweeps[{candidate, x, y, probability, views[url]}]}` |
| POST   | `/api/sessions/{id}/reply`     | `{text}`: simulated caller reply from the dashboard |
| POST   | `/api/inbound`                 | `{from, text}`: real caller reply, posted by the messaging sidecar |
| WS     | `/ws`                          | Pushes `{type: "session", session}` on every change |
| GET    | `/api/buildings/{model_id}`    | Cached `building.json` (`floors`, `floorplans` per floor, rooms with `floor`/walls/polygon/centroid, candidates) with asset URLs, plus `orientation` (`yaw_deg`, `street{name, distance_m, bearing_deg, side, direction_deg}`) once fetched |
| GET    | `/buildings/{model_id}/...`    | Cached floorplan and view images |

Session snapshot fields: `id, phone, address, status, status_detail, model_id`,
`match{id, name, address, score, exact}` (the Matterport property in use), `messages[{role, text, ts}]`
(`role` is `dispatch`, `caller` or `system`), `questions_asked, max_questions`,
`rooms[{index, name, floor, floor_label, probability}]` (sorted), `floors[{index, label, probability}]`,
`orientation` (true once north/street data is cached), `candidate_probabilities[]` (aligned with the building's
`candidates`), and `estimate{room, room_name, floor, floor_label, candidate, x, y, radius_m, room_probability,
runner_up_probability, confident}`. In `estimate`:
- `x, y` is the probability-weighted position of the top room's scan points.
- `radius_m` is its uncertainty (RMS spread, at least 0.75 m).
- `candidate` is the single most likely scan point.
Statuses: `loading → listening → located | best_guess`, or `no_model` / `error`.

## Matterport data

Track docs: <https://beta-app.matterport.com/2026-princeton-hackathon/quickstart> (sample model `3QuftVC2agD`,
a one-floor, **vacant** house at 13122 Chipman Glen Dr, Houston, TX). After filtering it has 8 indoor rooms,
16 sweeps with photos, and 4 photo-less rooms (two bathrooms, the hallway, one bedroom).

- **Endpoint:** `https://api.matterport.com/api/models/graph` (GraphQL, server-side only).
- **Auth:** HTTP Basic with `MATTERPORT_TOKEN_ID` as the username and `MATTERPORT_TOKEN_SECRET` as the password.
- **Address lookup:** see "Address matching" below.
- **Fields used:**
  - `model.floors { id label sequence }`: every floor that has rooms is kept, renumbered 0 (lowest) upward.
    Labels are usually just "Floor 1", "Floor 2", ...
  - `model.rooms { id label classifications { label } floor { id } boundary { edges { vertices { position } } } }`:
    room types (Bedroom, Kitchen, ...), the floor, and wall segments. Rooms classified only as `Void` are ignored.
    Unlabeled regions:
    - **In a model that labels its rooms:** they are outdoor areas (the sample's yard and driveway, which even have
      outdoor sweeps), so they are dropped.
    - **In a model that labels nothing** (many townhomes): the photographed regions are kept as "Room 1", "Room 2".

    Wall segments are chained into a polygon for shading; dead-end stub walls are pruned first.
  - `model.panoLocations { id position { x y z } anchor { room { id } floor { id } } skybox(resolution) { children } }`:
    sweeps. `children` holds 6 face URLs (`_skybox0` … `_skybox5`); faces 1–4 are the horizontal views,
    0 is up and 5 is down (verified visually). The `low` resolution (256 px faces) is enough for CLIP's 224 px input.
  - `model.assets.floorplans { format flags width height resolution origin { x y } floor { id } url }`: the plain
    `photogramy` JPG is downloaded for each floor. `resolution` is pixels per meter and `origin` is the model coordinate of the
    image's bottom-left corner (verified by overlaying sweeps). These rasters are **aerial photogrammetry that show the
    roof**, so the dashboard draws its own schematic from room walls (in model meters) and offers the photo only as an
    optional background layer.
- **Signed URLs expire after about an hour**, so images are downloaded at ingest time and served from `data/`.
- **Cache versioning:** `building.json` carries `schema` (`ingest.SCHEMA_VERSION`, currently 3). Older caches are
  re-ingested automatically on the next call. Already-downloaded images are reused, but embeddings are recomputed,
  which takes a few minutes.
- **Multi-floor test building:** `M7BdXkpzrrE`, 3012 Houston Ave, Houston: 4 floors and 60 rooms, including a
  kitchen on floors 1 and 2, a game room on 3 and a loft on 4. There are ~126 multi-floor models in the token's
  catalogue.
- **Rooms without sweeps** (often bathrooms and hallways) have no photos. They get one image-less candidate at the
  room outline's centroid; it can only be selected by room-type words in a reply (e.g. "bathroom").
- **Unusable models:** a model with no floors, no rooms, or no photographed rooms is rejected at ingest
  (`UnusableModel`), and the next-best address match is tried. A missing floorplan raster is fine (the dashboard
  hides the aerial-photo toggle).

## Address matching (`addresses.py`, `matterport.search_models` / `rank_matches`)

The dispatcher's address rarely matches Matterport's text exactly, and some callers' buildings have no model, but a
model at a very similar address does exist. The backend therefore picks the model with the most similar address.

1. **Candidates:**
   - Already-cached buildings.
   - Matterport `models(query:)` results for three query variants: the full address, "number + street", and the
     street name alone. The search ranks by word overlap, so city words like "Houston" can crowd out the right
     model. 20 results per query, de-duplicated.
   - If the search API is down, only the cache is used.
2. **Score (0–1):** both addresses are parsed into house number, street words, unit, city, state and ZIP. Suffixes
   (Dr/Drive) and directions (W/West) are normalized.

   | Component | Weight | Rule |
   |---|---|---|
   | Street name | 0.45 | Word coverage in both directions; words with spelling similarity of 0.8 or more count as the same (typos), others count 0. A differing direction (E vs W) multiplies by 0.8. |
   | House number | 0.30 | 1.0 if equal. Otherwise 0.7 falling linearly to 0 at 200 apart, so a neighbour never beats an exact number. 0 if the model has no number. |
   | City/state/ZIP | 0.25 | ZIP equal 1.0; neighbouring ZIP in the same city 0.6; city equal 0.9; state equal 0.4. |

   - **Missing pieces:** a component missing from the input is left out and the weights renormalized.
   - **Conflicting place:** if the input names a different city, state or ZIP, the score is halved.
   - **Unit:** it only breaks ties between models in the same building (±0.03).
3. **Accept:**
   - The best model scoring at least `ADDRESS_MATCH_THRESHOLD = 0.80` is used. If it has no usable interior, up to
     `ADDRESS_MATCH_ATTEMPTS = 3` matches are tried in order.
   - Below the threshold the call ends as `no_model`.
   - A non-exact match is shown on the dashboard (orange note under the address, plus a system message with the
     similarity) so the dispatcher can confirm it with the caller.

| Input | Matched model | Score |
|---|---|---|
| `13122 chipman glen drive` | 13122 Chipman Glen Dr, Houston (exact) | 1.00 |
| `13120 chipmen glen drive houston` | 13122 Chipman Glen Dr, Houston | 0.85 |
| `7925 Pointer St Houston` | 7921 Pointer St, Houston | 0.88 |
| `302 West Drew, Houston` | 302 W Drew St, Houston (exact) | 1.00 |
| `233 W 27th St, Houston` | none (best was 233 W 26th St) | < 0.80 |
| `1600 Pennsylvania Ave NW, Washington DC` | none | < 0.80 |

## Matching approach (`locator.py`)

- **Model:** open_clip `ViT-B-32` (`laion2b_s34b_b79k`), on CPU. All views are embedded once at ingest; replies are
  embedded at runtime in tens of milliseconds.
- **Candidates:** one per sweep (plus one per sweep-less room). Each candidate's score for a piece of text is the
  maximum cosine similarity over its 4 views. The caller may be facing any direction.
- **Prior:** uniform over rooms, split evenly across a room's candidates, so rooms with many sweeps are not favored.
- **Update:** `posterior ∝ prior × exp(LOGIT_SCALE × (score − max score))`, applied once per reply. If the reply
  names a room type, candidates in rooms with that classification are multiplied by `ROOM_KEYWORD_BOOST`.
- **Rooms without photos** (`Building.candidate_scores`) start from the median imaged score
  (`IMAGELESS_SCORE_QUANTILE`, roughly neutral). They are then lifted by how far photographed rooms *of the same
  type* stand out for the text (mean peer score minus mean imaged score).
  - **Why:** at 12331 Basilwood Dr the whole upper floor has no photos. Before this, "I'm in the bathroom" was
    "located" in the one photographed (downstairs) bathroom at 91%, because LOGIT_SCALE turned "has a photo" into
    ~20× odds. Now the three bathrooms tie and the floor question comes first.
  - **Mean, not max:** the best of several peers scores high for almost any text, and in the eval that let
    unphotographed closets and a 2 m² "bedroom" win.
- **Within a room** (`IN_ROOM_LOGIT_SCALE = 15`): each room's total moves exactly as above, but the room's share is
  split among its own scan points with a quarter of the sharpness.
  - **Why:** photo-score differences between sweeps of one room are mostly noise for a room-level description.
    "I'm in the living room" put 38% on one of eleven sweeps, and the dot started 2–5 m off.
  - **Rejected alternative:** ranking rooms by their best sweep favoured rooms with many sweeps, and "I'm in a
    narrow space" was "located" in the big living area instead of the hallway.
- **Free-text replies** are embedded and scored as above. A reply that only answers our question ("yes", "tile",
  "none of those") is not embedded; its meaning comes from the follow-up model below.
- **Confidence rule:** the top room's probability ≥ `CONFIDENCE_THRESHOLD` and it leads the runner-up by at least
  `CONFIDENCE_MARGIN`.
- **After "located", the call keeps listening.** New replies still update the estimate:
  - If the estimate moves to another room, or confidence drops, the call reopens and asks a follow-up.
  - If the room stays the same, nothing is re-sent.
  - An explicit correction ("sorry", "actually", "I meant", "no wait") resets the estimate to the prior before
    the new reply is applied. Without the reset, one correction after a confident wrong room (98%) only brought it
    down to ~50%.
- **Output:** the highest-probability candidate inside the top room; its `(x, y)` (model meters) is drawn on the
  floor plan. The dashboard shows every photo of that room, or of any room the dispatcher clicks.
- **Current values** (`config.py`): `LOGIT_SCALE=60`, `ROOM_KEYWORD_BOOST=20`, `IMAGELESS_SCORE_QUANTILE=0.5`,
  `CONFIDENCE_THRESHOLD=0.6`, `CONFIDENCE_MARGIN=0.25`, `MAX_QUESTIONS=6`, `MIN_INFORMATION_GAIN=0.05`. The comments
  in `config.py` say why.

## Orientation: north and the street out front (`orientation.py`)

Firefighters arrive from the street, so the dashboard draws the floor plan **north-up**, with a compass, and the
nearest street as a labelled band on the side where it really is ("Pointer Street · nearest street · 28 m east").
Once a call's building has loaded, the backend fetches this in the background (it never delays the questions) and
caches it in `data/buildings/<id>/orientation.json`:

1. **Heading and position:** Matterport's `model.geocoordinates` (`matterport.fetch_geo`) give a rigid transform
   from model meters to east/north meters. It was checked against Matterport's own `geoLocationOf` on two models
   (residuals around 1e-11 m), and the heading matched the roof lines in aerial photos.
2. **Nearest street:** one OpenStreetMap **Nominatim** reverse-geocoding request at the middle of the house
   (`zoom=17`, with the road's line geometry). The side, distance and road direction come from the closest point on
   that line. No key is needed. The requests send an identifying User-Agent and are cached, per Nominatim's usage
   policy.

| Building | Street (OSM) | Side | Distance (house middle → road centerline) | Road runs |
|---|---|---|---|---|
| 13122 Chipman Glen Dr | Chapman Glen Drive | south | 29 m | ENE (79°) |
| 7921 Pointer St | Pointer Street | east | 28 m | N–S |

Both match the aerial imagery. At Pointer St, the scanned "Front Porch" sits on the street side.

**Design notes:**
- **No imagery overlay:** an earlier Vexcel version laid the floor plan over aerial imagery. Matterport's position
  can be a few meters off, so the plan didn't sit exactly on the roof. Orientation only needs the heading, and a few
  meters of position error can't move a street to the wrong side.
- **Nominatim instead of Overpass:** Overpass (all nearby roads) timed out or was "at capacity" on every attempt
  during development. Nominatim answers in ~1 s but returns only the single nearest road, so a corner lot shows one
  street.
- **OSM spelling:** street names come from OSM and can differ from the address. OSM has "Chapman Glen Drive" where
  the address says "Chipman Glen Dr".

## Follow-up questions (`followups.py`)

There is no fixed question list. Each follow-up builds on what the caller has said so far.

- **Reading the caller:** every message is scanned for cues, and each cue becomes a likelihood over candidates:
  - **Size words:** "cramped", "tight" and "tiny" score rooms by floor area (computed from the room outline). Looser
    words like "small" are softer, because a 6.6 m² pantry is "small" but not "cramped". "Big" or "open" favours
    large rooms.
  - **Darkness:** "dark" counts as evidence that there is no window.
  - **Objects mentioned or denied:** "there's a toilet", "no windows".
  - **Room names:** as before, named room types boost matching rooms.
- **Feature model:** each feature (toilet, shelves, washer, stove, narrow shape, and so on) has a probability
  P(present | candidate), averaged from:
  - **CLIP:** the candidate's photos against the feature prompt, z-scored and passed through a sigmoid.
  - **Room label:** Matterport's classification or label, counted double. A toilet implies a bathroom, which lets
    rooms *without photos* be found.
  - **Geometry:** floor area for size questions; bounding-box aspect ratio for "long and narrow".
- **Question kinds:**
  - **Yes/no:** "Is there a washing machine or dryer?"
  - **Choice:** "What is the floor like: carpet, tile, or wood?" and "About how big is the room: closet-sized,
    bedroom-sized, or bigger?" (answered from geometry).
  - **Which of these:** a list built from the three features that best separate the rooms still in play, e.g.
    "What can you see around you: shelves or hanging clothes, a washer or dryer, or a toilet, tub or shower?"
- **Choosing:** the question with the highest expected information gain about the *room*
  (`locator.expected_information_gain`) is asked. Topics the caller already settled (mentioned, answered, or
  "I don't know") are never asked again. A "which of these" list is not asked twice in a row.
- **Stopping:** if no question is worth `MIN_INFORMATION_GAIN` (0.05 bits), the call stops with a best guess that
  lists the leading rooms, instead of asking irrelevant questions.
- **Floors (multi-storey buildings):**
  - **Question:** a choice question is built per building, e.g. "Which floor are you on: ground floor, 2nd floor,
    3rd floor, or 4th floor?".
  - **Cues in free text:** "upstairs", "downstairs", "second floor", "top floor", "floor 2". "Upstairs" means
    floor 2 in a two-storey building and any floor above ground in a taller one.
  - **Weight of a floor answer:** `FLOOR_ANSWER_ACCURACY = 0.9`, because callers confuse "first" and "ground".
  - **Floor first:** the floor question goes first while no floor has `FLOOR_FIRST_UNTIL = 0.8`. Responders act on
    the floor first, and callers know it.
  - **Located needs the floor too:** "located" also requires the room's floor to be that likely, so "I'm in the
    kitchen" in a building with two kitchens on different floors asks for the floor before declaring a location.
- **Where in the room** (`followups.refinement`): once the room is "located", the system asks up to
  `MAX_REFINEMENTS = 2` "what's closest to you?" questions. The conditions:
  - the room has several photographed scan points;
  - the position is spread more than `REFINE_MIN_SPREAD_M = 1.5 m`;
  - the question is worth at least `REFINE_MIN_GAIN = 0.15` bits.

  The first one is appended to the "located" text. How it works:
  - **Landmarks:** stove, sink, fridge, fireplace, front door, glass door, couch, table, TV, bed, toilet, stairs,
    and so on, limited to those the room probably has. Each scan point gets a close-up CLIP score
    ("a close-up photo of the stove"), z-scored across the room's scan points. In the sample house, kitchen items
    peak at the sweep between the sink, stove and fridge, and the fireplace at the sweep beside it (checked
    against the photos).
  - **Choosing the three landmarks:** every combination is scored, and the one with the highest expected
    information gain over the scan points is asked. Ranking landmarks one at a time picked weak sets (0.13 vs
    0.65 bits).
  - **Reading the answer:** each landmark is modelled independently as "right next to you" (`NEAR_SCALE = 1.5`),
    so naming one, several, or "none of those" are all evidence.
  - **No effect on the room:** the answer is applied with `locator.apply_within_rooms`, which keeps every room's
    total. It only moves the dot.

  Results for callers at three photo-checked spots in the sample house's main room, each opening with "I'm in the
  living room":

  | Spot | Answers | Error before → after | Uncertainty |
  |---|---|---|---|
  | By the sink/stove/fridge | "none of those"; "the stove and the sink" | 1.6 m → 0.6 m | ±3.0 → ±1.4 m |
  | By the fireplace and patio door | "the glass door to outside"; "the fireplace" | 3.7 m → 1.0 m | ±3.0 → ±2.0 m |
  | By the front door and table | "the table"; "the front door" | 4.0 m → 1.5 m | ±3.0 → ±2.6 m |
- **Phrasing:** each question starts by acknowledging the last cue: "You said it's a tight space. Can you see a
  sink or faucet?"

### Paraphrase understanding (`semantics.py`, no API key needed)

Keyword cues miss paraphrase: "I can touch both walls" or "the thing you wash clothes in". A small local
sentence-embedding model (`all-MiniLM-L6-v2`, ~90 MB, ~3 ms per message on CPU) compares each message with example
phrasings of every feature, plus "tight", "large" and "dark". A match must score at least 0.6 and beat a set of
neutral messages ("please help me") by 0.1.
- **Keywords win where both apply.**
- **Negations stay with the keywords**, because embeddings can't tell "toilet" from "no toilet".
- **Size from the model** counts only when it beats the opposite size by 0.15, because the model confuses opposites.
- **Why not CLIP:** its text encoder was tried first and rejected; it rated "a little space" as close to "large" as
  to "small".

### Optional Claude layer (`llm.py`)

For understanding beyond those examples, when
`ANTHROPIC_API_KEY` is set, each turn makes two Claude calls (`LLM_MODEL`, default `claude-opus-5-5`, effort `low`,
structured JSON output):

1. **Read:** the caller's message plus the recent conversation become the same structured `Cues` the keyword rules
   produce: the answer to the pending question, features present or absent, size (tight/small/medium/large),
   darkness, room types, and a short acknowledgement in the caller's terms. `followups.apply_cues` turns them into
   evidence exactly as before.
2. **Word:** the topic the planner chose is phrased as one short message built around what the caller said. It is
   rejected, and the template used instead, if it isn't a question or drops any of the offered options, so the
   answer stays parseable.

The language model never decides *what* to ask: that is still expected information gain over the building's rooms.
Every failure falls back to the local reading (keywords plus paraphrase) and the templates: no key, a timeout (`LLM_TIMEOUT_SECONDS`, default
10 s), an API error, a refusal (server-side `fallbacks: "default"` is on), or malformed output. `GET /api/config`
reports `followups: "claude" | "rules"`.

**Status:** covered by `tests/test_llm.py` with a stub client (request shape, parsing, every fallback). **Not run
against the live API**: no Claude credentials were available. Expect a few seconds of added latency per turn.

### Evaluation

`backend/scripts/eval_scenarios.py [-v]` runs scripted callers through the real session flow. Their answers come
from a fact sheet checked against the photos, and `-v` prints each conversation. Current result, with no API key:
**14/19 top room correct, and none confidently wrong**:
- 10/15 on the single-floor houses, including 3 of 4 open-ended paraphrases that contain no feature keywords.
- **4/4 rooms and 4/4 floors** on the 4-floor building.

Before the "rooms without photos" fix the score was 15/19, but two of the misses were declared "located" in the
wrong room. Those cases had won only because their closet was the one with a photo. They now end as an honest best
guess that lists the similar closets.

It needs the three buildings cached; it ingests them on first run, which takes several minutes.

| Building | Caller is in | Opening | Result |
|---|---|---|---|
| Chipman Glen (vacant) | Bedroom 1 | "empty room with a window and a ceiling fan" | Correct (best guess, 53%) |
| Chipman Glen | Bedroom 2 | "empty room with two windows with blinds" | Best guess: Bedroom 3 55%, Bedroom 2 31% (identical empty rooms) |
| Chipman Glen | Kitchen/living | "I can see a stove and a fridge" | Correct, no follow-ups |
| Chipman Glen | Kitchen/living | "there's a brick fireplace" | Correct, no follow-ups |
| Chipman Glen | Bathroom 1 | "I'm in a cramped space" | Best guess: Bathroom 2 58%, Bathroom 1 42% (neither has photos) |
| Chipman Glen | Hallway | "I'm in a narrow space" | Correct after 3 follow-ups |
| 7921 Pointer St | Pantry/Laundry | "small room with lots of shelves" | **Wrong:** best guess lists the closets (~33% each). Labeled laundry, but there is no machine, so a truthful "no washer" counts against it |
| 7921 Pointer St | Closet 4 | "I'm in a cramped space" | **Wrong:** best guess, Closet 4 third (22%) behind two unphotographed closets the caller can't be told apart from |
| 7921 Pointer St | Bathroom | "it's tiny in here" | Correct after 3 follow-ups |
| 7921 Pointer St | Kitchen | "I'm next to the stove" | Correct, no follow-ups |
| 7921 Pointer St | Bedroom 3 | "it's a big room with a bed" | Correct after 1 follow-up |
| 7921 Pointer St | Kitchen | "I'm right next to where you cook" | Correct, no follow-ups |
| 7921 Pointer St | Bathroom | "I can barely move, I'm squeezed in by the tub" | Correct after 4 follow-ups |
| Chipman Glen | Kitchen/living | "the room is massive, there's a brick mantel" | Correct, no follow-ups |
| 7921 Pointer St | Closet 4 | "I can touch both walls and my coats are hanging next to me" | **Wrong:** best guess across four closets (Closet 2 28%). Read correctly as a tight closet; a truthful "no window" counts against Closet 4, whose photo shows the bedroom window through its door |
| 3012 Houston Ave (4 floors) | Office, floor 1 | "I'm downstairs in a room with wood paneling and a bookcase" | Correct, floor 1 at 97%, no follow-ups |
| 3012 Houston Ave | Game Room, floor 3 | "I'm in an empty room with built-in shelves" | Asked the floor ("3rd floor"), then correct after more follow-ups |
| 3012 Houston Ave | Loft, floor 4 | "I'm on the top floor near the stairs, there's a couch" | Correct, floor 4 at 93%, no follow-ups |
| 3012 Houston Ave | Kitchen, floor 1 (another on floor 2) | "I'm in the kitchen" | Asked the floor ("ground floor"), then correct |

The remaining failures are rooms a truthful caller can't tell apart: identical rooms, rooms without photos, or a
label that disagrees with the photos. When rooms are tied, the call now ends as a best guess naming them, instead
of picking one confidently.

## Messaging (Photon)

- Our Photon project is on **Photon Beta (2.0)**: project id `pho_prj_…`, API key `pho_sk_…` sent as a Bearer token
  to `https://api.photon.codes/v1`. The Stable SDK `spectrum-ts` rejects Beta project ids (HTTP 422), so the sidecar
  (`messaging/index.mjs`) calls the REST API with plain `fetch` and has no npm dependencies. It reads the repo-root
  `.env` (`node --env-file=../.env`) and listens on `MESSAGING_PORT` (default 8787).
- **Shared line rule:** the project uses Photon's shared iMessage line, which can only *reply*. The caller's phone
  needs an active assignment (Photon dashboard), and must text the line once after being assigned. `GET /health`
  on the sidecar shows the line number. Sending to a phone with no conversation returns 409, which the dashboard
  shows as a system note. A dedicated line (available on the project) would remove this rule.
- **Sending:** `POST /v1/projects/{id}/messages` with `{spaceId, senderId, content: {type: "text", format: "plain"}}`.
  The caller's direct space id and our agent user id are learned from their inbound messages (the last 7 days are
  scanned at startup).
- **Receiving:** the sidecar polls `GET /v1/projects/{id}/messages?direction=received` every 2 s. It re-reads a
  30 s overlap window, because the API is eventually consistent, and dedupes by message id. Each new caller text
  is forwarded to `POST /api/inbound`. No webhook or tunnel needed.
- **Status:** verified end to end on 2026-11-14 with a real phone (`DEV_TEST_PHONE`), backend in `photon` mode:
  - the opening question and follow-up questions were `delivered`;
  - the caller's replies ("I see a tv", "No") reached `/api/inbound` within the 2 s poll;
  - the locator updated its estimate and asked the next question each time.
- `MESSAGING_MODE=simulated` (default) skips the sidecar entirely; `MESSAGING_MODE=photon` sends via the sidecar.

## Setup

Requirements: Python 3.11+, Node.js 20+, and the credentials below in `.env` (copy `.env.example`; never commit `.env`).

| Variable                  | Used by   | Notes |
|---------------------------|-----------|-------|
| `MATTERPORT_TOKEN_ID`     | backend   | From the track Discord (`/costar-keys`) |
| `MATTERPORT_TOKEN_SECRET` | backend   | |
| `PHOTON_PROJECT_ID`       | messaging | Photon Beta project id (`pho_prj_…`) |
| `PHOTON_API_KEY`          | messaging | Photon Beta project API key (`pho_sk_…`, needs project read/write) |
| `DEV_TEST_PHONE`          | dashboard default | Phone number prefilled in the simulated 911 form |
| `MESSAGING_MODE`          | backend   | `simulated` (default) or `photon` |
| `ANTHROPIC_API_KEY`       | backend   | Optional. Turns on Claude follow-ups (`llm.py`); leave empty for keyword rules |
| `MESSAGING_SIDECAR_URL`   | backend   | Default `http://localhost:8787` |
| `BACKEND_URL`             | messaging, dashboard proxy | Default `http://localhost:8000` |

```bash
# Backend
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # Windows path; use backend/.venv/bin/python on macOS/Linux
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt

# Pre-load the demo building (optional; otherwise it is loaded on the first call; downloads ~600 MB of model weights once)
cd backend && .venv/Scripts/python -m app.ingest 3QuftVC2agD

# Run the backend
cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000

# Dashboard (new terminal)
cd dashboard && npm install && npm run dev        # http://localhost:5173

# Messaging sidecar (only for MESSAGING_MODE=photon; new terminal)
cd messaging && npm start                         # no dependencies; the caller's phone must text the shared line first
```

Tests: `cd backend && .venv/Scripts/python -m pytest` (44 tests). `test_semantics.py` downloads the ~90 MB
paraphrase model on first run; the rest need no network.
Scenario check: `cd backend && .venv/Scripts/python scripts/eval_scenarios.py [-v]` (ingests the three test
buildings on first run).

## Demo script

1. Open the dashboard and enter a phone number and `13122 Chipman Glen Dr, Houston, TX`.
2. As the caller (simulated box, or a real phone in `photon` mode), describe the room. The sample house is empty,
   so describe fixtures: "there's a brick fireplace and a big window" (living area) or "I can see a stove and a
   fridge" (kitchen).
3. Answer the follow-ups (yes/no via the quick buttons, or type "tile", "closet-sized", "I see a stove") until the
   dot appears. Try a vague opening like "I'm in a cramped space" to see the follow-ups narrow it down.
4. Click any room (on the plan or in the probability list) to see all of its photos.
5. Where in the room: in the sample house say "I'm in the living room", then answer the "what's closest to you?"
   questions ("the glass door to outside", "the fireplace"). The dot moves to the fireplace corner and the circle
   shrinks.
6. Multi-floor: call `3012 Houston Ave, Houston, TX` and say "I'm in an empty room with built-in shelves". The
   floor question comes first; answer "3rd floor". Floor tabs above the plan show each floor's probability.

## Known limitations

- Floors are identified by order only: the lowest floor is the "ground floor", and basements aren't recognised.
- An approximate address match (e.g. the neighbouring house) is a *different building*. The dashboard flags it, and
  the dispatcher should confirm it with the caller.
- Address matching is tuned for US street addresses; other formats are untested.
- The spot within a room is only as fine as the scan points (about 1.5 m apart) and the landmarks in view. It
  can't place the caller between scan points beyond that, or in rooms without photos.
- Rooms without photos are scored like the photographed rooms of the same type, so they can't be told apart from
  each other (or from their photographed sibling) by appearance; floor, size and room-type words separate them.
- Visually similar rooms (the sample's empty bedrooms) are confused, sometimes with high confidence.
- Without `ANTHROPIC_API_KEY`, follow-ups use templates, and replies are read by keywords plus paraphrase matching
  against fixed example phrasings. Phrasing far from those examples is only understood through CLIP. The
  optional Claude layer is untested live.
- Room-label associations (toilet → bathroom, washer → laundry) are assumptions; a mislabeled room or a laundry
  room without a machine misleads them.
- `LOGIT_SCALE` and the confidence values in `config.py` were hand-tuned on the sample model, not calibrated.
- Sessions live in memory; restarting the backend clears them.
- Photon's shared iMessage line can't start a conversation: a caller must text it first (see Messaging).
