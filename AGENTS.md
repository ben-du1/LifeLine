# LifeLine: technical spec and status (for agents and developers)

> The full engineering reference: status of every part, module and API reference, matching maths, evaluation,
> setup and demo steps. The public overview for judges and visitors is [README.md](README.md). Keep this file
> current when behaviour changes.

Built for the CoStar × Matterport track ("Inside and Out: AI for the Built World") at HackPrinceton Fall 2026
(Nov 13–15, 2026; submissions close **Nov 15, 4:00 PM EST**).

## Status (2026-11-14)

| Area | State |
|---|---|
| Core loop (call → texts → room estimate → dot on plan), simulated caller | Working, verified |
| Address → Matterport model (fuzzy matching) | Working, verified against the live API |
| Follow-up questions (cues, information gain, paraphrase matching) | Working; scripted eval 15/19, no confidently wrong rooms |
| Room photos per call (API + dashboard gallery) | Working, verified |
| Photon iMessage (send + receive) | Working, verified with a real phone |
| Orientation: north-up plan + nearest street (OpenStreetMap) | Working, verified on two houses |
| Multi-floor buildings | Working, verified: eval 4/4 rooms and floors. In the browser, the floor tabs follow the estimate and the plan switches floors after the caller's answer. |
| Optional Claude layer (`llm.py`) | Written and unit-tested with a stub; **never run live** (no key) |
| Where inside a big room (`subroom.py`: region + zone label, up to 2 within-room questions) | Unit-tested on a synthetic building; within-room questions wait for `followups.choose_within_room`; not run on a real model |
| Demo preflight (`scripts/preflight_demo.py`) + "Demo buildings" menu (`/api/demo`) | Written; unit-tested on a synthetic cache and dry-run with stubs. **Not yet run against the live API** |
| Vexcel aerial overlay | Tried and removed (see "Orientation"). CoStar data: not used. |
| Map view data: GeoJSON rooms + Vexcel aerial photo (`geo.py`) | Written and tested with mocked Matterport/Vexcel; **never run live** |

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
6. **Orientation for responders.** The plan is drawn north-up, with the nearest street on the side where it really
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
| `adjacency.py`  | Shared walls between rooms and street-facing rooms, from the floor plan; "position" follow-ups ("next to the kitchen?"). |
| `ties.py`       | Rooms that are tied with the leader and look alike, named together to the dispatcher. |
| `semantics.py`  | Local paraphrase matching of caller messages (sentence embeddings). |
| `orientation.py`| North-up heading (Matterport) and the nearest street (OpenStreetMap Nominatim) for the floor plan. |
| `geo.py`        | Rooms, outline and sweeps in lon/lat (GeoJSON) for a map view; optional Vexcel aerial photo around the house. |
| `llm.py`        | Optional Claude reading of replies and wording of follow-ups (falls back to keyword rules). |
| `questions.py`  | Opening/closing messages and reply parsing (yes / no / free text / unsure). |
| `subroom.py`    | Where inside a big room: in-room region (centroid + radius) and a zone label ("kitchen end, near the stove"). |
| `sessions.py`   | Session state machine and conversation flow. |
| `messaging.py`  | Outbound messages: `simulated` (dashboard only) or `photon` (via the sidecar). |
| `demo.py`       | Demo building list (`demo_buildings.json`) and its cache readiness, for `/api/demo` and `scripts/preflight_demo.py`. |
| `main.py`       | FastAPI routes, WebSocket broadcast, static files for cached building assets. |

### API

| Method | Path                           | Body / notes |
|--------|--------------------------------|--------------|
| GET    | `/api/config`                  | `{messaging_mode, dev_test_phone, max_questions, followups, google_maps_key, aerial}` for the dashboard, plus `google_maps_api_key` when set (a browser key) |
| GET    | `/api/demo`                    | Demo buildings from `demo_buildings.json`, ready first: `[{address, label, note, model_id, ready, cached, orientation, match_score, problems[], score?}]`. Cache only (no network); `score` only if `compat.py` exists |
| POST   | `/api/calls`                   | `{phone, address}`: simulated 911 call; creates a session |
| GET    | `/api/sessions`                | All sessions (snapshots) |
| GET    | `/api/sessions/{id}`           | One session snapshot |
| GET    | `/api/sessions/{id}/rooms`     | Every room of the call's building, most likely first: `{index, name, types, floor, area_m2, probability, sweeps[{candidate, x, y, probability, views[url]}]}` |
| POST   | `/api/sessions/{id}/reply`     | `{text}`: simulated caller reply from the dashboard |
| POST   | `/api/inbound`                 | `{from, text}`: real caller reply, posted by the messaging sidecar |
| WS     | `/ws`                          | Pushes `{type: "session", session}` on every change |
| GET    | `/api/buildings/{model_id}`    | Cached `building.json` (`floors`, `floorplans` per floor, rooms with `floor`/walls/polygon/centroid, candidates) with asset URLs, plus `orientation` (`yaw_deg`, `street{name, distance_m, bearing_deg, side, direction_deg}`) once fetched |
| GET    | `/api/buildings/{model_id}/geo`    | GeoJSON `FeatureCollection` of rooms (`properties{kind:"room", index, name, floor, types}`) and the building outline (`kind:"building"`), plus `center{lat, lon}`, `yaw_deg`, `candidates[{candidate, lat, lon, room, floor}]`, `bounds[[s, w], [n, e]]`. 404 if not loaded or no geolocation |
| GET    | `/api/buildings/{model_id}/aerial` | `{url, bounds[[s, w], [n, e]], source:"Vexcel"}`, or 404 with the reason (no key, outside coverage, error) |
| GET    | `/buildings/{model_id}/...`    | Cached floorplan, view and aerial images |

Session snapshot fields: `id, phone, address, status, status_detail, model_id`,
`match{id, name, address, score, exact}` (the Matterport property in use), `messages[{role, text, ts}]`
(`role` is `dispatch`, `caller` or `system`), `questions_asked, max_questions`,
`rooms[{index, name, floor, floor_label, probability, has_photos, types, area_m2}]` (sorted), `floors[{index, label, probability}]`,
`orientation` (true once north/street data is cached), `candidate_probabilities[]` (aligned with the building's
`candidates`), and `estimate{room, room_name, floor, floor_label, candidate, x, y, room_probability,
runner_up_probability, confident}`, and `tie_group[{index, name, probability}]` (rooms that can't be told apart
from the leading one, leader first; empty when there is no tie, see `ties.py`),
`evidence[{ts, text, top_before, changes[{name, delta}]}]` (last 20
caller messages: the top-3 room probability changes each caused, shown under the message on the dashboard).
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
- **Update:** `posterior ∝ prior × L`, applied once per reply (`locator.clip_likelihood`). Per room, `L` is
  `exp(LOGIT_SCALE × (best sweep score − spread × expected max of N draws))`, where spread is the scores' robust (MAD)
  spread and N the room's sweep count, so a room with many sweeps isn't favored for one lucky view. It is clipped so no
  room gets more than `CLIP_MAX_LIKELIHOOD_RATIO` (4x) another's: photos alone can't locate a caller in one message.
  Inside a room, sweeps are reweighted by `exp(LOGIT_SCALE × score)` without changing the room's total. Image-less
  candidates borrow the score of photographed rooms that share their room type (log-mean-exp of each such room's
  sweeps, averaged over those rooms), so a photo-less bathroom and a photographed one split "there's a toilet" evenly;
  the follow-up model's CLIP presence is imputed the same way. With no photographed room of the same type they get
  the median imaged score (`IMAGELESS_SCORE_QUANTILE`, roughly neutral). If the reply names a room
  type, candidates in rooms with that classification are multiplied by `ROOM_KEYWORD_BOOST`.
- **Free-text replies** are embedded and scored as above. A reply that only answers our question ("yes", "tile",
  "none of those") is not embedded; its meaning comes from the follow-up model below.
- **Confidence rule:** the top room's probability ≥ `CONFIDENCE_THRESHOLD` and it leads the runner-up by at least
  `CONFIDENCE_MARGIN`. "Located" also needs (`locator.is_located`): no room of the same type with at least
  `SIBLING_RATIO` × the leader's probability (identical bedrooms), and at least `MIN_EVIDENCE_MESSAGES` caller
  messages that carried evidence, unless one message named the only room of its type or a feature only it has ("I'm in
  the kitchen" in a one-kitchen house). Short only of evidence, the best follow-up is asked even if its expected gain is
  below `MIN_INFORMATION_GAIN`.
- **Posterior floor:** after each evidence-bearing message, `POSTERIOR_FLOOR` (2%) of the uniform-room prior is mixed
  back in, so no room shows 0% or 100% and a later contradicting answer can still win.
- **Output:** the highest-probability candidate inside the top room; its `(x, y)` (model meters) is drawn on the
  floor plan. The dashboard shows every photo of that room, or of any room the dispatcher clicks.
- **Current values** (`config.py`): `LOGIT_SCALE=60`, `CLIP_MAX_LIKELIHOOD_RATIO=4`, `ROOM_KEYWORD_BOOST=20`, `IMAGELESS_SCORE_QUANTILE=0.5`,
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

### Map view data and aerial imagery (`geo.py`)

The same transform, plus a local meters-per-degree scale, puts every room, sweep and a convex-hull building outline
in lon/lat (`/api/buildings/{id}/geo`). Matterport's raw geocoordinates are cached in
`data/buildings/<id>/geo.json`, shared with `orientation.py`, so they cost one GraphQL call per building.

With `VEXCEL_API_KEY` set, the backend also clips one Vexcel orthophoto (7.5 cm/px, `AERIAL_SIZE_M` = 110 m square,
1400 px) around the house after orientation loads, in the background, and caches it as `aerial.jpg` +
`aerial.json` (bounds). Hackathon keys cover Houston and Los Angeles only; elsewhere Vexcel answers 403, logged
once, and the endpoint returns 404 with the reason. The photo is approach context (street, driveway, neighbours):
the few-meter position error that ruled out the old plan-on-roof overlay still applies, so the rooms are not
claimed to sit exactly on the roof.
### Approach card (`approach.py`, `ApproachCard.jsx`)

Under the plan, the dashboard says which side of the building and which floor the caller is on, in words for
responders outside: "2nd floor, rear (north) side, away from Pointer Street. Caller's room has an outside wall on the
north side, about 4 m from its east end."

- **Outside walls:** per floor, a room's wall is shared where another room on that floor is right behind it (a
  parallel wall within 0.45 m, the wall thickness, or a point 0.6 m outside it inside that room's outline). The rest
  is the building outline. An interior room says so and names the nearest outside wall and its distance.
- **Side and street:** the chosen wall's outward normal is turned into a compass bearing with the model's yaw; it is
  "front" within 60° of the street's bearing and "rear" beyond 120°.
- **Street View:** a keyless Google Maps link from ~18 m outside the wall, looking at it. This needs the model's
  geolocation, cached in `geo.json` next to `orientation.json`. If `GOOGLE_MAPS_API_KEY` is set, the card also shows
  a Street View Static image; that key is visible to the browser, so restrict it.
- The text the caller receives is unchanged; the dispatcher's status line adds "Approach: rear (north) side, ...".

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
  - **CLIP:** the candidate's photos against 3 phrasings of the feature, minus neutral prompts ("an empty room",
    "a white wall", ...) in the same view, best view per candidate, through a fixed logistic and nudged by the
    old z-score rank. A feature absent from the whole building now reads as absent everywhere. Constants in
    `config.py` (`CLIP_CONTRAST_*`) are reasoned, not yet measured on real CLIP; `CLIP_CALIBRATED=0` reverts.
  - **Room label:** Matterport's classification or label, counted double. A toilet implies a bathroom, which lets
    rooms *without photos* be found.
  - **Geometry:** floor area for size questions; bounding-box aspect ratio for "long and narrow".
  - **Commonsense guard** (`commonsense.py`): a room-type × feature table (typical / possible / implausible) bounds
    the result, so CLIP can't put a toilet in a garage or a bed in a kitchen (implausible ≤ 0.08, typical ≥ 0.6;
    also carpet in a garage, a "bigger" closet). Unrecognised room labels ("Room 1") get no bounds.
- **Question kinds:**
  - **Yes/no:** "Is there a washing machine or dryer?"
  - **Choice:** "What is the floor like: carpet, tile, wood, or concrete?" and "About how big is the room:
    closet-sized, bedroom-sized, or bigger?" (answered from geometry). For flooring, a room type that rules a
    material out (carpet in a garage, bathroom, kitchen or laundry) overrides CLIP, which confuses grey concrete with
    carpet. Floor materials named unprompted ("carpeted floor") count as the same answer, and a message that is only
    a floor description skips the free-text CLIP match.
  - **Which of these:** a list built from the three features that best separate the rooms still in play, e.g.
    "What can you see around you: shelves or hanging clothes, a washer or dryer, or a toilet, tub or shower?"
- **Choosing:** the question with the highest expected information gain about the *room*
  (`locator.expected_information_gain`) is asked. Topics the caller already settled (mentioned, answered, or
  "I don't know") are never asked again. A "which of these" list is not asked twice in a row.
- **Position questions (`adjacency.py`):** when the remaining rooms look alike, i.e. no content question is worth
  `POSITION_ONLY_BELOW_GAIN` (0.15 bits), questions about *where* the room is join the candidates:
  - **"Is the room you're in next to the kitchen, sharing a wall or a doorway?"**: two rooms on a floor are neighbours
    when their outlines run parallel within `ADJACENCY_WALL_GAP_M` (0.4 m) for at least `ADJACENCY_MIN_SHARED_M`
    (0.5 m). Only rooms with a name that is unique in the building are used as the reference ("the kitchen", never
    "Bedroom 2").
  - **"Is the room at the front of the house, on the <street> side?"**: once orientation is known, the street's
    bearing is rotated into model meters; a room is at the front if part of its street-facing outline has no room in
    front of it (ray cast) and is within `FRONT_MAX_SETBACK_M` (3 m) of the house's street-side edge.
  - Answers are soft, 0.85 / 0.15 like the room-label prior. Unit-tested on synthetic houses only; not yet run on a
    real model's walls.
- **Ties (`ties.py`):** a room is tied with the leader when it is at least half as likely and has the same room type
  or a near-identical feature profile, and the tied rooms hold at least half the belief. The best-guess (or located)
  detail then reads "Bathroom 1 or Bathroom 2 (58% / 42%), can't tell apart yet", and a system note tells the
  dispatcher to check each. The caller's message is unchanged.
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
**15/19**:
- 11/15 on the single-floor houses, including 3 of 4 open-ended paraphrases that contain no feature keywords.
- **4/4 rooms and 4/4 floors** on the 4-floor building.
- Since the Nov 14 reliability changes (re-run on the real API that day): all 4 misses end as a best guess naming
  the candidates; none is a confidently wrong "located" (the earlier code located Closet 4 and Closet 2 wrongly).
  More calls take 4-6 follow-ups before deciding. The per-scenario table below predates these changes.

It needs the three buildings cached; it ingests them on first run, which takes several minutes.

| Building | Caller is in | Opening | Result |
|---|---|---|---|
| Chipman Glen (vacant) | Bedroom 1 | "empty room with a window and a ceiling fan" | Correct, no follow-ups |
| Chipman Glen | Bedroom 2 | "empty room with two windows with blinds" | Best guess: Bedroom 3 55%, Bedroom 2 31% (identical empty rooms) |
| Chipman Glen | Kitchen/living | "I can see a stove and a fridge" | Correct, no follow-ups |
| Chipman Glen | Kitchen/living | "there's a brick fireplace" | Correct, no follow-ups |
| Chipman Glen | Bathroom 1 | "I'm in a cramped space" | Best guess: Bathroom 2 58%, Bathroom 1 42% (neither has photos) |
| Chipman Glen | Hallway | "I'm in a narrow space" | Correct after 3 follow-ups |
| 7921 Pointer St | Pantry/Laundry | "small room with lots of shelves" | **Wrong:** Closet 4. Labeled laundry, but there is no machine, so a truthful "no washer" counts against it |
| 7921 Pointer St | Closet 4 | "I'm in a cramped space" | Correct (best guess, 35%) |
| 7921 Pointer St | Bathroom | "it's tiny in here" | Correct after 3 follow-ups |
| 7921 Pointer St | Kitchen | "I'm next to the stove" | Correct, no follow-ups |
| 7921 Pointer St | Bedroom 3 | "it's a big room with a bed" | Correct after 1 follow-up |
| 7921 Pointer St | Kitchen | "I'm right next to where you cook" | Correct, no follow-ups |
| 7921 Pointer St | Bathroom | "I can barely move, I'm squeezed in by the tub" | Correct after 4 follow-ups |
| Chipman Glen | Kitchen/living | "the room is massive, there's a brick mantel" | Correct, no follow-ups |
| 7921 Pointer St | Closet 4 | "I can touch both walls and my coats are hanging next to me" | **Wrong:** Closet 2. Read correctly as a tight closet, but a truthful "no window" counts against Closet 4, whose photo shows the bedroom window through its door |
| 3012 Houston Ave (4 floors) | Office, floor 1 | "I'm downstairs in a room with wood paneling and a bookcase" | Correct, floor 1 at 97%, no follow-ups |
| 3012 Houston Ave | Game Room, floor 3 | "I'm in an empty room with built-in shelves" | Asked the floor ("3rd floor"), then correct |
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

Tests: `cd backend && .venv/Scripts/python -m pytest` (312 tests). `test_semantics.py` downloads the ~90 MB
paraphrase model on first run; the rest need no network.
Scenario check: `cd backend && .venv/Scripts/python scripts/eval_scenarios.py [-v]` (ingests the three test
buildings on first run).

## Demo script

0. **Preflight (the night before, and again on the morning of the demo, on the laptop with the `.env` keys):**
   `cd backend && .venv/Scripts/python scripts/preflight_demo.py`. For every building in the repo-root
   `demo_buildings.json` it resolves the address the way a call does, ingests the model if the cache is missing or
   stale (minutes per building, once), fetches orientation, runs two scripted caller messages through the real call
   flow, and prints a READY / NOT READY table with timings (exit code 0 only if all are READY). Messaging is forced to
   `simulated`, so it never texts anyone. `--only Chipman` checks one building; `--no-self-test` skips the scripted
   call. Add a building by appending `{address, label, model_id, note, self_test, expect_room}` to the JSON file.
   On the dashboard, the **Demo buildings** menu above the address fills it in (✓ cached, ⚠ not cached; a
   not-cached choice warns that the first call will take a few minutes to load).
1. Open the dashboard and enter a phone number and `13122 Chipman Glen Dr, Houston, TX`.
2. As the caller (simulated box, or a real phone in `photon` mode), describe the room. The sample house is empty,
   so describe fixtures: "there's a brick fireplace and a big window" (living area) or "I can see a stove and a
   fridge" (kitchen).
3. Answer the follow-ups (yes/no via the quick buttons, or type "tile", "closet-sized", "I see a stove") until the
   dot appears. Try a vague opening like "I'm in a cramped space" to see the follow-ups narrow it down.
4. Click any room (on the plan or in the probability list) to see all of its photos.
5. Multi-floor: call `3012 Houston Ave, Houston, TX` and say "I'm in an empty room with built-in shelves". The
   floor question comes first; answer "3rd floor". Floor tabs above the plan show each floor's probability.

## Known limitations

- Floors are identified by order only: the lowest floor is the "ground floor", and basements aren't recognised.
- An approximate address match (e.g. the neighbouring house) is a *different building*. The dashboard flags it, and
  the dispatcher should confirm it with the caller.
- Address matching is tuned for US street addresses; other formats are untested.
- Rooms without sweeps can only be chosen by room-type words.
- Visually similar rooms (the sample's empty bedrooms) are confused, sometimes with high confidence.
- Without `ANTHROPIC_API_KEY`, follow-ups use templates, and replies are read by keywords plus paraphrase matching
  against fixed example phrasings. Phrasing far from those examples is only understood through CLIP. The
  optional Claude layer is untested live.
- Room-label associations (toilet → bathroom, washer → laundry) are assumptions; a mislabeled room or a laundry
  room without a machine misleads them. The commonsense guard trusts labels too: in a room mislabeled "Garage", a
  real toilet is capped at 0.08 per answer (not ruled out).
- `LOGIT_SCALE` and the confidence values in `config.py` were hand-tuned on the sample model, not calibrated.
- Sessions live in memory; restarting the backend clears them.
- Photon's shared iMessage line can't start a conversation: a caller must text it first (see Messaging).

## Voice demo

See [voice setup and verification](voice/README.md).
