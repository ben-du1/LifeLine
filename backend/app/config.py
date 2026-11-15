"""Settings loaded from the repo-root .env, plus the locator's tunable constants."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
BUILDINGS_DIR = DATA_DIR / "buildings"


def _load_dotenv(path: Path) -> None:
    # Tiny .env reader so we don't need python-dotenv. Real environment variables win.
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(REPO_ROOT / ".env")

VOICE_TOOL_TOKEN = os.environ.get("VOICE_TOOL_TOKEN", "")

MATTERPORT_API_URL = "https://api.matterport.com/api/models/graph"
MATTERPORT_TOKEN_ID = os.environ.get("MATTERPORT_TOKEN_ID", "")
MATTERPORT_TOKEN_SECRET = os.environ.get("MATTERPORT_TOKEN_SECRET", "")

MESSAGING_MODE = os.environ.get("MESSAGING_MODE", "simulated")  # "simulated" or "photon"
MESSAGING_SIDECAR_URL = os.environ.get("MESSAGING_SIDECAR_URL", "http://localhost:8787")
DEV_TEST_PHONE = os.environ.get("DEV_TEST_PHONE", "")

# Local paraphrase matching of caller messages (semantics.py). Thresholds from a 17-message calibration set: true
# paraphrases scored 0.63-0.92 against their cue's examples while unrelated messages stayed under 0.55, and any
# message that is closer to a neutral one ("please help me") than to a cue by less than 0.1 is ignored.
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MIN_SCORE = 0.6
SEMANTIC_NEUTRAL_MARGIN = 0.1
# A semantic size cue counts only if it beats the opposite size by this much: the model confuses opposites.
SEMANTIC_SIZE_MARGIN = 0.15

# Optional Claude layer for follow-ups (llm.py). Off unless a key is set; the keyword rules are the fallback.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-opus-5-5")
# A caller is waiting on each reply, so a slow answer is worse than the keyword fallback: give up after 10 s.
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "10"))

# Optional: a Street View Static API image on the approach card (approach.py). The key is sent to the browser inside
# the image URL, so use a key restricted to that API (and the dashboard's referrer). Without it the card still links
# to Street View in Google Maps, which needs no key.
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Embedding model. ViT-B-32 is the smallest open_clip model that matches reliably and stays fast on a laptop CPU.
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Address matching (addresses.py). 0.80 accepts typos, a missing suffix/city, and a neighbouring house on the same
# street (13120 vs 13122 scores 0.88), but rejects a different street or a different city with the same street
# name (0.30-0.53 in tests/test_addresses.py). Anything below exact is shown to the dispatcher as an approximate match.
ADDRESS_MATCH_THRESHOLD = 0.80
# Search results per Matterport query; several query variants are run (see matterport.search_models).
ADDRESS_SEARCH_PAGE_SIZE = 20
# How many matching models to try before giving up. Some models have no usable interior (no rooms or no photos on
# any floor), so the next-best match is tried; each failed attempt costs one GraphQL call.
ADDRESS_MATCH_ATTEMPTS = 3

# Skybox resolution to download. "low" faces (~512 px) are plenty for CLIP's 224 px input and download quickly.
SKYBOX_RESOLUTION = "low"

# Locator tuning, hand-tuned on the sample model (3QuftVC2agD). CLIP cosine similarities for good vs. poor
# matches differ by only ~0.02-0.08, so LOGIT_SCALE converts cosine differences into log-likelihood (nats per cosine
# unit): 0.01 of cosine ≈ 0.6 nats ≈ 1.8x. At 40 a clear "stove and refrigerator" reply gave the kitchen only 0.29; at
# 80 a single vague reply could almost decide alone. Since the room-level update is now capped by
# CLIP_MAX_LIKELIHOOD_RATIO, 60 mainly sets how fast evidence saturates: a ~0.025 cosine lead (a "clear" match)
# reaches the cap, a ~0.01 lead (noise level) gives under 2x. It also spreads mass between sweeps inside a room.
LOGIT_SCALE = 60.0
# Most one free-text photo match can favor any room over any other (locator.clip_likelihood). Uncapped, one lucky view
# (a caller's "carpeted floor" matching a garage's grey concrete) put 90%+ on a wrong room. At 4x, one match moves a
# uniform prior to at most 4/7 = 57% in a 4-room building (36% with 8 rooms), below CONFIDENCE_THRESHOLD, so photos
# alone need a second clue (a cue, an answered question, a room-type word via ROOM_KEYWORD_BOOST; none are capped).
# 6x was tried first: in a 5-room building it reached exactly 60% and "located" the caller from one photo match.
CLIP_MAX_LIKELIHOOD_RATIO = 4.0
# A caller naming the room type ("I'm in the bathroom") is strong evidence. At 5x it lost to image noise: a room with
# many sweeps (the kitchen has 11) always has one that happens to score well against any text.
ROOM_KEYWORD_BOOST = 20.0
# A caller denying a room type ("I'm not in the garage", "not the kitchen") multiplies that type's candidates by this.
# Weaker than the 1/20 mirror of the boost: Matterport labels can be wrong (a "kitchen" area labeled "dining room")
# and a denial only rules out one name, but 0.1 still takes a 50% room down to ~9%, so it stops being the answer.
ROOM_DENIAL_FACTOR = 0.1
# Score given to rooms without sweeps (no photos), as a quantile of the imaged candidates' scores. In scripted
# truthful-caller runs on the sample model, 0.5 kept kitchen/living/hallway callers correct; 0.75 let photo-less rooms
# soak up probability, and averaging likelihoods instead (more "Bayesian") made them win far too often.
IMAGELESS_SCORE_QUANTILE = 0.5
CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_MARGIN = 0.25
# After each caller message, mix this share of the uniform-room prior back into the posterior (locator.apply_floor).
# Without it one strong cue (x20 room-type boost, CLIP at scale 60) drove rooms to 100%/0%, which overstates what a
# single text tells us and leaves a later contradicting answer nothing to recover. 0.02 caps a room at ~98%, well above
# CONFIDENCE_THRESHOLD, and gives every room >= 0.02/n_rooms, enough for a few contradicting answers to bring it back.
POSTERIOR_FLOOR = 0.02
# "Located" needs this many caller messages that carried evidence: one vague reply that CLIP happens to like is too
# thin. Exception: a single message that names the only room of its type ("I'm in the kitchen", "next to the stove"
# in a one-kitchen house) may decide alone (locator.is_located).
MIN_EVIDENCE_MESSAGES = 2
# A same-type room (e.g. Bedroom 2 vs Bedroom 3) holding at least this share of the leader's probability blocks
# "located": identical rooms look alike to CLIP, so 75/20 between siblings is mostly noise, not evidence.
SIBLING_RATIO = 0.25
MAX_QUESTIONS = 6
# Stop asking follow-ups when the best one is expected to resolve less than this many bits about the room. 0.05 bits
# is roughly a question that would move a 50/50 pair of rooms to 60/40 only half the time: not worth a caller's text.
MIN_INFORMATION_GAIN = 0.05
# How often a caller's floor answer is right. Not 1: callers confuse "first" and "ground" floor and sometimes count
# from 0. At 0.9 a single answer moves a 50/50 two-floor split to 90/10, enough to focus the follow-ups on that floor
# while a later contradicting answer (e.g. naming a room only on the other floor) can still win.
FLOOR_ANSWER_ACCURACY = 0.9
# The floor question goes first while no floor is at least this likely (see followups.choose).
FLOOR_FIRST_UNTIL = 0.8

# Calibrated CLIP feature presence (followups._clip_presence). Set CLIP_CALIBRATED=0 in .env to go back to the old
# relative z-score (e.g. if the eval scenarios get worse before the constants below are checked on real CLIP).
CLIP_CALIBRATED = os.environ.get("CLIP_CALIBRATED", "1") != "0"
# A view's "contrast" is its cosine with a feature's prompts minus its mean cosine with neutral prompts ("a photo of
# an empty room", ...); a candidate takes its best view. Expected ViT-B-32 magnitudes, from CLIP's usual behaviour
# rather than measured here: a matching specific prompt beats the neutral ones by ~0.05-0.10, an absent object trails
# them by 0-0.04. The specific prompt has a small head start even when the object is absent (it still says "a room"),
# and the max over 4 faces adds a little more, so the 50% point sits well above zero: a view that only ties the
# neutral prompts reads as absent. NOT YET CHECKED ON REAL CLIP: print view_contrast() for a known building (e.g. the
# washer in a house without one) and move the center to between the absent and present clusters.
CLIP_CONTRAST_CENTER = 0.04
# Contrast per unit of log-odds: 0.015 maps +0.08 (clear match) to ~0.93, a tie with the baseline (0) to ~0.07, and
# a weak match such as a small mirror (+0.03) to ~0.34, which a "yes" still favours 5x over a clearly absent room.
CLIP_CONTRAST_SCALE = 0.015
# The old relative signal (z-score of the raw score across imaged candidates) is added in log-odds at half its old
# slope of 1.5: it still ranks candidates when the absolute calibration is off, but it can no longer make a feature
# that trails the neutral prompts everywhere (no washer in the house) look present in some room. z is clipped to +-2:
# with 50 candidates one odd sweep can sit 7 std out, which at full weight would outvote the calibration.
CLIP_RELATIVE_SLOPE = 0.75
CLIP_RELATIVE_MAX_Z = 2.0
# Bound on the absolute term's log-odds (+-3 = 0.05-0.95), so a mis-set center can't saturate every candidate.
CLIP_ABSOLUTE_MAX_LOGIT = 3.0
# Commonsense guard (commonsense.py): the allowed range of P(feature present | room) from the room's type. An
# "implausible" feature (a toilet in a garage) is capped at 0.08, not 0: labels can be wrong and callers misspeak, so
# a stubborn caller can still win, but one CLIP hit can't. 0.08 is just above _combine's 0.05 floor, so it only
# binds when CLIP pushed the value up. A "typical" feature (a bed in a bedroom) is at least 0.6, so a bad photo
# can't argue it away. Both chosen so a single "there's a toilet" leaves a garage under 10% in a 5-room house.
COMMONSENSE_IMPLAUSIBLE_MAX = 0.08
COMMONSENSE_TYPICAL_MIN = 0.6
# Flooring answers (followups._flooring_presence). CLIP's floor prompts mostly see colour and texture: a grey concrete
# garage floor out-scored real carpet on "carpet floor". A room type that rules a material out (carpet in a garage,
# bathroom, kitchen or laundry) gives that answer 0.03 regardless of CLIP: rare, not impossible, since labels can be
# wrong. With it, "carpet" leaves a garage at ~3% of a typical house instead of letting its photos decide.
FLOORING_IMPLAUSIBLE = 0.03
# Inside a big room (see followups.choose_within_room): ask "which are you closest to: the stove or the fireplace?"
# only when it is expected to resolve at least this many bits about the caller's SWEEP. Higher than
# MIN_INFORMATION_GAIN because the room is already known: a responder walking in finds the caller within seconds
# anyway, so a near-question is only worth a text when it clearly splits the room (0.2 bits ~ a 50/50 pair of ends
# moving to 75/25 on average).
MIN_WITHIN_ROOM_GAIN = 0.2
# Ties and position questions (ties.py, adjacency.py).
# A room is "tied" with the leading room when it is at least this fraction as likely and looks the same to the
# feature model (same room type, or a near-identical feature profile). 0.5: the README's identical-bedroom case ended at
# 55% vs 31% (ratio 0.56), which a dispatcher should hear as "one of these two", not as a pick.
TIE_PROBABILITY_RATIO = 0.5
# Root-mean-square difference of two rooms' P(yes) over the content features below which they count as alike. Identical
# empty bedrooms differ only by CLIP noise (one third of the weight, see followups._combine): ~0.03-0.08.
TIE_PROFILE_RMS = 0.1
# ...and only once the tied rooms together hold at least half the belief: at the start of a call (uniform prior) three
# bedrooms holding 30% are not yet "it's one of these".
TIE_MIN_GROUP_PROBABILITY = 0.5
# Shared walls: adjacent rooms' outlines are a wall's thickness apart (interior walls ~0.1-0.2 m plus scan error), and
# must run together for at least ~half a doorway; shorter contacts are corners touching.
ADJACENCY_WALL_GAP_M = 0.4
ADJACENCY_MIN_SHARED_M = 0.5
# Front of the house: at least this much outline must face the street unobstructed, within this distance of the
# house's street-side edge (a porch or bay window can stick out a few meters).
FRONT_MIN_EXTERIOR_M = 0.5
FRONT_MAX_SETBACK_M = 3.0
# P(yes | candidate) for position questions, matching the room-label prior (followups._type_presence): geometry from
# scans is good but doorways, open plans and callers' sense of "next to" are not exact.
POSITION_P_YES = 0.85
POSITION_P_NO = 0.15
# Position questions ("next to the kitchen?") are harder for a stressed caller than "is there a toilet?", so they are
# only considered once the best question about the room's contents is expected to resolve less than this many bits:
# in practice when the remaining rooms look alike. Keeps the content-first flow of the scripted evals unchanged.
POSITION_ONLY_BELOW_GAIN = 0.15
# Where inside a big room (subroom.py). Open-plan living/kitchen spaces in the sample models are 30-60 m² with 5-11
# sweeps; a bedroom is 10-15 m². 25 m² is about where a responder at the door can no longer see the whole room at a
# glance (smoke, furniture, an L-shape), so only rooms at least this big get within-room questions and a zone.
BIG_ROOM_M2 = 25.0
# A region radius above this still covers ~20 m² (a whole bedroom's worth of floor), so it is worth narrowing. Sweeps
# are typically 1.5-2.5 m apart, so two adjacent likely sweeps give ~1 m and pass easily.
NARROW_RADIUS_M = 2.5
# The caller is near a sweep, not on it: the region never shrinks below 1 m, about the spacing error of a scan point.
REGION_MIN_RADIUS_M = 1.0
# Within-room questions asked after the room is confident, before "located". 2 keeps the whole call under the usual
# MAX_QUESTIONS budget, and the caller has already answered the room questions: more reads like stalling.
MAX_WITHIN_ROOM_QUESTIONS = 2
# Zone wording: the centroid must be this far (fraction of the room's north-up extent) from the middle to be called
# "west side" etc.; an axis shorter than ZONE_MIN_AXIS_M is too short to have sides. A landmark is named only if the
# likely sweeps' P(yes) beats the room average by ZONE_LANDMARK_MIN_CONTRAST. CLIP is 1/3 of P(yes) (the room label
# is the other 2/3 and constant within a room), so 0.05 is about a 0.15 CLIP-probability lead: a visible difference.
ZONE_SIDE_FRACTION = 0.17
ZONE_MIN_AXIS_M = 2.5
ZONE_LANDMARK_MIN_CONTRAST = 0.05
# ---- Geo-referenced map view and aerial imagery (geo.py) --------------------------------------------------------
# Vexcel orthophotos (7.5 cm/px). Hackathon keys cover Houston and Los Angeles only; elsewhere the API answers 403
# and the map view falls back to the plan alone. Empty key = aerial imagery off (no request is ever made).
VEXCEL_API_KEY = os.environ.get("VEXCEL_API_KEY", "")
VEXCEL_EXTRACT_URL = os.environ.get("VEXCEL_EXTRACT_URL", "https://api.vexcelgroup.com/v2/ortho/extract")
# Side of the square clipped around the building, in meters. 110 m covers a house lot plus the street and the
# neighbours on every side (responders orient by those), and at the 1400 px request size it is ~7.9 cm/px, close to
# Vexcel's native 7.5 cm, so the image is neither upsampled much nor wastefully large.
AERIAL_SIZE_M = float(os.environ.get("AERIAL_SIZE_M", "110"))
# Google Maps JavaScript API key for the dashboard's map view. Browser keys are public by design (they ship in page
# source), so this one is sent to the dashboard; restrict it by HTTP referrer in the Google Cloud console.
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
