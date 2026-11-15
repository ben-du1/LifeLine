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
# matches differ by only ~0.02-0.08. At 40 a clear "stove and refrigerator" reply gave the kitchen only 0.29; at 80 a
# single vague reply could almost decide alone. 60 lets one distinctive reply dominate while bedrooms (which look
# alike) still need follow-up questions.
LOGIT_SCALE = 60.0
# Sharpness for splitting a room's probability among its own sweeps (locator.update): photo-score differences between
# sweeps of one room are mostly noise for a room-level description, so they get a quarter of LOGIT_SCALE. Each room's
# total is computed exactly as before, so room-level results don't change.
IN_ROOM_LOGIT_SCALE = 15.0
# A caller naming the room type ("I'm in the bathroom") is strong evidence. At 5x it lost to image noise: a room with
# many sweeps (the kitchen has 11) always has one that happens to score well against any text. It is no longer taken
# on trust, though: the leading room is checked with a verification question before it is reported (see
# followups.verification), and that check (~7x at LAYOUT_YES/LAYOUT_NO) can overturn a claim. Only names said as a
# location count (questions.room_mentions), not "big like a living room".
ROOM_KEYWORD_BOOST = 20.0
# "I'm not in the bathroom": rooms of that type keep this fraction of their probability.
ROOM_DENIAL_FACTOR = 0.2
# Corroboration (followups.verification): the best check must be worth at least this many bits about "this room or
# not" (judged at 50/50). Below it nothing in the bank can check the room (two identical photo-less closets), and the
# room is reported unverified rather than asking a question that can't help.
VERIFY_MIN_GAIN = 0.15
# A check corroborates the room only if its answer multiplied the room's odds by at least this much (a layout answer
# that fits only this room gives ~7x; one that fits the runner-up equally gives ~1x and doesn't count).
VERIFY_MIN_ODDS = 2.0
# After the caller is told where responders will look, questions continue (to confirm, then where in the room) while
# the room is less than this likely. Answers can still move the estimate; the caller is re-told if the room changes.
CONFIRM_UNTIL = 0.9
# When the dispatcher presses "Ask next best question", any question worth more than this (bits) is asked: they decide
# it's worth the caller's time, so only questions that can't change anything at all are refused.
DISPATCHER_MIN_GAIN = 0.005
# Score given to rooms without sweeps (no photos), as a quantile of the imaged candidates' scores. In scripted
# truthful-caller runs on the sample model, 0.5 kept kitchen/living/hallway callers correct; 0.75 let photo-less rooms
# soak up probability, and averaging likelihoods instead (more "Bayesian") made them win far too often.
IMAGELESS_SCORE_QUANTILE = 0.5
CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_MARGIN = 0.25
# Automatic follow-ups per call (verification and confirmation questions included; "what's closest to you?" and
# questions the dispatcher asks for are not counted). Raised from 6 when verification was added: each checked room
# costs one question.
MAX_QUESTIONS = 8
# Stop asking follow-ups when the best one is expected to resolve less than this many bits about the room. 0.05 bits
# is roughly a question that would move a 50/50 pair of rooms to 60/40 only half the time: not worth a caller's text.
MIN_INFORMATION_GAIN = 0.05
# How often a caller's floor answer is right. Not 1: callers confuse "first" and "ground" floor and sometimes count
# from 0. At 0.9 a single answer moves a 50/50 two-floor split to 90/10, enough to focus the follow-ups on that floor
# while a later contradicting answer (e.g. naming a room only on the other floor) can still win.
FLOOR_ANSWER_ACCURACY = 0.9
# The floor question goes first while no floor is at least this likely (see followups.choose).
FLOOR_FIRST_UNTIL = 0.8
# "How many flights of stairs did you climb?" (asked when the caller doesn't know the floor) is weaker evidence than
# a floor answer: the caller may have entered from a raised porch or a lower street level, come down from above, or
# miscounted landings. 0.75 still moves a three-floor split to 75% on the floor it names: one more answer settles it.
STAIRS_ANSWER_ACCURACY = 0.75

# Where in the located room (followups.refinement): at most MAX_REFINEMENTS "what's closest to you?" questions, asked
# only while the probability-weighted spread of the room's scan points exceeds REFINE_MIN_SPREAD_M (1.5 m: about the
# spacing of Matterport sweeps, so below it the plan can't be more precise anyway) and the question is worth
# REFINE_MIN_GAIN bits over scan points. NEAR_SCALE: 1.5 makes a scan point whose close-up score for a landmark is one
# standard deviation above the room's others ~4x likelier to name it among three offered landmarks.
MAX_REFINEMENTS = 2
REFINE_MIN_SPREAD_M = 1.5
REFINE_MIN_GAIN = 0.15
NEAR_SCALE = 1.5

VOICE_TOOL_TOKEN = os.environ.get("VOICE_TOOL_TOKEN", "")

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

CLIP_MAX_LIKELIHOOD_RATIO = 4.0

POSTERIOR_FLOOR = 0.02

MIN_EVIDENCE_MESSAGES = 2

SIBLING_RATIO = 0.25

CLIP_CALIBRATED = os.environ.get("CLIP_CALIBRATED", "1") != "0"

CLIP_CONTRAST_CENTER = 0.04

CLIP_CONTRAST_SCALE = 0.015

CLIP_RELATIVE_SLOPE = 0.75

CLIP_RELATIVE_MAX_Z = 2.0

CLIP_ABSOLUTE_MAX_LOGIT = 3.0

COMMONSENSE_IMPLAUSIBLE_MAX = 0.08

COMMONSENSE_TYPICAL_MIN = 0.6

FLOORING_IMPLAUSIBLE = 0.03

MIN_WITHIN_ROOM_GAIN = 0.2

TIE_PROBABILITY_RATIO = 0.5

TIE_PROFILE_RMS = 0.1

TIE_MIN_GROUP_PROBABILITY = 0.5

ADJACENCY_WALL_GAP_M = 0.4

ADJACENCY_MIN_SHARED_M = 0.5

FRONT_MIN_EXTERIOR_M = 0.5

FRONT_MAX_SETBACK_M = 3.0

POSITION_P_YES = 0.85

POSITION_P_NO = 0.15

POSITION_ONLY_BELOW_GAIN = 0.15

BIG_ROOM_M2 = 25.0

NARROW_RADIUS_M = 2.5

REGION_MIN_RADIUS_M = 1.0

MAX_WITHIN_ROOM_QUESTIONS = 2

ZONE_SIDE_FRACTION = 0.17

ZONE_MIN_AXIS_M = 2.5

ZONE_LANDMARK_MIN_CONTRAST = 0.05

VEXCEL_API_KEY = os.environ.get("VEXCEL_API_KEY", "")

VEXCEL_EXTRACT_URL = os.environ.get("VEXCEL_EXTRACT_URL", "https://api.vexcelgroup.com/v2/ortho/extract")

AERIAL_SIZE_M = float(os.environ.get("AERIAL_SIZE_M", "110"))

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")


# ==== caller-initiated calls: one block per workstream; edit only your own block ====
# ---- A: address matching (resolve.py, addresses.py)
# resolve.py bands, over the same 0..1 score as ADDRESS_MATCH_THRESHOLD. "match" needs an exact address (house number
# and street) scoring at least ADDRESS_MATCH_THRESHOLD. Anything else at or above ADDRESS_CLARIFY_THRESHOLD is asked
# about once: "clarify" when the best model leads the next one by at least ADDRESS_CLARIFY_MARGIN, "choose" between
# the top two when it doesn't. 0.65 keeps a swapped-digit house number (13212 for 13122 scores 0.79) and a missing
# house number, but not a street that shares one word ("5819 Eldridge Glen Dr" scores 0.30). A neighbouring house
# scores ~0.09 below the exact one (13120 vs 13122), so 0.10 makes two neighbours a choice when neither is exact.
ADDRESS_CLARIFY_THRESHOLD = 0.65
ADDRESS_CLARIFY_MARGIN = 0.10
# How many catalog rows (catalog.py, ~15,000 models) are scored per caller address after the street-word prefilter.
# Scoring is ~50 us per row, so 60 keeps resolve() in single milliseconds on a laptop.
ADDRESS_CATALOG_CANDIDATES = 60
# ---- B: intake and handoff (sessions.py, handoff.py, main.py)
# ---- C: caller language (questions.py, semantics.py)
# questions.looks_like_address: how many words a street name may have between the house number and the street
# word ("13122 Chipman Glen Dr" is 2). Larger accepts longer names but also more sentences that merely contain a
# number ("2 people are in the back bedroom by the door").
ADDRESS_MAX_NAME_WORDS = 4
# ---- D: voice and platform (voice.py, phones.py, .env loading)
