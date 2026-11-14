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


_load_dotenv(REPO_ROOT / ".env.local")
_load_dotenv(REPO_ROOT / ".env")

# Shared bearer credential for ElevenLabs voice tools; provisioning stores it in .env.local.
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
# many sweeps (the kitchen has 11) always has one that happens to score well against any text.
ROOM_KEYWORD_BOOST = 20.0
# Score given to rooms without sweeps (no photos), as a quantile of the imaged candidates' scores. In scripted
# truthful-caller runs on the sample model, 0.5 kept kitchen/living/hallway callers correct; 0.75 let photo-less rooms
# soak up probability, and averaging likelihoods instead (more "Bayesian") made them win far too often.
IMAGELESS_SCORE_QUANTILE = 0.5
CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_MARGIN = 0.25
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

# Where in the located room (followups.refinement): at most MAX_REFINEMENTS "what's closest to you?" questions, asked
# only while the probability-weighted spread of the room's scan points exceeds REFINE_MIN_SPREAD_M (1.5 m: about the
# spacing of Matterport sweeps, so below it the plan can't be more precise anyway) and the question is worth
# REFINE_MIN_GAIN bits over scan points. NEAR_SCALE: 1.5 makes a scan point whose close-up score for a landmark is one
# standard deviation above the room's others ~4x likelier to name it among three offered landmarks.
MAX_REFINEMENTS = 2
REFINE_MIN_SPREAD_M = 1.5
REFINE_MIN_GAIN = 0.15
NEAR_SCALE = 1.5
