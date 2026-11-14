"""Local paraphrase matching for caller messages (no API key needed).

The keyword rules in followups.py only see words they list. A small sentence-embedding model (all-MiniLM-L6-v2,
~90 MB, ~3 ms per message on a laptop CPU) compares the message with example phrasings of each cue, so "the thing
you wash clothes in" counts as a washer and "I can touch both walls" as a tight space.

CLIP's text encoder was tried first and rejected: it rated "I'm stuck in a little space" as close to "a large open
room" as to "a small cramped space". MiniLM is trained on paraphrase pairs and gets objects and darkness right, but
it still confuses opposites (small vs. large), so followups.rule_cues lets size keywords win and only takes a
semantic size when it clearly beats the opposite size.
"""

import threading

import numpy as np

from . import config

# Example phrasings per cue. Feature keys match followups.FEATURES; "tight", "large" and "dark" are size/light cues.
EXAMPLES = {
    "tight": ["a very small cramped space", "I can touch both walls", "there is barely room to move",
              "I'm squeezed into a tiny spot", "I can't stretch out", "it's very confined in here"],
    "large": ["a large open room", "it's a huge spacious room", "there is a lot of space around me",
              "the room is really big"],
    "dark": ["it is completely dark", "there are no lights on", "I can't see anything, it's pitch black"],
    "toilet": ["there's a toilet", "I see a bathtub", "there is a shower"],
    "sink": ["there's a sink", "I see a faucet", "where you wash your hands"],
    "mirror": ["there's a mirror", "I can see my reflection"],
    "stove": ["there's a stove", "I see the oven and kitchen counters", "where you cook food"],
    "fridge": ["there's a refrigerator", "I see the fridge", "where the food is kept cold"],
    "washer": ["there's a washing machine", "I see a dryer", "the machine you wash clothes in"],
    "shelves": ["there are shelves", "clothes are hanging on a rod", "I see coats and hangers"],
    "bed": ["there's a bed", "I see a mattress", "where you sleep"],
    "sofa": ["there's a couch", "I see a sofa", "somewhere to sit and watch TV"],
    "table": ["there's a table with chairs", "where you eat dinner"],
    "tv": ["there's a TV", "I see a television screen"],
    "fireplace": ["there's a fireplace", "I see a brick mantel"],
    "window": ["there's a window", "I can see outside through the glass"],
    "fan": ["there's a ceiling fan", "a fan is spinning above me"],
    "stairs": ["there are stairs", "I see steps going up"],
    "front_door": ["I can see the front door", "I'm by the entrance of the house"],
    "sliding_door": ["there's a sliding glass door", "a glass door to the backyard"],
    "narrow": ["a long narrow corridor", "a long thin passage", "it's long and skinny"],
}
# Messages that carry no location cue. A cue must beat the best of these by NEUTRAL_MARGIN, so "please help me"
# never matches anything.
NEUTRAL = ["please help me", "I'm scared", "my phone battery is low", "I'm in a room", "someone is outside",
           "hurry", "I don't know", "yes", "no", "I'm hiding"]

_model = None
_examples: dict[str, np.ndarray] = {}
_neutral: np.ndarray | None = None
_lock = threading.Lock()


def _load():
    global _model, _neutral
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer  # lazy: torch import is slow

            model = SentenceTransformer(config.SEMANTIC_MODEL, device="cpu")
            for key, phrases in EXAMPLES.items():
                _examples[key] = model.encode(phrases, normalize_embeddings=True)
            _neutral = model.encode(NEUTRAL, normalize_embeddings=True)
            _model = model
    return _model


def warm_up() -> None:
    _load()


def scores(text: str) -> dict[str, float]:
    """Best similarity of `text` to each cue's examples, keeping only cues that clear SEMANTIC_MIN_SCORE and beat
    the neutral messages by SEMANTIC_NEUTRAL_MARGIN."""
    model = _load()
    vector = model.encode([text], normalize_embeddings=True)[0]
    neutral = float((_neutral @ vector).max())
    out = {}
    for key, examples in _examples.items():
        score = float((examples @ vector).max())
        if score >= config.SEMANTIC_MIN_SCORE and score >= neutral + config.SEMANTIC_NEUTRAL_MARGIN:
            out[key] = score
    return out
