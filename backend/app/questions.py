"""Opening/closing messages and reply parsing. Follow-up questions are planned in followups.py."""

import re
from dataclasses import dataclass

OPENING_MESSAGE = (
    "This is 911 emergency services. We're working to find you inside the building. "
    "Please describe the room you're in: what furniture, windows, doors or appliances can you see?"
)
LOCATED_MESSAGE = "Thank you. Responders now know which room you're in ({room}). Stay where you are; help is on the way."
BEST_GUESS_MESSAGE = (
    "Thank you. Responders have your approximate location. Stay where you are and keep your phone with you."
)
NO_MODEL_MESSAGE = "Thank you. Stay where you are if it is safe; responders are on the way to your address."


# Words in a reply that name a room type, mapped to Matterport room classification labels (lowercase).
ROOM_KEYWORDS = {
    r"\b(bath ?room|restroom|toilet|shower|bathtub|washroom)\b": "bathroom",
    r"\b(bed ?room|master suite)\b": "bedroom",
    r"\bkitchen\b": "kitchen",
    r"\b(living ?room|family room|lounge)\b": "living room",
    r"\bdining\b": "dining room",
    r"\b(hallway|hall|corridor)\b": "hallway",
    r"\bgarage\b": "garage",
    r"\b(office|study)\b": "office",
    r"\bcloset\b": "closet",
    r"\blaundry\b": "laundry room",
    r"\b(entry|foyer|entryway)\b": "entryway",
}

_YES = re.compile(r"^\s*(yes|yeah|yep|yup|ya|yea|y|sure|correct|there is|i do|i can)\b[\s,.!:-]*", re.I)
_NO = re.compile(r"^\s*(no|nope|nah|n|not really|there isn'?t|i don'?t see|i can'?t see|none)\b[\s,.!:-]*", re.I)
_UNSURE = re.compile(r"\b(don'?t know|not sure|unsure|idk|can'?t tell|no idea|maybe)\b", re.I)


@dataclass(frozen=True)
class Reply:
    kind: str   # "yes", "no", "unsure", or "free"
    extra: str  # remaining free text ("yes, next to the bed" -> "next to the bed"); full text for "free"


def parse_reply(text: str) -> Reply:
    text = text.strip()
    if _UNSURE.search(text) and len(text.split()) <= 6:
        return Reply("unsure", "")
    for kind, pattern in (("no", _NO), ("yes", _YES)):
        match = pattern.match(text)
        if match:
            return Reply(kind, text[match.end():].strip())
    return Reply("free", text)


def room_types_mentioned(text: str) -> set[str]:
    return {label for pattern, label in ROOM_KEYWORDS.items() if re.search(pattern, text, re.I)}


def has_detail(extra: str) -> bool:
    """True when leftover text after yes/no is worth embedding on its own (not just 'there is one')."""
    return len(re.findall(r"[a-z]+", extra.lower())) >= 3
