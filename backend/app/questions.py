"""Opening/closing messages and reply parsing. Follow-up questions are planned in followups.py."""

import re
from dataclasses import dataclass

# Texts to the caller. The building may be on fire, so no message says "stay where you are" unconditionally: getting
# out comes first, staying put is only for when they can't, and the caller is told to text if they move or get out
# (handled by is_out and the correction/move check below). Each text is short and asks for one thing, because the
# caller may be reading it in smoke or panic.
OPENING_MESSAGE = (
    "911: We're texting to help responders find you inside the building. "
    "If you can get out safely, get out now and text OUT. "
    "If you can't, tell us what room you're in and what you can see around you."
)
LOCATED_MESSAGE = (
    "Got it. Responders will look for you in the {room}. "
    "If it becomes unsafe there, get out or move, then text us where you are."
)
# Sent once with LOCATED_MESSAGE. Standard advice for someone trapped by a fire; harmless for other emergencies.
SAFETY_MESSAGE = (
    "If there is smoke: stay low near the floor, keep the door closed, and block the gap under it with cloth. "
    "If you have a window, signal from it."
)
# Leads into the first "which is closest to you?" question, so the caller knows it is optional detail.
REFINE_INTRO = "To find you faster, one more question."
# Before a room is reported: one independent fact is checked (followups.verification).
VERIFY_INTRO = "To make sure we send responders to the right room:"
# After the caller is told the room, while it is still under config.CONFIRM_UNTIL.
CONFIRM_INTRO = "To be sure, one more question."
BEST_GUESS_MESSAGE = (
    "Thank you. Responders know roughly where you are. "
    "If you move or get out, text us. Keep your phone with you."
)
REFINED_MESSAGE = "Thank you, that tells responders where to look. If you move or get out, text us."
NO_MODEL_MESSAGE = (
    "Responders are on the way to your address. If you can get out safely, get out now. "
    "If you can't, text us the room and floor you're on."
)
OUT_MESSAGE = (
    "Good. Stay away from the building and go to the responders when they arrive. "
    "If anyone is still inside, text us where they are."
)


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
_UNSURE = re.compile(r"\b(don'?t know|do not know|dunno|not sure|unsure|idk|can'?t tell|no idea|no clue|maybe"
                     r"|can'?t remember|don'?t remember|not certain)\b", re.I)

# Phones send typographic punctuation (iOS "smart punctuation" turns "don't" into "don’t"), and every pattern in this
# file and followups.py is written with plain ASCII quotes. A live "Don’t know" was read as a room description and
# the floor question was asked three more times.
_SMART_PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Caller text as the reply patterns expect it: ASCII quotes, trimmed."""
    return text.translate(_SMART_PUNCTUATION).strip()


# Prefix when a question is asked again because its answer couldn't be read (at most once per question).
NOT_UNDERSTOOD = "Sorry, I didn't catch that."


@dataclass(frozen=True)
class Reply:
    kind: str   # "yes", "no", "unsure", or "free"
    extra: str  # remaining free text ("yes, next to the bed" -> "next to the bed"); full text for "free"


# The caller takes back what they said before ("sorry, I'm actually upstairs"), or has moved ("I moved to the
# bedroom"). Both mean the earlier description no longer applies, so the estimate starts over.
_CORRECTION = re.compile(
    r"\b(actually|i meant|i mean|no wait|wait no|correction|scratch that|my mistake"
    r"|i('ve| have)? moved|we('ve| have)? moved|i'?m now|now i'?m)\b|^\s*sorry\b", re.I)

# The caller has left the building: "OUT", "I'm out", "we got outside", "made it out of the house".
_OUT = re.compile(
    r"^\s*out\s*[.!]*\s*$"
    r"|\b(i'?m|i am|we'?re|we are|got|made it|gotten)\s+(out|outside)\b(?!\s+of\b)"  # "out of breath / of the bedroom"
    r"|\b(left|outside|out of) the (building|house|home|apartment)\b", re.I)
# "I can't get out", "we're not out": a negation means they're still inside.
_NOT_OUT = re.compile(r"\b(can'?t|cannot|couldn'?t|unable|not|no way|trying to)\b", re.I)


# The caller reports danger where they are. Not used for locating; it flags the call and triggers SAFETY_MESSAGE.
_DANGER = re.compile(r"\b(smoke|smoky|fire|flames?|burning|can'?t breathe|hard to breathe|hot door|door is hot)\b", re.I)


def mentions_danger(text: str) -> bool:
    return bool(_DANGER.search(text)) and not re.search(r"\b(fire ?place|fire ?pit|no (smoke|fire))\b", text, re.I)


def is_correction(text: str) -> bool:
    return bool(_CORRECTION.search(text))


def is_out(text: str) -> bool:
    return bool(_OUT.search(text)) and not _NOT_OUT.search(text)


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


@dataclass(frozen=True)
class RoomMentions:
    claimed: set[str]   # rooms the caller says they are in: "I'm in the kitchen", "hiding in a closet"
    nearby: set[str]    # rooms they are next to: "next to the kitchen", "the closet off the bedroom" (bedroom)
    denied: set[str]    # rooms they say they are not in: "I'm not in the bathroom"


# Words just before a room name that make it something other than where the caller is. Checked on the ~40
# characters before the name, allowing one extra word ("like a big living room").
_COMPARISON = re.compile(r"\b(like|as|than|size of|similar to|kind of|sort of)\s+(an?\s+|the\s+|my\s+)?(\w+\s+)?$", re.I)
_RELATION = re.compile(r"\b(next to|near|nearby|beside|by|across from|off|off of|outside|outside of|close to|"
                       r"opposite|toward|towards|behind|through|down|end of|from)\s+(the\s+|a\s+|an\s+|my\s+|our\s+)?"
                       r"(\w+\s+)?$", re.I)
_DENIAL = re.compile(r"\b(not|isn'?t|never)\s+(in\s+)?(the\s+|a\s+|an\s+|my\s+)?(\w+\s+)?$", re.I)
# Fixtures in the bathroom pattern ("by the toilet") place the caller in the room even after a relation word.
_FIXTURES = {"toilet", "shower", "bathtub"}


def room_mentions(text: str) -> RoomMentions:
    """Which room names are a claim of location, and which are something else. Every room name used to count as a
    claim: "big like a living room" (an answer about size) put the caller in the living room at 20x, and "the closet
    off the bedroom" claimed both rooms. Now:
      - comparisons ("like a living room", "bedroom-sized", "bigger than a closet") are ignored;
      - relations ("next to the kitchen", "down the hall") are nearby rooms, used as adjacency evidence;
      - a room name directly followed by another ("bedroom closet") describes the second one, so it is nearby;
      - "not in the bathroom" is a denial."""
    claimed, nearby, denied = set(), set(), set()
    for pattern, label in ROOM_KEYWORDS.items():
        for match in re.finditer(pattern, text, re.I):
            before = text[max(0, match.start() - 40):match.start()]
            after = text[match.end():match.end() + 20]
            if _COMPARISON.search(before) or re.match(r"[\s-]*(sized|size)\b", after, re.I):
                continue
            followed_by_room = any(re.match(rf"\s+{p}", after, re.I) for p in ROOM_KEYWORDS)
            if _DENIAL.search(before):
                denied.add(label)
            elif match.group(0).lower() not in _FIXTURES and (_RELATION.search(before) or followed_by_room):
                nearby.add(label)
            else:
                claimed.add(label)
    return RoomMentions(claimed, nearby - claimed, denied - claimed)


def has_detail(extra: str) -> bool:
    """True when leftover text after yes/no is worth embedding on its own (not just 'there is one')."""
    return len(re.findall(r"[a-z]+", extra.lower())) >= 3


def describes_something(text: str) -> bool:
    """True when free text could describe a room and is worth scoring against the photos. "3", "3rd", "ok", "what?"
    are not: scored against the photos they still favour whichever rooms happen to match noise, and four such texts
    once pushed one floor past 80% before the caller had said which floor they were on. Single words that do
    describe ("bedroom", "stove") are read by the room-name and feature keyword rules instead."""
    return len(re.findall(r"[a-z]{3,}", text.lower())) >= 2
