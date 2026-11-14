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


# Words in a reply that name a room type, mapped to Matterport room classification labels (lowercase). Common
# typos are listed because callers text in a panic. A name followed by another room name is a modifier and is skipped
# ("hall closet", "bedroom closet": the caller is in the closet), see _room_readings.
ROOM_KEYWORDS = {
    r"\b(bath ?(room|rom|rm|roon)s?|bathrooom|rest ?rooms?|washrooms?|powder room|half bath|master bath|lavatory|"
    r"\bloo)\b": "bathroom",
    r"\b(bed ?(room|rom|rm)s?|bdrm|master suite|guest room|spare room|nursery|(my|his|her|their) room|"
    r"(kid|son|daughter|baby)'?s'? room|kids room)\b": "bedroom",
    r"\b(kitchen(ette)?s?|kitchn|kichen|kitchin|kithcen|kicthen)\b": "kitchen",
    r"\b(living ?(room|rm|rom)|family room|lounge|den|sitting room|tv room|front room|great room)\b": "living room",
    r"\b(dining|dinning)\b": "dining room",
    r"\b(hallways?|halls?|halway|corridors?)\b": "hallway",
    r"\b(garage|garag)\b": "garage",
    r"\b(office|study)\b": "office",
    r"\b(closets?|closit|walk[- ]in)\b": "closet",
    # "laundry basket" is in a bedroom, not a laundry room.
    r"\blaundry\b(?!\s+(basket|hamper|bag|pile|detergent))": "laundry room",
    r"\b(entry ?way|entry(?! door)|foyer|front hall|mud ?room|vestibule)\b": "entryway",
}
# Fixtures that imply a room type when present, but whose absence doesn't rule it out ("no shower": a half bath).
ROOM_FIXTURE_KEYWORDS = {
    r"\b(toilet|shower|bath ?tub|tub|bath)\b": "bathroom",
}

# Filler a panicked caller puts before the answer: "um yes", "oh god no", "please, yes".
_FILLER = (r"^\s*(?:(?:u+m+|u+h+|e+r+|o+h+|ok(?:ay)?|well|so|hm+|please|help|omg|oh my god|god|sorry|hello|hi)\b"
           r"[\s,.!-]*)*")
_AFTER = r"\b[\s,.!?:-]*"
_YES = re.compile(
    _FILLER + r"(ye+s*a*h*|yeh|ya+[hs]*|yu+p+|ye+p+|ys|yh|y|sure|correct|affirmative|uh[- ]?huh|mhm+|mm[- ]?hm+|"
    r"definitely|absolutely|i think so|i believe so|i guess so|i (?:do|can)(?=\s*$|\s*[,.!])|"
    r"i (?:can )?see (?:one|it|them)|"
    # "there is" is a yes only when the rest doesn't deny it: "there is no toilet" is a description, read by cues.
    r"there(?:'s| is| are)(?!\s+(?:no|not|nothing|none)\b)(?=\s*$|\s*[,.!]|\s+(?:one|some|a couple)\b))" + _AFTER,
    re.I)
_NO = re.compile(
    # A bare "no" is an answer only when nothing follows or a new statement does ("no, I'm in the kitchen",
    # "no there's a sink"). "no window" is a description; rule_cues reads it as a denied feature.
    _FILLER + r"((?:no+|n)(?=\s*$|\s*[,.!?;:-]|\s+(?:i|im|i'm|it|its|it's|there|theres|there's|nothing|none|not|"
    r"sorry|but|just|only|that|thats|that's|one|we|my|no+|nope)\b)|no+pe+|na+h+|nay|negative|not really|"
    r"not that i (?:can )?see|(?:i )?don'?t think so|"
    r"there (?:isn'?t|aren'?t)(?=\s*$|\s*[,.!?]|\s+(?:one|any|anything)\b)|"
    r"i (?:don'?t|can'?t|cannot) see(?=\s*$|\s*[,.!?]|\s+(?:one|it|any|anything|that|them)\b)|none)" + _AFTER,
    re.I)
_UNSURE = re.compile(
    r"\b(don'?t know|dont know|not sure|unsure|idk|idek|dunno|can'?t tell|no idea|no clue|maybe|not certain|"
    r"hard to (?:say|tell)|i guess(?! so)|possibly|yes and no|yes or no)\b|^\s*(kind of|kinda|sort of|sorta)\b",
    re.I)


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


# ---- words in context -----------------------------------------------------------------------------------------
# A negation, hedge or "next to" reaches only the words after it in the same clause. Clauses end at punctuation, at
# contrast/sequence words ("no window but there's a toilet") and where a new statement starts ("no window theres a
# toilet": panicked callers skip punctuation). "think/guess/sure/if/know there's" doesn't start one: "I don't think
# there's a window" is still a denial.
_CLAUSE_BREAK = re.compile(
    r"[.;:!?\n]|,(?!\s*(?:or|nor)\b)|\b(?:but|though|although|however|except|and|plus|also|then|while|whereas|yet|"
    r"wait|actually)\b|(?<!think )(?<!guess )(?<!sure )(?<!if )(?<!know )(?<!whether )"
    r"\b(?:there'?s|theres|there is|there are|there'?re|i see|i can see|i got|i have|i'?m|im|i am|it'?s|its)\b",
    re.I)
# Words that deny what follows. Contractions are only negations of a thing with a verb of seeing or having:
# "I can't see a window" denies the window, "I can't get out, the bed is in the way" doesn't deny the bed.
NEGATOR = re.compile(
    r"\b(no|not|nor|neither|none|nothing|without|never|cannot|zero)\b|\b(is|are|was|were|ai)n'?t\b|"
    r"\b(do|does|did|ca|can|could|has|have|had|wo|would)n'?t\s+(see|find|spot|notice|have|has|got|think|look|seem|"
    r"really|any|a|an)\b", re.I)
_HEDGE = re.compile(r"\b(not sure|unsure|idk|dunno|don'?t know|dont know|no idea|no clue|can'?t tell|maybe|might|"
                    r"possibly|not certain)\b", re.I)
# "next to the kitchen", "across from the bathroom", "I ran from the garage": a room named as a landmark or a place
# the caller left, not the room they're in.
_RELATIVE = re.compile(
    r"\b(next to|near|nearby|by|beside|behind|across from|across|down|opposite|outside|outside of|off|off of|past|"
    r"from|out of|left|toward|towards|close to|away from|was in|were in|was at)\s+((the|a|my|our|this|that)\s+)?$",
    re.I)
# A negation covers one item, plus items joined to it by "or"/"nor" ("no toilet or sink"); "not the kitchen the
# bathroom" names a second room, which is not denied.
_JOINED = re.compile(r"\s*,?\s*(or|nor)\s+((a|an|any|the)\s+)?", re.I)
NEGATION_REACH_WORDS = 6


def polarity(text: str, start: int, blockers: list[tuple[int, int]] = (), relative: bool = False) -> str | None:
    """How the words at text[start:] are meant: "yes" (present / where I am), "no" (denied) or None (hedged, or a
    landmark when `relative`). `blockers` are spans of other items the scope of a negation stops at."""
    clause_start = max((m.end() for m in _CLAUSE_BREAK.finditer(text, 0, start)), default=0)
    before = text[clause_start:start]
    if relative and _RELATIVE.search(before):
        return None
    if _HEDGE.search(before):
        return None
    negations = list(NEGATOR.finditer(before))
    if not negations:
        return "yes"
    neg_end = clause_start + negations[-1].end()
    if len(text[neg_end:start].split()) > NEGATION_REACH_WORDS:
        return "yes"
    between = [(s, e) for s, e in blockers if neg_end <= s and e <= start]
    if between and not _JOINED.fullmatch(text[max(e for _, e in between):start]):
        return "yes"
    return "no"


# "looks like a bedroom" still names the room: only comparisons of size are dropped ("bigger, like a living room",
# "smaller than a bedroom", "the size of a closet").
SIZE_COMPARISON = re.compile(r"\b\w+(\s+room)?[\s-]*size[ds]?\b(?!\s+of\b)|\b(bigger|smaller|larger|size of)\b,?\s*((like|than|as)\s+)?"
                             r"(a|an|the)\s+\w+(\s+room)?\b", re.I)


def _room_readings(text: str) -> list[tuple[str, str | None, bool]]:
    """(label, polarity, is_name) for each room word in the text. Fixture hits have is_name False."""
    # A room named only as a size ("closet-sized", "about bedroom size", "like a living room") is a comparison, not
    # where the caller is: our own size question offers these words, and they gave the named type a 20x boost
    # (a bathroom caller answering "closet-sized" ended up located in the closet).
    text = SIZE_COMPARISON.sub(" ", text)
    hits = [(m.start(), m.end(), label, True) for pattern, label in ROOM_KEYWORDS.items()
            for m in re.finditer(pattern, text, re.I)]
    hits += [(m.start(), m.end(), label, False) for pattern, label in ROOM_FIXTURE_KEYWORDS.items()
             for m in re.finditer(pattern, text, re.I)]
    # Drop hits inside longer ones ("front hall" is an entry, not a hallway; "master bath" is one name).
    hits = [h for h in hits if not any(o is not h and o[0] <= h[0] and h[1] <= o[1] and o[1] - o[0] > h[1] - h[0]
                                       for o in hits)]
    hits.sort()
    spans = [(s, e) for s, e, _, _ in hits]
    readings = []
    for s, e, label, is_name in hits:
        # A room name right before another one is a modifier: "hall closet", "bedroom's closet", "kitchen pantry".
        if is_name and any(o[3] and o[0] > s and re.fullmatch(r"('s)?\s*", text[e:o[0]]) for o in hits):
            continue
        # Fixtures are in the room even when named as a landmark: "squeezed in by the tub".
        readings.append((label, polarity(text, s, spans, relative=is_name), is_name))
    return readings


def room_types_mentioned(text: str) -> set[str]:
    """Room types the caller says they are in. Denied ("I'm not in the garage"), hedged ("maybe the kitchen") and
    landmark mentions ("next to the kitchen") are left out."""
    return {label for label, said, _ in _room_readings(text) if said == "yes"}


def room_types_denied(text: str) -> set[str]:
    """Room types the caller says they are not in ("not the kitchen"). Only room names count: "no shower" doesn't
    rule out a bathroom. A type both named and denied in one message is left out of both answers' effect here."""
    readings = _room_readings(text)
    named = {label for label, said, _ in readings if said == "yes"}
    return {label for label, said, is_name in readings if said == "no" and is_name} - named


def has_detail(extra: str) -> bool:
    """True when leftover text after yes/no is worth embedding on its own (not just 'there is one')."""
    return len(re.findall(r"[a-z]+", extra.lower())) >= 3
