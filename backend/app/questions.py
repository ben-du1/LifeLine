"""Opening/closing messages and reply parsing. Follow-up questions are planned in followups.py."""

import re
from dataclasses import dataclass

from . import config

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
    # Emphatic denials start with a word _YES reads as yes ("absolutely"), so they must be matched here first.
    r"(?:absolutely|definitely|certainly|of course|surely)\s+not|not at all|hell no|no way(?!\s+out)|"
    r"there (?:isn'?t|aren'?t)(?=\s*$|\s*[,.!?]|\s+(?:one|any|anything)\b)|"
    r"i (?:don'?t|can'?t|cannot) see(?=\s*$|\s*[,.!?]|\s+(?:one|it|any|anything|that|them)\b)|none)" + _AFTER,
    re.I)
_UNSURE = re.compile(
    r"\b(don'?t know|dont know|not sure|unsure|idk|idek|dunno|can'?t tell|no idea|no clue|maybe|not certain|"
    r"hard to (?:say|tell)|i guess(?! so)|possibly|yes and no|yes or no)\b|^\s*(kind of|kinda|sort of|sorta)\b",
    re.I)

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

# Whether the caller has left the building is read by out_signal (below).


# The caller reports danger where they are. Not used for locating; it flags the call and triggers SAFETY_MESSAGE.
# Heat is a warning before smoke reaches the caller: a hot or warm door means fire on the other side, and an orange
# glow under the door or through a window is flames.
_DANGER = re.compile(
    r"\b(smoke|smoky|fire|flames?|burning|can'?t breathe|hard to breathe|"
    r"(?:hot|warm) door|door(?:knob| handle)? (?:is|feels|getting|is getting) (?:really |so |very )?(?:hot|warm)|"
    r"(?:it'?s|its|it is|getting|so|really|very) (?:getting )?(?:really |so |very )?hot(?: in here)?|"
    r"orange (?:light|glow)|glow(?:ing|s)?)\b", re.I)
# Phrases that contain a danger word but are not danger. They are removed before _DANGER is checked, so
# "no smoke, but the door is hot" still counts while "no smoke here" and "the fireplace" don't.
_NOT_DANGER = re.compile(
    r"\b(fire ?place|fire ?pit|fire ?(?:alarm|extinguisher|escape|truck|department|fighters?)|"
    r"(?:no|not any|isn'?t any|there'?s no) (?:smoke|fire|flames?)|"
    r"(?:isn'?t|is not|not|never) (?:really |very )?(?:hot|warm|glowing)|"
    r"hot (?:tub|water|chocolate|sauce|plate)|glow[- ]in[- ]the[- ]dark|glow ?sticks?|night ?light)\b", re.I)


def mentions_danger(text: str) -> bool:
    return bool(_DANGER.search(_NOT_DANGER.sub(" ", text)))


def is_correction(text: str) -> bool:
    return bool(_CORRECTION.search(text))


# ---- caller-initiated calls (contract; owner: workstream C)
# The caller made contact, so the first message asks for the building before anything else. It still leads with
# getting out, for the same reason as OPENING_MESSAGE.
ADDRESS_QUESTION = (
    "911: If you can get out safely, get out now and text OUT. "
    "If you can't, what is the address of the building you're in?"
)
ADDRESS_RETRY = "I couldn't find that address. Please send the house number, street name and city."
ADDRESS_CONFIRM = "Did you mean {address}?"
ADDRESS_CHOOSE = "Is it {a} or {b}?"
OUT_OF_AREA_MESSAGE = (
    "That address isn't in this system's service area. "
    "I'm connecting you to standard emergency services now. Stay on the line."
)
# Asked when a message might mean the caller got out ("I'm outside", "almost out") but might not, because treating
# it as out ends the questions for someone who may still be inside.
OUT_CONFIRM = "Are you outside the building now? Yes or no."


# Spoken versions for a voice call, keyed by the text constant. Only what is wrong on a phone call changes:
# "text OUT" -> "say you're out", "texting" -> "on this call", "text us" / "send" -> "tell me". Constants that read
# the same aloud (ADDRESS_CONFIRM, SAFETY_MESSAGE, ...) are not listed and pass through unchanged.
VOICE_VERSIONS = {
    OPENING_MESSAGE: (
        "911: We're on this call to help responders find you inside the building. "
        "If you can get out safely, get out now and say you're out. "
        "If you can't, tell me what room you're in and what you can see around you."
    ),
    ADDRESS_QUESTION: (
        "911: If you can get out safely, get out now and say you're out. "
        "If you can't, what is the address of the building you're in?"
    ),
    ADDRESS_RETRY: "I couldn't find that address. Please tell me the house number, street name and city.",
    LOCATED_MESSAGE: (
        "Got it. Responders will look for you in the {room}. "
        "If it becomes unsafe there, get out or move, then tell me where you are."
    ),
    BEST_GUESS_MESSAGE: (
        "Thank you. Responders know roughly where you are. "
        "If you move or get out, tell me. Keep your phone with you."
    ),
    REFINED_MESSAGE: "Thank you, that tells responders where to look. If you move or get out, tell me.",
    NO_MODEL_MESSAGE: (
        "Responders are on the way to your address. If you can get out safely, get out now. "
        "If you can't, tell me the room and floor you're on."
    ),
    OUT_MESSAGE: (
        "Good. Stay away from the building and go to the responders when they arrive. "
        "If anyone is still inside, tell me where they are."
    ),
}


def _template_pattern(template: str) -> re.Pattern:
    """A regex that finds `template` in a sent message, with each {name} placeholder captured, so a message built by
    formatting a constant (LOCATED_MESSAGE.format(room=...)) is still recognised."""
    parts = re.split(r"\{(\w+)\}", template)
    return re.compile("".join(re.escape(part) if i % 2 == 0 else f"(?P<{part}>.+?)" for i, part in enumerate(parts)))


# Longest first, so a template is never replaced inside a longer one that contains it.
_VOICE_PATTERNS = [(_template_pattern(text), voice, "{" in text)
                   for text, voice in sorted(VOICE_VERSIONS.items(), key=lambda item: -len(item[0]))]


def for_channel(text: str, channel: str) -> str:
    """The message as it should be sent on `channel` ("messaging" or "voice"). On voice, every known constant in the
    message (also one formatted with values, or joined with other text) becomes its spoken version; anything else
    is unchanged."""
    if channel != "voice":
        return text
    for pattern, voice, has_fields in _VOICE_PATTERNS:
        if has_fields:
            text = pattern.sub(lambda m, voice=voice: voice.format(**m.groupdict()), text)
        else:
            text = pattern.sub(lambda m, voice=voice: voice, text)
    return text


# out_signal. A trapped caller can easily write "I'm outside the bathroom door"; reading that as out would stop the
# questions for someone still inside, so a place inside the building after out/outside is not out.
_OUT_NEGATED = re.compile(
    r"\b(can'?t|cannot|couldn'?t|unable|not|no way|trying to|try to|tried to|how (?:do|can) (?:i|we)|"
    r"(?:get|help|let) (?:me|us)|need to|have to|want to|stuck|trapped)\b", re.I)
# Hedges are removed before the negation check ("not sure if I'm out" is a hedge, not "not out").
_OUT_HEDGE = re.compile(r"\b(i think|i guess|maybe|probably|almost|nearly|not sure(?: if)?|i'?m not sure(?: if)?|"
                        r"kind of|kinda|sort of|i believe)\b", re.I)
# Clearly outside the building.
_OUT_CLEAR = re.compile(
    r"^\s*(?:i'?m |we'?re |im |were )?(?:out|outside)(?: now| safe| safely)?\s*[.!]*\s*$"
    r"|\b(?:left|outside(?: of)?|out of|got out of|made it out of) (?:the |my |our |this )?"
    r"(?:building|house|home|apartment|apt|condo|complex|office building)\b"
    r"|\b(?:out|outside) (?:now|safe|safely)\b|\b(?:safe|safely) (?:out|outside)\b"
    r"|\b(?:on|across) the (?:street|sidewalk|lawn)\b|\bin the (?:front yard|back ?yard|yard|parking lot|street)\b",
    re.I)
# "out"/"outside" said of the caller: "I'm out", "we got outside", "made it out" (no subject: the caller's own). Only
# a first-person subject counts: "my kids are outside, I'm in the bedroom" and "smoke is outside my window" come
# from callers who are still inside.
_OUT_SAID = re.compile(
    r"(?:\b(?:i'm|im|i am|i was|we're|we are|we were|we|i)\s+(?:(?:got|made it|gotten|went|are|all)\s+)*"
    r"|^\s*(?:all\s+)?(?:got|made it|gotten|went)\s+(?:all\s+)?)(?:out|outside)\b(?!\s+of\b)", re.I)
# Someone or something else is out/outside ("the kids are outside", "power is out", "the lights went out"). Removed
# before the checks, so "the kids are outside now" isn't read as the caller being out either.
_OUT_THIRD = re.compile(
    r"\b(?!(?:i|we)\b)\w+(?:\s+(?:is|are|was|were|went)|'s)\s+(?:all\s+)?(?:out|outside)(?:\s+(?:now|safe|safely))?\b",
    re.I)
# Somewhere inside the building, named right after out/outside: "outside the bathroom door", "out of my room".
_INSIDE_PLACE = (r"(?:room|bathroom|bedroom|kitchen|closet|hall(?:way)?|corridor|office|stairs|stairwell|basement|"
                 r"attic|garage|laundry|elevator|unit|apartment door|bath|den|study|lobby|"
                 r"(?:bedroom |bathroom |kitchen |closet |back )?door)")
_OUT_INSIDE = re.compile(
    rf"\b(?:out|outside)(?:\s+of)?\s+(?:the |my |our |a |his |her |their |this |that )?(?:\w+\s+)?{_INSIDE_PLACE}\b",
    re.I)
# The way out the caller used or is at, which may or may not be outside: "out the window", "on the balcony".
_OUT_UNCLEAR = re.compile(
    r"\b(?:went|got|climbed|go|going|jumped) out (?:the |a |through the )?window\b"
    r"|\b(?:on|onto) the (?:roof|balcony|fire escape|ledge|porch)\b", re.I)
_FRONT_DOOR = re.compile(r"\b(?:out|outside)(?:\s+of)?\s+(?:the |our |my )?(?:front|main) (?:door|entrance)\b", re.I)


def out_signal(text: str):
    """How sure it is that the caller has left the building: "out" (act on it now), "maybe" (ask OUT_CONFIRM first)
    or None (not about getting out, or still inside: "I can't get out", "I'm outside the bathroom door")."""
    text = normalize(text)
    hedged = bool(_OUT_HEDGE.search(text))
    plain = _OUT_THIRD.sub(" ", _OUT_HEDGE.sub(" ", text))
    if _OUT_NEGATED.search(plain):
        return None
    if _OUT_CLEAR.search(plain):
        return "maybe" if hedged else "out"
    if _FRONT_DOOR.search(plain):
        return "maybe"   # the house's front door, or the apartment's door onto the hallway
    if _OUT_INSIDE.search(plain):
        return None
    if _OUT_UNCLEAR.search(plain):
        return "maybe"   # before _OUT_SAID: "I'm out on the balcony" is still on the building
    if _OUT_SAID.search(plain):
        return "maybe" if hedged else "out"
    return None


# looks_like_address. A regex, not addresses.parse_caller: that parser was being written in parallel (workstream A)
# and can replace this at integration. A house number (digits, or spoken number words), then up to
# config.ADDRESS_MAX_NAME_WORDS words of street name, then a street word.
_NUMBER_WORDS = (r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                 r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
                 r"ninety|hundred|thousand)")
_HOUSE_NUMBER = rf"(?:\d{{1,6}}[a-z]?(?:-\d{{1,5}})?|{_NUMBER_WORDS}(?:[\s-]+{_NUMBER_WORDS})*)"
# Street words that are also common words ("way", "park", "green", "run") are left out: "there's one way out" and
# "three people in the green room" are not addresses.
_STREET_WORD = (r"(?:street|st|avenue|ave|av|road|rd|drive|dr|lane|ln|boulevard|blvd|court|ct|place|pl|circle|"
                r"cir|terrace|ter|parkway|pkwy|highway|hwy|trail|trl|plaza|plz|pike|crescent|cres|alley|aly|"
                r"expressway|expy|freeway|fwy|turnpike|tpke)")
# Words of a sentence, not of a street name, between a number and a street word ("2 rooms down the hall").
_NOT_NAME_WORD = r"(?:in|on|at|the|a|an|my|our|by|near|of|to|and|or|is|are|was|people|rooms?|floors?|minutes?|me|us)"
_ADDRESS = re.compile(
    rf"\b{_HOUSE_NUMBER}\s+(?:(?!{_NOT_NAME_WORD}\b)[\w'.-]+\s+){{1,{config.ADDRESS_MAX_NAME_WORDS}}}?"
    rf"{_STREET_WORD}\b\.?", re.I)
# "Route 1", "Highway 27": the number comes after the street word.
_NUMBERED_ROAD = re.compile(r"\b\d{1,6}\s+(?:route|rte|highway|hwy|state road|county road)\s+\d{1,4}\b", re.I)


def looks_like_address(text: str) -> bool:
    """True when the text reads like a street address: a house number (digits or spoken number words), then a
    street name and a street word ("13122 Chipman Glen Dr", "thirty one twelve Houston Avenue")."""
    text = normalize(text)
    return bool(_ADDRESS.search(text) or _NUMBERED_ROAD.search(text))


# clearly_not_address. Anything with a digit, a spoken number ("oh" aside: "oh god") or a street word might be an
# address; so might a bare street name ("chipman glen"), so only a plea, a greeting or a question is ruled out.
_ADDRESS_HINT = re.compile(rf"\d|\b{_NUMBER_WORDS}\b|\b{_STREET_WORD}\b", re.I)
_NOT_ADDRESS_CUE = re.compile(r"\?|\b(?:help|hurry|please|hello|hi|hey|what|who|why|are you there)\b", re.I)


def clearly_not_address(text: str) -> bool:
    """True for a reply that can't be an address ("please hurry", "hello?", "there's smoke"): no number and no street
    word, and a plea, greeting, question or danger. Such text is not looked up (resolve may search Matterport)."""
    text = re.sub(r"\boh\b", " ", normalize(text), flags=re.I)
    if _ADDRESS_HINT.search(text):
        return False
    return bool(_NOT_ADDRESS_CUE.search(text)) or mentions_danger(text) or parse_reply(text).kind != "free"


def is_out(text: str) -> bool:
    """True when the caller has clearly left the building (out_signal "out"). A "maybe" is not out here; callers
    that can ask OUT_CONFIRM use out_signal."""
    return out_signal(text) == "out"


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
    """Room types the caller says they are in. Denied ("I'm not in the garage"), hedged ("maybe the kitchen") and
    landmark mentions ("next to the kitchen") are left out."""
    return {label for label, said, _ in _room_readings(text) if said == "yes"}


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
    for pattern, label in (ROOM_KEYWORDS | ROOM_FIXTURE_KEYWORDS).items():
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

def room_types_denied(text: str) -> set[str]:
    """Room types the caller says they are not in ("not the kitchen"). Only room names count: "no shower" doesn't
    rule out a bathroom. A type both named and denied in one message is left out of both answers' effect here."""
    readings = _room_readings(text)
    named = {label for label, said, _ in readings if said == "yes"}
    return {label for label, said, is_name in readings if said == "no" and is_name} - named
ROOM_FIXTURE_KEYWORDS = {
    r"\b(toilet|shower|bath ?tub|tub|bath)\b": "bathroom",
}





_CLAUSE_BREAK = re.compile(
    r"[.;:!?\n]|,(?!\s*(?:or|nor)\b)|\b(?:but|though|although|however|except|and|plus|also|then|while|whereas|yet|"
    r"wait|actually)\b|(?<!think )(?<!guess )(?<!sure )(?<!if )(?<!know )(?<!whether )"
    r"\b(?:there'?s|theres|there is|there are|there'?re|i see|i can see|i got|i have|i'?m|im|i am|it'?s|its)\b",
    re.I)

NEGATOR = re.compile(
    r"\b(no|not|nor|neither|none|nothing|without|never|cannot|zero)\b|\b(is|are|was|were|ai)n'?t\b|"
    r"\b(do|does|did|ca|can|could|has|have|had|wo|would)n'?t\s+(see|find|spot|notice|have|has|got|think|look|seem|"
    r"really|any|a|an)\b", re.I)

_HEDGE = re.compile(r"\b(not sure|unsure|idk|dunno|don'?t know|dont know|no idea|no clue|can'?t tell|maybe|might|"
                    r"possibly|not certain)\b", re.I)

_RELATIVE = re.compile(
    r"\b(next to|near|nearby|by|beside|behind|across from|across|down|opposite|outside|outside of|off|off of|past|"
    r"from|out of|left|toward|towards|close to|away from|was in|were in|was at)\s+((the|a|my|our|this|that)\s+)?$",
    re.I)

_JOINED = re.compile(r"\s*,?\s*(or|nor)\s+((a|an|any|the)\s+)?", re.I)

NEGATION_REACH_WORDS = 6

SIZE_COMPARISON = re.compile(r"\b\w+(\s+room)?[\s-]*size[ds]?\b(?!\s+of\b)|\b(bigger|smaller|larger|size of)\b,?\s*((like|than|as)\s+)?"
                             r"(a|an|the)\s+\w+(\s+room)?\b", re.I)
