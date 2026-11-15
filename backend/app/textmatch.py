"""The visual part of a caller's message, as a CLIP prompt for photo matching.

CLIP used to embed the whole reply ("I'm so scared please hurry, carpeted floor, someone is downstairs"). Panic words,
people and actions match no photo, but they still shift the text embedding, and at LOGIT_SCALE 60 one noisy update
can move the room. `visual_query` keeps only what a camera could see: fixtures, furniture, household objects,
materials, colours, light and room words, minus anything negated ("no window" must not send "window"), and minus
whatever was already read as a cue (a mentioned feature, a flooring answer, a named room type), so the same words
are not counted twice. Nothing left means no CLIP update at all.
"""

import re
from dataclasses import dataclass
from typing import Iterable

from . import followups, questions


@dataclass(frozen=True)
class Term:
    phrase: str                # how it reads in the prompt: "a desk", "curtains"; "" = the caller's own words
    words: str                 # regex over the lowercased message
    keys: tuple[str, ...] = ()  # cue keys that already carry this evidence (feature/option/choice/room type)
    kind: str = "object"       # "object", "room" (the prompt's subject) or "light" (an adjective on the room)


# Household objects the feature bank doesn't ask about. Rugs come before the flooring options so "a rug" reads as a
# rug, not "carpet floor"; it is still carpet evidence when the caller answered the flooring question.
_OBJECTS = [
    ("a rug", r"\brugs?\b", ("carpet", "flooring")),
    ("a desk", r"\bdesks?\b"), ("counters", r"\b(counter ?tops?|counters?)\b"), ("a dresser", r"\bdressers?\b|chest of drawers"), ("a nightstand", r"\bnightstands?\b"),
    ("a bookshelf", r"\bbook ?(shel(f|ves)|cases?)\b"), ("a piano", r"\bpianos?\b"), ("a crib", r"\bcribs?\b"),
    ("a car", r"\bcars?\b"), ("tools", r"\btools?\b"), ("a workbench", r"\bwork ?bench(es)?\b"),
    ("boxes", r"\bbox(es)?\b|\bcardboard\b"), ("toys", r"\btoys?\b"), ("plants", r"\b(house ?)?plants?\b"),
    ("curtains", r"\bcurtains?\b|\bdrapes\b"), ("blinds", r"\bblinds\b"), ("a painting", r"\bpaintings?\b"),
    ("a picture", r"\bpictures?\b|\bphotos?\b|\bframes?\b"), ("a poster", r"\bposters?\b"),
    ("a lamp", r"\blamps?\b"), ("a chandelier", r"\bchandeliers?\b"), ("an armchair", r"\barm ?chairs?\b|recliner"),
    ("a bench", r"\bbench(es)?\b"), ("a wardrobe", r"\bwardrobes?\b|\barmoire\b"), ("an ottoman", r"\bottomans?\b"),
    ("pillows", r"\b(pillows?|cushions?)\b"), ("a blanket", r"\bblankets?\b"), ("a clock", r"\bclocks?\b"),
    ("a computer", r"\b(computers?|laptops?|monitors?|printers?)\b"), ("a treadmill", r"\btreadmills?\b"),
    ("a bike", r"\b(bikes?|bicycles?)\b"), ("exercise weights", r"\b(weights|dumbbells?|exercise equipment)\b"),
    ("shoes", r"\bshoes?\b|\bboots\b"), ("a basket", r"\b(laundry )?baskets?\b|\bhampers?\b"),
    ("a trash can", r"\b(trash|garbage) (can|bin)s?\b"), ("a vanity", r"\bvanit(y|ies)\b"),
    ("towels", r"\btowels?\b"), ("a stroller", r"\bstrollers?\b"), ("a guitar", r"\bguitars?\b"),
    ("an aquarium", r"\baquariums?\b|fish ?tanks?"), ("a skylight", r"\bskylights?\b"),
    ("wallpaper", r"\bwallpaper\b"), ("a dishwasher", r"\bdishwashers?\b"), ("a water heater", r"water heater"),
    ("a door", r"\bdoors?\b"), ("walls", r"\bwalls?\b"), ("a ceiling", r"\bceilings?\b"),
]
# Rooms questions.ROOM_KEYWORDS doesn't name (so they reach CLIP instead of the room-type boost).
_ROOMS = [("a basement", r"\bbasement\b"), ("an attic", r"\battic\b"),
          ("a pantry", r"\bpantry\b"), ("a mudroom", r"\bmud ?room\b"), ("a playroom", r"\bplay ?room\b"),
          ("a home gym", r"\bgym\b"), ("a sunroom", r"\bsun ?room\b"), ("a porch", r"\bporch\b"),
          ("a utility room", r"\butility\b"), ("a den", r"\bden\b")]
_LIGHT = [("bright", r"\b(bright|sunny|sunlit|well lit)\b|lots of (sun)?light"), ("dim", r"\bdim(ly lit)?\b")]

COLOURS = ("red", "blue", "green", "yellow", "white", "black", "grey", "gray", "brown", "beige", "pink", "purple",
           "orange", "tan", "cream", "navy", "teal", "gold", "silver")
MATERIALS = ("wooden", "wood", "glass", "metal", "leather", "marble", "granite", "brick", "stone", "stainless",
             "steel", "plastic", "velvet", "wicker")
_MODIFIER = re.compile(rf"\b({'|'.join(COLOURS + MATERIALS)})\b")
# "the walls are blue": a colour after the noun it describes.
_TRAILING_COLOUR = re.compile(rf"^\s+(?:is|are)\s+(?:painted\s+)?({'|'.join(COLOURS)})\b")

# Negation scope is the clause: "no window but a desk" and "no window, there's a desk" keep the desk, while
# "no window or door" drops both. "the window isn't there" negates a word before it.
_CLAUSE = re.compile(r"[.,;!?]+|\b(?:but|and|also|plus|though|although|except)\b")
_NEGATED_BEFORE = re.compile(r"\b(no|not|none|nothing|without|never|neither|nor|cannot|don'?t|doesn'?t|can'?t|"
                             r"isn'?t|aren'?t|there'?s no)\b|n't\b")
_NEGATED_AFTER = re.compile(r"^\s+(?:isn'?t|aren'?t|is not|are not)\b")

_MASS = {"shelving", "laundry", "hanging clothes", "clothes hanging", "sliding", "patio"}
_FLOORS = {"carpet", "tile", "wood"}  # flooring option keys
_WORD_REST = re.compile(r"\w*\s+")    # the rest of a word and the space after it
MAX_TERMS = 6  # CLIP reads 77 tokens; a short list of the caller's own objects matches photos best


def _terms() -> list[Term]:
    terms = [Term(phrase, words, tuple(keys[0]) if keys else ()) for phrase, words, *keys in _OBJECTS]
    # Feature words and nouns come from the feature bank, so the two never drift apart. "narrow" is a shape read
    # from the floor plan, not something a photo shows.
    # The caller's own word is the prompt ("counters", not "a stove or kitchen counters"), and a feature word must
    # start a word: the stairs pattern also matches "upstairs", which is a floor, not a staircase in view.
    terms += [Term("", rf"(?<![a-z])(?:{f.words})", (f.key,)) for f in followups.FEATURES if f.key != "narrow"]
    terms += [Term(f"{o.label} floor", o.words, (o.key, c.key)) for c in followups.CHOICES if c.key == "flooring"
              for o in c.options]
    terms += [Term(f"{'an' if label[0] in 'aeiou' else 'a'} {label}", words, (label,), "room")
              for words, label in questions.ROOM_KEYWORDS.items()]
    terms += [Term(phrase, words, (), "room") for phrase, words in _ROOMS]
    terms += [Term(phrase, words, (), "light") for phrase, words in _LIGHT]
    return terms


TERMS = _terms()


def _with_article(phrase: str, modifiers: list[str]) -> str:
    """'a desk' + ['blue'] -> 'a blue desk'; 'an armchair' + ['red'] -> 'a red armchair'; 'curtains' + ['blue'] ->
    'blue curtains'."""
    if not modifiers:
        return phrase
    head, _, rest = phrase.partition(" ")
    if head not in ("a", "an"):
        return f"{' '.join(modifiers)} {phrase}"
    article = "an" if modifiers[0][0] in "aeiou" else "a"
    return f"{article} {' '.join(modifiers)} {rest}"


def _own_words(words: str) -> str:
    """A matched feature word as a prompt phrase: 'toilet' -> 'a toilet', 'cabinets' -> 'cabinets'."""
    if words.endswith("s") or words in _MASS:
        return words
    return f"{'an' if words[0] in 'aeiou' else 'a'} {words}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _clause_terms(clause: str) -> list[tuple[Term, str, set[str], list[str], bool]]:
    """(term, prompt phrase, every cue key behind its words, modifiers, negated) for each visual term in one clause,
    in order of appearance."""
    found = []  # (start, end, term, list position) for every match; overlaps are resolved below
    for order, term in enumerate(TERMS):
        for match in re.finditer(term.words, clause):
            found.append((match.start(), match.end(), term, order))
    # Longest match wins ("ceiling fan" over "fan", "bookshelf" over "shelf"); ties go to the earlier list entry
    # ("a rug" over the carpet option). Words under an overlapped match still count as cues ("toilet" is both the
    # toilet feature and a bathroom).
    found.sort(key=lambda m: (m[0] - m[1], m[3]))
    chosen: list[tuple[int, int, Term, set[str]]] = []
    for start, end, term, _ in found:
        overlap = next((c for c in chosen if start < c[1] and c[0] < end), None)
        if overlap is None:
            chosen.append((start, end, term, set(term.keys)))
        else:
            overlap[3].update(term.keys)
    # A modifier right before a noun ("blue curtains", "wooden desk") describes it; a material word that is also a
    # flooring option ("wood" in "wooden desk") is then not a floor.
    modifiers_of: dict[int, list[str]] = {}
    for i, (start, end, term, _) in enumerate(chosen):
        if term.kind != "object":
            continue
        words = re.findall(r"[a-z']+", clause[:start])[-2:]
        mods = [w for w in words if _MODIFIER.fullmatch(w)]
        trailing = _TRAILING_COLOUR.match(clause[end:])
        if trailing:
            mods.append(trailing.group(1))
        modifiers_of[i] = mods
    object_starts = {c[0] for c in chosen if c[2].kind == "object" and not set(c[2].keys) & _FLOORS}
    out = []
    for i, (start, end, term, keys) in sorted(enumerate(chosen), key=lambda item: item[1][0]):
        after = _WORD_REST.match(clause, end)
        if set(term.keys) & _FLOORS and after and after.end() in object_starts:
            continue  # "wooden desk", "stone counters": a material of the next object, not the floor
        negated = bool(_NEGATED_BEFORE.search(clause[:start]) or _NEGATED_AFTER.match(clause[end:]))
        phrase = term.phrase or _own_words(clause[start:end])
        out.append((term, phrase, keys, modifiers_of.get(i, []), negated))
    return out


def visual_query(text: str, cues: Iterable[str] = ()) -> str | None:
    """A CLIP prompt made of the message's visual content not already captured as a cue, or None.

    `cues` are the keys this message already turned into evidence: feature keys ("window"), choice and option keys
    ("flooring", "carpet") and room-type labels ("bathroom"); in sessions that is Reading.covered plus the room types
    mentioned. Negated terms are never sent. Captured terms are left out, so a message whose every visual word was a
    cue returns None and gets no CLIP update."""
    cues = set(cues)
    room, light, items = None, None, []
    for clause in _CLAUSE.split(text.lower()):
        for term, phrase, keys, mods, negated in _clause_terms(clause or ""):
            if negated or keys & cues:
                continue
            if term.kind == "room":
                room = room or phrase
            elif term.kind == "light":
                light = light or phrase
            else:
                phrase = _with_article(phrase, mods)
                if phrase not in items:
                    items.append(phrase)
    if not (room or light or items):
        return None
    subject = room or "a room"
    if light:
        subject = _with_article(subject, [light])
    prompt = f"a photo of {subject}"
    return f"{prompt} with {_join(items[:MAX_TERMS])}" if items else prompt
