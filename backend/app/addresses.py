"""Fuzzy address similarity, used to pick the Matterport model for a caller's address.

Matterport's model search is a loose text search that always returns results (e.g. "1600 Pennsylvania Ave,
Washington DC" comes back as "2520 Washington Ave, Houston"), so every result is re-scored here and only results
above config.ADDRESS_MATCH_THRESHOLD are used.

The two sides are parsed differently on purpose. Matterport addresses are consistently formatted
("4010 W Bellfort Ave, 227, Houston, TX, 77025, US"), so they are split on commas. Caller/dispatcher input is free-form
("13122 chipman glen drive houston"), so it is treated as a bag of tokens, and the street is whatever sits between
the house number and the first street suffix, comma, or state/ZIP. Callers who text or speak the address go through
parse_caller first (spoken numbers, filler, then the trained usaddress parser), which falls back to parse_freeform.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import usaddress
from text_to_num import alpha2digit

try:  # rapidfuzz is ~20x faster than difflib for the same ratio; difflib stays as a fallback
    from rapidfuzz.fuzz import ratio as _fuzz_ratio

    def _ratio(a: str, b: str) -> float:
        return _fuzz_ratio(a, b) / 100
except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

# Spelled-out and abbreviated forms are mapped to one canonical token so "Drive" == "Dr".
SUFFIXES = {
    "drive": "dr", "dr": "dr", "street": "st", "st": "st", "lane": "ln", "ln": "ln", "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "boulevard": "blvd", "blvd": "blvd", "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir", "way": "way", "place": "pl", "pl": "pl", "cove": "cv", "cv": "cv",
    "cte": "cte", "parkway": "pkwy", "pkwy": "pkwy", "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl",
    "highway": "hwy", "hwy": "hwy", "square": "sq", "sq": "sq", "loop": "loop", "path": "path", "row": "row",
}
DIRECTIONS = {"north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
              "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
              "ne": "ne", "nw": "nw", "se": "se", "sw": "sw"}
UNIT_WORDS = {"apt", "apartment", "unit", "suite", "ste", "#", "no", "fl", "floor", "rm", "room", "bldg"}
STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la",
    "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or",
    "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "pr",
}
COUNTRY_WORDS = {"us", "usa", "united", "states"}
STATE_NAMES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
    "district of columbia": "dc",
}

# Component weights for the final score. The street name is the strongest identity signal; the house number
# separates houses on that street; city/state/ZIP mainly rejects same-named streets in other cities.
STREET_WEIGHT, NUMBER_WEIGHT, LOCALITY_WEIGHT = 0.45, 0.30, 0.25
# Two street words count as the same if their spelling similarity is at least this high (catches "Chipmen" for
# "Chipman"); below it they count as different, so a shared generic word like "Glen" can't carry a match alone.
TOKEN_MATCH_MIN = 0.8
# House numbers this far apart score 0. Within it the score falls linearly from 0.7 (so an exact number always
# beats a neighbour) to 0, so a house on the same block still counts as "sufficiently similar".
NUMBER_SPAN = 200


@dataclass
class Address:
    number: str | None
    street: list[str]        # canonical street words without suffix, e.g. ["w", "bellfort"]
    unit: str | None
    city: list[str]
    state: str | None
    zip: str | None
    tokens: list[str]        # every canonical token, for free-form lookups


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9#]+", text.lower().replace("'", ""))
    return [DIRECTIONS.get(w, SUFFIXES.get(w, w)) for w in words]


def _street_words(words: list[str]) -> list[str]:
    # Suffixes are dropped because callers and Matterport abbreviate (or omit) them inconsistently.
    return [w for w in words if w not in SUFFIXES.values()]


def parse_matterport(text: str) -> Address:
    """Parse Matterport's comma-separated format: street, [unit,] city, ST, ZIP, US."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    parts = [p for p in parts if p.lower() not in COUNTRY_WORDS and p.lower() != "united states"]
    zip_code = next((p[:5] for p in parts if re.fullmatch(r"\d{5}(-\d{4})?", p)), None)
    state = next((p.lower() for p in parts if p.lower() in STATES), None)
    rest = [p for p in parts if not re.fullmatch(r"\d{5}(-\d{4})?", p) and p.lower() != state]

    number, street, unit, city = None, [], None, []
    if rest:
        words = _tokens(rest[0])
        if words and words[0].isdigit():
            number, words = words[0], words[1:]
        unit_at = next((i for i, w in enumerate(words) if w in UNIT_WORDS), None)
        if unit_at is not None:  # "5200 Wilshire Blvd Apt 457"
            unit = "".join(words[unit_at + 1:]) or None
            words = words[:unit_at]
        street = _street_words(words)
        middle = rest[1:]
        if middle:
            city = _tokens(middle[-1])
            for part in middle[:-1]:  # "4010 W Bellfort Ave, 227, Houston": the bare part between is the unit
                unit = unit or "".join(w for w in _tokens(part) if w not in UNIT_WORDS) or None
    return Address(number, street, unit, city, state, zip_code, _tokens(text))


def parse_freeform(text: str) -> Address:
    """Parse whatever a dispatcher typed. Only the house number, street, unit, state and ZIP are located; the city
    is left empty and matched against all tokens instead (see _locality_score)."""
    tokens = _tokens(text)
    first_line = _tokens(text.split(",")[0])
    number = first_line[0] if first_line and first_line[0].isdigit() else None
    zip_code = next((t for t in tokens[1:] if re.fullmatch(r"\d{5}", t)), None)
    # Two-letter states collide with words ("in", "or", "me", "la"), so only the last word counts as a state,
    # ignoring a trailing ZIP or country.
    tail = [t for t in tokens[1:] if t != zip_code and t not in COUNTRY_WORDS]
    state = tail[-1] if tail and tail[-1] in STATES and len(tokens) > 2 else None

    words = first_line[1:] if number else first_line
    street = []
    for word in words:
        if word in SUFFIXES.values() and street:
            break  # the suffix ends the street; whatever follows is the city (no-comma input)
        if word in UNIT_WORDS or word.startswith("#") or word == zip_code or (word == state and street):
            break
        street.append(word)

    # Unit: the word after "apt"/"unit"/..., or "#227", anywhere in the input.
    unit = next((t[1:] for t in tokens if t.startswith("#") and len(t) > 1), None)
    for i, token in enumerate(tokens[:-1]):
        if token in UNIT_WORDS and unit is None:
            unit = tokens[i + 1].lstrip("#")

    # City: only knowable when the input has commas ("<street>, <city>, <state> <zip>").
    parts = [p for p in text.split(",")[1:] if p.strip()]
    city_words = _tokens(parts[0]) if parts else []
    city = [w for w in city_words if w not in UNIT_WORDS and not w.isdigit() and w != state and not w.startswith("#")]
    if city_words and city_words[0] in UNIT_WORDS:  # "123 Oak St, Apt 4, Houston": the city is the next part
        city = [w for w in (_tokens(parts[1]) if len(parts) > 1 else []) if w != state and not w.isdigit()]
    return Address(number, _street_words(street), unit, city, state, zip_code, tokens)


# ---- caller input: spoken or typed, possibly with filler ("um it's 13122 chipman glen in houston texas")

# Words a caller puts before the address. Only used when the address has no house number to start at.
_LEAD_FILLER = re.compile(r"^(?:(?:um+|uh+|er+|so|okay|ok|yeah|yes|hi|hello|well|it'?s|it is|i'?m|i am|we'?re|we are|"
                          r"i live|we live|my address is|the address is|address is|at|on|in|the)\b[\s,.:]*)+", re.I)
# Words that join the street and the city in speech; usaddress reads "glen in houston" as street type "in".
_JOINERS = re.compile(r"\s+(?:in|at|near)\s+", re.I)
# Words between the street words that are never part of a street name.
_STOP_WORDS = {"in", "at", "on", "of", "near", "by", "the"}


def _group_house_number(text: str) -> str:
    """Join the digit groups a caller says before the street into one house number: "13 1 22 chipman" -> "13122",
    "79 21 pointer" -> "7921", "5 oh 5 oak" -> "505". Only leading groups are joined, so "233 W 27th St" is unchanged."""
    words = text.split()
    groups = []
    while words:
        word = words[0].lower()
        if word.isdigit():
            groups.append(word)
        elif groups and word in ("oh", "o") and len(words) > 1 and words[1].isdigit():
            groups.append("0")   # "five oh five": a spoken zero between digits
        else:
            break
        words.pop(0)
    return " ".join(["".join(groups)] + words) if groups else text


def clean_caller(text: str) -> str:
    """The caller's text reduced to the address: spoken numbers as digits, digit groups joined, filler before the
    house number dropped, and speech joiners ("in houston") turned into commas. Case is kept for display."""
    text = " ".join(text.replace("\n", " ").split())
    try:
        converted = alpha2digit(text, "en")
    except Exception:  # text2num rejects some odd inputs; the plain text still parses
        converted = text
    # The house number: digits followed by a word ("13122 chipman", "13 1 22 chipman", "5 oh 5 oak"). A number
    # followed by a comma is a ZIP or unit ("Princeton, NJ, 08544, US") and is not a place to start.
    house = re.search(r"\b\d+(?:\s+(?:\d+|oh|o)\b)*\s+[A-Za-z]", converted, re.I)
    if house:
        converted = converted[house.start():]   # start at the house number: drops "um it's", "I'm at"
    else:
        converted = _LEAD_FILLER.sub("", converted)
    converted = _group_house_number(converted)
    converted = _JOINERS.sub(", ", converted)
    return converted.strip(" ,.")


def _split_state(words: list[str]) -> tuple[list[str], str | None]:
    """Take a trailing state ("houston texas", "princeton nj") off a city."""
    for size in (3, 2, 1):
        tail = " ".join(words[-size:]) if len(words) >= size else ""
        if tail in STATE_NAMES:
            return words[:-size], STATE_NAMES[tail]
    if len(words) > 1 and words[-1] in STATES:
        return words[:-1], words[-1]
    return words, None


@lru_cache(maxsize=512)  # one caller address is compared with every candidate; callers must not mutate the result
def parse_caller(text: str) -> Address:
    """Parse a caller's address with usaddress (a trained US parser) after clean_caller. Falls back to
    parse_freeform when usaddress can't label it or finds no street."""
    cleaned = clean_caller(text)
    try:
        tagged, _kind = usaddress.tag(cleaned)
    except usaddress.RepeatedLabelError:
        return parse_freeform(cleaned)
    street_text = " ".join(tagged.get(k, "") for k in (
        "StreetNamePreDirectional", "StreetNamePreModifier", "StreetNamePreType", "StreetName",
        "StreetNamePostType", "StreetNamePostDirectional"))
    street = [w for w in _street_words(_tokens(street_text)) if w not in _STOP_WORDS]
    number_match = re.match(r"\d+", tagged.get("AddressNumber", ""))
    if not street or any(w.isdigit() for w in street) or (tagged.get("AddressNumber") and not number_match):
        # No street, a number inside the street ("1 22 chipman"), or a word read as the house number ("I'm on
        # Chipman Glen"): the trained parser got lost, so use the hand-written one.
        return parse_freeform(cleaned)
    unit = "".join(w for w in _tokens(tagged.get("OccupancyIdentifier", "")) if w not in UNIT_WORDS).lstrip("#")
    city = [w for w in _tokens(tagged.get("PlaceName", "")) if not w.isdigit()]
    city, state_from_city = _split_state(city)
    state = tagged.get("StateName", "").lower().strip(". ") or state_from_city
    state = STATE_NAMES.get(state, state)
    zip_match = re.match(r"\d{5}", tagged.get("ZipCode", ""))
    return Address(number=number_match.group() if number_match else None, street=street, unit=unit or None,
                   city=city, state=state if state in STATES else None,
                   zip=zip_match.group() if zip_match else None, tokens=_tokens(cleaned))


def _token_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a.isdigit() or b.isdigit():  # "26th" vs "27th" or "2nd" vs "3rd" are different streets, not typos
        return 0.0
    ratio = _ratio(a, b)
    return ratio if ratio >= TOKEN_MATCH_MIN else 0.0


def _coverage(words: list[str], pool: list[str]) -> float:
    """Average best-match similarity of each word in `words` against `pool`."""
    if not words or not pool:
        return 0.0
    return sum(max(_token_similarity(w, p) for p in pool) for w in words) / len(words)


def _street_score(query: Address, candidate: Address) -> float:
    # Directions ("W", "N") are compared separately: callers often omit them, and as ordinary words a shared "W"
    # gave "W 27th St" half credit against "W 26th St".
    q_words = [w for w in query.street if w not in DIRECTIONS.values()]
    c_words = [w for w in candidate.street if w not in DIRECTIONS.values()]
    # Both directions of coverage, so neither a candidate with extra words ("Post Oak Blvd Paw Spa") nor a query
    # with extra words scores as a full match.
    score = (_coverage(q_words, c_words) + _coverage(c_words, q_words)) / 2
    q_dirs = {w for w in query.street if w in DIRECTIONS.values()}
    c_dirs = {w for w in candidate.street if w in DIRECTIONS.values()}
    if q_dirs and c_dirs and q_dirs != c_dirs:
        score *= 0.8  # "E Bellfort" vs "W Bellfort": same name, other side of town
    return score


def _number_score(query: Address, candidate: Address) -> float | None:
    if query.number is None:
        return None  # nothing to compare; the component is left out
    if candidate.number is None:
        return 0.0   # a street-level model ("Post Oak Blvd Sports Lounge") is not a specific house
    if query.number == candidate.number:
        return 1.0
    diff = abs(int(query.number) - int(candidate.number))
    return max(0.0, 0.7 * (1 - diff / NUMBER_SPAN))


def _locality_score(query: Address, candidate: Address) -> float | None:
    """Most specific evidence wins: ZIP, then city, then state. 0 means the query names a different place."""
    if query.zip and candidate.zip:
        if query.zip == candidate.zip:
            return 1.0
        # A neighbouring ZIP in the same city is a common slip; a different ZIP region is a different place.
        same_city = candidate.city and _coverage(candidate.city, query.tokens) >= TOKEN_MATCH_MIN
        return 0.6 if same_city or query.zip[:3] == candidate.zip[:3] else 0.0
    if query.city and candidate.city:
        return 0.9 if _coverage(candidate.city, query.city) >= TOKEN_MATCH_MIN else 0.0
    if candidate.city and _coverage(candidate.city, query.tokens) >= TOKEN_MATCH_MIN:
        return 0.9  # no-comma input: "13122 chipman glen dr houston"
    if query.state and candidate.state:
        return 0.4 if query.state == candidate.state else 0.0
    return None


def similarity(query_text: str, candidate_text: str) -> float:
    """0..1 similarity between a free-form address and a Matterport-formatted one."""
    query, candidate = parse_caller(query_text), parse_matterport(candidate_text)
    if not candidate.street:
        return 0.0
    parts = [(STREET_WEIGHT, _street_score(query, candidate)),
             (NUMBER_WEIGHT, _number_score(query, candidate)),
             (LOCALITY_WEIGHT, _locality_score(query, candidate))]
    known = [(w, s) for w, s in parts if s is not None]
    score = sum(w * s for w, s in known) / sum(w for w, _ in known)
    locality = parts[2][1]
    if locality == 0.0:
        score *= 0.5  # the query names another city/ZIP/state: same street name elsewhere is a different building
    # Unit is only a tie-breaker between models in the same building (e.g. many units at 700 Dunson Glen Dr).
    if query.unit and candidate.unit:
        score += 0.03 if query.unit == candidate.unit else -0.03
    return max(0.0, min(1.0, score))


def with_locality(name: str, address: str) -> str:
    """A model's name followed by the city/state/ZIP parts of its address field (everything after the street), so
    a name like "2810 Princeton St" is still compared as being in Houston. Returns the name as is when the address
    field has nothing after the street."""
    if not name:
        return ""
    locality = address.split(",", 1)[1].strip() if "," in address else ""
    return f"{name}, {locality}" if locality else name


def is_exact(query_text: str, candidate_text: str) -> bool:
    """Same house number and street words (suffix and city formatting aside), and no conflicting unit: "700 Dunson
    Glen Dr Unit 1013" is not exactly "..., Unit 107", but a query without a unit is exact for every unit."""
    query, candidate = parse_caller(query_text), parse_matterport(candidate_text)
    if query.unit and candidate.unit and query.unit != candidate.unit:
        return False
    return query.number is not None and query.number == candidate.number and query.street == candidate.street
