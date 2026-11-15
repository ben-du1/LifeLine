"""Follow-up questions that build on what the caller has said, and how their answers become evidence.

Instead of walking a fixed yes/no list, every turn:
  1. The caller's words are read for cues: size ("cramped", "huge"), darkness, objects they mention or deny, room names.
     Each cue is a likelihood over candidates (see `cue_evidence`).
  2. Possible next questions are generated from the building: yes/no features ("a toilet or tub"), choice
     questions ("carpet, tile or wood?", "closet-sized, bedroom-sized or bigger?") and a "which of these do you see"
     list built from the features that best separate the rooms still in play.
  3. The question with the highest expected information gain over the room distribution is asked, phrased to
     acknowledge what the caller just said ("You said it's a tight space. What can you see around you: ...").

Every feature has a model P(yes | candidate) that combines up to three sources (see `build_model`):
  - CLIP: how well the candidate's photos match the feature prompt (z-scored across candidates).
  - Room type: Matterport's classification/label (a toilet implies a bathroom). This is what lets photo-less rooms
    (often bathrooms and closets) be found at all.
  - Geometry: room area and shape from the floor plan outline (for size and "long and narrow").
"""

import itertools
import re
from dataclasses import dataclass, field

import numpy as np

from . import config, layout, locator, questions, commonsense

# ---- feature and question bank --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
    key: str
    noun: str                         # short phrase for lists and acknowledgements: "a toilet or tub"
    question: str                     # yes/no phrasing
    words: str                        # regex: the caller mentioned it
    prompt: str | None                # CLIP prompt for a view where it is present (None: geometry only)
    room_types: tuple[str, ...] = ()  # room types/labels where it is typically found (lowercase substrings)


FEATURES = [
    Feature("toilet", "a toilet, tub or shower", "Is there a toilet, bathtub or shower?",
            r"toilet|bath ?tub|\btub\b|shower", "a photo of a bathroom with a toilet and bathtub", ("bath",)),
    Feature("sink", "a sink", "Can you see a sink or faucet?", r"\bsinks?\b|faucet",
            "a photo of a room with a sink and faucet", ("bath", "kitchen", "laundry")),
    Feature("mirror", "a mirror", "Is there a mirror on the wall?", r"mirror",
            "a photo of a room with a large mirror", ("bath",)),
    Feature("stove", "a stove or kitchen counters", "Can you see a stove, oven or kitchen counters?",
            # Not bare "cabinets" or "counter": bathrooms have a vanity cabinet and counter too.
            r"stove|oven|\b(cooktop|burners?|microwave|dishwasher)\b|kitchen (cabinets?|counters?)",
            "a photo of a kitchen with a stove and cabinets", ("kitchen",)),
    Feature("fridge", "a fridge", "Can you see a refrigerator?", r"fridge|refrigerator|freezer",
            "a photo of a kitchen with a refrigerator", ("kitchen",)),
    Feature("washer", "a washer or dryer", "Is there a washing machine or dryer?",
            # Not "dishwasher", "hair dryer" (bathroom) or "laundry basket" (bedroom).
            r"\bwashers?\b|(?<!hair )(?<!blow )\bdryers?\b|washing machine|\blaundry\b(?!\s+(basket|hamper|pile))",
            "a photo of a laundry room with a washing machine and dryer", ("laundry", "utility")),
    Feature("shelves", "shelves or hanging clothes", "Are there shelves, or a rod with clothes hanging?",
            # Not bare "clothes": "the thing you wash clothes in" is a washer, not a closet.
            # Not "bookshelf": that's a living room or office.
            r"(?<!book)(?<!book )shel(f|ves|ving)|\bhangers?\b|hanging clothes|clothes hanging|\brod\b|\bcoats?\b",
            "a photo of a small closet with shelves and a clothes rod", ("closet", "pantry", "laundry")),
    Feature("bed", "a bed", "Is there a bed?", r"\bbeds?\b|mattress", "a photo of a bedroom with a bed",
            ("bedroom",)),
    Feature("sofa", "a couch", "Is there a couch or sofa?", r"couch|sofa", "a photo of a living room with a sofa",
            ("living", "family")),
    Feature("table", "a table with chairs", "Is there a table with chairs?",
            # Not a bedside, end or coffee table (bedroom, living room) or a wheelchair/armchair.
            r"(?<!bedside )(?<!night )(?<!side )(?<!end )(?<!coffee )(?<!changing )\btables?\b|\bchairs?\b",
            "a photo of a dining table with chairs", ("dining", "kitchen")),
    Feature("tv", "a TV", "Can you see a TV?", r"\btv\b|television", "a photo of a room with a television",
            ("living", "family", "bedroom")),
    Feature("fireplace", "a fireplace", "Is there a fireplace?", r"fireplace|mantel|hearth",
            "a photo of a living room with a brick fireplace", ("living", "family")),
    Feature("window", "a window", "Is there a window in the room?", r"\bwindow(s|less)?\b",
            "a photo of a room with a large window"),
    Feature("fan", "a ceiling fan", "Is there a ceiling fan?",
            # Not an exhaust/vent fan (bathrooms) or a box/desk fan.
            r"ceiling fan|(?<!exhaust )(?<!vent )(?<!box )(?<!desk )(?<!bathroom )\bfans?\b",
            "a photo of a room with a ceiling fan", ("bedroom", "living")),
    Feature("stairs", "stairs", "Can you see stairs?",
            # Whole words: "upstairs bathroom" is a floor, not stairs in the room; "steps away" is a distance.
            r"\b(stairs?|staircase|stairway|stairwell)\b|\bsteps\b(?!\s+(away|from|back))",
            "a photo of a staircase inside a house", ("stair", "entry", "foyer")),
    Feature("front_door", "the front door", "Can you see the front door of the building?",
            r"front door|entrance|entry door|main door", "a photo of a house entrance with a front door",
            ("entry", "foyer", "living")),
    Feature("sliding_door", "a glass door to outside", "Is there a glass sliding door to the outside?",
            # Not "sliding closet door".
            r"sliding (glass )?doors?|\bslider\b|patio|glass door|back door",
            "a photo of a room with a sliding glass patio door",
            ("living", "dining", "kitchen", "family")),
    Feature("narrow", "a long narrow space", "Is it long and narrow, like a hallway?",
            r"narrow|corridor|hallway|long and thin", None, ("hall",)),
    Feature("outside_door", "a door to the outside", "Is there a door in your room that leads straight outside?",
            r"door (that )?(leads?|goes|opens?) (straight )?(to the )?outside|door to the (yard|backyard|patio|deck|"
            r"garden|outside)|back door|side door|exterior door", None),
    Feature("bath_door", "a door into a bathroom", "Is there a door from your room straight into a bathroom?",
            r"(own|attached|connected|private|en.?suite) bath(room)?|bathroom (door )?(in|off|inside|attached to) "
            r"(my|the|this) room", None),
    Feature("closet_door", "a closet in the room", "Is there a closet in your room, with its own door?",
            r"closet door|(walk.?in )?closet (in|inside) (my|the|this) room|my closet", None),
    Feature("street_view", "the street out a window", "From a window in your room, can you see the street in "
            "front of the house?", r"see the street|street outside|see (the )?cars|front yard|"
            r"windows? (face|faces|facing|looks?( out)? on(to)?) the street", None),
]
FEATURES_BY_KEY = {f.key: f for f in FEATURES}
LAYOUT_FEATURES = ("outside_door", "bath_door", "closet_door", "street_view")


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    words: str
    prompt: str | None = None
    room_types: tuple[str, ...] = ()
    # Also accepted, only as the reply to this option's own question and only when no option's `words` matched: a bare
    # "3" answers "which floor?", but "3 windows" said anywhere else must not be read as the 3rd floor.
    excluded: tuple[str, ...] = ()
    reply_words: str = ""


@dataclass(frozen=True)
class Choice:
    key: str
    question: str
    options: tuple[Option, ...]


_NEVER_CARPETED = ("garage", "bath", "kitchen", "laundry", "utility")

CHOICES = [
    Choice("flooring", "What is the floor like: carpet, tile, wood, or concrete?", (
        Option("carpet", "carpet", r"carpet|\brugs?\b", "a photo of a room with carpet floor",
               ("bedroom", "closet", "living", "family"), _NEVER_CARPETED),
        Option("tile", "tile", r"\btiles?\b|tiled|linoleum|vinyl|stone|ceramic", "a photo of a room with tile floor",
               ("bath", "kitchen", "laundry", "utility", "entry"), ("garage",)),
        Option("wood", "wood", r"wood|hardwood|laminate|planks?", "a photo of a room with hardwood floor",
               ("living", "dining", "hall", "kitchen"), ("garage",)),
        Option("concrete", "concrete", r"concrete|cement|epoxy", "a photo of a garage with a bare concrete floor",
               ("garage", "utility", "basement", "laundry")),
    )),
    # Size is answered from the floor plan geometry, not from photos (see _size_likelihoods). The question names no
    # rooms: "bigger, like a living room" invited "big like a living room", which was then read as "I'm in the living
    # room". Room words in the answer are still understood as sizes (and never as a room claim: see
    # questions.room_mentions).
    Choice("size", "About how big is the room: tiny (you can almost touch both walls), medium (a few steps across), "
           "or large (lots of open space)?", (
        Option("small", "tiny", r"closet|small|tiny|cramped|tight|little|touch both|bathroom.sized"),
        Option("medium", "medium", r"bedroom|medium|normal|average|regular|few steps"),
        Option("large", "large", r"big|bigger|large|living|huge|open|spacious|lots of"),
    )),
]
CHOICES_BY_KEY = {c.key: c for c in CHOICES}

# Layout choices (layout.py). Their options are fixed so answers can be parsed (and Claude can be given them), but
# each building only offers the door destinations it actually has.
WINDOWS_CHOICE = Choice("windows", "How many windows does the room have: none, one, or more than one?", (
    Option("none", "none", r"\b(none|no windows?|zero|0|not any)\b"),
    # Not after "than": "more than one" matched both "one" and "more than one", which made the answer worthless.
    Option("one", "one", r"(?<!than )\b(one|1|a single|just one|single|a window)\b"),
    Option("several", "more than one", r"\b(two|three|four|five|several|many|multiple|more than one|lots|[2-9])\b"),
))
DOOR_OPTIONS = (  # key, label, words, layout kinds it covers
    ("hallway", "a hallway", r"hall|corridor|passage", ("hallway",)),
    ("bedroom", "a bedroom", r"bed ?room", ("bedroom",)),
    ("living", "the kitchen or living room", r"kitchen|living|dining|family|lounge|main room", ("living",)),
    ("bathroom", "a bathroom", r"bath|toilet", ("bathroom",)),
    ("stairs", "stairs or a landing", r"stair|steps|landing", ("stairs",)),
    ("entry", "the entryway", r"entry|foyer|front door|porch", ("entry",)),
    ("outside", "outside", r"outside|yard|garden|patio|deck|garage|street", ("outside", "garage")),
)
DOOR_QUESTION = "What is right outside the door of the room you're in: {listed}?"
# Where the room is in the house, relative to the street (needs orientation.py's street direction). On 1348 W 35th St
# the three top-floor bedrooms have the same doors (hallway + own bathroom) and no annotated windows; their position
# front-to-back (0.84 / 0.57 / 0.21 of the floor's depth) is the one thing that tells them apart.
POSITION_CHOICE = Choice("position", "Is your room at the front of the house (the street side), in the middle, or at "
                                     "the back?", (
    Option("front", "the front", r"\bfront\b|street|road"),
    Option("middle", "the middle", r"\b(middle|center|centre|between)\b"),
    Option("back", "the back", r"\b(back|rear)\b|back ?yard|\byard\b"),
))
# Centres of "front" / "middle" / "back" as a fraction of the floor's depth from the back (0) to the street (1), and
# how far a room can be from a centre and still be called that. POSITION_NOISE of answers are spread evenly: "front"
# may mean the front-door side when that isn't the street, and callers in a panic misjudge.
POSITION_CENTRES, POSITION_WIDTH, POSITION_NOISE = (0.8, 0.5, 0.2), 0.15, 0.25

# Cues in free text. Size cues reuse the size likelihoods; "dark" counts as evidence of no window.
# "Narrow" is deliberately not a size cue: it's a shape (the "narrow" feature), and hallways can be long.
SMALL_CUE = re.compile(r"\b(cramped|(?<!hold )(?<!sit )(?<!holding )(?<!sitting )tight|tiny|small|confined|enclosed|"
                       r"squeez\w*|barely fit|no room to move|can'?t move)\b|"
                       r"\blittle\s+(room|space|area|spot|place|closet|bathroom|bedroom|office|nook)\b", re.I)
STRICT_SMALL_CUE = re.compile(r"\b(cramped|(?<!hold )(?<!sit )(?<!holding )(?<!sitting )tight|tiny|confined|"
                              r"squeez\w*|barely fit|no room to move|can'?t move)\b", re.I)
LARGE_CUE = re.compile(r"\b(big|large|huge|spacious|giant|massive|open (space|area|plan|concept|room))\b", re.I)
DARK_CUE = re.compile(r"\b(dark|pitch black|no light|can'?t see anything|lights? (are |is )?(off|out))\b", re.I)
ANY_NEGATION = re.compile(r"\b(no|not|none|nothing|without|never)\b|n't\b", re.I)
NEGATION = re.compile(r"\b(no|not|isn'?t|aren'?t|without|don'?t see|can'?t see|nothing)\b[^.,;]*$", re.I)

# Free-form "which of these" question: how many features to list.
LIST_SIZE = 3


# ---- per-building model ---------------------------------------------------------------------------------------


@dataclass
class FeatureModel:
    p_yes: dict[str, np.ndarray]           # feature key -> P(yes | candidate), shape (n_candidates,)
    p_option: dict[str, np.ndarray]        # choice key -> P(option | candidate), shape (n_options, n_candidates)
    room_area: np.ndarray                  # (n_rooms,) square meters, NaN if unknown
    candidate_room: np.ndarray             # (n_candidates,) index into rooms
    floor_options: tuple = ()              # one Option per floor (lowest first) when the building has 2+ floors
    candidate_floor: np.ndarray | None = None  # (n_candidates,) floor index of each candidate
    near: dict | None = None               # landmark key -> raw close-up CLIP score per candidate (NaN: no photos)
    choices: dict = field(default_factory=dict)    # building-specific Choices (layout: "windows", "door_to")
    has_views: np.ndarray | None = None
    p_local: dict | None = None
    positions: dict = field(default_factory=dict)
    adjacent: dict = field(default_factory=dict)   # door-option key -> P(room has a door to that kind | candidate)


# Landmarks for the in-room "what's closest to you?" question: fixtures a person stands next to, with how they are
# named to the caller. Their close-up CLIP prompt ("a close-up photo of the stove") scores highest at the scan points
# standing next to them: in the sample house's main room, kitchen items peaked at the sweep between sink, stove and
# fridge, and the fireplace at the sweep beside it (checked against the photos).
NEAR_NAMES = {
    "stove": "the stove", "fridge": "the fridge", "sink": "the sink", "toilet": "the toilet or tub",
    "washer": "the washer or dryer", "shelves": "shelves or hanging clothes", "bed": "the bed", "sofa": "the couch",
    "table": "the table", "tv": "the TV", "fireplace": "the fireplace", "window": "a window", "stairs": "the stairs",
    "front_door": "the front door", "sliding_door": "the glass door to outside", "mirror": "the mirror",
}


def near_prompt(key: str) -> str:
    return f"a close-up photo of {NEAR_NAMES[key]}"


def clip_prompts() -> list[str]:
    """Every CLIP prompt the model needs, in a fixed order (embedded once per building)."""
    prompts = [f.prompt for f in FEATURES if f.prompt]
    prompts += [o.prompt for c in CHOICES for o in c.options if o.prompt]
    # Ensemble phrasings and baselines after the original prompts, so the originals keep their order.
    extra = [v for p in prompts for v in PROMPT_VARIANTS.get(p, ())] + list(BASELINE_PROMPTS)
    return list(dict.fromkeys(prompts + extra))


def _polygon_area(polygon: list[list[float]] | None) -> float:
    if not polygon or len(polygon) < 3:
        return float("nan")
    xs, ys = np.array(polygon).T
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2)


def _aspect_ratio(room: dict) -> float:
    """Long side / short side of the room's bounding box (walls are mostly axis-aligned in these scans)."""
    points = np.array(room.get("polygon") or [p for w in room.get("walls") or [] for p in (w[:2], w[2:])])
    if len(points) < 2:
        return float("nan")
    extent = points.max(axis=0) - points.min(axis=0)
    return float(extent.max() / max(extent.min(), 0.3))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _size_likelihoods(area: np.ndarray) -> np.ndarray:
    """P(small/medium/large answer | room area). Breakpoints from the sample buildings: closets 0.5-5 m², bathrooms
    5-6 m², bedrooms 12-21 m², living areas 26-60 m². Unknown area gives no information."""
    small = _sigmoid((6.5 - area) / 1.5)
    large = _sigmoid((area - 20.0) / 3.0)
    medium = np.clip(1.0 - small - large, 0.05, None)
    probs = np.stack([small, medium, large])
    probs = np.where(np.isnan(probs), 1.0 / 3, probs)
    return probs / probs.sum(axis=0, keepdims=True)


def _room_kinds(room: dict) -> str:
    return " ".join([*room.get("types", []), room.get("label") or "", room.get("name") or ""]).lower()


def _type_presence(room_kinds: list[str], types: tuple[str, ...]) -> np.ndarray | None:
    """Per-room P(feature present) from the room's type. Matterport labels are reliable but not exhaustive (a
    kitchen may still have a window), so a match gives 0.85 and a non-match 0.15, not 1 and 0."""
    if not types:
        return None
    return np.array([0.85 if any(t in kinds for t in types) else 0.15 for kinds in room_kinds])


def _clip_presence(raw: np.ndarray, has_views: np.ndarray, peers: dict[int, list[np.ndarray]] | None = None,
                   contrast: np.ndarray | None = None) -> np.ndarray | None:
    """Per-candidate P(feature present) from CLIP.

    Without `contrast` (old behaviour): z-score over imaged candidates, then a sigmoid; slope 1.5 makes a candidate
    1 std above average ~0.82. That is relative: about half the rooms look "above average" even when the feature is
    nowhere in the building. With `contrast` (how much the feature beats the neutral prompts in the candidate's best
    view, see view_contrast): a fixed logistic gives an absolute log-odds, and the z-score is added at a reduced slope
    so candidates are still ranked if the calibration is off (constants and reasons in config).

    A candidate without photos takes the mean presence of the photographed rooms of its type (`peers`, from
    Building.imageless_peers): otherwise a photographed bathroom's CLIP "toilet" pushes its P(yes) above a photo-less
    bathroom's, and "yes, a toilet" favours it for no real reason. NaN (label only) when no room of its type has
    photos."""
    if not has_views.any():
        return None
    out = np.full(len(raw), np.nan)
    imaged = raw[has_views]
    z = (imaged - imaged.mean()) / (imaged.std() + 1e-6)
    if contrast is None:
        out[has_views] = _sigmoid(1.5 * z)
    else:
        absolute = (contrast[has_views] - config.CLIP_CONTRAST_CENTER) / config.CLIP_CONTRAST_SCALE
        absolute = np.clip(absolute, -config.CLIP_ABSOLUTE_MAX_LOGIT, config.CLIP_ABSOLUTE_MAX_LOGIT)
        relative = config.CLIP_RELATIVE_SLOPE * np.clip(z, -config.CLIP_RELATIVE_MAX_Z, config.CLIP_RELATIVE_MAX_Z)
        out[has_views] = _sigmoid(absolute + relative)
    for c, groups in (peers or {}).items():
        if groups:
            out[c] = np.mean([out[g].mean() for g in groups])
    return out


def _combine(clip: np.ndarray | None, by_type: np.ndarray | None, n: int) -> np.ndarray:
    """Weighted average of the available sources per candidate (NaN = unavailable); 0.5 if none. The room label
    counts double: in scripted runs, an equal average let CLIP noise (white closet shelves scoring as a laundry room)
    drown out a reliable label, so a "yes, there's a fridge" barely moved the kitchen. Clipped so no single answer
    rules a candidate out completely: callers misspeak and labels are imperfect."""
    total, weight = np.zeros(n), np.zeros(n)
    for source, w in ((clip, 1.0), (by_type, 2.0)):
        if source is not None:
            known = ~np.isnan(source)
            total[known] += w * source[known]
            weight[known] += w
    mean = np.where(weight > 0, total / np.maximum(weight, 1e-9), 0.5)
    return np.clip(mean, 0.05, 0.95)


ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def floor_options(n_floors: int) -> tuple:
    """One Option per floor, lowest first, with the words callers use for it. Matterport floor labels are just
    "Floor 1", "Floor 2", ... (lowest = 1), so floor 1 is taken to be the ground floor; a basement can't be told
    apart. "Upstairs" means floor 2 in a two-storey building and any floor above the ground in a taller one."""
    if n_floors < 2:
        return ()
    options = []
    for i in range(n_floors):
        n = i + 1
        word = ORDINALS[i] if i < len(ORDINALS) else str(n)
        number = rf"\b({n}|{_ordinal(n)}|{word})"
        words = [rf"{number}\s*(floor|story|storey|level)\b", rf"\b(floor|level)\s*{n}\b"]
        if i == 0:
            words.append(r"\b(ground|main|bottom|lowest|street)[\s-]*(floor|level)\b|\bdownstairs\b|\bground\b")
        else:
            words.append(r"\bupstairs\b")
        if i == n_floors - 1:
            words.append(r"\b(top|highest|upper)\s*(floor|level)\b|\battic\b")
        # The whole reply is just the floor: "3", "3rd", "third", "the 3rd one". A live caller answered "3" and then
        # "3rd" and was asked the floor question again both times.
        bare = rf"^\W*(i'?m\s+)?(on\s+)?(the\s+)?{number}(\s+(one|floor|story|storey|level))?\W*$"
        if i == n_floors - 1:
            bare += r"|^\W*(the\s+)?top(\s+one)?\W*$"
        options.append(Option(f"floor{i}", "ground floor" if i == 0 else f"{_ordinal(n)} floor", "|".join(words),
                              reply_words=bare))
    return tuple(options)


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]


def flights_options(n_floors: int) -> tuple:
    """Fallback when the caller doesn't know the floor: "how many flights of stairs did you climb?" One Option per
    floor (same keys as floor_options, so the same likelihoods apply). A plain "yes" / "upstairs" means some floor
    above the ground (every upper option), like "upstairs" in floor_options."""
    options = []
    for i in range(n_floors):
        word = NUMBER_WORDS[i] if i < len(NUMBER_WORDS) else str(i)
        if i == 0:
            words, reply = r"\b(no|none|nope|zero|0|didn'?t|did not|no stairs)\b", ""
            label = "none"
        else:
            words = rf"\b({i}|{word})\b" + (r"|\ba (flight|set)\b|\bonce\b" if i == 1 else "")
            words += r"|\btwice\b" if i == 2 else ""
            reply = r"^\W*(yes|yeah|yep|yup|up|upstairs|i did)\b"
            label = f"{word} flight{'s' if i > 1 else ''}"
        options.append(Option(f"floor{i}", label, words, reply_words=reply))
    return tuple(options)


def flights_question(n_floors: int) -> str:
    if n_floors == 2:
        return "Did you go up any stairs to get to the room you're in?"
    return "Did you go up any stairs to get to the room you're in? If yes, how many flights?"


def answer_hint(plan: "Plan") -> str:
    """How to answer, added when a question is asked again because the first answer couldn't be read."""
    if plan.key == "floor":
        return "Just the number is fine (1 is the ground floor)."
    if plan.key == "flights":
        return "Just the number of flights is fine (0 if none)."
    if plan.kind in ("yesno", "position"):
        return "Yes or no is fine."
    if plan.kind == "choice":
        return "One word is fine."
    return ""


def _floor_likelihoods(candidate_floor: np.ndarray, n_floors: int,
                       acc: float = config.FLOOR_ANSWER_ACCURACY) -> np.ndarray:
    """P(caller names floor j | candidate on floor i): `acc` when j == i, the rest spread evenly.
    Callers do mix up floor numbers (is the ground floor 1 or 0?), so a floor answer is strong evidence, not proof."""
    probs = np.full((n_floors, len(candidate_floor)), (1 - acc) / (n_floors - 1))
    probs[candidate_floor, np.arange(len(candidate_floor))] = acc
    return probs


def floor_question(options: tuple) -> str:
    labels = [o.label for o in options]
    if len(labels) == 2:
        return f"Which floor are you on: {labels[0]} or {labels[1]}?"
    if len(labels) <= 4:
        return f"Which floor are you on: {', '.join(labels[:-1])}, or {labels[-1]}?"
    return f"Which floor are you on? (The ground floor is floor 1; the building has {len(labels)} floors.)"


# How far a layout answer is trusted. Matterport's door/window annotations are good but not perfect (a doorway can be
# missed, a window drawn on the wrong wall), and callers misjudge too, so a matching answer is ~6x evidence, not proof.
LAYOUT_YES, LAYOUT_NO = 0.88, 0.12
# Share of "what's outside your door?" answers treated as noise. At 0.2 one wrong answer was ~17x evidence and, with
# 15% simulated caller errors, sent a Bedroom 2 caller to Bedroom 3 (the door is all that separates them); at 0.3 it
# is ~10x, and truthful results were unchanged (eval_scenarios.py).
DOOR_NOISE = 0.3
# A window "faces the street" when its outward direction is within this angle of the direction to the street.
STREET_FACING_DEG = 60.0


def _toward_street(orientation: dict | None) -> np.ndarray | None:
    """Unit vector toward the nearest street in model coordinates, from orientation.json (compass bearing of the
    street from the house, and the model's yaw: east = c*x - s*y, north = s*x + c*y)."""
    if not orientation or not orientation.get("street"):
        return None
    b = np.radians(orientation["street"]["bearing_deg"])
    t = np.radians(orientation.get("yaw_deg") or 0.0)
    east, north = np.sin(b), np.cos(b)
    return np.array([np.cos(t) * east + np.sin(t) * north, -np.sin(t) * east + np.cos(t) * north])


def _layout_features(rooms: list[dict], layout: dict, toward_street: np.ndarray | None):
    """Per-room P(yes) for the layout yes/no features, P(window count), and P(door to each kind), from layout.json.
    Rooms the layout doesn't cover get 0.5 / uniform (no information)."""
    n = len(rooms)
    yes = {k: np.full(n, 0.5) for k in LAYOUT_FEATURES}
    windows = np.full((3, n), 1 / 3)
    door_kinds = []                         # per room: list of door-option keys its doors lead to
    kind_to_option = {kind: key for key, _, _, kinds in DOOR_OPTIONS for kind in kinds}
    cos_facing = np.cos(np.radians(STREET_FACING_DEG))
    for i, room in enumerate(rooms):
        info = layout.get("rooms", {}).get(room["id"])
        if info is None:
            door_kinds.append(None)
            continue
        kinds = [d["to_kind"] for d in info["doors"]]
        options = [kind_to_option[k] for k in kinds if k in kind_to_option]
        door_kinds.append(options)
        yes["outside_door"][i] = LAYOUT_YES if "outside" in options else LAYOUT_NO
        yes["bath_door"][i] = LAYOUT_YES if "bathroom" in kinds and info["kind"] != "bathroom" else LAYOUT_NO
        # A caller standing in a closet won't say their room "has a closet"; closets themselves count as no.
        yes["closet_door"][i] = LAYOUT_YES if "closet" in kinds and info["kind"] != "closet" else LAYOUT_NO
        count = len(info["windows"])
        windows[:, i] = (1 - LAYOUT_YES) / 2
        windows[min(count, 2), i] = LAYOUT_YES
        if toward_street is not None:
            facing = any(np.dot(w["normal"], toward_street) >= cos_facing for w in info["windows"])
            yes["street_view"][i] = LAYOUT_YES if facing else (0.2 if count else 0.05)
    if toward_street is None:
        del yes["street_view"]
    return yes, windows, door_kinds


def _door_choice(door_kinds: list, cand_room: np.ndarray):
    """The "what is right outside your door?" choice for this building: the door destinations at least two rooms
    have (others can't help), and P(answer | candidate). A room with several doors is equally likely to be described
    by any of them (we don't know which door the caller means); DOOR_NOISE of answers are spread evenly."""
    counts = {key: sum(1 for kinds in door_kinds if kinds and key in kinds) for key, *_ in DOOR_OPTIONS}
    offered = [(key, label, words) for key, label, words, _ in DOOR_OPTIONS if counts[key] >= 2]
    if len(offered) < 2:
        return None, {}
    keys = [k for k, _, _ in offered]
    probs = np.full((len(keys), len(door_kinds)), 1 / len(keys))
    adjacent = {k: np.full(len(door_kinds), 0.5) for k in keys}
    for i, kinds in enumerate(door_kinds):
        if not kinds:  # no layout, or no annotated doors (open-plan rooms): no information
            continue
        named = [k for k in kinds if k in keys]
        for j, key in enumerate(keys):
            share = named.count(key) / len(named) if named else 0.0
            probs[j, i] = (1 - DOOR_NOISE) * share + DOOR_NOISE / len(keys)
            adjacent[key][i] = LAYOUT_YES if key in kinds else LAYOUT_NO
    labels = [label for _, label, _ in offered]
    listed = f"{', '.join(labels[:-1])}, or {labels[-1]}"
    choice = Choice("door_to", DOOR_QUESTION.format(listed=listed),
                    tuple(Option(k, label, words) for k, label, words in offered))
    return (choice, probs[:, cand_room]), {k: v[cand_room] for k, v in adjacent.items()}


def _position_likelihoods(rooms: list[dict], toward_street: np.ndarray) -> np.ndarray:
    """P(caller says front / middle / back | room), shape (3, n_rooms). Each room's centre is placed between the back
    (0) and the street side (1) of its own floor (floors can have different footprints). Rooms without a polygon get
    a uniform row (no information)."""
    probs = np.full((3, len(rooms)), 1 / 3)
    polygons = [np.asarray(r.get("polygon") or [], dtype=float).reshape(-1, 2) for r in rooms]
    for floor in {r["floor"] for r in rooms}:
        members = [i for i, r in enumerate(rooms) if r["floor"] == floor and len(polygons[i]) >= 3]
        if len(members) < 2:
            continue
        depth = np.concatenate([polygons[i] for i in members]) @ toward_street
        lo, hi = depth.min(), depth.max()
        if hi - lo < 1.0:  # a floor less than a metre deep has no front or back
            continue
        for i in members:
            t = (polygons[i].mean(axis=0) @ toward_street - lo) / (hi - lo)
            fit = np.exp(-0.5 * ((t - np.array(POSITION_CENTRES)) / POSITION_WIDTH) ** 2)
            probs[:, i] = (1 - POSITION_NOISE) * fit / fit.sum() + POSITION_NOISE / 3
    return probs


def _layout_model(building, prompt_scores: dict[str, np.ndarray], layout: dict | None = None,
                orientation: dict | None = None) -> FeatureModel:
    """`prompt_scores` maps each prompt from clip_prompts() to per-candidate raw CLIP scores. `layout` (layout.py)
    and `orientation` (orientation.py) add the door/window questions when available."""
    rooms, cand_room = building.rooms, building.candidate_room
    kinds = [_room_kinds(r) for r in rooms]
    area = np.array([_polygon_area(r.get("polygon")) for r in rooms])
    aspect = np.array([_aspect_ratio(r) for r in rooms])

    def presence(prompt: str | None, types: tuple[str, ...]) -> np.ndarray:
        clip = _clip_presence(prompt_scores[prompt], building.has_views) if prompt else None
        by_type = _type_presence(kinds, types)
        return _combine(clip, by_type[cand_room] if by_type is not None else None, len(cand_room))

    layout_yes, window_counts, door_kinds = _layout_features(rooms, layout, _toward_street(orientation)) \
        if layout else ({}, None, [])

    p_yes = {}
    for feature in FEATURES:
        if feature.key in LAYOUT_FEATURES:
            if feature.key in layout_yes:
                p_yes[feature.key] = layout_yes[feature.key][cand_room]
        elif feature.key == "window" and window_counts is not None:
            # The layout's window annotations are more reliable than "a photo of a room with a large window"
            # (CLIP rates any bright wall as a window), so they replace the label/photo guess where known.
            known = ~np.isclose(window_counts[0], 1 / 3)       # per room: the layout covers it
            from_layout = 1 - window_counts[0]                  # per room: P(at least one window)
            guess = presence(feature.prompt, feature.room_types)
            p_yes[feature.key] = np.where(known[cand_room], from_layout[cand_room], guess)
        elif feature.key == "narrow":
            # Long and narrow: aspect ratio over ~2.2, or a hallway label.
            shape = np.where(np.isnan(aspect), 0.5, _sigmoid((aspect - 2.2) / 0.5))
            by_type = _type_presence(kinds, feature.room_types)
            p_yes[feature.key] = np.clip(((shape + by_type) / 2)[cand_room], 0.05, 0.95)
        else:
            p_yes[feature.key] = presence(feature.prompt, feature.room_types)

    p_option = {}
    for choice in CHOICES:
        if choice.key == "size":
            probs = _size_likelihoods(area)[:, cand_room]
        else:
            probs = np.stack([presence(o.prompt, o.room_types) for o in choice.options])
            probs = probs / probs.sum(axis=0, keepdims=True)
        p_option[choice.key] = probs

    floors = floor_options(len(building.floors))
    if floors:
        p_option["floor"] = _floor_likelihoods(building.candidate_floor, len(floors))
        p_option["flights"] = _floor_likelihoods(building.candidate_floor, len(floors), config.STAIRS_ANSWER_ACCURACY)
    choices, adjacent = {}, {}
    if window_counts is not None:
        choices["windows"] = WINDOWS_CHOICE
        p_option["windows"] = window_counts[:, cand_room]
    toward_street = _toward_street(orientation)
    if toward_street is not None:
        choices["position"] = POSITION_CHOICE
        p_option["position"] = _position_likelihoods(rooms, toward_street)[:, cand_room]
    if door_kinds:
        door, adjacent = _door_choice(door_kinds, cand_room)
        if door is not None:
            choices["door_to"], p_option["door_to"] = door
    near = {k: np.where(building.has_views, prompt_scores[near_prompt(k)], np.nan) for k in NEAR_NAMES}
    return FeatureModel(p_yes, p_option, area, cand_room, floors, building.candidate_floor, near=near, choices=choices, adjacent=adjacent)


# ---- planned questions ----------------------------------------------------------------------------------------


@dataclass
class Plan:
    kind: str                  # "yesno", "choice", "list", or "nearest" (where in the located room)
    key: str                   # feature key (yesno), choice key (choice), or "list"
    text: str
    items: tuple[str, ...] = ()  # feature keys offered in a "list" question
    options: tuple = ()          # Options offered in a "choice" question
    room: int = -1               # the located room ("nearest"), or the room being checked ("verify")
    purpose: str = ""            # "verify" (checks `room`), "dispatcher" (asked on request), or "" (automatic)
    before: float = 0.0          # for "verify": the room's probability when the check was asked


def answer_matrix(plan: Plan, model: FeatureModel) -> np.ndarray:
    """P(answer | candidate) for each possible answer, shape (n_answers, n_candidates)."""
    if plan.kind in ("yesno", "position"):
        p = model.p_yes[plan.key]
        return np.stack([p, 1 - p])
    if plan.kind == "choice":
        return model.p_option[plan.key]
    # "Which of these do you see?" approximated as one categorical answer: one of the items, or none of them.
    ps = np.stack([model.p_yes[k] for k in plan.items])
    none = np.prod(1 - ps, axis=0)
    probs = np.vstack([ps, none[None]])
    return probs / probs.sum(axis=0, keepdims=True)


def _candidate_plans(model: FeatureModel, covered: set[str], allow_list: bool,
                     posterior: np.ndarray, cand_room: np.ndarray, n_rooms: int) -> list[Plan]:
    plans = [Plan("yesno", f.key, f.question) for f in FEATURES if f.key in model.p_yes and f.key not in covered]
    plans += [Plan("choice", c.key, c.question, options=c.options)
              for c in [*CHOICES, *model.choices.values()] if c.key not in covered]
    if model.floor_options and "floor" not in covered:
        plans.append(Plan("choice", "floor", floor_question(model.floor_options), options=model.floor_options))
    elif model.floor_options and "flights" not in covered:
        # Only once the floor question is settled without a usable answer ("I don't know"): the stairs question
        # gets at the floor another way.
        n = len(model.floor_options)
        plans.append(Plan("choice", "flights", flights_question(n), options=flights_options(n)))
    if allow_list:
        # The list offers the features that individually split the plausible rooms best.
        singles = sorted((p for p in plans if p.kind == "yesno"),
                         key=lambda p: -locator.expected_information_gain(
                             posterior, answer_matrix(p, model), cand_room, n_rooms))
        # Layout features don't fit "what can you see around you" (a door into a bathroom isn't something you see).
        items = tuple(p.key for p in singles if p.key not in LAYOUT_FEATURES)[:LIST_SIZE]
        if len(items) == LIST_SIZE:
            nouns = [FEATURES_BY_KEY[k].noun for k in items]
            text = f"What can you see around you: {nouns[0]}, {nouns[1]}, or {nouns[2]}? Or something else?"
            plans.append(Plan("list", "list", text, items))
    return plans


def choose(model: FeatureModel, posterior: np.ndarray, cand_room: np.ndarray, n_rooms: int,
           covered: set[str], last_kind: str | None, min_gain: float = config.MIN_INFORMATION_GAIN) -> Plan | None:
    """Highest expected information gain among questions not yet covered. A "which of these" list is not offered
    twice in a row: it is the natural first follow-up to a vague description, but repeated lists read like a form.
    `min_gain` is lowered when the dispatcher explicitly asks for another question."""
    plans = _candidate_plans(model, covered, last_kind != "list", posterior, cand_room, n_rooms)
    if not plans and not model.positions:
        return None
    # Floor first: in a multi-storey building the floor is what responders act on first (which stairs, which
    # windows to ladder) and callers know it reliably, so it is asked before room details, even when another
    # question would split the rooms slightly better. It is asked even when the descriptions already favour one
    # floor: a live call reached 80% on Floor 1 from photo matches alone and named a ground-floor bedroom to a caller
    # on the 3rd floor. After "I don't know", the stairs question is asked instead while no floor is likely enough.
    floor_plan = next((p for p in plans if p.key in ("floor", "flights")), None)
    if floor_plan is not None:
        floor_probs = np.bincount(model.candidate_floor, weights=posterior, minlength=len(model.floor_options))
        if floor_plan.key == "floor" or floor_probs.max() < config.FLOOR_FIRST_UNTIL:
            return floor_plan
    gains = [locator.expected_information_gain(posterior, answer_matrix(p, model), cand_room, n_rooms)
             for p in plans]
    plans, gains = _with_position_plans(model, posterior, cand_room, n_rooms, covered, plans, gains)
    if not plans:
        return None
    best = int(np.argmax(gains))
    # Nothing left that would tell the remaining rooms apart (e.g. two photo-less bathrooms): stop asking rather than
    # keep the caller typing answers to irrelevant questions.
    return plans[best] if gains[best] >= min_gain else None


def verification(model: FeatureModel, posterior: np.ndarray, room: int, covered: set[str],
                 min_gain: float = config.VERIFY_MIN_GAIN) -> Plan | None:
    """The question that best checks the leading room before it is reported: the one whose answer is expected to
    tell "the caller is in `room`" from "somewhere else" best, scored as if the two were equally likely (judged on
    the real posterior, a room at 80% leaves almost no information to gain, so every check looks worthless).

    This is what corroboration means here: a room reached from one description or one claim ("I'm in the kitchen")
    gets one independent fact checked ("Can you see a stove?" / "What is right outside your door?") before the caller
    is told responders will look there. A wrong answer then moves the estimate instead of confirming a guess.
    Floor questions are excluded (the floor is settled before a room can lead), and so are lists, whose answers are
    too loose to confirm anything."""
    in_room = model.candidate_room == room
    top, rest = posterior[in_room].sum(), posterior[~in_room].sum()
    if top <= 0 or rest <= 0:
        return None
    balanced = np.where(in_room, posterior / top, posterior / rest) / 2
    split = in_room.astype(int)
    plans = [p for p in _candidate_plans(model, covered, False, posterior, model.candidate_room, 0)
             if p.key not in ("floor", "flights")]
    if not plans:
        return None
    gains = [locator.expected_information_gain(balanced, answer_matrix(p, model), split, 2) for p in plans]
    best = int(np.argmax(gains))
    return plans[best] if gains[best] >= min_gain else None


# ---- where in the room ----------------------------------------------------------------------------------------


def _room_scans(model: FeatureModel, room: int) -> np.ndarray:
    """Indices of the photographed candidates (scan points) in `room`."""
    return np.flatnonzero((model.candidate_room == room) & ~np.isnan(model.near["stove"]))


def _near_each(model: FeatureModel, items: tuple[str, ...], scans: np.ndarray) -> np.ndarray:
    """P(item i is right next to the caller | standing at scan s), shape (n_items, n_scans), each item on its own:
    a sigmoid of its close-up score z-scored across the room's scans. Scans about one standard deviation above the
    room's average count as "next to it" (NEAR_SCALE sets how sharply)."""
    raw = np.stack([model.near[k][scans] for k in items])
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-6)
    return np.clip(1 / (1 + np.exp(-config.NEAR_SCALE * (z - 1.0))), 0.05, 0.95)


def _answer_probs(p: np.ndarray, named: tuple[bool, ...]) -> np.ndarray:
    """P(caller names exactly the `named` subset of the offered items | scan), from independent per-item p."""
    out = np.ones(p.shape[1])
    for pi, said in zip(p, named):
        out = out * (pi if said else 1 - pi)
    return out


def _nearest_probs(model: FeatureModel, items: tuple[str, ...], scans: np.ndarray) -> np.ndarray:
    """P(answer | scan) for every possible answer (each subset of the offered items, including "none of those"),
    shape (2**n_items, n_scans). Used to score how informative a set of landmarks is."""
    p = _near_each(model, items, scans)
    return np.stack([_answer_probs(p, named) for named in itertools.product((False, True), repeat=len(items))])


def position_spread(posterior: np.ndarray, xy: np.ndarray, members: np.ndarray) -> tuple[np.ndarray, float]:
    """Probability-weighted position of `members` (candidate indices) and its RMS spread in meters."""
    w = posterior[members] / posterior[members].sum()
    center = w @ xy[members]
    return center, float(np.sqrt(w @ ((xy[members] - center) ** 2).sum(axis=1)))


def refinement(model: FeatureModel, posterior: np.ndarray, room: int, xy: np.ndarray,
               asked: set[str]) -> Plan | None:
    """A "what's closest to you?" question inside the located room, or None when it can't help: fewer than two
    scan points, the position already tight (REFINE_MIN_SPREAD_M), or no landmarks that separate the scan points."""
    scans = _room_scans(model, room)
    if len(scans) < 2:
        return None
    _, spread = position_spread(posterior, xy, scans)
    if spread < config.REFINE_MIN_SPREAD_M:
        return None
    w = posterior[scans] / posterior[scans].sum()
    # Landmarks this room probably has (presence model), not yet offered.
    present = [k for k in NEAR_NAMES if k not in asked and model.p_yes[k][scans].mean() >= 0.5]
    if len(present) < 2:
        return None
    # The three landmarks whose answer is expected to tell the likely scan points apart best. Every combination is
    # scored (at most C(16, 3) = 560): ranking landmarks one by one by how much they vary picked TV/sofa/table
    # (0.13 bits) where sofa/fireplace/glass door was worth 0.65 bits in the sample house's main room.
    def gain_of(items):
        return locator.expected_information_gain(w, _nearest_probs(model, items, scans),
                                                 np.arange(len(scans)), len(scans))
    size = min(3, len(present))
    gain, items = max((gain_of(combo), combo) for combo in itertools.combinations(present, size))
    if gain < config.REFINE_MIN_GAIN:
        return None
    names = [NEAR_NAMES[k] for k in items]
    listed = f"{names[0]} or {names[1]}" if len(names) == 2 else f"{', '.join(names[:-1])}, or {names[-1]}"
    text = f"Which is closest to you: {listed}? Or reply \"none\"."
    return Plan("nearest", "nearest", text, items=items, room=room)


def nearest_likelihood(model: FeatureModel, plan: Plan, said: list[str]) -> np.ndarray:
    """Likelihood over candidates from which offered landmarks the caller named as next to them ("none of those"
    counts too: not next to any of them). Only the located room's scan points are reweighted; everything else is
    left as is (1), so this never moves the room estimate."""
    scans = _room_scans(model, plan.room)
    likelihood = np.ones(len(model.candidate_room))
    likelihood[scans] = _answer_probs(_near_each(model, plan.items, scans), tuple(k in said for k in plan.items))
    return likelihood


# ---- reading the caller ---------------------------------------------------------------------------------------


@dataclass
class Cues:
    """What one caller message says, independent of who read it: the keyword rules below, or a language model
    (llm.py) when one is configured. apply_cues() turns either into evidence the same way."""
    answer: str | None = None                  # reply to a yes/no or list question: "yes", "no", "none", "unsure"
    options: tuple[str, ...] = ()              # reply to a choice question: option keys ("tile",)
    mentioned: dict[str, str] = field(default_factory=dict)   # feature key -> "yes" / "no" ("no windows")
    size: str | None = None                    # "tight" (cramped), "small" (loose), "medium", "large"
    dark: bool = False
    room_types: tuple[str, ...] = ()           # room classification labels the caller named ("bathroom",)
    floors: tuple[int, ...] = ()               # floor indices the caller said they are on (0 = lowest)
    nearby: tuple[str, ...] = ()               # door-option keys of rooms the caller is next to ("near the kitchen")
    acknowledgement: str = ""                  # e.g. "You said it's a tight space."
    flooring: tuple[str, ...] = ()
    explained: bool = False                    # nothing beyond the answer itself, so no CLIP free-text match


@dataclass
class Reading:
    likelihood: np.ndarray | None        # product of all cue likelihoods (n_candidates,), None if no cues
    covered: set[str]                    # feature/choice keys this message settled (don't ask about them again)
    acknowledgement: str
    explained: bool
    position: np.ndarray | None = None   # in-room likelihood from a "closest" answer (apply with apply_within_rooms)


def _mentioned(feature: Feature, text: str, blockers: list[tuple[int, int]] = ()) -> str | None:
    """'yes' / 'no' if the caller mentioned the feature (negated: 'no windows', "there isn't a toilet"), else None.
    A hedged mention ("maybe a window") is None. If the feature is both present and denied ("no toilet but there's a
    tub", both the toilet feature), present wins. `blockers` are spans of other items (see questions.polarity)."""
    said = set()
    for match in re.finditer(feature.words, text, re.I):
        if match.group(0).lower().endswith("less"):  # "windowless"
            said.add("no")
        else:
            said.add(questions.polarity(text, match.start(), blockers))
    said.discard(None)
    if not said:
        return None
    return "yes" if "yes" in said else "no"


def rule_cues(text: str, pending: Plan | None, reply_kind: str, semantic: dict[str, float] | None = None,
              floors: tuple = ()) -> Cues:
    """Local reading of a message: keyword rules, plus paraphrase matches from semantics.scores() when given.
    `reply_kind` is questions.parse_reply(text).kind; `floors` is the building's FeatureModel.floor_options."""
    cues = _keyword_cues(text, pending, reply_kind)
    if pending is not None and pending.key == "door_to":
        # The answer names the room *outside*: "a hallway" is not a narrow-space cue, and "a bedroom" not a claim.
        return cues
    cues.nearby = tuple(sorted({_door_option(label) for label in questions.room_mentions(text).nearby} - {None}))
    if floors and not (pending is not None and pending.key in ("floor", "flights")):  # those answers are answers
        named = stated_floors(text, floors)
        if named and len(named) < len(floors):
            cues.floors = named
            where = floors[named[0]].label if len(named) == 1 else None
            cues.acknowledgement = f"You said you're on the {where}." if where else "You said you're upstairs."
    if semantic:
        _add_semantic(cues, semantic, text, pending)
    return cues


def _door_option(room_label: str) -> str | None:
    """Door-option key ("living", "hallway", ...) for a room label the caller used ("kitchen", "hall")."""
    kind = layout.room_kind(room_label)
    return next((key for key, _, _, kinds in DOOR_OPTIONS if kind in kinds), None)


def _add_semantic(cues: Cues, semantic: dict[str, float], text: str, pending: Plan | None) -> None:
    """Paraphrase matches fill in what the keywords missed; keywords always win where both speak."""
    # Embeddings can't tell "a toilet" from "no toilet", so a message with any negation only uses keywords.
    if not ANY_NEGATION.search(text):
        for key, _ in sorted(semantic.items(), key=lambda item: -item[1]):
            feature = FEATURES_BY_KEY.get(key)
            if feature is None or key in cues.mentioned:
                continue
            if pending is not None and pending.kind == "yesno" and pending.key == key:
                continue  # an answer to the pending question is handled as the answer
            cues.mentioned[key] = "yes"
            if not cues.acknowledgement:
                cues.acknowledgement = f"Okay, {feature.noun.split(' or ')[0].split(',')[0]}."
    # A shape the caller named ("narrow") isn't a size, but MiniLM puts "I'm in a narrow space" close to "large"
    # (0.70 on the eval): a hallway caller was read as "a big space" and sent to a bedroom. Shape words win.
    if cues.size is None and "narrow" not in cues.mentioned:
        tight, large = semantic.get("tight", 0.0), semantic.get("large", 0.0)
        if tight - large >= config.SEMANTIC_SIZE_MARGIN:
            cues.size, cues.acknowledgement = "tight", "You said it's a tight space."
        elif large - tight >= config.SEMANTIC_SIZE_MARGIN:
            cues.size, cues.acknowledgement = "large", "You said it's a big space."
    if not cues.dark and "dark" in semantic:
        cues.dark, cues.acknowledgement = True, "You said it's dark."


def _keyword_cues(text: str, pending: Plan | None, reply_kind: str) -> Cues:
    cues = Cues()
    short = len(text.split())
    spans = _feature_spans(text)
    if pending is not None:
        if reply_kind == "unsure":
            cues.answer, cues.acknowledgement = "unsure", "That's okay."
        elif pending.kind in ("yesno", "position") and reply_kind in ("yes", "no"):
            cues.answer, cues.explained = reply_kind, True
        elif pending.kind == "yesno":
            # "no toilet", "there's a tub", "i don't see any windows": the answer is in the description.
            said = _mentioned(FEATURES_BY_KEY[pending.key], text, spans) if pending.key in FEATURES_BY_KEY else None
            if said:
                cues.answer, cues.explained = said, short <= 4
        elif pending.kind == "choice":
            if pending.key == "floor":
                cues.options = tuple(pending.options[i].key for i in stated_floors(text, pending.options))
            else:
                cues.options = tuple(o.key for o in pending.options if re.search(o.words, text, re.I))
            if not cues.options:
                cues.options = tuple(o.key for o in pending.options if o.reply_words and re.search(o.reply_words, text, re.I))
            if pending.key == "door_to":
                cues.explained = bool(cues.options)
                return cues
            if not cues.options:  # "the second one"
                index = _position(text, len(pending.options))
                cues.options = (pending.options[index].key,) if index is not None else ()
            cues.explained = bool(cues.options) and (short <= 4 or (pending.key == "flooring"
                                                                    and only_floor_words(text)))
        elif pending.kind == "list" and (reply_kind == "no" or re.search(r"\b(none|nothing|neither)\b", text, re.I)):
            cues.answer, cues.explained = "none", short <= 4
        elif pending.kind == "nearest" and (reply_kind == "no" or re.search(r"\b(none|nothing|neither)\b", text, re.I)):
            cues.answer, cues.explained = "none", short <= 4
        elif pending.kind == "near":
            cues.options = _near_answer(text, pending)
            if cues.options:
                cues.explained = short <= 6
            elif reply_kind == "no" or NEAR_NONE.search(text):
                cues.answer, cues.explained = "none", short <= 4

    for feature in FEATURES:
        said = _mentioned(feature, text, spans)
        if said and not (pending and pending.kind == "yesno" and pending.key == feature.key):
            cues.mentioned[feature.key] = said
            if said == "yes" and not cues.acknowledgement:
                cues.acknowledgement = f"Okay, {feature.noun.split(' or ')[0].split(',')[0]}."
    if pending is not None and pending.kind == "list" and cues.answer is None:
        index = _position(text, len(pending.items))
        picked = [pending.items[index]] if index is not None else list(pending.items) if _ALL.fullmatch(text) else []
        for key in picked:  # "the second one", "all of them"
            cues.mentioned.setdefault(key, "yes")
            if not cues.acknowledgement:
                cues.acknowledgement = f"Okay, {FEATURES_BY_KEY[key].noun.split(' or ')[0].split(',')[0]}."
    if pending is not None and pending.kind == "list" and set(cues.mentioned) & set(pending.items):
        cues.explained = short <= 6

    small, large = _room_cue(SMALL_CUE, text), _room_cue(LARGE_CUE, text)
    if small and not large:
        cues.size = "tight" if _room_cue(STRICT_SMALL_CUE, text) else "small"
        cues.acknowledgement = f"You said it's {_size_phrase(small.group(0))}."
    elif large and not small:
        cues.size, cues.acknowledgement = "large", "You said it's a big space."
    if _room_cue(DARK_CUE, text):
        cues.dark, cues.acknowledgement = True, "You said it's dark."
    if not (pending is not None and pending.key == "flooring"):  # an answer to that question is an answer
        _add_flooring(cues, text)
    return cues


def apply_cues(cues: Cues, model: FeatureModel, pending: Plan | None) -> Reading:
    """Turn cues into one likelihood over candidates."""
    likelihood = None
    position = None
    covered: set[str] = set()

    def apply(values: np.ndarray) -> None:
        nonlocal likelihood
        likelihood = values if likelihood is None else likelihood * values

    # 1. The answer to the question we asked.
    if pending is not None:
        if cues.answer == "unsure":
            covered.add(pending.key)  # don't ask the same thing again
        elif pending.kind in ("yesno", "position") and cues.answer in ("yes", "no"):
            p = model.p_yes[pending.key]
            apply(p if cues.answer == "yes" else 1 - p)
            covered.add(pending.key)
        elif pending.kind == "choice" and cues.options:
            keys = [o.key for o in pending.options]
            hits = [keys.index(k) for k in cues.options if k in keys]
            if hits:
                apply(model.p_option[pending.key][hits].sum(axis=0))
                covered.add(pending.key)
                if pending.key == "windows":
                    covered.add("window")  # "how many windows" already answered "is there a window"
                if pending.key in ("floor", "flights"):
                    covered.update(("floor", "flights"))  # the floor is answered; the other one isn't needed
        elif pending.kind == "list" and cues.answer in ("none", "no"):
            for key in pending.items:
                apply(1 - model.p_yes[key])
                covered.add(key)

        elif pending.kind == "nearest":
            said = [k for k in pending.items if cues.mentioned.get(k) == "yes"]
            if said or cues.answer in ("none", "no"):
                position = nearest_likelihood(model, pending, said)
            covered.update(pending.items)  # named landmarks are about position here, not room presence

    # 1b. The answer to a "which are you closest to?" question: items by keyword (options) or, from Claude, under
    # 'mentioned'. "None" / "not sure" moves nothing. The items' presence in the room is already known, so they are
    # not also applied as mentioned features below.
    near_items: tuple[str, ...] = ()
    if pending is not None and pending.kind == "near":
        near_items = pending.items
        hits = [k for k in cues.options if k in near_items] or \
            [k for k, said in cues.mentioned.items() if said == "yes" and k in near_items]
        if hits:
            rows = _near_matrix(model, pending.room, near_items)
            position = rows[[near_items.index(k) for k in hits]].sum(axis=0)
            covered |= set(hits)
        if hits or cues.answer in ("none", "no", "unsure"):
            covered |= {f"near:{k}" for k in near_items}

    # 2. Features mentioned anywhere in the message ("there's a toilet", "no windows").
    for key, said in cues.mentioned.items():
        if key in covered or key not in model.p_yes or key in near_items:
            continue
        p = model.p_yes[key]
        apply(p if said == "yes" else 1 - p)
        if said == "yes":   # "by the stove": also which part of a big room, from the photos (room totals unchanged)
            factor = _within_room_factor(model, key)
            position = factor if position is None else position * factor
        covered.add(key)

    # 3. Rooms the caller is next to ("I'm near the kitchen"). Softened: "near" is often a shared wall, not a door.
    for key in cues.nearby:
        if key in model.adjacent:
            apply(0.25 + 0.5 * model.adjacent[key])

    # 3. Floors named outside the floor question ("I'm upstairs").
    if cues.floors and "floor" not in covered and "floor" in model.p_option:
        apply(model.p_option["floor"][list(cues.floors)].sum(axis=0))
        covered.add("floor")

    # 4. Floor materials named outside the flooring question ("carpeted floor").
    if cues.flooring and "flooring" not in covered and "flooring" in model.p_option:
        keys = [o.key for o in CHOICES_BY_KEY["flooring"].options]
        hits = [keys.index(k) for k in cues.flooring if k in keys]
        if hits:
            apply(model.p_option["flooring"][hits].sum(axis=0))
            covered.add("flooring")

    # 5. Size and darkness.
    if cues.size and "size" not in covered:
        sizes = _size_likelihoods(model.room_area)[:, model.candidate_room]
        if cues.size == "small":
            # A loose "small room" covers more than a closet: a 6.6 m² pantry or a 6 m² bathroom is "small" to a
            # caller, so the cut-off is ~9 m² and softer than for "cramped"/"tight".
            area = np.where(np.isnan(model.room_area), 9.0, model.room_area)
            apply(np.clip(_sigmoid((9.0 - area) / 2.5), 0.05, 0.95)[model.candidate_room])
        else:
            apply(sizes[{"tight": 0, "medium": 1, "large": 2}[cues.size]])
        covered.add("size")
    if cues.dark and "window" not in covered:
        apply(1 - model.p_yes["window"])
        covered.add("window")

    return Reading(likelihood, covered, cues.acknowledgement, cues.explained, position)


def read_message(text: str, model: FeatureModel, pending: Plan | None, reply_kind: str,
                 cues: Cues | None = None, semantic: dict[str, float] | None = None) -> Reading:
    """Turn one caller message into likelihoods over candidates: from Claude's cues when given, otherwise from the
    keyword rules plus local paraphrase matches."""
    if cues is None:
        cues = rule_cues(text, pending, reply_kind, semantic, model.floor_options)
    return apply_cues(cues, model, pending)


def _size_phrase(word: str) -> str:
    word = word.lower()
    if word.startswith(("cramped", "tight", "tiny", "confined", "enclosed", "squeez", "barely", "can")):
        return "a tight space"
    return "a small space"


def phrase(plan: Plan, acknowledgement: str) -> str:
    return f"{acknowledgement} {plan.text}".strip()


def model_from_building(building, embed_texts, layout: dict | None = None,
                        orientation: dict | None = None) -> FeatureModel:
    """Build the feature model for a loaded building; `embed_texts` is embeddings.embed_texts."""
    prompts = list(dict.fromkeys([*clip_prompts(), *(near_prompt(k) for k in NEAR_NAMES)]))
    vectors = embed_texts(prompts)
    return build_model(building, {p: building.candidate_scores(v) for p, v in zip(prompts, vectors)},
                       layout, orientation, contrast=view_contrast(building, dict(zip(prompts, vectors))))


def _ensemble(prompt: str) -> tuple[str, ...]:
    """The prompt and its extra phrasings."""
    return (prompt, *PROMPT_VARIANTS.get(prompt, ()))


def _ensemble_scores(prompt: str, prompt_scores: dict[str, np.ndarray]) -> np.ndarray:
    """Mean per-candidate score over the prompt's phrasings that were scored (score-space ensemble)."""
    return np.mean([prompt_scores[p] for p in _ensemble(prompt) if p in prompt_scores], axis=0)


def view_contrast(building, prompt_vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-candidate contrast for each original prompt: in every view, the mean cosine over the prompt's phrasings
    minus the mean cosine over BASELINE_PROMPTS, then the best view of each candidate. Subtracting the baseline per
    view (not per candidate) means a cluttered face that scores well against everything doesn't win by default.
    Candidates without views get 0 (ignored: _clip_presence only reads imaged candidates)."""
    views = building.view_embeddings
    baseline = np.mean([views @ prompt_vectors[b] for b in BASELINE_PROMPTS], axis=0)
    originals = [f.prompt for f in FEATURES if f.prompt] + [o.prompt for c in CHOICES for o in c.options if o.prompt]
    out = {}
    for prompt in originals:
        sims = np.mean([views @ prompt_vectors[p] for p in _ensemble(prompt)], axis=0)
        best = np.full(len(building.candidates), -np.inf)
        np.maximum.at(best, building.view_candidate, sims - baseline)
        out[prompt] = np.where(np.isfinite(best), best, 0.0)
    return out


def _flooring_presence(options: tuple[Option, ...], room_kinds: list[str], cand_room: np.ndarray,
                       clip: list[np.ndarray | None]) -> np.ndarray:
    """P(flooring answer | candidate), shape (n_options, n_candidates). Like other features (CLIP + type prior), except
    that an option the room's type excludes gets FLOORING_IMPLAUSIBLE whatever CLIP says: CLIP's floor prompts mostly
    respond to colour and texture (grey concrete scores as carpet), while "garage" or "bathroom" says what the floor
    is. Rooms whose label says nothing about flooring (e.g. "Room 3") are decided by their photos as before."""
    rows = []
    for option, option_clip in zip(options, clip):
        by_type = _type_presence(room_kinds, option.room_types)[cand_room]
        excluded = np.array([any(t in kinds for t in option.excluded) for kinds in room_kinds], dtype=bool)
        combined = _combine(option_clip, by_type, len(cand_room))
        rows.append(np.where(excluded[cand_room], config.FLOORING_IMPLAUSIBLE, combined))
    probs = np.stack(rows)
    return probs / probs.sum(axis=0, keepdims=True)


def _near_noun(key: str) -> str:
    """'a stove or kitchen counters' -> 'the stove'; 'a glass door to outside' -> 'the glass door to outside'."""
    noun = FEATURES_BY_KEY[key].noun.split(" or ")[0].split(",")[0]
    for article in ("a ", "an "):
        noun = noun.removeprefix(article)
    return noun if noun.startswith("the ") else f"the {noun}"


def _near_mask(model: FeatureModel, room: int) -> np.ndarray:
    """The room's imaged sweeps: the only ones whose photos say anything about position inside the room."""
    in_room = model.candidate_room == room
    return in_room & model.has_views if model.has_views is not None else in_room


def _near_matrix(model: FeatureModel, room: int | None, items: tuple[str, ...]) -> np.ndarray:
    """P(caller says they are closest to item k | candidate), plus a last "none / not sure" row. Inside the room each
    item's P(yes) is rescaled to 0..1 over the room's imaged sweeps (see NEAR_FULL_SPREAD) (the room label is constant there, so only the
    photos vary); other candidates get the same answer distribution whatever they are, so the answer doesn't move
    probability between rooms."""
    n = len(model.candidate_room)
    rows = np.ones((len(items), n))
    mask = _near_mask(model, room) if room is not None else np.zeros(n, dtype=bool)
    if mask.any():
        for i, key in enumerate(items):
            values = (model.p_local or model.p_yes)[key][mask]
            # Rescaled by the spread, but a landmark that barely varies stays weak evidence.
            rows[i, mask] = NEAR_FLOOR + (values - values.min()) / max(values.max() - values.min(), NEAR_FULL_SPREAD)
    rows = rows / rows.sum(axis=0, keepdims=True) * (1 - NEAR_UNSURE)
    return np.vstack([rows, np.full((1, n), NEAR_UNSURE)])


def sweep_information_gain(posterior: np.ndarray, answer_probs: np.ndarray, mask: np.ndarray) -> float:
    """Expected reduction (bits) in uncertainty about the caller's *sweep*, among the candidates in `mask` only (the
    posterior restricted to the room and renormalised). locator.expected_information_gain is over rooms, where a
    question that only splits one room's sweeps is worth nothing; here the room is settled and the sweep is the goal."""
    prior = np.where(mask, posterior, 0.0)
    if prior.sum() <= 0:
        return 0.0
    prior = prior / prior.sum()
    before = locator._entropy(prior)
    after = 0.0
    for row in answer_probs:
        joint = prior * row
        p_answer = joint.sum()
        if p_answer > 0:
            after += p_answer * locator._entropy(joint / p_answer)
    return before - after


def choose_within_room(model: FeatureModel, posterior: np.ndarray, candidate_room: np.ndarray, room: int,
                       covered: set[str]) -> Plan | None:
    """A "which are you closest to: the stove, the fireplace, or ...?" question for the given room, or None when the
    room has fewer than 2 imaged sweeps or no question would place the caller well enough (MIN_WITHIN_ROOM_GAIN).
    Landmarks already asked about this way are in `covered` as "near:<key>"."""
    from itertools import combinations

    mask = _near_mask(model, room) & (candidate_room == room)
    if mask.sum() < 2:
        return None
    spreads = {}
    for key in NEAR_FEATURES:
        if f"near:{key}" in covered or key not in model.p_yes:
            continue
        values = (model.p_local or model.p_yes)[key][mask]
        if values.max() - values.min() >= NEAR_MIN_SPREAD:
            spreads[key] = values.max() - values.min()
    pool = sorted(spreads, key=lambda k: -spreads[k])[:NEAR_POOL]
    best, best_gain = None, 0.0
    for size in (2, 3):
        for items in combinations(pool, size):
            gain = sweep_information_gain(posterior, _near_matrix(model, room, items), mask)
            if gain > best_gain + 1e-9:  # a third item must add something to be worth the longer question
                best, best_gain = items, gain
    if best is None or best_gain < config.MIN_WITHIN_ROOM_GAIN:
        return None
    nouns = [_near_noun(k) for k in best]
    listed = f"{nouns[0]} or {nouns[1]}" if len(nouns) == 2 else f"{nouns[0]}, {nouns[1]}, or {nouns[2]}"
    return Plan("near", "near", f"Which are you closest to: {listed}?", best, room=room)


def _near_answer(text: str, pending: Plan) -> tuple[str, ...]:
    """Items a reply to a "near" question picks: by name ("by the fireplace"), else by position ("the second one")."""
    named = tuple(k for k in pending.items if _mentioned(FEATURES_BY_KEY[k], text) == "yes")
    if named:
        return named
    match = NEAR_ORDINAL.search(text)
    if not match:
        return ()
    word = match.group(1).lower()
    index = len(pending.items) - 1 if word == "last" else {"first": 0, "1st": 0, "second": 1, "2nd": 1}.get(word, 2)
    return (pending.items[index],) if index < len(pending.items) else ()


def _feature_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for f in FEATURES for m in re.finditer(f.words, text, re.I)]


def _describes_thing(text: str, end: int) -> bool:
    """True when the word ending at `end` describes an object, not the room: "a big bed", "small window"."""
    after = text[end:end + 40]
    things = "|".join([_DESCRIBED] + [f"(?:{f.words})" for f in FEATURES])
    return bool(re.match(rf"\s+(\w+\s+)?(?:{things})", after, re.I))


def _room_cue(pattern: re.Pattern, text: str) -> re.Match | None:
    """First match of a size/dark cue that is about the room: not negated ("not big"), not describing a thing."""
    for match in pattern.finditer(text):
        if questions.polarity(text, match.start()) == "yes" and not _describes_thing(text, match.end()):
            return match
    return None


def _position(text: str, n: int) -> int | None:
    match = _ORDINAL.fullmatch(text)
    if not match:
        return None
    index = _ORDINAL_INDEX[match.group(1).lower()]
    index = n - 1 if index < 0 else index
    return index if index < n else None


def stated_floors(text: str, options: tuple) -> tuple[int, ...]:
    """Find affirmative floor statements, excluding destinations and landmarks."""
    return tuple(i for i, option in enumerate(options) if any(
        questions.polarity(text, match.start(), relative=True) == "yes"
        and not re.search(r"\b(?:to|from|towards?)\s+(?:the\s+)?$", text[:match.start()], re.I)
        for match in re.finditer(option.words, text, re.I)
    ))


def flooring_mentioned(text: str) -> tuple[str, ...]:
    """Flooring option keys named (not negated) anywhere in the text: "carpeted floor" -> ("carpet",)."""
    found = []
    for key, pattern in FLOORING_CUES.items():
        match = pattern.search(text)
        if not match:
            continue
        clause = re.split(r"[.,;!?]|\bbut\b", text[:match.start()])[-1]   # "not carpet, it's tile": tile counts
        if not NEGATION.search(clause + " "):
            found.append(key)
    return tuple(found)


def only_floor_words(text: str) -> bool:
    """True when the message is just a floor description ("it's carpeted", "grey concrete floor")."""
    words = re.findall(r"[a-z][a-z'-]*", text.lower())
    return bool(words) and all(w in _FLOOR_FILLER for w in words)


def _add_flooring(cues: Cues, text: str) -> None:
    named = flooring_mentioned(text)
    if not named:
        return
    cues.flooring = named
    if only_floor_words(text):
        cues.explained = True  # nothing else for CLIP to match; its floor prompts are unreliable (see _NEVER_CARPETED)
    if not cues.acknowledgement:
        labels = {o.key: o.label for o in CHOICES_BY_KEY["flooring"].options}
        cues.acknowledgement = f"You said the floor is {' and '.join(labels[k] for k in named)}."


def _within_room_factor(model: FeatureModel, key: str) -> np.ndarray:
    """Per-candidate factor that moves probability toward the sweeps whose photos show `key`, inside each room only:
    each imaged sweep's photo-only P(yes) over its room's mean, so a room's total is unchanged when its sweeps are
    equally likely (the prior), and the room-level evidence stays with p_yes. Without it "I'm by the stove" names
    the kitchen but leaves the dot anywhere in a 50 m² open-plan room. 1 for photo-less candidates and one-sweep rooms."""
    local = (model.p_local or model.p_yes)[key]
    views = model.has_views if model.has_views is not None else np.ones(len(local), dtype=bool)
    factor = np.ones(len(local))
    for room in np.unique(model.candidate_room):
        mask = (model.candidate_room == room) & views
        if mask.sum() >= 2:
            factor[mask] = local[mask] / local[mask].mean()
    return factor


def _photo_model(building, prompt_scores: dict[str, np.ndarray],
                contrast: dict[str, np.ndarray] | None = None) -> FeatureModel:
    """`prompt_scores` maps each prompt from clip_prompts() to per-candidate raw CLIP scores. `contrast` (from
    view_contrast) maps each original prompt to its per-view baseline contrast; without it, the contrast is
    approximated per candidate from the baseline prompts' scores, and if those weren't scored either, CLIP presence
    is the old relative z-score."""
    rooms, cand_room = building.rooms, building.candidate_room
    kinds = [_room_kinds(r) for r in rooms]
    area = np.array([_polygon_area(r.get("polygon")) for r in rooms])
    aspect = np.array([_aspect_ratio(r) for r in rooms])
    peers = building.imageless_peers()
    if not config.CLIP_CALIBRATED:
        contrast = None
    elif contrast is None and all(b in prompt_scores for b in BASELINE_PROMPTS):
        baseline = np.mean([prompt_scores[b] for b in BASELINE_PROMPTS], axis=0)
        contrast = {p: _ensemble_scores(p, prompt_scores) - baseline for p in prompt_scores}

    clip_only: dict[str | None, np.ndarray | None] = {}   # prompt -> CLIP-only presence, for within-room questions

    def presence(prompt: str | None, types: tuple[str, ...]) -> np.ndarray:
        clip = None
        if prompt:
            clip = _clip_presence(_ensemble_scores(prompt, prompt_scores), building.has_views, peers,
                                  contrast=contrast.get(prompt) if contrast is not None else None)
            clip_only[prompt] = clip
        by_type = _type_presence(kinds, types)
        return _combine(clip, by_type[cand_room] if by_type is not None else None, len(cand_room))

    p_yes = {}
    for feature in FEATURES:
        if feature.key == "narrow":
            # Long and narrow: aspect ratio over ~2.2, or a hallway label.
            shape = np.where(np.isnan(aspect), 0.5, _sigmoid((aspect - 2.2) / 0.5))
            by_type = _type_presence(kinds, feature.room_types)
            p_yes[feature.key] = np.clip(((shape + by_type) / 2)[cand_room], 0.05, 0.95)
        else:
            p_yes[feature.key] = presence(feature.prompt, feature.room_types)

    p_option = {}
    for choice in CHOICES:
        if choice.key == "size":
            probs = _size_likelihoods(area)[:, cand_room]
        elif choice.key == "flooring":
            clip = [_clip_presence(prompt_scores[o.prompt], building.has_views) if o.prompt else None
                    for o in choice.options]
            probs = _flooring_presence(choice.options, kinds, cand_room, clip)
        else:
            probs = np.stack([presence(o.prompt, o.room_types) for o in choice.options])
            probs = probs / probs.sum(axis=0, keepdims=True)
        p_option[choice.key] = probs

    # ---- commonsense guard (commonsense.py): keep each candidate's P(yes) inside what its room type allows, so a
    # CLIP hit can't put a toilet in the garage or a bed in the kitchen. Post-processes the combined sources above;
    # rooms with no recognised type are untouched.
    # Within one room the label and the commonsense bounds are the same for every sweep, so they only dilute the photo
    # differences that tell one end of a big room from the other: "near" questions use CLIP alone where it exists.
    p_local = {}
    for feature in FEATURES:
        clip = clip_only.get(feature.prompt)
        p_local[feature.key] = (p_yes[feature.key].copy() if clip is None
                                else np.where(np.isnan(clip), p_yes[feature.key], np.clip(clip, 0.05, 0.95)))
    commonsense.guard(p_yes, p_option, {c.key: [o.key for o in c.options] for c in CHOICES}, kinds, cand_room)

    floors = floor_options(len(building.floors))
    if floors:
        p_option["floor"] = _floor_likelihoods(building.candidate_floor, len(floors))
    return FeatureModel(p_yes, p_option, area, cand_room, floors, building.candidate_floor,
                        has_views=np.asarray(building.has_views, dtype=bool), p_local=p_local)



BASELINE_PROMPTS = ("a photo of an empty room", "a photo of a white wall", "a photo of a hallway",
                    "a photo of a room with furniture")

_DESCRIBED = (r"kids?|child(ren)?|baby|dog|cat|bit|lamp|lights?|desk|dresser|box(es)?|bags?|pile|rugs?|fire|smoke|"
              r"man|guy|hole|crack|plant|picture|painting|doors?|wood|wooden|tiles?|floors?|blue|green|brown|gr[ae]y|"
              r"red|black|colou?red|carpet|curtains?|walls?|paint|cabinets?|furniture|hair|bruise")

ANY_NEGATION = re.compile(r"\b(no|not|nor|none|nothing|without|never|neither|cannot)\b|n't\b|"
                          r"\b(is|are|was|were|do|does|did|ca|can|could|has|have|had|wo|would|should|ai)nt\b", re.I)

_ORDINAL = re.compile(
    r"\W*(?:(?:um+|uh+|it'?s|its|the|number|option|#)\s*)*(first|1st|second|2nd|third|3rd|fourth|4th|last|[1-4])"
    r"(?:\s+(?:one|option|thing))?\W*", re.I)

_ORDINAL_INDEX = {"first": 0, "1st": 0, "1": 0, "second": 1, "2nd": 1, "2": 1, "third": 2, "3rd": 2, "3": 2,
                  "fourth": 3, "4th": 3, "4": 3, "last": -1}

_ALL = re.compile(r"\W*(all|all of (them|those|it)|all three|every ?thing|everything you said)\W*", re.I)


def build_model(building, prompt_scores, layout=None, orientation=None, contrast=None):
    if layout and not {"rooms", "doors", "windows"}.intersection(layout) and all(isinstance(value, np.ndarray) for value in layout.values()):
        contrast, layout = layout, None
    keys = [near_prompt(k) for k in NEAR_NAMES]
    scores = {**{key: np.zeros(len(building.candidate_room)) for key in keys}, **prompt_scores}
    model = _photo_model(building, scores, contrast)
    additions = _layout_model(building, scores, layout, orientation)
    model.near = additions.near
    model.choices = additions.choices
    model.adjacent = additions.adjacent
    model.p_yes.update({key: value for key, value in additions.p_yes.items()
                        if key in LAYOUT_FEATURES or (key == "window" and "windows" in additions.choices)})
    model.p_option.update({key: value for key, value in additions.p_option.items()
                           if key == "flights" or key in additions.choices})
    return model

FLOORING_CUES = {
    "carpet": re.compile(r"\bcarpet(s|ed|ing)?\b|\brugs?\b", re.I),
    "tile": re.compile(r"\btile[sd]?\b|\btiling\b|\blinoleum\b|\bvinyl\b|ceramic floor", re.I),
    "wood": re.compile(r"hardwood|\bwood(en)? floor|\blaminate\b|floor ?boards?", re.I),
    "concrete": re.compile(r"\bconcrete\b|\bcement\b", re.I),
}

_FLOOR_FILLER = set("""
    floor floors flooring ground the a an it its it's is are was there there's here i i'm im on of with and or kind
    sort like some all just very really looks look covered gray grey beige brown white black dark light soft hard cold
    old new my me under under-foot underfoot feet thick thin wall-to-wall wall to tan red blue green cream
    carpet carpets carpeted carpeting rug rugs tile tiles tiled tiling linoleum vinyl ceramic hardwood wood wooden
    laminate floorboards floorboard boards concrete cement
""".split())

NEAR_FEATURES = ("stove", "fridge", "sink", "fireplace", "sofa", "tv", "table", "bed", "window", "sliding_door",
                 "front_door", "stairs")

NEAR_MIN_SPREAD = 0.1

NEAR_POOL = 6

NEAR_FLOOR = 0.25

NEAR_FULL_SPREAD = 0.2

NEAR_UNSURE = 0.1

NEAR_ORDINAL = re.compile(r"\b(first|1st|second|2nd|third|3rd|last)\b", re.I)

NEAR_NONE = re.compile(r"\b(none|nothing|neither|not sure|don'?t know|no idea|middle)\b", re.I)

PROMPT_VARIANTS: dict[str, tuple[str, ...]] = {
    "a photo of a bathroom with a toilet and bathtub": ("a photo of a toilet", "a photo of a bathtub or shower"),
    "a photo of a room with a sink and faucet": ("a photo of a sink", "a photo of a faucet over a basin"),
    "a photo of a room with a large mirror": ("a photo of a mirror on a wall", "a photo of a bathroom mirror"),
    "a photo of a kitchen with a stove and cabinets": ("a photo of a stove and oven",
                                                       "a photo of kitchen counters and cabinets"),
    "a photo of a kitchen with a refrigerator": ("a photo of a refrigerator", "a photo of a fridge in a kitchen"),
    "a photo of a laundry room with a washing machine and dryer": ("a photo of a washing machine",
                                                                   "a photo of a clothes dryer"),
    "a photo of a small closet with shelves and a clothes rod": ("a photo of clothes hanging in a closet",
                                                                 "a photo of storage shelves"),
    "a photo of a bedroom with a bed": ("a photo of a bed", "a photo of a bed with pillows and a blanket"),
    "a photo of a living room with a sofa": ("a photo of a sofa", "a photo of a couch"),
    "a photo of a dining table with chairs": ("a photo of a table and chairs", "a photo of a dining room"),
    "a photo of a room with a television": ("a photo of a TV", "a photo of a flat screen television on a wall"),
    "a photo of a living room with a brick fireplace": ("a photo of a fireplace", "a photo of a fireplace mantel"),
    "a photo of a room with a large window": ("a photo of a window", "a photo of a room with daylight from windows"),
    "a photo of a room with a ceiling fan": ("a photo of a ceiling fan", "a photo of a fan on the ceiling"),
    "a photo of a staircase inside a house": ("a photo of stairs", "a photo of a stairway with a railing"),
    "a photo of a house entrance with a front door": ("a photo of a front door", "a photo of an entryway"),
    "a photo of a room with a sliding glass patio door": ("a photo of a sliding glass door",
                                                          "a photo of a glass door to a patio"),
    "a photo of a room with carpet floor": ("a photo of a carpeted floor", "a photo of a room with wall to wall carpet"),
    "a photo of a room with tile floor": ("a photo of a tiled floor", "a photo of a room with ceramic floor tiles"),
    "a photo of a room with hardwood floor": ("a photo of a hardwood floor", "a photo of a wooden plank floor"),
}


def _with_position_plans(model: FeatureModel, posterior: np.ndarray, cand_room: np.ndarray, n_rooms: int,
                         covered: set[str], plans: list[Plan], gains: list[float]) -> tuple[list[Plan], list[float]]:
    """Position hook (adjacency.py): "next to the kitchen?" / "front of the house?" join the candidates only once no
    question about the room's contents is worth much (config.POSITION_ONLY_BELOW_GAIN), i.e. the rooms look alike."""
    if not model.positions or max(gains, default=0.0) >= config.POSITION_ONLY_BELOW_GAIN:
        return plans, gains
    extra = [Plan("position", key, text) for key, text in model.positions.items() if key not in covered]
    extra_gains = [locator.expected_information_gain(posterior, answer_matrix(p, model), cand_room, n_rooms)
                   for p in extra]
    return plans + extra, gains + extra_gains
