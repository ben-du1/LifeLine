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

from . import config, layout, locator, questions

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
            r"stove|oven|cabinets?|counters?|countertop|microwave", "a photo of a kitchen with a stove and cabinets",
            ("kitchen",)),
    Feature("fridge", "a fridge", "Can you see a refrigerator?", r"fridge|refrigerator|freezer",
            "a photo of a kitchen with a refrigerator", ("kitchen",)),
    Feature("washer", "a washer or dryer", "Is there a washing machine or dryer?",
            r"washer|dryer|washing machine|laundry", "a photo of a laundry room with a washing machine and dryer",
            ("laundry", "utility")),
    Feature("shelves", "shelves or hanging clothes", "Are there shelves, or a rod with clothes hanging?",
            # Not bare "clothes": "the thing you wash clothes in" is a washer, not a closet.
            r"shel(f|ves|ving)|hangers?|hanging clothes|clothes hanging|\brod\b|coats?",
            "a photo of a small closet with shelves and a clothes rod", ("closet", "pantry", "laundry")),
    Feature("bed", "a bed", "Is there a bed?", r"\bbeds?\b|mattress", "a photo of a bedroom with a bed",
            ("bedroom",)),
    Feature("sofa", "a couch", "Is there a couch or sofa?", r"couch|sofa", "a photo of a living room with a sofa",
            ("living", "family")),
    Feature("table", "a table with chairs", "Is there a table with chairs?", r"\btables?\b|chairs?",
            "a photo of a dining table with chairs", ("dining", "kitchen")),
    Feature("tv", "a TV", "Can you see a TV?", r"\btv\b|television", "a photo of a room with a television",
            ("living", "family", "bedroom")),
    Feature("fireplace", "a fireplace", "Is there a fireplace?", r"fireplace|mantel|hearth",
            "a photo of a living room with a brick fireplace", ("living", "family")),
    Feature("window", "a window", "Is there a window in the room?", r"windows?",
            "a photo of a room with a large window"),
    Feature("fan", "a ceiling fan", "Is there a ceiling fan?", r"ceiling fan|\bfan\b",
            "a photo of a room with a ceiling fan", ("bedroom", "living")),
    Feature("stairs", "stairs", "Can you see stairs?", r"stairs?|staircase|steps",
            "a photo of a staircase inside a house", ("stair", "entry", "foyer")),
    Feature("front_door", "the front door", "Can you see the front door of the building?",
            r"front door|entrance|entry door|main door", "a photo of a house entrance with a front door",
            ("entry", "foyer", "living")),
    Feature("sliding_door", "a glass door to outside", "Is there a glass sliding door to the outside?",
            r"sliding|patio|glass door|back door", "a photo of a room with a sliding glass patio door",
            ("living", "dining", "kitchen", "family")),
    Feature("narrow", "a long narrow space", "Is it long and narrow, like a hallway?",
            r"narrow|corridor|hallway|long and thin", None, ("hall",)),
    # Layout features: answered from Matterport's doors and windows (layout.py), not from photos. They tell apart
    # rooms that look the same (empty bedrooms, photo-less bathrooms). Only offered when the building's layout is
    # known; "street_view" also needs the street's direction (orientation.py).
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
    reply_words: str = ""


@dataclass(frozen=True)
class Choice:
    key: str
    question: str
    options: tuple[Option, ...]


CHOICES = [
    Choice("flooring", "What is the floor like: carpet, tile, or wood?", (
        Option("carpet", "carpet", r"carpet|\brugs?\b", "a photo of a room with carpet floor",
               ("bedroom", "closet", "living", "family")),
        Option("tile", "tile", r"\btiles?\b|tiled|linoleum|vinyl|stone|ceramic", "a photo of a room with tile floor",
               ("bath", "kitchen", "laundry", "utility", "entry")),
        Option("wood", "wood", r"wood|hardwood|laminate|planks?", "a photo of a room with hardwood floor",
               ("living", "dining", "hall", "kitchen")),
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
SMALL_CUE = re.compile(r"cramped|tight|tiny|small|little|confined|enclosed|can'?t move|barely fit|squeez", re.I)
STRICT_SMALL_CUE = re.compile(r"cramped|tight|tiny|confined|can'?t move|barely fit|squeez", re.I)
LARGE_CUE = re.compile(r"\b(big|large|huge|spacious|open|wide|giant)\b", re.I)
DARK_CUE = re.compile(r"\b(dark|pitch black|no light|can'?t see anything)\b", re.I)
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
    prompts += [near_prompt(k) for k in NEAR_NAMES]
    return prompts


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


def _clip_presence(raw: np.ndarray, has_views: np.ndarray) -> np.ndarray | None:
    """Per-candidate P(feature present) from CLIP: z-score over imaged candidates, then a sigmoid. Slope 1.5 makes
    a candidate 1 std above average ~0.82. NaN for candidates without photos."""
    if not has_views.any():
        return None
    out = np.full(len(raw), np.nan)
    imaged = raw[has_views]
    out[has_views] = _sigmoid(1.5 * (imaged - imaged.mean()) / (imaged.std() + 1e-6))
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
    if plan.kind == "yesno":
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


def build_model(building, prompt_scores: dict[str, np.ndarray], layout: dict | None = None,
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
    return FeatureModel(p_yes, p_option, area, cand_room, floors, building.candidate_floor, near, choices, adjacent)


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
    if plan.kind == "yesno":
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
    plans = [Plan("yesno", f.key, f.question) for f in FEATURES if f.key not in covered and f.key in model.p_yes]
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
    if not plans:
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
    explained: bool = False                    # nothing beyond the answer itself, so no CLIP free-text match


@dataclass
class Reading:
    likelihood: np.ndarray | None        # product of all cue likelihoods (n_candidates,), None if no cues
    covered: set[str]                    # feature/choice keys this message settled (don't ask about them again)
    acknowledgement: str
    explained: bool
    position: np.ndarray | None = None   # in-room likelihood from a "closest" answer (apply with apply_within_rooms)


def _mentioned(feature: Feature, text: str) -> str | None:
    """'yes' / 'no' if the caller mentioned the feature (negated: 'no windows'), else None."""
    match = re.search(feature.words, text, re.I)
    if not match:
        return None
    return "no" if NEGATION.search(text[:match.start()].split(".")[-1] + " ") else "yes"


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
        named = tuple(i for i, option in enumerate(floors) if re.search(option.words, text, re.I))
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
    # "Narrow" is a shape, not a size, but the paraphrase model put "I'm in a narrow space" closer to "there is a lot
    # of space around me" than to "a small cramped space" and the Hallway caller was told "You said it's a big space".
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
    if pending is not None:
        if reply_kind == "unsure":
            cues.answer, cues.acknowledgement = "unsure", "That's okay."
        elif pending.kind == "yesno" and reply_kind in ("yes", "no"):
            cues.answer, cues.explained = reply_kind, True
        elif pending.kind == "choice":
            cues.options = tuple(o.key for o in pending.options if re.search(o.words, text, re.I))
            if not cues.options:
                cues.options = tuple(o.key for o in pending.options
                                     if o.reply_words and re.search(o.reply_words, text, re.I))
            cues.explained = bool(cues.options) and short <= 4
            if pending.key == "door_to":  # see rule_cues: the room words describe the next room, nothing else
                cues.explained = bool(cues.options) and short <= 8
                return cues
        elif pending.kind in ("list", "nearest") and (reply_kind == "no" or re.search(r"\b(none|nothing|neither)\b", text, re.I)):
            cues.answer, cues.explained = "none", short <= 4

    for feature in FEATURES:
        said = _mentioned(feature, text)
        if said and not (pending and pending.kind == "yesno" and pending.key == feature.key):
            cues.mentioned[feature.key] = said
            if said == "yes" and not cues.acknowledgement:
                cues.acknowledgement = f"Okay, {feature.noun.split(' or ')[0].split(',')[0]}."
    if pending is not None and pending.kind == "list" and set(cues.mentioned) & set(pending.items):
        cues.explained = short <= 6

    small, large = SMALL_CUE.search(text), LARGE_CUE.search(text)
    if small and not large:
        cues.size = "tight" if STRICT_SMALL_CUE.search(text) else "small"
        cues.acknowledgement = f"You said it's {_size_phrase(small.group(0))}."
    elif large and not small:
        cues.size, cues.acknowledgement = "large", "You said it's a big space."
    if DARK_CUE.search(text):
        cues.dark, cues.acknowledgement = True, "You said it's dark."
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
        elif pending.kind == "yesno" and cues.answer in ("yes", "no"):
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

    # 2. Features mentioned anywhere in the message ("there's a toilet", "no windows").
    for key, said in cues.mentioned.items():
        if key in covered or key not in model.p_yes:
            continue
        p = model.p_yes[key]
        apply(p if said == "yes" else 1 - p)
        covered.add(key)

    # 3. Rooms the caller is next to ("I'm near the kitchen"). Softened: "near" is often a shared wall, not a door.
    for key in cues.nearby:
        if key in model.adjacent:
            apply(0.25 + 0.5 * model.adjacent[key])

    # 4. Floors named outside the floor question ("I'm upstairs").
    if cues.floors and "floor" not in covered and "floor" in model.p_option:
        apply(model.p_option["floor"][list(cues.floors)].sum(axis=0))
        covered.add("floor")

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
    prompts = clip_prompts()
    vectors = embed_texts(prompts)
    return build_model(building, {p: building.candidate_scores(v) for p, v in zip(prompts, vectors)},
                       layout, orientation)
