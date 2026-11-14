"""Follow-up questions that build on what the caller has said, and how their answers become evidence.

Instead of walking a fixed yes/no list, every turn:
  1. The caller's words are read for cues: size ("cramped", "huge"), darkness, objects they mention or deny, room names.
     Each cue is a likelihood over candidates (see `cue_evidence`).
  2. Possible next questions are generated from the building: yes/no features ("a toilet or tub"), choice
     questions ("carpet, tile, wood or concrete?", "closet-sized, bedroom-sized or bigger?") and a "which of these do
     you see" list built from the features that best separate the rooms still in play.
  3. The question with the highest expected information gain over the room distribution is asked, phrased to
     acknowledge what the caller just said ("You said it's a tight space. What can you see around you: ...").

Every feature has a model P(yes | candidate) that combines up to three sources (see `build_model`):
  - CLIP: how much the candidate's best photo matches the feature's prompts beyond neutral prompts ("an empty
    room"), through a fixed logistic, nudged by its rank among candidates (see _clip_presence).
  - Room type: Matterport's classification/label (a toilet implies a bathroom). This is what lets photo-less rooms
    (often bathrooms and closets) be found at all.
  - Geometry: room area and shape from the floor plan outline (for size and "long and narrow").
"""

import re
from dataclasses import dataclass, field

import numpy as np

from . import commonsense, config, locator, questions

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
]
FEATURES_BY_KEY = {f.key: f for f in FEATURES}


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    words: str
    prompt: str | None = None
    room_types: tuple[str, ...] = ()
    # Room types where this answer is implausible (a carpeted garage or kitchen). Only used for flooring, where the
    # room label vetoes CLIP for these rooms (see _flooring_presence).
    excluded: tuple[str, ...] = ()


@dataclass(frozen=True)
class Choice:
    key: str
    question: str
    options: tuple[Option, ...]


# Room types that are essentially never carpeted: wet rooms and garages. A grey concrete garage floor can out-score
# real carpet on the CLIP carpet prompt, which once put the dot in the garage for "carpeted floor".
_NEVER_CARPETED = ("garage", "bath", "kitchen", "laundry", "utility")

CHOICES = [
    # Concrete is always offered (not only when the building has a labelled garage): an unlabelled basement or storage
    # room is concrete too, and a caller who says "concrete" to a carpet/tile/wood question would otherwise give no
    # usable answer. In a building without such rooms the option simply carries little probability.
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
    # Size is answered from the floor plan geometry, not from photos (see _size_likelihoods).
    Choice("size", "About how big is the room: closet-sized, bedroom-sized, or bigger, like a living room?", (
        Option("small", "closet-sized", r"closet|small|tiny|cramped|tight|little|bathroom.sized"),
        Option("medium", "bedroom-sized", r"bedroom|medium|normal|average|regular"),
        Option("large", "bigger", r"big|bigger|large|living|huge|open|spacious"),
    )),
]
CHOICES_BY_KEY = {c.key: c for c in CHOICES}

# ---- CLIP prompt ensembles and neutral baselines (see _clip_presence) ------------------------------------------
# Extra phrasings per Feature/Option prompt (the prompt itself stays first). One wording is brittle: CLIP may tie
# "washing machine" to a laundromat, so a feature's score is the mean over its phrasings. A prompt missing here
# (e.g. a new option) simply has no extra phrasings.
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
# Neutral prompts a feature prompt has to beat in the same view. "A room with furniture" is there so a cluttered
# view, which scores well against any object prompt, also scores well against the baseline.
BASELINE_PROMPTS = ("a photo of an empty room", "a photo of a white wall", "a photo of a hallway",
                    "a photo of a room with furniture")

# Cues in free text. Size cues reuse the size likelihoods; "dark" counts as evidence of no window.
# "Narrow" is deliberately not a size cue: it's a shape (the "narrow" feature), and hallways can be long.
# Not bare "open" or "wide" ("the door is open", "wide open"), not "little" alone ("a little scared"), not
# "hold/sit tight". A size word describing a thing ("a big bed", "small window") is skipped, see _room_size_cue.
SMALL_CUE = re.compile(r"\b(cramped|(?<!hold )(?<!sit )(?<!holding )(?<!sitting )tight|tiny|small|confined|enclosed|"
                       r"squeez\w*|barely fit|no room to move|can'?t move)\b|"
                       r"\blittle\s+(room|space|area|spot|place|closet|bathroom|bedroom|office|nook)\b", re.I)
STRICT_SMALL_CUE = re.compile(r"\b(cramped|(?<!hold )(?<!sit )(?<!holding )(?<!sitting )tight|tiny|confined|"
                              r"squeez\w*|barely fit|no room to move|can'?t move)\b", re.I)
LARGE_CUE = re.compile(r"\b(big|large|huge|spacious|giant|massive|open (space|area|plan|concept|room))\b", re.I)
DARK_CUE = re.compile(r"\b(dark|pitch black|no light|can'?t see anything|lights? (are |is )?(off|out))\b", re.I)
# Nouns that a size or darkness word may describe instead of the room: "a big bed", "small kid", "dark wood floor".
_DESCRIBED = (r"kids?|child(ren)?|baby|dog|cat|bit|lamp|lights?|desk|dresser|box(es)?|bags?|pile|rugs?|fire|smoke|"
              r"man|guy|hole|crack|plant|picture|painting|doors?|wood|wooden|tiles?|floors?|blue|green|brown|gr[ae]y|"
              r"red|black|colou?red|carpet|curtains?|walls?|paint|cabinets?|furniture|hair|bruise")
ANY_NEGATION = re.compile(r"\b(no|not|nor|none|nothing|without|never|neither|cannot)\b|n't\b|"
                          r"\b(is|are|was|were|do|does|did|ca|can|could|has|have|had|wo|would|should|ai)nt\b", re.I)
NEGATION = questions.NEGATOR  # kept for callers of the old name; scope rules live in questions.polarity

# Floor materials named anywhere in free text, by flooring option. Stricter than the option words used for answers to
# the flooring question: unprompted, "wood" alone is usually furniture ("a wooden table") and "stone" a fireplace.
FLOORING_CUES = {
    "carpet": re.compile(r"\bcarpet(s|ed|ing)?\b|\brugs?\b", re.I),
    "tile": re.compile(r"\btile[sd]?\b|\btiling\b|\blinoleum\b|\bvinyl\b|ceramic floor", re.I),
    "wood": re.compile(r"hardwood|\bwood(en)? floor|\blaminate\b|floor ?boards?", re.I),
    "concrete": re.compile(r"\bconcrete\b|\bcement\b", re.I),
}
# Words that may surround a floor material without adding anything CLIP should match ("it's a carpeted floor"). A
# message made only of these and material words is fully explained by the flooring cue, so the raw text is not sent
# to the free-text CLIP match (which is what favoured the garage for "carpeted floor").
_FLOOR_FILLER = set("""
    floor floors flooring ground the a an it its it's is are was there there's here i i'm im on of with and or kind
    sort like some all just very really looks look covered gray grey beige brown white black dark light soft hard cold
    old new my me under under-foot underfoot feet thick thin wall-to-wall wall to tan red blue green cream
    carpet carpets carpeted carpeting rug rugs tile tiles tiled tiling linoleum vinyl ceramic hardwood wood wooden
    laminate floorboards floorboard boards concrete cement
""".split())

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
    has_views: np.ndarray | None = None    # (n_candidates,) bool, sweep has photos (None: assume all do)
    # Position questions (adjacency.register): key ("adj:<room id>", "front") -> question text; P(yes) is in p_yes.
    positions: dict[str, str] = field(default_factory=dict)
    # P(yes) from the photos alone (see build_model); read by the within-room ("near") questions only.
    p_local: dict[str, np.ndarray] | None = None


def clip_prompts() -> list[str]:
    """Every CLIP prompt the model needs, in a fixed order (embedded once per building)."""
    prompts = [f.prompt for f in FEATURES if f.prompt]
    prompts += [o.prompt for c in CHOICES for o in c.options if o.prompt]
    # Ensemble phrasings and baselines after the original prompts, so the originals keep their order.
    extra = [v for p in prompts for v in PROMPT_VARIANTS.get(p, ())] + list(BASELINE_PROMPTS)
    return list(dict.fromkeys(prompts + extra))


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
        options.append(Option(f"floor{i}", "ground floor" if i == 0 else f"{_ordinal(n)} floor", "|".join(words)))
    return tuple(options)


def _floor_likelihoods(candidate_floor: np.ndarray, n_floors: int) -> np.ndarray:
    """P(caller names floor j | candidate on floor i): FLOOR_ANSWER_ACCURACY when j == i, the rest spread evenly.
    Callers do mix up floor numbers (is the ground floor 1 or 0?), so a floor answer is strong evidence, not proof."""
    acc = config.FLOOR_ANSWER_ACCURACY
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


def build_model(building, prompt_scores: dict[str, np.ndarray],
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


# ---- planned questions ----------------------------------------------------------------------------------------


@dataclass
class Plan:
    kind: str                  # "yesno", "choice", "list", "near" (see choose_within_room) or "position" (where
                               # the room is; see adjacency.py)
    key: str                   # feature key (yesno), choice key (choice), or "list"
    text: str
    items: tuple[str, ...] = ()  # feature keys offered in a "list" question
    options: tuple = ()          # Options offered in a "choice" question
    room: int | None = None      # room a "near" question is about


def answer_matrix(plan: Plan, model: FeatureModel) -> np.ndarray:
    """P(answer | candidate) for each possible answer, shape (n_answers, n_candidates)."""
    if plan.kind in ("yesno", "position"):
        p = model.p_yes[plan.key]
        return np.stack([p, 1 - p])
    if plan.kind == "choice":
        return model.p_option[plan.key]
    if plan.kind == "near":
        return _near_matrix(model, plan.room, plan.items)
    # "Which of these do you see?" approximated as one categorical answer: one of the items, or none of them.
    ps = np.stack([model.p_yes[k] for k in plan.items])
    none = np.prod(1 - ps, axis=0)
    probs = np.vstack([ps, none[None]])
    return probs / probs.sum(axis=0, keepdims=True)


def _candidate_plans(model: FeatureModel, covered: set[str], allow_list: bool,
                     posterior: np.ndarray, cand_room: np.ndarray, n_rooms: int) -> list[Plan]:
    plans = [Plan("yesno", f.key, f.question) for f in FEATURES if f.key not in covered]
    plans += [Plan("choice", c.key, c.question, options=c.options) for c in CHOICES if c.key not in covered]
    if model.floor_options and "floor" not in covered:
        plans.append(Plan("choice", "floor", floor_question(model.floor_options), options=model.floor_options))
    if allow_list:
        # The list offers the features that individually split the plausible rooms best.
        singles = sorted((p for p in plans if p.kind == "yesno"),
                         key=lambda p: -locator.expected_information_gain(
                             posterior, answer_matrix(p, model), cand_room, n_rooms))
        items = tuple(p.key for p in singles[:LIST_SIZE])
        if len(items) == LIST_SIZE:
            nouns = [FEATURES_BY_KEY[k].noun for k in items]
            text = f"What can you see around you: {nouns[0]}, {nouns[1]}, or {nouns[2]}? Or something else?"
            plans.append(Plan("list", "list", text, items))
    return plans


def choose(model: FeatureModel, posterior: np.ndarray, cand_room: np.ndarray, n_rooms: int,
           covered: set[str], last_kind: str | None, min_gain: float = config.MIN_INFORMATION_GAIN) -> Plan | None:
    """Highest expected information gain among questions not yet covered. A "which of these" list is not offered
    twice in a row: it is the natural first follow-up to a vague description, but repeated lists read like a form."""
    plans = _candidate_plans(model, covered, last_kind != "list", posterior, cand_room, n_rooms)
    if not plans and not model.positions:
        return None
    # Floor first: in a multi-storey building the floor is what responders act on first (which stairs, which
    # windows to ladder) and callers know it reliably, so it is asked before room details while no floor is likely
    # enough, even when another question would split the rooms slightly better.
    floor_plan = next((p for p in plans if p.key == "floor"), None)
    if floor_plan is not None:
        floor_probs = np.bincount(model.candidate_floor, weights=posterior, minlength=len(model.floor_options))
        if floor_probs.max() < config.FLOOR_FIRST_UNTIL:
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


# ---- within a big room: "which are you closest to?" -----------------------------------------------------------
# Once the room is known, a big open-plan room (kitchen + living area) still leaves responders searching. Each sweep
# has its own photos, so a landmark that is prominent in some sweeps' photos and not others (the stove at one end,
# the fireplace at the other) tells the sweeps apart. Called by sessions once the room is settled (see subroom.py).

# Landmarks a caller can say they are next to. Not "toilet"/"washer"/"narrow": those identify rooms, not positions.
NEAR_FEATURES = ("stove", "fridge", "sink", "fireplace", "sofa", "tv", "table", "bed", "window", "sliding_door",
                 "front_door", "stairs")
# A landmark whose P(yes) varies less than this across the room's imaged sweeps is flat (CLIP sees it equally from
# everywhere, or only the room label speaks): asking about it can't place the caller. 0.1 on the probability scale is
# about a third of a CLIP standard deviation after _combine's averaging with the room label.
NEAR_MIN_SPREAD = 0.1
# How many of the most varying landmarks to try in 2- and 3-item questions (at most 20 combinations).
NEAR_POOL = 6
# Closeness of a sweep to a landmark is its P(yes) rescaled to 0..1 within the room, plus this floor, so that the
# sweep where the landmark is least visible is still possible (callers misjudge "closest", and CLIP is noisy): the
# best-matching sweep is at most 5x as likely as the worst for that answer.
NEAR_FLOOR = 0.25
# A landmark whose P(yes) rises by at least this much across the room counts at full strength; one that varies less
# is scaled down proportionally (e.g. a 0.1 spread gives half the contrast). 0.2 is what a CLIP score ~1.5 std apart
# at the two ends of a room gives after _combine.
NEAR_FULL_SPREAD = 0.2
# P("none of them" / "not sure") for every sweep: such a reply carries no position information.
NEAR_UNSURE = 0.1
NEAR_ORDINAL = re.compile(r"\b(first|1st|second|2nd|third|3rd|last)\b", re.I)
NEAR_NONE = re.compile(r"\b(none|nothing|neither|not sure|don'?t know|no idea|middle)\b", re.I)


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
    flooring: tuple[str, ...] = ()             # floor materials named outside the flooring question ("carpet",)
    acknowledgement: str = ""                  # e.g. "You said it's a tight space."
    explained: bool = False                    # nothing beyond the answer itself, so no CLIP free-text match


@dataclass
class Reading:
    likelihood: np.ndarray | None        # product of all cue likelihoods (n_candidates,), None if no cues
    covered: set[str]                    # feature/choice keys this message settled (don't ask about them again)
    acknowledgement: str
    explained: bool


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


# "the second one", "#2", "last one": answers to a list or choice question by position.
_ORDINAL = re.compile(
    r"\W*(?:(?:um+|uh+|it'?s|its|the|number|option|#)\s*)*(first|1st|second|2nd|third|3rd|fourth|4th|last|[1-4])"
    r"(?:\s+(?:one|option|thing))?\W*", re.I)
_ORDINAL_INDEX = {"first": 0, "1st": 0, "1": 0, "second": 1, "2nd": 1, "2": 1, "third": 2, "3rd": 2, "3": 2,
                  "fourth": 3, "4th": 3, "4": 3, "last": -1}
_ALL = re.compile(r"\W*(all|all of (them|those|it)|all three|every ?thing|everything you said)\W*", re.I)


def _position(text: str, n: int) -> int | None:
    match = _ORDINAL.fullmatch(text)
    if not match:
        return None
    index = _ORDINAL_INDEX[match.group(1).lower()]
    index = n - 1 if index < 0 else index
    return index if index < n else None


def rule_cues(text: str, pending: Plan | None, reply_kind: str, semantic: dict[str, float] | None = None,
              floors: tuple = ()) -> Cues:
    """Local reading of a message: keyword rules, plus paraphrase matches from semantics.scores() when given.
    `reply_kind` is questions.parse_reply(text).kind; `floors` is the building's FeatureModel.floor_options."""
    cues = _keyword_cues(text, pending, reply_kind)
    if floors and not (pending is not None and pending.key == "floor"):  # an answer to that question is an answer
        named = tuple(i for i, option in enumerate(floors) if re.search(option.words, text, re.I))
        if named and len(named) < len(floors):
            cues.floors = named
            where = floors[named[0]].label if len(named) == 1 else None
            cues.acknowledgement = f"You said you're on the {where}." if where else "You said you're upstairs."
    if semantic:
        _add_semantic(cues, semantic, text, pending)
    return cues


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
            cues.answer = "unsure"
        elif pending.kind in ("yesno", "position") and reply_kind in ("yes", "no"):
            cues.answer, cues.explained = reply_kind, True
        elif pending.kind == "yesno":
            # "no toilet", "there's a tub", "i don't see any windows": the answer is in the description.
            said = _mentioned(FEATURES_BY_KEY[pending.key], text, spans) if pending.key in FEATURES_BY_KEY else None
            if said:
                cues.answer, cues.explained = said, short <= 4
        elif pending.kind == "choice":
            cues.options = tuple(o.key for o in pending.options if re.search(o.words, text, re.I))
            if not cues.options:  # "the second one"
                index = _position(text, len(pending.options))
                cues.options = (pending.options[index].key,) if index is not None else ()
            cues.explained = bool(cues.options) and (short <= 4 or (pending.key == "flooring"
                                                                    and only_floor_words(text)))
        elif pending.kind == "list" and (reply_kind == "no" or re.search(r"\b(none|nothing|neither)\b", text, re.I)):
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


def apply_cues(cues: Cues, model: FeatureModel, pending: Plan | None) -> Reading:
    """Turn cues into one likelihood over candidates."""
    likelihood = None
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
        elif pending.kind == "list" and cues.answer in ("none", "no"):
            for key in pending.items:
                apply(1 - model.p_yes[key])
                covered.add(key)

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
            apply(rows[[near_items.index(k) for k in hits]].sum(axis=0))
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
            apply(_within_room_factor(model, key))
        covered.add(key)

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

    return Reading(likelihood, covered, cues.acknowledgement, cues.explained)


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


def read_message(text: str, model: FeatureModel, pending: Plan | None, reply_kind: str,
                 cues: Cues | None = None, semantic: dict[str, float] | None = None) -> Reading:
    """Turn one caller message into likelihoods over candidates: from Claude's cues when given, otherwise from the
    keyword rules plus local paraphrase matches."""
    if cues is None:
        cues = rule_cues(text, pending, reply_kind, semantic, model.floor_options)
    return apply_cues(cues, model, pending)


def _size_phrase(word: str) -> str:
    word = word.lower()
    if word.startswith(("cramped", "tight", "tiny", "confined", "enclosed", "squeez", "barely", "can", "no room")):
        return "a tight space"
    return "a small space"


def phrase(plan: Plan, acknowledgement: str) -> str:
    return f"{acknowledgement} {plan.text}".strip()


def model_from_building(building, embed_texts) -> FeatureModel:
    """Build the feature model for a loaded building; `embed_texts` is embeddings.embed_texts."""
    prompts = clip_prompts()
    vectors = embed_texts(prompts)
    scores = {p: building.candidate_scores(v) for p, v in zip(prompts, vectors)}
    return build_model(building, scores, view_contrast(building, dict(zip(prompts, vectors))))
