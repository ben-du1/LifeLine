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

import re
from dataclasses import dataclass, field

import numpy as np

from . import config, locator

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
]
FEATURES_BY_KEY = {f.key: f for f in FEATURES}


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    words: str
    prompt: str | None = None
    room_types: tuple[str, ...] = ()


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
    # Size is answered from the floor plan geometry, not from photos (see _size_likelihoods).
    Choice("size", "About how big is the room: closet-sized, bedroom-sized, or bigger, like a living room?", (
        Option("small", "closet-sized", r"closet|small|tiny|cramped|tight|little|bathroom.sized"),
        Option("medium", "bedroom-sized", r"bedroom|medium|normal|average|regular"),
        Option("large", "bigger", r"big|bigger|large|living|huge|open|spacious"),
    )),
]
CHOICES_BY_KEY = {c.key: c for c in CHOICES}

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


def clip_prompts() -> list[str]:
    """Every CLIP prompt the model needs, in a fixed order (embedded once per building)."""
    prompts = [f.prompt for f in FEATURES if f.prompt]
    prompts += [o.prompt for c in CHOICES for o in c.options if o.prompt]
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


def build_model(building, prompt_scores: dict[str, np.ndarray]) -> FeatureModel:
    """`prompt_scores` maps each prompt from clip_prompts() to per-candidate raw CLIP scores."""
    rooms, cand_room = building.rooms, building.candidate_room
    kinds = [_room_kinds(r) for r in rooms]
    area = np.array([_polygon_area(r.get("polygon")) for r in rooms])
    aspect = np.array([_aspect_ratio(r) for r in rooms])

    def presence(prompt: str | None, types: tuple[str, ...]) -> np.ndarray:
        clip = _clip_presence(prompt_scores[prompt], building.has_views) if prompt else None
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
        else:
            probs = np.stack([presence(o.prompt, o.room_types) for o in choice.options])
            probs = probs / probs.sum(axis=0, keepdims=True)
        p_option[choice.key] = probs

    floors = floor_options(len(building.floors))
    if floors:
        p_option["floor"] = _floor_likelihoods(building.candidate_floor, len(floors))
    return FeatureModel(p_yes, p_option, area, cand_room, floors, building.candidate_floor)


# ---- planned questions ----------------------------------------------------------------------------------------


@dataclass
class Plan:
    kind: str                  # "yesno", "choice", or "list"
    key: str                   # feature key (yesno), choice key (choice), or "list"
    text: str
    items: tuple[str, ...] = ()  # feature keys offered in a "list" question
    options: tuple = ()          # Options offered in a "choice" question


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
           covered: set[str], last_kind: str | None) -> Plan | None:
    """Highest expected information gain among questions not yet covered. A "which of these" list is not offered
    twice in a row: it is the natural first follow-up to a vague description, but repeated lists read like a form."""
    plans = _candidate_plans(model, covered, last_kind != "list", posterior, cand_room, n_rooms)
    if not plans:
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
    best = int(np.argmax(gains))
    # Nothing left that would tell the remaining rooms apart (e.g. two photo-less bathrooms): stop asking rather than
    # keep the caller typing answers to irrelevant questions.
    return plans[best] if gains[best] >= config.MIN_INFORMATION_GAIN else None


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
    acknowledgement: str = ""                  # e.g. "You said it's a tight space."
    explained: bool = False                    # nothing beyond the answer itself, so no CLIP free-text match


@dataclass
class Reading:
    likelihood: np.ndarray | None        # product of all cue likelihoods (n_candidates,), None if no cues
    covered: set[str]                    # feature/choice keys this message settled (don't ask about them again)
    acknowledgement: str
    explained: bool


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
    if cues.size is None:
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
            cues.answer = "unsure"
        elif pending.kind == "yesno" and reply_kind in ("yes", "no"):
            cues.answer, cues.explained = reply_kind, True
        elif pending.kind == "choice":
            cues.options = tuple(o.key for o in pending.options if re.search(o.words, text, re.I))
            cues.explained = bool(cues.options) and short <= 4
        elif pending.kind == "list" and (reply_kind == "no" or re.search(r"\b(none|nothing|neither)\b", text, re.I)):
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
        elif pending.kind == "list" and cues.answer in ("none", "no"):
            for key in pending.items:
                apply(1 - model.p_yes[key])
                covered.add(key)

    # 2. Features mentioned anywhere in the message ("there's a toilet", "no windows").
    for key, said in cues.mentioned.items():
        if key in covered or key not in model.p_yes:
            continue
        p = model.p_yes[key]
        apply(p if said == "yes" else 1 - p)
        covered.add(key)

    # 3. Floors named outside the floor question ("I'm upstairs").
    if cues.floors and "floor" not in covered and "floor" in model.p_option:
        apply(model.p_option["floor"][list(cues.floors)].sum(axis=0))
        covered.add("floor")

    # 4. Size and darkness.
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


def model_from_building(building, embed_texts) -> FeatureModel:
    """Build the feature model for a loaded building; `embed_texts` is embeddings.embed_texts."""
    prompts = clip_prompts()
    vectors = embed_texts(prompts)
    return build_model(building, {p: building.candidate_scores(v) for p, v in zip(prompts, vectors)})
