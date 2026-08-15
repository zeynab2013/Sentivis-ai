"""Claim-level factuality checks for caption sentences.

Statuses:
  SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core.contracts.reasoning import SceneUnderstanding

if TYPE_CHECKING:
    from core.contracts.verified_evidence import VerifiedSceneEvidence


class ClaimSupport(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


@dataclass(frozen=True)
class ClaimVerdict:
    sentence: str
    status: ClaimSupport
    reason: str = ""


# High-risk invention patterns — reject unless evidence explicitly supports them.
_INVENTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:teacher|student|coworker|colleague|doctor|nurse|police)\b", re.I), "role"),
    (re.compile(r"\b(?:happy|sad|angry|excited|depressed|lonely)\b", re.I), "emotion"),
    (re.compile(r"\b(?:discussing|negotiating|arguing about|planning their)\b", re.I), "intent"),
    (re.compile(r"\b(?:named|called)\s+[A-Z][a-z]+\b"), "name"),
    (re.compile(r"\b(?:talking to|in conversation with)\b", re.I), "speculative_talk"),
    (re.compile(r"\b(?:gazing at|staring at)\b", re.I), "speculative_gaze"),
    (re.compile(r"\b(?:man|men|woman|women|girl|boy)\b", re.I), "gender"),
    (re.compile(r"\b(?:on (?:his|her|their) shoulder|over (?:his|her|their) shoulder)\b", re.I), "shoulder"),
    # Venue / school claims must appear in verified evidence — never free invent.
    (
        re.compile(
            r"\b(?:classroom|laboratory|school|library|restaurant|cafe|office workspace)\b",
            re.I,
        ),
        "venue",
    ),
    (
        re.compile(r"\b(?:studying|schoolwork|homework|attending (?:a |the )?class)\b", re.I),
        "intent_activity",
    ),
)


_ROBOTIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\ba person talking to (?:a |an )?person\b", re.I),
    re.compile(r"\bsecond person stands farther back in the frame\b", re.I),
    re.compile(r"\b(?:a |an |the )?person is present\.?\b", re.I),
    re.compile(r"\bthere is (?:a |an )?person\b", re.I),
    re.compile(r"^(?:the image shows|in this image)\b", re.I),
)


def _evidence_blob(understanding: SceneUnderstanding) -> str:
    parts = [
        understanding.evidence_brief or "",
        " ".join(understanding.ranked_subjects),
        " ".join(understanding.environment_keys),
        " ".join(understanding.activity_keys),
        " ".join(understanding.ocr_text),
    ]
    for fact in understanding.facts:
        parts.append(f"{fact.subject} {fact.predicate} {fact.value}")
    return " ".join(parts).lower()


def _verified_blob(verified: VerifiedSceneEvidence) -> str:
    parts = [verified.as_evidence_brief()]
    for ent in verified.entities:
        if ent.narrative_safe:
            parts.append(f"{ent.entity_id} {ent.label}")
    for attr in verified.narrative_attributes():
        parts.append(f"{attr.entity_id} {attr.name} {attr.value}")
    for rel in verified.narrative_relations():
        parts.append(f"{rel.subject_id} {rel.relation_type} {rel.object_id}")
    for act in verified.activities:
        if act.narrative_safe:
            parts.append(act.activity)
    parts.extend(verified.ocr_text)
    # Environment evidence lines (e.g. hazard detected: fire) are authoritative.
    parts.extend(getattr(verified.scene, "evidence", ()) or ())
    if verified.scene.setting:
        parts.append(verified.scene.setting)
    if verified.scene.indoor_outdoor:
        parts.append(verified.scene.indoor_outdoor)
    return " ".join(parts).lower()


def _classify_against_blob(sentence: str, blob: str) -> ClaimVerdict:
    body = (sentence or "").strip()
    if not body:
        return ClaimVerdict(body, ClaimSupport.UNSUPPORTED, "empty")

    lower = body.lower()
    for pattern in _ROBOTIC_PATTERNS:
        if pattern.search(lower):
            return ClaimVerdict(body, ClaimSupport.UNSUPPORTED, "robotic_detector_phrasing")

    for pattern, kind in _INVENTION_PATTERNS:
        if pattern.search(body):
            hits = [m.group(0).lower() for m in pattern.finditer(body)]
            if any(h not in blob for h in hits):
                return ClaimVerdict(body, ClaimSupport.UNSUPPORTED, f"invented_{kind}")

    tokens = {
        t
        for t in re.findall(r"[a-z]{4,}", lower)
        if t
        not in {
            "with",
            "from",
            "that",
            "this",
            "into",
            "onto",
            "while",
            "where",
            "appears",
            "visible",
            "scene",
            "image",
            "person",
            "people",
            "another",
            "toward",
            "farther",
            "nearby",
            "foreground",
            "background",
        }
    }
    if not tokens:
        return ClaimVerdict(body, ClaimSupport.PARTIALLY_SUPPORTED, "generic")
    supported = sum(1 for t in tokens if t in blob or t.rstrip("s") in blob)
    ratio = supported / max(1, len(tokens))
    if ratio >= 0.45:
        return ClaimVerdict(body, ClaimSupport.SUPPORTED, f"overlap={ratio:.2f}")
    if ratio >= 0.25:
        return ClaimVerdict(body, ClaimSupport.PARTIALLY_SUPPORTED, f"overlap={ratio:.2f}")
    return ClaimVerdict(body, ClaimSupport.UNSUPPORTED, f"overlap={ratio:.2f}")


def classify_sentence(sentence: str, understanding: SceneUnderstanding) -> ClaimVerdict:
    """Classify one sentence against SceneUnderstanding evidence."""
    return _classify_against_blob(sentence, _evidence_blob(understanding))


def classify_sentence_against_verified(
    sentence: str,
    verified: VerifiedSceneEvidence,
) -> ClaimVerdict:
    """Classify one sentence against canonical VerifiedSceneEvidence."""
    blob = _verified_blob(verified)
    neutralized = _neutralize_unverified_gender(sentence, blob)
    return _classify_against_blob(neutralized, blob)


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

# Color adjectives that must not block count lookup ("4 brown refrigerators").
_COUNT_COLOR_MULTI = (
    "light gray",
    "dark gray",
    "sky blue",
    "navy blue",
    "olive green",
    "forest green",
)
_COUNT_COLOR_TOKENS = frozenset(
    {
        "black",
        "charcoal",
        "gray",
        "grey",
        "white",
        "cream",
        "beige",
        "tan",
        "khaki",
        "brown",
        "maroon",
        "burgundy",
        "red",
        "orange",
        "mustard",
        "yellow",
        "blond",
        "green",
        "cyan",
        "blue",
        "purple",
        "pink",
        "navy",
        "silver",
        "gold",
    }
)
_PLURAL_TO_SINGULAR = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "knives": "knife",
    "leaves": "leaf",
    "mice": "mouse",
    "geese": "goose",
    "refrigerators": "refrigerator",
    "chairs": "chair",
    "bicycles": "bicycle",
    "motorcycles": "motorcycle",
    "horses": "horse",
    "cars": "car",
    "trucks": "truck",
    "buses": "bus",
    "bowls": "bowl",
    "bottles": "bottle",
    "cups": "cup",
    "vases": "vase",
    "sinks": "sink",
    "ovens": "oven",
    "tables": "table",
    "dining tables": "dining table",
    "stop signs": "stop sign",
    "sports balls": "sports ball",
    "tennis rackets": "tennis racket",
    "handbags": "handbag",
    "backpacks": "backpack",
}
_PERSON_COUNT_LABELS = frozenset(
    {"person", "people", "man", "woman", "child", "boy", "girl"}
)
_VAGUE_QUANTIFIERS = ("several", "multiple", "many", "a few", "numerous")
# Nouns that are plural without a trailing "s" (people, children, …).
_SPECIAL_PLURAL_NOUNS = ("people", "children", "men", "women", "mice", "geese")
_COUNTABLE_NOUN_RE = (
    r"(?:people|children|men|women|mice|geese|"
    r"[A-Za-z][A-Za-z\s-]{0,40}?s)"
)


def _strip_count_color_modifiers(phrase: str) -> str:
    text = (phrase or "").strip().lower()
    if not text:
        return ""
    for prefix in _COUNT_COLOR_MULTI:
        if text.startswith(prefix + " "):
            return text[len(prefix) + 1 :].strip()
    parts = text.split()
    if parts and parts[0] in _COUNT_COLOR_TOKENS:
        return " ".join(parts[1:]).strip()
    return text


def _singularize_count_noun(noun: str) -> str:
    bare = (noun or "").strip().lower()
    if not bare:
        return ""
    if bare in _PLURAL_TO_SINGULAR:
        return _PLURAL_TO_SINGULAR[bare]
    if bare.endswith("ies") and len(bare) > 4:
        return bare[:-3] + "y"
    if bare.endswith(("ches", "shes", "xes", "zes", "sses")):
        return bare[:-2]
    if bare.endswith("s") and not bare.endswith(("ss", "us", "is", "oes")):
        return bare[:-1]
    return bare


def _canonical_count_label(raw: str) -> str:
    """Map a subject / noun phrase to a stable singular object label for counting."""
    key = (raw or "").strip().lower().replace("_", " ")
    if not key or key in {"scene", "vlm"}:
        return ""
    for prefix in ("a ", "an ", "the "):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    # entity#1 / stop_sign_1 / "stop sign 1"
    key = re.sub(r"\s*#\d+\s*$", "", key).strip()
    key = re.sub(r"\s+\d+\s*$", "", key).strip()
    key = _strip_count_color_modifiers(key)
    if not key:
        return ""
    if key in _PLURAL_TO_SINGULAR:
        key = _PLURAL_TO_SINGULAR[key]
    else:
        key = _singularize_count_noun(key)
    if key in {"people", "person", "man", "woman", "child", "boy", "girl"}:
        return "person"
    return key


def _entity_uid(raw: str, label: str) -> str:
    text = (raw or "").strip().lower()
    m = re.search(r"#(\d+)", text)
    if m:
        return f"{label}#{m.group(1)}"
    m = re.search(r"(?:_|\s)(\d+)\s*$", text.replace("#", " "))
    if m:
        return f"{label}#{m.group(1)}"
    return f"{label}#bare"


def _collapse_bare_duplicates(by_label: dict[str, set[str]]) -> dict[str, int]:
    """If numbered ids exist for a label, ignore unlabeled bare duplicates."""
    out: dict[str, int] = {}
    for label, uids in by_label.items():
        numbered = {u for u in uids if not u.endswith("#bare")}
        if numbered:
            out[label] = len(numbered)
        else:
            out[label] = len(uids)
    return out


def _verified_label_counts(understanding: SceneUnderstanding) -> dict[str, int]:
    """Count distinct narrative entities per canonical label from scene understanding."""
    by_label: dict[str, set[str]] = {}
    subjects: set[str] = set()
    for fact in understanding.facts:
        subj = (fact.subject or "").strip().lower()
        if not subj or subj in {"scene", "vlm"}:
            continue
        subjects.add(subj)
    for subj in understanding.ranked_subjects:
        token = (subj or "").strip().lower()
        if token and token not in {"scene", "vlm"}:
            subjects.add(token)
    for subj in subjects:
        label = _canonical_count_label(subj)
        if not label:
            continue
        by_label.setdefault(label, set()).add(_entity_uid(subj, label))
    return _collapse_bare_duplicates(by_label)


def label_counts_from_verified(verified: VerifiedSceneEvidence) -> dict[str, int]:
    """Authoritative distinct-entity counts from the final verified entity set."""
    by_label: dict[str, set[str]] = {}
    for ent in verified.entities:
        if not getattr(ent, "narrative_safe", True):
            continue
        label = _canonical_count_label(ent.label or "")
        if not label:
            continue
        uid = (ent.entity_id or "").strip() or f"{label}#{getattr(ent, 'object_index', 'x')}"
        by_label.setdefault(label, set()).add(uid)
    return {label: len(uids) for label, uids in by_label.items()}


def _resolve_count_map(
    understanding: SceneUnderstanding | None,
    verified: VerifiedSceneEvidence | None,
    counts: dict[str, int] | None,
) -> dict[str, int]:
    if counts is not None:
        return {str(k).lower(): int(v) for k, v in counts.items() if int(v) >= 0}
    if verified is not None:
        return label_counts_from_verified(verified)
    if understanding is not None:
        return _verified_label_counts(understanding)
    return {}


def _lookup_verified_count(counts: dict[str, int], noun_phrase: str) -> int:
    phrase = (noun_phrase or "").strip().lower()
    if not phrase:
        return 0
    bare = _strip_count_color_modifiers(phrase)
    singular = _canonical_count_label(bare) or _singularize_count_noun(bare)
    if singular in _PERSON_COUNT_LABELS or bare in {"people", "persons"}:
        return int(counts.get("person", 0))
    for key in (bare, singular, phrase):
        if key in counts:
            return int(counts[key])
    return int(counts.get(singular, 0))


def _format_verified_count_phrase(verified: int, noun_phrase: str) -> str:
    """Natural phrase using the verified count; preserves color modifiers."""
    lower = (noun_phrase or "").strip().lower()
    color_prefix = ""
    bare = _strip_count_color_modifiers(lower)
    if bare != lower:
        color_prefix = lower[: len(lower) - len(bare)].strip()
    singular = _canonical_count_label(bare) or _singularize_count_noun(bare) or bare
    if singular == "person" and verified != 1:
        noun = "people"
    elif verified == 1:
        noun = singular
    elif bare.endswith("s") and not bare.endswith(("ss", "us", "is")):
        noun = bare
    else:
        noun = bare if bare.endswith("s") else f"{bare}s"
    colored = f"{color_prefix} {noun}".strip() if color_prefix else noun
    if verified <= 1:
        art = "an" if singular[:1] in "aeiou" else "a"
        return f"{art} {colored}".strip()
    return f"{verified} {colored}".strip()


# Pure scene-inventory predicates after "N people/person [are/is] …".
# Anything else (playing/holding/riding/preparing/talking/…) is treated as an
# activity or relationship subject and must keep its actor quantity.
_PERSON_CENSUS_PREDICATE = re.compile(
    r"^(?:(?:are|is|appear|appears)\s+)?"
    r"(?:"
    r"visible(?:\s+(?:nearby|here|there|in\s+the\s+\w+))?"
    r"|present(?:\s+in\s+(?:the\s+)?\w+)?"
    r"|nearby|here|there|shown|detected"
    r"|in\s+(?:the\s+)?(?:scene|image|photo|picture|frame|view|background|foreground|"
    r"kitchen|room|field|street|park|area|outdoor|outdoors|indoor|indoors)"
    r"|outdoor|outdoors|indoor|indoors|outside|inside"
    r"|around(?:\s+(?:the\s+)?\w+)?"
    r"|with\s+(?:a|an|the|\d+)\s+\w+(?:\s+\w+){0,4}"
    r")"
    r"(?:\s|$|[.,;:!?])",
    re.IGNORECASE,
)


def _is_person_census_clause(text: str, phrase_end: int) -> bool:
    """True when ``N people/person`` is a scene census, not an activity/relation subject.

    Distinguishes inventory statements such as ``Four people are visible`` from
    actor-bound clauses such as ``Two people are playing football`` without
    hard-coding sport or activity names.
    """
    rest = (text or "")[phrase_end:]
    cut = re.search(r"[.!?]", rest)
    if cut is not None:
        rest = rest[: cut.start()]
    rest = rest.strip()
    if not rest:
        # Bare "Four people." / trailing inventory fragment → census.
        return True
    return _PERSON_CENSUS_PREDICATE.match(rest) is not None


def _quantity_token_to_int(token: str) -> int:
    raw = (token or "").strip().lower()
    if not raw:
        return -1
    if raw.isdigit():
        return int(raw)
    if raw in _NUM_WORDS:
        return int(_NUM_WORDS[raw])
    if raw in _VAGUE_QUANTIFIERS:
        return -2
    return -1


def _cleanup_after_phrase_removal(text: str) -> str:
    """Repair list/sentence grammar after deleting a duplicate quantity phrase."""
    updated = text
    updated = re.sub(r"\s{2,}", " ", updated)
    updated = re.sub(r"\s+,", ",", updated)
    updated = re.sub(r",\s*,+", ", ", updated)
    updated = re.sub(r"\band\s+and\b", "and", updated, flags=re.IGNORECASE)
    updated = re.sub(r"(?:,\s*)?\band\s+(?=[.!?])", "", updated, flags=re.IGNORECASE)
    updated = re.sub(r"^\s*and\s+", "", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\.\s*and\s+", ". ", updated, flags=re.IGNORECASE)
    updated = re.sub(
        r"\b((?:a|an|one)(?:\s+[A-Za-z]+){1,3})\s+appear\b",
        r"\1 appears",
        updated,
        flags=re.IGNORECASE,
    )
    # Do NOT rewrite "are"→"is" globally: compound subjects like
    # "a vase, a cup, and a dining table are arranged" must keep "are"
    # for downstream dining-table coalesce rewrites.
    updated = re.sub(r"\.\s*(?:nearby|farther back|behind them)\s*\.", ".", updated, flags=re.I)
    updated = re.sub(r"(?:(?<=\.\s)|^)\s*(?:is|are|appears?)\s+nearby\s*\.", ".", updated, flags=re.I)
    # Capitalize sentence starts after removals.
    updated = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        updated,
    )
    updated = re.sub(r",\s*\.", ".", updated)
    updated = re.sub(r"\s+\.", ".", updated)
    updated = re.sub(r"\.{2,}", ".", updated)
    updated = re.sub(r"\s{2,}", " ", updated)
    return updated.strip()


def _color_prefix_pattern() -> str:
    multi = "|".join(_COUNT_COLOR_MULTI)
    singles = "|".join(sorted(_COUNT_COLOR_TOKENS, key=len, reverse=True))
    return rf"(?:(?:{multi}|{singles})\s+)?"


def extract_caption_quantity_mentions(text: str) -> list[tuple[str, int]]:
    """Return (canonical_label, stated_count) for explicit quantity phrases."""
    word_alt = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
    vague_alt = "|".join(_VAGUE_QUANTIFIERS)
    color = _color_prefix_pattern()
    pattern = re.compile(
        rf"\b(?P<quant>\d+|{word_alt}|{vague_alt})\s+(?P<noun>{color}{_COUNTABLE_NOUN_RE})\b",
        flags=re.IGNORECASE,
    )
    out: list[tuple[str, int]] = []
    for match in pattern.finditer(text or ""):
        noun = match.group("noun")
        label = _canonical_count_label(_strip_count_color_modifiers(noun)) or _canonical_count_label(
            noun
        )
        if not label:
            continue
        out.append((label, _quantity_token_to_int(match.group("quant"))))
    return out


def clamp_caption_object_counts(
    text: str,
    understanding: SceneUnderstanding | None = None,
    *,
    verified: VerifiedSceneEvidence | None = None,
    counts: dict[str, int] | None = None,
) -> str:
    """Reconcile caption object quantities to distinct verified entity counts.

    General rules (any class, no image-specific branches):
    * Explicit digit/word quantities must equal verified distinct-entity counts.
    * Vague plurals (several/many) become the verified exact count.
    * Mentions are processed left-to-right; once a class is fully accounted for,
      later quantity/indefinite recounts of that class are removed.
    * Indefinite singular "a person" is never promoted to a plural headcount.
    * Person *census* phrases (visible/present/in the scene/…) use the global
      verified people count; person *activity/relationship subjects*
      (``Two people are playing…``, ``One person is holding…``) keep their
      actor quantity and are never rewritten to the scene headcount.
    """
    updated = (text or "").strip()
    if not updated:
        return updated
    count_map = _resolve_count_map(understanding, verified, counts)
    if not count_map:
        return updated

    word_alt = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
    vague_alt = "|".join(_VAGUE_QUANTIFIERS)
    color = _color_prefix_pattern()
    quant_pat = re.compile(
        rf"\b(?P<quant>\d+|{word_alt}|{vague_alt})\s+(?P<noun>{color}{_COUNTABLE_NOUN_RE})"
        rf"(?:\s+(?P<verb>are|appear))?\b",
        flags=re.IGNORECASE,
    )
    indef_pat = re.compile(
        rf"\b(?P<article>a|an)\s+(?P<noun>{color}[A-Za-z][A-Za-z-]{{1,30}})\b",
        flags=re.IGNORECASE,
    )

    events: list[tuple[int, int, str, str, str]] = []
    for match in quant_pat.finditer(updated):
        events.append(
            (match.start(), match.end(), "quant", match.group("noun"), match.group("verb") or "")
        )
    for match in indef_pat.finditer(updated):
        noun = match.group("noun").strip()
        label = _canonical_count_label(_strip_count_color_modifiers(noun)) or _canonical_count_label(
            noun
        )
        if not label or label not in count_map:
            continue
        if any(not (match.end() <= s or match.start() >= e) for s, e, *_ in events):
            continue
        events.append((match.start(), match.end(), "indef", noun, ""))
    events.sort(key=lambda item: item[0])

    accounted: dict[str, int] = {lab: 0 for lab in count_map}
    pieces: list[str] = []
    cursor = 0
    for start, end, kind, noun, verb in events:
        if start < cursor:
            continue
        pieces.append(updated[cursor:start])
        label = _canonical_count_label(_strip_count_color_modifiers(noun)) or _canonical_count_label(
            noun
        )
        verified_n = int(count_map.get(label, 0)) if label else 0
        if not label or verified_n <= 0:
            pieces.append(updated[start:end])
            cursor = end
            continue

        already = int(accounted.get(label, 0))
        if kind == "quant":
            # Person quantities: only force the global census onto inventory
            # statements. Activity / relationship subjects keep their actor count
            # ("Two people are playing…" must not become "Four people are playing…").
            if label == "person":
                if _is_person_census_clause(updated, end):
                    phrase = _format_verified_count_phrase(verified_n, noun)
                    if verb:
                        phrase = f"{phrase} {verb}"
                    pieces.append(phrase)
                    accounted[label] = verified_n
                else:
                    pieces.append(updated[start:end])
                cursor = end
                continue
            remaining = verified_n - already
            if remaining <= 0:
                pieces.append("")
                cursor = end
                continue
            phrase = _format_verified_count_phrase(remaining, noun)
            if verb:
                v = verb.lower()
                if remaining <= 1 and v == "are":
                    phrase = f"{phrase} is"
                elif remaining <= 1 and v == "appear":
                    phrase = f"{phrase} appears"
                else:
                    phrase = f"{phrase} {verb}"
            pieces.append(phrase)
            accounted[label] = already + remaining
        else:
            # Indefinite singular.
            if label == "person":
                # Activity/role heads — do not consume or alter people census.
                pieces.append(updated[start:end])
            elif already >= verified_n:
                pieces.append("")
                cursor = end
                continue
            else:
                pieces.append(updated[start:end])
                accounted[label] = min(verified_n, already + 1)
        cursor = end

    pieces.append(updated[cursor:])
    return _cleanup_after_phrase_removal("".join(pieces))

def filter_unsupported_claims(text: str, understanding: SceneUnderstanding) -> str:
    """Drop UNSUPPORTED / CONTRADICTED sentences; keep the rest in order."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    kept: list[str] = []
    for sentence in sentences:
        verdict = classify_sentence(sentence, understanding)
        if verdict.status in {ClaimSupport.UNSUPPORTED, ClaimSupport.CONTRADICTED}:
            continue
        kept.append(sentence if sentence.endswith((".", "!", "?")) else f"{sentence}.")
    return " ".join(kept).strip()


def _neutralize_unverified_gender(sentence: str, blob: str) -> str:
    """Rewrite gender words to person/people when gender is not in evidence.

    Prefer rewrite over dropping an otherwise factual sentence (e.g. riding).
    After rewrite, collapse awkward \"person and person\" into natural plurals.
    """
    if re.search(r"\b(?:man|men|woman|women|girl|boy|girls|boys)\b", blob, re.I):
        return sentence
    out = sentence
    out = re.sub(r"\bwomen\b", "people", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmen\b", "people", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:woman|girl|man|boy)\b", "person", out, flags=re.IGNORECASE)
    # Pronouns left behind after gendered nouns are stripped (or invented by VLM).
    out = re.sub(r"\bto (?:her|his) (?:left|right|side)\b", "nearby", out, flags=re.IGNORECASE)
    out = re.sub(r"\bon (?:her|his) (?:left|right)\b", "nearby", out, flags=re.IGNORECASE)
    out = re.sub(r"\bshe\b", "they", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhe\b", "they", out, flags=re.IGNORECASE)
    out = re.sub(r"\bher\b", "their", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhis\b", "their", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhim\b", "them", out, flags=re.IGNORECASE)
    out = re.sub(r"\bthey is\b", "they are", out, flags=re.IGNORECASE)
    out = re.sub(r"\bthey was\b", "they were", out, flags=re.IGNORECASE)
    # Gender strip must not leave detector-like \"A person and person\".
    out = re.sub(
        r"\b(?:(?:an?|the)\s+)?person(?:\s*,)?\s+and\s+(?:(?:an?|the)\s+)?person\b",
        "two people",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\btwo people and (?:(?:an?|the)\s+)?person\b", "people", out, flags=re.IGNORECASE)
    out = re.sub(r"\ba person and two people\b", "people", out, flags=re.IGNORECASE)
    if out and out[0].islower() and sentence[:1].isupper():
        out = out[0].upper() + out[1:]
    return out


def filter_unsupported_claims_verified(text: str, verified: VerifiedSceneEvidence) -> str:
    """Filter caption sentences against canonical verified evidence.

    Prefer neutralizing or stripping unsupported clauses over discarding an
    otherwise factual rich sentence.
    """
    blob = _verified_blob(verified)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    kept: list[str] = []
    for sentence in sentences:
        neutralized = _neutralize_unverified_gender(sentence, blob)
        neutralized = _strip_unverified_invention_clauses(neutralized, blob)
        if not neutralized.strip():
            continue
        verdict = classify_sentence_against_verified(neutralized, verified)
        if verdict.status in {ClaimSupport.UNSUPPORTED, ClaimSupport.CONTRADICTED}:
            # Last salvage: keep only clauses that independently classify as supported.
            salvaged_parts: list[str] = []
            for clause in re.split(r",\s+", neutralized):
                clause = clause.strip(" ,")
                if not clause:
                    continue
                if classify_sentence_against_verified(clause, verified).status in {
                    ClaimSupport.UNSUPPORTED,
                    ClaimSupport.CONTRADICTED,
                }:
                    continue
                salvaged_parts.append(clause)
            if not salvaged_parts:
                continue
            neutralized = ", ".join(salvaged_parts)
            if neutralized and not neutralized[0].isupper():
                neutralized = neutralized[0].upper() + neutralized[1:]
        body = neutralized if neutralized.endswith((".", "!", "?")) else f"{neutralized}."
        kept.append(body)
    return " ".join(kept).strip()


def _strip_unverified_invention_clauses(sentence: str, blob: str) -> str:
    """Remove high-risk invented nouns/roles when absent from verified evidence."""
    out = sentence
    # Drop catcher/umpire/referee clauses when not in evidence.
    for role in ("catcher", "umpire", "referee", "goalkeeper", "batter"):
        if role in blob:
            continue
        out = re.sub(
            rf"(?:^|[.]\s*)(?:[^.]*?\b(?:a |an |the )?{role}\b[^.]*?)(?=[.!?]|$)",
            "",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf",\s*(?:a |an |the )?{role}\b[^,.]*",
            "",
            out,
            flags=re.IGNORECASE,
        )
    # Drop helmet/glove/mask clothing claims when those tokens are not evidenced.
    for noun in ("helmet", "glove", "face mask", "goggles"):
        token = noun.split()[0]
        if token in blob or noun in blob:
            continue
        out = re.sub(
            rf"(?:,\s*)?(?:wearing |with )?(?:a |an |the )?(?:\w+\s+){{0,2}}{re.escape(noun)}\b",
            "",
            out,
            flags=re.IGNORECASE,
        )
    # Repair grammar holes left by removals: "He is, a blue shirt" → "He is wearing a blue shirt"
    out = re.sub(r"\bis\s*,\s+(a|an|the)\s+", r"is wearing \1 ", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(wearing)\s*,\s*", r"\1 ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*,+", ",", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    out = re.sub(r"\s+([.!?])", r"\1", out)
    # Drop empty husks like "He is." / "He is wearing."
    if re.fullmatch(r"(?:he|she|they|a person|the person)\s+is(?:\s+wearing)?\.?", out, flags=re.I):
        return ""
    return out


@dataclass(frozen=True)
class CaptionQualitySignals:
    evidence_coverage: float
    unsupported_claim_count: int
    semantic_redundancy: float
    information_density: float
    sentence_count: int
    word_count: int


def quality_signals(text: str, understanding: SceneUnderstanding) -> CaptionQualitySignals:
    """Internal quality signals — never shown as fake accuracy scores."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    unsupported = 0
    for sentence in sentences:
        if classify_sentence(sentence, understanding).status in {
            ClaimSupport.UNSUPPORTED,
            ClaimSupport.CONTRADICTED,
        }:
            unsupported += 1
    words = (text or "").split()
    blob = _evidence_blob(understanding)
    content = {t for t in re.findall(r"[a-z]{4,}", (text or "").lower())}
    coverage = (sum(1 for t in content if t in blob) / max(1, len(content))) if content else 0.0
    token_sets = [
        {t for t in re.findall(r"[a-z]{3,}", s.lower()) if len(t) >= 3} for s in sentences
    ]
    overlaps: list[float] = []
    for i, a in enumerate(token_sets):
        for b in token_sets[i + 1 :]:
            if not a or not b:
                continue
            overlaps.append(len(a & b) / max(1, len(a | b)))
    redundancy = sum(overlaps) / max(1, len(overlaps)) if overlaps else 0.0
    density = min(1.0, len(content) / max(12.0, len(words) * 0.55)) if words else 0.0
    return CaptionQualitySignals(
        evidence_coverage=round(coverage, 3),
        unsupported_claim_count=unsupported,
        semantic_redundancy=round(redundancy, 3),
        information_density=round(density, 3),
        sentence_count=len(sentences),
        word_count=len(words),
    )
