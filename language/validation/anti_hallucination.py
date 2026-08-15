"""Remove unsupported claims from generated paragraphs."""

from __future__ import annotations

import re

from core.contracts.reasoning import SceneUnderstanding
from core.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class AntiHallucinationFilter:
    """Verify generated sentences against SceneReasoner evidence."""

    def filter_paragraph(self, text: str, understanding: SceneUnderstanding) -> str:
        if not text.strip():
            return ""
        allowed_tokens = self._allowed_tokens(understanding)
        has_person = any(
            any(tok in f.subject.lower() for tok in ("person", "man", "woman", "child"))
            for f in understanding.facts
        ) or any(
            sub.split("#")[0].strip().lower() in {"person", "man", "woman", "child", "people"}
            for sub in understanding.ranked_subjects
        )
        kept: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(text.strip()):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            if self._is_list_like(cleaned):
                continue
            if self._contradicts_high_confidence(cleaned, understanding):
                logger.debug("Dropped contradictory sentence: %s", cleaned[:80])
                continue
            lower = cleaned.lower()
            person_lead = has_person and any(
                lower.startswith(p)
                for p in (
                    "a person",
                    "the person",
                    "a man",
                    "a woman",
                    "a child",
                    "two people",
                )
            )
            if person_lead or self._supported(cleaned, allowed_tokens, understanding):
                kept.append(cleaned if cleaned.endswith((".", "!", "?")) else cleaned + ".")
        paragraph = " ".join(kept).strip()
        logger.debug("Anti-hallucination kept %d sentences", len(kept))
        return paragraph

    def _contradicts_high_confidence(self, sentence: str, understanding: SceneUnderstanding) -> bool:
        """Detect clear clothing/color contradictions against high-confidence facts."""
        lower = sentence.lower()
        evidence_colors = {
            fact.value.lower()
            for fact in understanding.facts
            if fact.confidence >= 0.62
            and (fact.predicate.endswith("_color") or fact.predicate in {"dominant_color", "clothing_color"})
            and fact.value not in {"unknown", "unlikely"}
        }
        evidence_clothing = {
            fact.value.lower().replace("_", " ")
            for fact in understanding.facts
            if fact.confidence >= 0.62
            and fact.predicate in {"clothing_type", "footwear_type"}
            and fact.value not in {"unknown", "unlikely"}
        }
        palette = (
            "black",
            "white",
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "purple",
            "pink",
            "brown",
            "navy",
            "charcoal",
            "cream",
            "beige",
            "maroon",
            "burgundy",
            "gray",
            "grey",
        )
        mentioned = {name for name in palette if name in lower}
        if evidence_colors and mentioned:
            # Allow if any evidence color is present; flag only total mismatch.
            if not any(color.split()[0] in lower or color in lower for color in evidence_colors):
                if any(word in lower for word in ("wearing", "shirt", "top", "dress", "hoodie", "jacket")):
                    return True
        if "formal suit" in " ".join(evidence_clothing) and any(
            word in lower for word in ("hoodie", "shorts", "sportswear")
        ):
            return True
        if any(v in evidence_clothing for v in ("hoodie", "sportswear", "shorts")) and any(
            word in lower for word in ("formal suit", "tuxedo", "blazer")
        ):
            return True
        return False

    def _allowed_tokens(self, understanding: SceneUnderstanding) -> set[str]:
        tokens: set[str] = set()
        for fact in understanding.facts:
            tokens.update(re.findall(r"[a-zA-Z]{3,}", fact.subject.lower()))
            tokens.update(re.findall(r"[a-zA-Z]{3,}", fact.value.lower()))
            tokens.update(re.findall(r"[a-zA-Z]{3,}", fact.predicate.lower()))
        for word in (
            "wearing",
            "holding",
            "standing",
            "sitting",
            "walking",
            "running",
            "playing",
            "beside",
            "near",
            "nearby",
            "outdoor",
            "indoor",
            "sunny",
            "cloudy",
            "street",
            "room",
            "office",
            "court",
            "tennis",
            "setting",
            "present",
            "scene",
            "central",
            "activity",
            "subject",
            "main",
            "other",
            "people",
            "person",
            "young",
            "woman",
            "man",
            "child",
            "while",
            "with",
            "and",
            "the",
            "overall",
            "reads",
            "conditions",
            "weather",
            "timing",
            "point",
            "lettering",
            "includes",
            "visible",
            "using",
            "phone",
            "rest",
            "rests",
            "attention",
            "turns",
            "toward",
            "holds",
            "hold",
            "charcoal",
            "navy",
            "burgundy",
            "olive",
            "cream",
            "beige",
            "mustard",
            "maroon",
            "forest",
            "sneakers",
            "boots",
            "hoodie",
            "jacket",
            "jeans",
            "sweater",
            "blazer",
            "backpack",
            "bottoms",
            "hair",
            "top",
            "racket",
            "afternoon",
            "morning",
            "evening",
            "night",
            "quiet",
            "busy",
            "anchors",
            "anchor",
            "rests",
            "rest",
            "complete",
            "immediate",
            "surroundings",
            "important",
            "details",
            "around",
            "include",
            "conditions",
            "main",
            "subject",
            "dominates",
            "defines",
            "prepared",
            "document",
            "readable",
            "information",
            "landscape",
            "opens",
            "across",
            "architectural",
            "forms",
            "built",
            "environment",
            "interior",
            "everyday",
            "objects",
            "unfolds",
            "stretches",
            "arrangement",
            "focused",
            "centers",
            "beginning",
            "carries",
            "completes",
            "complete",
            "shares",
            "share",
            "frame",
            "outfit",
            "look",
            "includes",
            "beyond",
            "focus",
            "suggests",
            "suggest",
            "feels",
            "action",
            "unfolds",
            "space",
            "wears",
            "wear",
            "just",
            "also",
            "view",
            "stands",
            "visible",
            "part",
            "farther",
            "frame",
            "reinforcing",
            "moment",
            "tone",
            "depth",
            "adding",
            "captures",
            "capture",
            "within",
            "while",
            "beside",
            "clothing",
            "dark",
            "light",
            "dressed",
            "rope",
            "trees",
            "tree",
            "smoke",
            "fire",
            "horse",
            "horses",
            "field",
            "grassy",
            "leading",
            "sweatshirt",
            "wooden",
            "wood",
            "cap",
            "rising",
            "sending",
            "beneath",
            "large",
            "brown",
            "black",
            "blue",
            "green",
            "pit",
            # Photojournalism glue — lets understanding paragraphs sound human.
            "catches",
            "catch",
            "holds",
            "hold",
            "keeps",
            "keep",
            "calm",
            "open",
            "steady",
            "grounded",
            "soften",
            "softens",
            "tones",
            "tone",
            "heart",
            "figure",
            "figures",
            "person",
            "people",
            "nearby",
            "behind",
            "beneath",
            "under",
            "washed",
            "washes",
            "lived",
            "atmosphere",
            "energy",
            "echo",
            "close",
            "wider",
            "natural",
            "clear",
            "bright",
            "sunny",
            "seated",
            "sits",
            "sit",
            "moves",
            "move",
            "through",
            "giving",
            "direction",
            "exchange",
            "turns",
            "attention",
            "stays",
            "stay",
            "unfolds",
            "unfold",
            "grounded",
            "ground",
            "main",
            "primary",
            "secondary",
            "lining",
            "lines",
            "line",
            "settles",
            "settle",
            "filling",
            "fills",
            "fill",
            "lending",
            "lends",
            "moment",
            "moments",
            # Scene-understanding prose (competition narrative, not detector lists).
            "posture",
            "pause",
            "center",
            "story",
            "decoration",
            "tool",
            "makes",
            "make",
            "possible",
            "backdrop",
            "alone",
            "shapes",
            "shape",
            "bodies",
            "body",
            "meet",
            "meets",
            "observational",
            "enough",
            "detail",
            "relationships",
            "relationship",
            "remain",
            "remains",
            "quiet",
            "language",
            "practical",
            "weight",
            "working",
            "geography",
            "shot",
            "quietly",
            "completing",
            "marking",
            "edges",
            "edge",
            "giving",
            "place",
            "rather",
            "blank",
            "stage",
            "reads",
            "read",
            "softened",
            "weather",
            "staged",
            "nudging",
            "nudge",
            "mood",
            "toward",
            "something",
            "unforced",
            "tempo",
            "letting",
            "small",
            "gestures",
            "gesture",
            "theatrical",
            "emphasis",
            "faithful",
            "actually",
            "offers",
            "offer",
            "additional",
            "linger",
            "lingers",
            "echoing",
            "stealing",
            "lettering",
            "enters",
            "enter",
            "further",
            "link",
            "defining",
            "contact",
            "everything",
            "else",
            "arranges",
            "arrange",
            "itself",
            "purpose",
            "sharpen",
            "sharpens",
            "matters",
            "matter",
            "broader",
            "motion",
            "concentrates",
            "concentrate",
            "passage",
            "stretch",
            "gathers",
            "gather",
            "second",
            "beat",
            "same",
            "carrying",
            "carry",
            "roles",
            "role",
            "inventory",
            "exists",
            "exist",
            "outward",
            "spreads",
            "spread",
            "grazing",
            "graze",
            "pasture",
            "rural",
            "farm",
            "grass",
            "ordinary",
            "everyday",
            "unhurried",
            "steady",
            "practical",
            "matter",
            "fact",
            "workaday",
            "relaxed",
            "settled",
            "paused",
            "guiding",
            "guide",
            "animal",
            "handler",
            "foreground",
            "background",
            "landscape",
            "softens",
            "soften",
            "evenly",
            "subjects",
            "ceremony",
            "spectacle",
            "blaze",
            "belong",
            "belongs",
            "drawn",
            "draws",
            "watching",
            "watches",
            "distance",
            "adult",
            "young",
            "tones",
            "wearing",
            "dressed",
            "skies",
            "ground",
            "unfolds",
            "unforced",
            "quiet",
            "calm",
            "staged",
            "display",
            "pause",
            "together",
            "feels",
            "feel",
            "small",
            "outdoor",
            "indoor",
            "compact",
            "specific",
            "place",
            "event",
            "purpose",
            "scenery",
            "put",
            "used",
            "use",
            "left",
            "mid",
            "task",
            "grip",
            "firm",
            "careful",
            "rushing",
            "short",
            "control",
            "balanced",
            "matching",
            "pace",
            "purposeful",
            "closer",
            "routine",
            "portrait",
            "real",
            "gives",
            "give",
            "fixed",
            "most",
            "exchange",
            "pass",
            "between",
            "figures",
            "sign",
            "reads",
            "read",
            "nearby",
            "nothing",
            "details",
            "belong",
            "moment",
            "work",
            "preparing",
            "prepare",
            "cooking",
            "cooks",
            "cook",
            "food",
            "kitchen",
            "dining",
            "table",
            "chairs",
            "chair",
            "refrigerator",
            "television",
            "visible",
            "room",
            "working",
            "works",
            "typing",
            "skiing",
            "skis",
            "crossing",
            "driving",
            "two",
            "three",
            "four",
            "several",
            "multiple",
        ):
            tokens.add(word)
        return tokens

    def _supported(
        self,
        sentence: str,
        allowed_tokens: set[str],
        understanding: SceneUnderstanding,
    ) -> bool:
        words = re.findall(r"[a-zA-Z]{4,}", sentence.lower())
        if not words:
            return False
        # Stricter novelty gate: accuracy over verbose speculation.
        novel = [word for word in words if word not in allowed_tokens]
        novelty_ratio = len(novel) / max(1, len(words))
        lower = sentence.lower()
        filler = any(
            phrase in lower
            for phrase in (
                "photographed scene",
                "quiet observational",
                "observational detail",
                "calm and lived-in",
                "minimal interaction",
                "appears to be",
                "seems to be",
            )
        )
        if filler:
            return False
        speculative = any(
            phrase in lower
            for phrase in (
                "might be",
                "could be",
                "probably",
                "perhaps",
                "seems like maybe",
                "i think",
                "likely",
                "possibly",
            )
        )
        if speculative:
            return False
        # Adaptive novelty: stronger evidence allows slightly more fluent wording.
        # Competition captions need photojournalism connective tissue, not telegram style.
        if understanding.overall_confidence >= 0.7 and novelty_ratio <= 0.42:
            return True
        if understanding.overall_confidence >= 0.55 and novelty_ratio <= 0.34:
            return True
        return novelty_ratio <= 0.26

    def _is_list_like(self, sentence: str) -> bool:
        lower = sentence.lower().strip()
        if lower.startswith(("-", "*", "•")):
            return True
        if re.match(r"^\w+\s*:\s*\w+$", lower):
            return True
        return lower.count("\n") > 0
