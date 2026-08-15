"""Caption candidate arbitration grounded on VerifiedSceneEvidence.

Policy (priority order):
1. Factuality vs verified evidence (highest)
2. Coverage of verified entities/attributes
3. Entity consistency / contradiction-free
4. Low repetition
5. Naturalness / fluency
6. Stylistic quality

A fluent hallucinated caption must never beat a slightly less fluent factual one.
Equal-factuality short stubs lose to richer evidence-covering candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.contracts.verified_evidence import ClaimStatus, VerifiedSceneEvidence
from language.refinement.caption_coverage import missing_salient_hazard_count
from language.refinement.caption_sanity import caption_sanity_score, sanitize_caption
from language.validation.caption_factuality import (
    ClaimSupport,
    classify_sentence_against_verified,
)

_INTERACTION_VERBS = frozenset(
    {
        "holding",
        "talking",
        "looking at",
        "gazing",
        "using",
        "playing",
        "touching",
        "interacting",
        "wearing",
        "sitting on",
        "riding",
        "leading",
        "carrying",
    }
)
_ROBOTIC = (
    re.compile(r"\ba person talking to (?:a |an )?person\b", re.I),
    re.compile(r"\b(?:a |an |the )?person(?:\s*,)?\s+and\s+(?:(?:an?|the)\s+)?person\b", re.I),
    re.compile(r"\bsecond person stands farther back in the frame\b", re.I),
    re.compile(r"^(?:the image shows|in this image)\b", re.I),
    re.compile(r"\bobserved activity\s*:", re.I),
    re.compile(r"\bthe location is outdoor\b", re.I),
    re.compile(r"\bare present in an indoor\b", re.I),
    re.compile(r"\bconfirmed\s*:", re.I),
    re.compile(r"\bsupported\s*:", re.I),
    re.compile(r"\bverified\s*:", re.I),
    re.compile(
        r"^(?:[a-z]+(?:\s+[a-z]+)?,\s*){3,}[a-z]+(?:\s+[a-z]+)?\.?$",
        re.I,
    ),  # inventory list
    re.compile(
        r"^(?:[a-z]+(?:\s+[a-z]+)?,\s*)+[a-z]+(?:\s+[a-z]+)?,\s*and\s+[a-z]+(?:\s+[a-z]+)?\.?"
        r"(?:\s+the location is\b.*)?$",
        re.I,
    ),  # "bicycle, person, handbag, and person. The location is outdoor."
    re.compile(
        r"^(?:person|bicycle|motorcycle|car|horse)(?:,\s*(?:and\s+)?(?:person|bicycle|motorcycle|car|horse))+\.?\s*$",
        re.I,
    ),  # "Person, and bicycle."
)
_THIN_PROXIMITY = re.compile(
    r"\b(?:a |an |the )?person(?: wearing [\w\s]+)? "
    r"(?:stands?|is standing|is) (?:near|beside|next to)\b",
    re.I,
)
_NEAR_SYNONYMS = frozenset({"near", "beside", "nearby", "next"})


@dataclass(frozen=True)
class CaptionCandidateScore:
    text: str
    total: float
    factuality: float
    coverage: float
    naturalness: float
    unsupported: int
    rejected: bool
    reason: str = ""


def _verified_token_blob(verified: VerifiedSceneEvidence) -> set[str]:
    parts: list[str] = [verified.as_evidence_brief().lower()]
    for ent in verified.entities:
        if ent.narrative_safe:
            parts.append(ent.label.lower())
            parts.append(ent.entity_id.lower())
    for attr in verified.narrative_attributes():
        parts.append(attr.value.lower())
        parts.append(attr.name.lower())
    for rel in verified.narrative_relations():
        parts.append(rel.relation_type.lower().replace("_", " "))
    for act in verified.activities:
        if act.narrative_safe:
            parts.append(act.activity.lower())
    if verified.scene.setting:
        parts.append(verified.scene.setting.lower())
    if verified.scene.indoor_outdoor:
        parts.append(verified.scene.indoor_outdoor.lower())
    for line in getattr(verified.scene, "evidence", ()) or ():
        parts.append((line or "").lower())
    blob = " ".join(parts)
    return {t for t in re.findall(r"[a-z]{3,}", blob)}


def _unsupported_interaction(text: str, verified: VerifiedSceneEvidence) -> bool:
    lower = text.lower()
    allowed = {
        r.relation_type.lower().replace("_", " ")
        for r in verified.narrative_relations()
    }
    allowed |= {
        r.relation_type.lower().replace("_", " ")
        for r in verified.qa_relations()
        if r.status == ClaimStatus.OBSERVED
    }
    for verb in _INTERACTION_VERBS:
        if verb not in lower:
            continue
        if verb not in allowed and verb.replace(" ", "_") not in {
            r.relation_type.lower() for r in verified.relations if r.narrative_safe
        }:
            # "wearing" / clothing colors may appear without explicit wearing relation.
            if verb == "wearing" and any(
                a.name in {"clothing_color", "shirt_color", "clothing_type"}
                for a in verified.narrative_attributes()
            ):
                continue
            return True
    return False


def _is_thin_proximity_stub(text: str, verified: VerifiedSceneEvidence) -> bool:
    """Factual but robotic stubs like 'A person stands near a table.'"""
    words = text.split()
    if len(words) >= 36:
        return False
    if not _THIN_PROXIMITY.search(text):
        # Ultra-short when the scene has several narrative-safe entities.
        narr_ents = [e for e in verified.entities if e.narrative_safe]
        if len(words) < 14 and len(narr_ents) >= 3:
            return True
        return False
    # Thin proximity with little extra content.
    lower = text.lower()
    has_color = any(
        c in lower
        for c in (
            "red",
            "blue",
            "green",
            "black",
            "white",
            "gray",
            "grey",
            "navy",
            "brown",
            "yellow",
            "orange",
            "pink",
            "purple",
            "charcoal",
            "maroon",
        )
    )
    narr_ents = [e for e in verified.entities if e.narrative_safe]
    mentioned = sum(1 for e in narr_ents if e.label.lower() in lower)
    if len(words) < 28 and not has_color and mentioned <= 2:
        return True
    if len(words) < 22:
        return True
    return False


def _repetition_ratio(sentences: list[str]) -> float:
    token_sets = [{t for t in re.findall(r"[a-z]{3,}", s.lower())} for s in sentences]
    redundancy = 0.0
    pairs = 0
    for i, a in enumerate(token_sets):
        for b in token_sets[i + 1 :]:
            if a and b:
                # Treat near/beside synonyms as overlapping spatial claims.
                a_norm = {(t if t not in _NEAR_SYNONYMS else "near") for t in a}
                b_norm = {(t if t not in _NEAR_SYNONYMS else "near") for t in b}
                redundancy += len(a_norm & b_norm) / max(1, len(a_norm | b_norm))
                pairs += 1
    return redundancy / pairs if pairs else 0.0


def score_caption_candidate(
    text: str,
    verified: VerifiedSceneEvidence,
) -> CaptionCandidateScore:
    cleaned = sanitize_caption(text)
    if not cleaned:
        return CaptionCandidateScore(
            text="",
            total=-100.0,
            factuality=0.0,
            coverage=0.0,
            naturalness=0.0,
            unsupported=0,
            rejected=True,
            reason="empty",
        )
    # Salvage: drop only unsupported sentences; keep the rest of a rich caption.
    from language.validation.caption_factuality import filter_unsupported_claims_verified

    salvaged = filter_unsupported_claims_verified(cleaned, verified)
    if salvaged.strip():
        cleaned = salvaged
    for pattern in _ROBOTIC:
        if pattern.search(cleaned):
            return CaptionCandidateScore(
                text=cleaned,
                total=-50.0,
                factuality=0.0,
                coverage=0.0,
                naturalness=0.0,
                unsupported=1,
                rejected=True,
                reason="robotic_detector_phrasing",
            )
    thin_stub = _is_thin_proximity_stub(cleaned, verified)
    if _unsupported_interaction(cleaned, verified):
        # Soft reject: heavy factuality penalty, may still lose to factual rivals.
        interaction_penalty = 2.5
    else:
        interaction_penalty = 0.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    unsupported = 0
    supported = 0
    for sentence in sentences:
        verdict = classify_sentence_against_verified(sentence, verified)
        if verdict.status in {ClaimSupport.UNSUPPORTED, ClaimSupport.CONTRADICTED}:
            unsupported += 1
        elif verdict.status == ClaimSupport.SUPPORTED:
            supported += 1
    total_sents = max(1, len(sentences))
    factuality = max(0.0, (supported + 0.5 * (total_sents - supported - unsupported)) / total_sents)
    factuality -= 0.35 * unsupported
    factuality -= interaction_penalty * 0.15
    factuality = max(0.0, min(1.0, factuality))

    blob = _verified_token_blob(verified)
    content = {t for t in re.findall(r"[a-z]{4,}", cleaned.lower())}
    coverage = (sum(1 for t in content if t in blob) / max(1, len(content))) if content else 0.0
    # Reward mentioning narrative-safe entities.
    entity_hits = 0
    narr_ents = [e for e in verified.entities if e.narrative_safe]
    for ent in narr_ents:
        if ent.label.lower() in cleaned.lower():
            entity_hits += 1
    entity_coverage = entity_hits / max(1, len(narr_ents)) if narr_ents else 0.5
    coverage = 0.6 * coverage + 0.4 * entity_coverage

    # Penalize dropping high-salience verified hazards (fire/smoke) even if shorter.
    missing_hazards = missing_salient_hazard_count(cleaned, verified=verified)
    if missing_hazards:
        coverage = max(0.0, coverage - 0.22 * missing_hazards)

    # Reward covering CONFIRMED narrative activities (riding, leading, …).
    confirmed_acts = [
        a.activity.lower()
        for a in verified.activities
        if a.narrative_safe and a.evidence_level.value == "CONFIRMED"
    ]
    act_hits = 0.0
    for act in confirmed_acts[:4]:
        tokens = [t for t in re.findall(r"[a-z]{4,}", act) if t not in {"with", "from", "person"}]
        if tokens and all(t in cleaned.lower() for t in tokens[:2]):
            act_hits += 1.0
        elif tokens and any(t in cleaned.lower() for t in tokens[:2]):
            act_hits += 0.5

    naturalness = caption_sanity_score(cleaned)
    words = len(cleaned.split())
    # Prefer detailed but not list-like captions.
    if 28 <= words <= 160:
        naturalness += 0.15
    # Prefer 3–5 factual sentences when the scene text already supports them.
    sent_n = len(sentences)
    if 3 <= sent_n <= 5:
        naturalness += 0.12
        coverage = min(1.0, coverage + 0.06)
    elif sent_n == 2 and words >= 40:
        naturalness += 0.04
    if words < 12 and len(narr_ents) >= 3:
        naturalness -= 0.55  # collapsed caption when scene is rich
    if thin_stub:
        naturalness -= 0.55
        coverage *= 0.85
    if re.search(r"^(?:[a-z]+,\s*){3,}[a-z]+\.?$", cleaned.lower()):
        naturalness -= 0.6  # detector list
    if re.search(
        r"^(?:[a-z]+(?:\s+[a-z]+)?,\s*)+[a-z]+(?:\s+[a-z]+)?,\s*and\s+[a-z]+",
        cleaned.lower(),
    ):
        naturalness -= 0.75  # "bicycle, person, handbag, and person"
    if "the location is" in cleaned.lower() and "," in cleaned:
        naturalness -= 0.35
    if confirmed_acts:
        coverage = min(
            1.0, coverage + 0.18 * min(1.0, act_hits / max(1.0, float(len(confirmed_acts[:2]))))
        )
        if act_hits < 0.5:
            # Inventory that names objects but drops the defining verified action.
            naturalness -= 0.45
            coverage *= 0.88
    # Repetition penalty
    redundancy = _repetition_ratio(sentences)
    naturalness -= 0.35 * redundancy

    # Factuality dominates; verified coverage + activity next so rich factual
    # captions beat short inventory stubs with similar risk.
    total = (
        3.0 * factuality
        + 1.55 * coverage
        + 0.85 * naturalness
        - 0.45 * unsupported
        - interaction_penalty
        - (0.65 if thin_stub else 0.0)
        - (0.85 * missing_hazards)
    )
    rejected = factuality < 0.20 or (unsupported >= 3 and factuality < 0.45)
    reason = ""
    if interaction_penalty:
        reason = "interaction_unsupported"
    elif missing_hazards:
        reason = "missing_salient_hazard"
    elif thin_stub:
        reason = "thin_proximity_stub"
    return CaptionCandidateScore(
        text=cleaned,
        total=total,
        factuality=factuality,
        coverage=coverage,
        naturalness=naturalness,
        unsupported=unsupported,
        rejected=rejected,
        reason=reason,
    )


def arbitrate_captions(
    candidates: list[str] | tuple[str, ...],
    verified: VerifiedSceneEvidence,
) -> str:
    """Select the safest high-quality caption candidate."""
    scored = [score_caption_candidate(c, verified) for c in candidates if (c or "").strip()]
    usable = [s for s in scored if not s.rejected and s.text]
    if not usable:
        usable = [s for s in scored if s.text]
    if not usable:
        return ""
    # Prefer coverage when factuality is near-equal — do not reward shortest stub.
    usable.sort(
        key=lambda s: (
            round(s.factuality, 2),
            s.coverage,
            s.total,
            min(1.0, len(s.text.split()) / 80.0),
        ),
        reverse=True,
    )
    # Tie-break: richer verified coverage wins when factuality is close.
    if len(usable) >= 2:
        a, b = usable[0], usable[1]
        if abs(a.factuality - b.factuality) <= 0.12 and (b.coverage - a.coverage) >= 0.10:
            if b.unsupported <= a.unsupported + 1 and not b.rejected:
                return b.text
        # Prefer higher total when coverage gap is small but activity-rich.
        if abs(a.factuality - b.factuality) <= 0.05 and b.total > a.total + 0.35:
            if b.unsupported <= a.unsupported and not b.rejected:
                return b.text
    return usable[0].text


def choose_better_caption_grounded(
    verified: VerifiedSceneEvidence,
    *candidates: str,
) -> str:
    """Drop-in grounded replacement for choose_better_caption when verified exists."""
    return arbitrate_captions(candidates, verified)
