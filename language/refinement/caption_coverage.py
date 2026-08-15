"""Salient verified-element coverage for final captions.

Ensures scene-defining verified hazards/objects are not dropped solely because
a shorter caption candidate won arbitration. Never invents unsupported details.
"""

from __future__ import annotations

import re
from typing import Iterable

from core.contracts.reasoning import SceneUnderstanding
from core.contracts.verified_evidence import VerifiedSceneEvidence
from language.evaluation.caption_quality_evaluator import (
    _ordered_activity_tokens_match,
    _strip_articles,
)

# Scene-defining hazards / events — high salience when verified.
_SALIENT_HAZARDS = ("fire", "smoke", "flame")
_PERSON_LABELS = frozenset(
    {"person", "man", "woman", "child", "people", "skier", "rider"}
)
_COUNT_WORDS = {
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
}


def _activity_already_expressed(caption_lower: str, activity: str) -> bool:
    """True when caption already states this verified activity (inflection/modifiers OK).

    Reuses the quality evaluator's controlled ordered-token matcher so
    "holding a rope" matches "holds a rope" / "hold a rope", and
    "leading a horse" matches "leading a brown horse", without treating a
    shared verb alone (holding rope vs holding phone) as coverage.
    """
    name = (activity or "").strip().lower()
    if not name:
        return True
    lower = (caption_lower or "").lower()
    if name in lower:
        return True
    norm_phrase = _strip_articles(name)
    norm_text = _strip_articles(lower)
    if norm_phrase and norm_phrase in norm_text:
        return True
    if _ordered_activity_tokens_match(norm_text, norm_phrase):
        return True
    # Riding a vehicle: accept common surface variants (dirt bike ↔ motorcycle).
    # Keep bicycle separate from motorcycle-family wording.
    if re.match(r"^riding\b", name) and re.search(r"\brid(?:e|es|ing)\b", lower):
        obj = re.sub(r"^riding\s+", "", norm_phrase).strip()
        if not obj:
            return True
        motor = bool(
            re.search(r"\b(?:motorcycle|motorbike|dirt\s+bike)\b", lower)
            or ("dirt bike" in lower)
        )
        cycle = bool(re.search(r"\b(?:bicycle|cycle)\b", lower)) or (
            bool(re.search(r"\bbike\b", lower)) and not motor
        )
        if obj in {"motorcycle", "motorbike"} or "motorcycle" in obj or "motorbike" in obj:
            return motor
        if obj in {"bicycle", "bike", "cycle"} or "bicycle" in obj:
            return cycle
        if "dirt" in obj and "bike" in obj:
            return motor or ("dirt bike" in lower)
    return False


def _activity_person_actors(
    act,
    verified: VerifiedSceneEvidence,
) -> list:
    """Person entities that are verified actors of this activity (not global people)."""
    actors = []
    seen: set[str] = set()
    for eid in act.entity_ids or ():
        ent = verified.entity_by_id(eid)
        if ent is None or not ent.narrative_safe:
            continue
        if ent.label.lower() not in _PERSON_LABELS:
            continue
        if eid in seen:
            continue
        seen.add(eid)
        actors.append(ent)
    for idx in act.object_indices or ():
        for ent in verified.entities:
            if ent.object_index != idx or not ent.narrative_safe:
                continue
            if ent.label.lower() not in _PERSON_LABELS:
                continue
            if ent.entity_id in seen:
                continue
            seen.add(ent.entity_id)
            actors.append(ent)
    return actors


def _coverage_activity_sentence(who: str, name: str, *, plural: bool) -> str:
    """Build a natural coverage sentence with correct is/are agreement."""
    activity = (name or "").strip()
    if activity.startswith(("is ", "are ")):
        activity = re.sub(r"^(?:is|are)\s+", "", activity, count=1)
    if plural:
        return f"{who} are {activity}."
    return f"{who} is {activity}."


def _collect_salient_hazards(
    *,
    understanding: SceneUnderstanding | None = None,
    verified: VerifiedSceneEvidence | None = None,
    environment_evidence: Iterable[str] = (),
) -> dict[str, float]:
    """Return label → best confidence for verified salient hazards."""
    found: dict[str, float] = {}

    def _add(label: str, conf: float) -> None:
        key = (label or "").strip().lower()
        if key == "flame":
            key = "fire"
        if key not in {"fire", "smoke"}:
            return
        if conf < 0.60:
            return
        found[key] = max(found.get(key, 0.0), float(conf))

    if understanding is not None:
        for fact in understanding.facts:
            if fact.predicate == "hazard" and fact.value:
                _add(fact.value, fact.confidence)
            if fact.subject.lower() in _SALIENT_HAZARDS and fact.predicate in {
                "is",
                "detected",
                "present",
                "visible",
            }:
                _add(fact.subject, fact.confidence)
        brief = (understanding.evidence_brief or "").lower()
        for lab in ("fire", "smoke"):
            if f"hazard detected: {lab}" in brief or re.search(
                rf"\b{lab}\b.*confidence", brief
            ):
                _add(lab, 0.70)
        for key in understanding.environment_keys:
            low = key.lower()
            for lab in ("fire", "smoke"):
                if lab in low:
                    _add(lab, 0.68)

    if verified is not None:
        for ent in verified.entities:
            lab = (ent.label or "").lower()
            if lab in _SALIENT_HAZARDS and ent.confidence >= 0.60:
                _add(lab, ent.confidence)
        scene = verified.scene
        for line in getattr(scene, "evidence", ()) or ():
            low = (line or "").lower()
            for lab in ("fire", "smoke"):
                if lab == "fire" and "fire hydrant" in low and "hazard detected: fire" not in low:
                    continue
                if lab not in low and not (lab == "fire" and "flame" in low):
                    continue
                conf = 0.70
                if "confidence:" in low:
                    try:
                        pct = low.split("confidence:", 1)[1].strip().rstrip(").%")
                        conf = float(pct) / (100.0 if float(pct) > 1.0 else 1.0)
                        if conf > 1.0:
                            conf = conf / 100.0
                    except ValueError:
                        conf = 0.70
                _add(lab, conf)
        # Hazard facts often land in the composed brief / notes, not as COCO entities.
        brief = (verified.evidence_brief or verified.as_evidence_brief() or "").lower()
        for lab in ("fire", "smoke"):
            if f"hazard detected: {lab}" in brief or re.search(
                rf"\bhazard\b[^\n]*\b{lab}\b|\b{lab}\b[^\n]*\bhazard\b", brief
            ):
                _add(lab, 0.72)
            elif lab == "fire" and re.search(
                rf"\b(?:fire|flame|campfire|bonfire)\b", brief
            ) and "fire hydrant" not in brief:
                _add(lab, 0.68)
            elif lab == "smoke" and re.search(rf"\bsmoke\b", brief):
                _add(lab, 0.68)

    for line in environment_evidence:
        low = (line or "").lower()
        for lab in ("fire", "smoke"):
            if lab in low or f"hazard detected: {lab}" in low:
                conf = 0.70
                if "confidence:" in low:
                    try:
                        pct = low.split("confidence:", 1)[1].strip().rstrip(").%")
                        conf = float(pct)
                        if conf > 1.0:
                            conf = conf / 100.0
                    except ValueError:
                        conf = 0.70
                _add(lab, conf)

    return found


_REDUNDANT_COUNT_RE = re.compile(
    r"^\s*(?:Two|There are two)\s+people\s+and\s+two\s+horses\s+"
    r"(?:are\s+)?(?:visible|present)?\s*(?:across|in|on)?\s*"
    r"(?:the\s+)?(?:open\s+)?(?:field|scene|setting)?\.?\s*$",
    re.IGNORECASE,
)


def strip_redundant_caption_sentences(text: str) -> str:
    """Drop count/inventory restatements that add no new verified detail."""
    updated = " ".join((text or "").split()).strip()
    if not updated:
        return updated
    kept: list[str] = []
    seen_norm: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", updated):
        body = sentence.strip()
        if not body:
            continue
        if _REDUNDANT_COUNT_RE.match(body):
            continue
        # Exact / near-exact duplicate sentences.
        norm = re.sub(r"[^a-z0-9\s]", "", body.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        if not body.endswith((".", "!", "?")):
            body += "."
        if body[0].islower():
            body = body[0].upper() + body[1:]
        kept.append(body)
    return re.sub(r"\s{2,}", " ", " ".join(kept)).strip()


def expand_verified_information_density(text: str) -> str:
    """Split packed multi-entity clauses into natural sentences — facts only.

    Does not invent subjects. Used so rich scenes land as 3–5 sentences when
    the evidence is already present in the caption text.
    """
    updated = " ".join((text or "").split()).strip()
    if not updated:
        return updated

    # Split common packed secondary-subject clauses into their own sentence.
    # General pattern: ", while another X ..." → separate depth sentence.
    updated = re.sub(
        r",\s*while another ([a-z ]{3,40}?) stand(?:s)? farther back(?: in the [\w\s-]{1,20})?",
        r". Farther back, another \1 stand nearby",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r",\s*while another ([a-z ]{3,40}?) stand(?:s)? nearby",
        r". Farther back, another \1 stand nearby",
        updated,
        flags=re.IGNORECASE,
    )
    # Fix agreement when the captured phrase is a single noun.
    updated = re.sub(
        r"\bFarther back, another ([a-z]+) stand nearby\b",
        r"Farther back, another \1 stands nearby",
        updated,
        flags=re.IGNORECASE,
    )

    # Normalize clothing phrasing without inventing garment type.
    updated = re.sub(
        r"\ba person in a ([a-z ]{3,40}?) sweatshirt\b",
        r"a person wearing a \1 sweatshirt",
        updated,
        flags=re.IGNORECASE,
    )

    # Prefer fuller fire wording when a bare foreground fire clause is present.
    updated = re.sub(
        r"\.\s*In the foreground, a fire burns\.?",
        ". A fire is burning in the foreground.",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\.\s*A fire burns in the foreground\.?",
        ". A fire is burning in the foreground.",
        updated,
        flags=re.IGNORECASE,
    )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", updated) if s.strip()]
    cleaned: list[str] = []
    for sentence in sentences:
        body = sentence.strip()
        if not body.endswith((".", "!", "?")):
            body += "."
        if body[0].islower():
            body = body[0].upper() + body[1:]
        cleaned.append(body)

    # Do NOT append scene-category summary sentences (people/horses/fire).
    # Density expansion only restructures facts already present in the text.
    return strip_redundant_caption_sentences(" ".join(cleaned))


def ensure_salient_verified_coverage(
    text: str,
    *,
    understanding: SceneUnderstanding | None = None,
    verified: VerifiedSceneEvidence | None = None,
    environment_evidence: Iterable[str] = (),
) -> str:
    """Append short natural sentences for missing high-salience verified facts.

    Covers hazards (fire/smoke) and CONFIRMED narrative-safe activities.
    Does not invent unsupported details.
    """
    updated = (text or "").strip()
    if not updated:
        return updated
    hazards = _collect_salient_hazards(
        understanding=understanding,
        verified=verified,
        environment_evidence=environment_evidence,
    )

    # Drop smoke wording when smoke is not independently verified.
    if hazards.get("smoke", 0) < 0.60 and re.search(r"\bsmoke\b", updated, re.I):
        updated = re.sub(
            r",?\s*sending smoke(?:\s+into the air)?",
            "",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"(?:,?\s*with smoke rising(?:\s+into the air)?)",
            "",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"(?:A |The )?smoke is visible nearby\.?",
            "",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\bsmoke rising\b[^.]*\.?",
            "",
            updated,
            flags=re.IGNORECASE,
        )
        # If a fire sentence was only about smoke, keep a fire-only line when fire is verified.
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        updated = re.sub(r"\s+([,.;])", r"\1", updated)
        updated = re.sub(r"\.{2,}", ".", updated)

    lower = updated.lower()
    extras: list[str] = []
    has_fire_mention = any(tok in lower for tok in ("fire", "flame", "fire pit", "firepit", "campfire"))
    has_smoke_mention = "smoke" in lower

    if hazards.get("fire", 0) >= 0.60 and not has_fire_mention:
        extras.append("A fire is burning nearby.")
        has_fire_mention = True
    if hazards.get("smoke", 0) >= 0.60 and not has_smoke_mention:
        # Only assert smoke when independently verified — never invent from fire.
        extras.append("Smoke is visible nearby.")

    # CONFIRMED narrative activities must survive arbitration stubs (e.g. clothing-only).
    if verified is not None:
        from core.contracts.verified_evidence import ActivityEvidenceLevel

        _PERSON = _PERSON_LABELS
        people = [
            e
            for e in verified.entities
            if e.narrative_safe and e.label.lower() in _PERSON
        ]
        # Caption + QA must share the same verified person count.
        # Only add a count when the draft never established plurality at all.
        if len(people) >= 2:
            person_hits = len(re.findall(r"\bpersons?\b", lower))
            has_plural = bool(
                re.search(
                    r"\b(?:two|three|four|five|six|several|multiple)\s+people\b"
                    r"|\bpeople\b|\banother person\b|\bboth people\b|\bthey\b"
                    r"|\bperson(?:\s*,)?\s+and\s+(?:(?:an?|the)\s+)?person\b",
                    lower,
                )
            ) or person_hits >= 2
            if not has_plural:
                n = len(people)
                if person_hits == 1 and n == 2:
                    extras.append("Another person is also visible.")
                else:
                    phrase = _COUNT_WORDS.get(n, str(n))
                    extras.append(f"{phrase} people are visible.")
                lower = (updated + " " + " ".join(extras)).lower()

        for act in verified.activities:
            if act.evidence_level != ActivityEvidenceLevel.CONFIRMED:
                continue
            if not act.narrative_safe:
                continue
            name = (act.activity or "").strip().lower()
            if not name:
                continue
            # Skip only when the activity is actually expressed — verb inflection
            # and brief modifiers count; a shared verb alone must not suppress a
            # distinct second activity (ordered token match, not any-token overlap).
            if _activity_already_expressed(lower, name):
                continue
            # Subject comes from this activity's verified person actors — never from
            # global person count alone (background people must not inflate actors).
            person_actors = _activity_person_actors(act, verified)
            n_actors = len(person_actors)
            if n_actors >= 2:
                if n_actors in _COUNT_WORDS:
                    who = f"{_COUNT_WORDS[n_actors]} people"
                else:
                    who = "People"
                extras.append(_coverage_activity_sentence(who, name, plural=True))
                lower = (updated + " " + " ".join(extras)).lower()
                continue
            # Natural coverage sentence — distinguish people when multiple scene people
            # but this activity has a single verified actor.
            who = "A person"
            actor_is_person = n_actors == 1
            if person_actors:
                who = "A person"
            elif act.entity_ids:
                ent = verified.entity_by_id(act.entity_ids[0])
                if ent is not None and ent.label.lower() in _PERSON_LABELS:
                    actor_is_person = True
                    who = "A person"
                elif ent is not None:
                    who = f"The {ent.label}"
            if actor_is_person and len(people) >= 2:
                # Already described one person's activity → name the next distinctly.
                if re.search(
                    r"\b(?:one person|a person|another person)\b.{0,40}\b"
                    r"(?:is|are)\b.{0,40}\b(?:cooking|preparing|riding|leading|holding|"
                    r"playing|looking|using|walking|running|sitting|standing|carrying|"
                    r"talking|pushing|pulling|watching)\b",
                    lower,
                ) or re.search(
                    r"\b(?:is|are)\s+(?:cooking|preparing|riding|leading|holding|playing|"
                    r"looking|using|walking|running|sitting|standing|carrying)\b",
                    lower,
                ):
                    who = "Another person"
                else:
                    who = "One person"
            extras.append(_coverage_activity_sentence(who, name, plural=False))
            lower = (updated + " " + " ".join(extras)).lower()

        # Activity-only stubs lose clothing context after arbitration salvage.
        # Restore one OBSERVED person clothing color when the caption is too thin.
        draft = f"{updated} {' '.join(extras)}".strip()
        if len(draft.split()) < 12:
            actor_ids: list[str] = []
            for act in verified.activities:
                if act.evidence_level != ActivityEvidenceLevel.CONFIRMED:
                    continue
                if not act.narrative_safe:
                    continue
                actor_ids.extend(list(act.entity_ids))
            _MUTED = {"olive", "khaki", "tan", "beige", "cream", "unknown", ""}
            for eid in actor_ids:
                if not (eid or "").startswith("person"):
                    continue
                shirt = ""
                clothing = ""
                for attr in verified.narrative_attributes():
                    if attr.entity_id != eid:
                        continue
                    if attr.status.value != "OBSERVED":
                        continue
                    if attr.name == "shirt_color":
                        shirt = (attr.value or "").strip()
                    elif attr.name == "clothing_color" and not clothing:
                        clothing = (attr.value or "").strip()
                color = shirt or clothing
                if not color or color.lower() in _MUTED:
                    continue
                if color.lower() in draft.lower():
                    continue
                extras.append(f"The person is wearing {color} clothing.")
                break

    if extras:
        updated = updated.rstrip(".") + ". " + " ".join(extras)
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        if updated and updated[0].islower():
            updated = updated[0].upper() + updated[1:]

    updated = _resolve_riding_spatial_contradiction(updated, verified)
    updated = _strip_unsupported_pose_claims(updated, verified)
    return updated


def _resolve_riding_spatial_contradiction(
    text: str,
    verified: VerifiedSceneEvidence | None,
) -> str:
    """Prefer CONFIRMED riding over contradictory 'next to/beside' for the same vehicle."""
    updated = (text or "").strip()
    if not updated:
        return updated
    lower = updated.lower()
    ride_targets: set[str] = set()
    if verified is not None:
        from core.contracts.verified_evidence import ActivityEvidenceLevel

        for act in verified.activities:
            if not act.narrative_safe:
                continue
            if act.evidence_level != ActivityEvidenceLevel.CONFIRMED and not act.qa_safe:
                continue
            name = (act.activity or "").lower()
            for veh in ("motorcycle", "bicycle", "bike", "horse"):
                if f"riding" in name and veh in name:
                    ride_targets.add("bicycle" if veh == "bike" else veh)
        for rel in verified.narrative_relations():
            if rel.relation_type.lower() != "riding":
                continue
            obj = verified.entity_by_id(rel.object_id)
            if obj is not None:
                lab = obj.label.lower()
                if lab in {"motorcycle", "bicycle", "horse"}:
                    ride_targets.add(lab)
                elif lab == "bike":
                    ride_targets.add("bicycle")
    # Caption text itself may already assert riding.
    for veh in ("motorcycle", "bicycle", "horse"):
        if re.search(rf"\briding (?:a |an |the )?{veh}\b", lower):
            ride_targets.add(veh)
        if veh == "bicycle" and re.search(r"\briding (?:a |an |the )?bike\b", lower):
            ride_targets.add("bicycle")

    if not ride_targets:
        return updated

    for veh in sorted(ride_targets, key=len, reverse=True):
        alt = "bike" if veh == "bicycle" else veh
        # Drop weak spatial clauses about the ridden vehicle.
        patterns = [
            rf",?\s*is positioned (?:next to|beside|near|to the (?:left|right) of|in front of|behind) (?:a |an |the )?(?:\w+\s+){{0,3}}{veh}\b",
            rf",?\s*is positioned (?:next to|beside|near|to the (?:left|right) of|in front of|behind) (?:a |an |the )?(?:\w+\s+){{0,3}}{alt}\b",
            rf",?\s*(?:stands?|is standing|standing) (?:next to|beside|near|to the (?:left|right) of) (?:a |an |the )?(?:\w+\s+){{0,3}}{veh}\b",
            rf",?\s*(?:stands?|is standing|standing) (?:next to|beside|near|to the (?:left|right) of) (?:a |an |the )?(?:\w+\s+){{0,3}}{alt}\b",
            rf"\bpositioned (?:next to|beside|near|to the (?:left|right) of|in front of|behind) (?:a |an |the )?(?:\w+\s+){{0,3}}{veh}\b",
            rf"\bpositioned (?:next to|beside|near|to the (?:left|right) of|in front of|behind) (?:a |an |the )?(?:\w+\s+){{0,3}}{alt}\b",
            rf",?\s*while (?:the |a |an )?(?:\w+\s+){{0,2}}(?:{veh}|{alt}) remains? stationary(?:\s+(?:beside|next to|near)\s+\w+)?\b",
            rf"\b(?:the |a |an )?(?:{veh}|{alt}) remains? stationary(?:\s+(?:beside|next to|near)(?:\s+\w+){{0,3}})?\b",
            rf"\b(?:remains?|is) stationary (?:beside|next to|near) (?:a |an |the )?(?:\w+\s+){{0,3}}{veh}\b",
            rf"\b(?:remains?|is) stationary (?:beside|next to|near) (?:a |an |the )?(?:\w+\s+){{0,3}}{alt}\b",
        ]
        for pat in patterns:
            updated = re.sub(pat, "", updated, flags=re.IGNORECASE)
        # Clean orphaned clause fragments after removals.
        updated = re.sub(r",?\s*while the\s+(?:beside|next to|near)\s+\w+\b", "", updated, flags=re.I)
        updated = re.sub(r",?\s*while the\.?", "", updated, flags=re.I)
        updated = re.sub(r"\s{2,}", " ", updated).strip(" ,")
        # Full sentences that only assert spatial proximity to the ridden vehicle.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", updated) if s.strip()]
        kept: list[str] = []
        for sentence in sentences:
            s_low = sentence.lower()
            only_spatial = bool(
                re.search(
                    rf"\b(?:positioned|stands?|standing|next to|beside|near|left of|right of)\b.*\b(?:{veh}|{alt})\b",
                    s_low,
                )
            ) and not re.search(r"\briding\b", s_low)
            mentions_ride_veh = bool(re.search(rf"\b(?:{veh}|{alt})\b", s_low))
            if only_spatial and mentions_ride_veh and re.search(
                rf"\b(?:positioned|stands?|standing)\b.*\b(?:next to|beside|near|to the (?:left|right) of|left of|right of)\b",
                s_low,
            ):
                # If clothing remains in the sentence, rewrite to riding.
                if re.search(r"\b(?:wearing|jersey|pants|jacket|shirt)\b", s_low):
                    rewritten = re.sub(
                        rf"\b(?:is )?positioned (?:next to|beside|near|to the (?:left|right) of) (?:a |an |the )?(?:\w+\s+){{0,3}}(?:{veh}|{alt})\b",
                        f"is riding a {veh}",
                        sentence,
                        flags=re.IGNORECASE,
                    )
                    rewritten = re.sub(
                        rf"\b(?:stands?|is standing|standing) (?:next to|beside|near|to the (?:left|right) of) (?:a |an |the )?(?:\w+\s+){{0,3}}(?:{veh}|{alt})\b",
                        f"is riding a {veh}",
                        rewritten,
                        flags=re.IGNORECASE,
                    )
                    if rewritten != sentence and "riding" in rewritten.lower():
                        kept.append(rewritten if rewritten.endswith((".", "!", "?")) else rewritten + ".")
                        continue
                continue  # drop pure spatial contradiction
            kept.append(sentence if sentence.endswith((".", "!", "?")) else sentence + ".")
        updated = " ".join(kept)
        # Deduplicate double riding sentences.
        if len(re.findall(rf"\briding (?:a |an |the )?{veh}\b", updated.lower())) >= 2:
            parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", updated) if s.strip()]
            seen_ride = False
            deduped: list[str] = []
            for part in parts:
                is_ride = bool(re.search(rf"\briding (?:a |an |the )?{veh}\b", part.lower()))
                if is_ride and seen_ride and len(part.split()) <= 8:
                    continue
                if is_ride:
                    seen_ride = True
                deduped.append(part if part.endswith((".", "!", "?")) else part + ".")
            updated = " ".join(deduped)

    updated = re.sub(r"\s{2,}", " ", updated).strip()
    updated = re.sub(r"\s+([,.;])", r"\1", updated)
    updated = re.sub(r"\.{2,}", ".", updated)
    return updated


def _strip_unsupported_pose_claims(
    text: str,
    verified: VerifiedSceneEvidence | None,
) -> str:
    """Remove standing/sitting/lying/kneeling claims without verified pose/relation support."""
    updated = (text or "").strip()
    if not updated or verified is None:
        return updated

    allowed: set[str] = set()
    for attr in verified.narrative_attributes():
        if attr.name.lower() != "pose":
            continue
        val = (attr.value or "").strip().lower()
        if val in {"standing", "sitting", "lying", "kneeling"}:
            allowed.add(val)
    for rel in verified.narrative_relations():
        rt = rel.relation_type.lower()
        if rt == "sitting_on":
            allowed.add("sitting")
        if rt == "riding":
            # Riding implies mounted posture — not "standing next to".
            pass

    # Never keep lying/kneeling without narrative pose evidence.
    for pose in ("lying", "kneeling", "sitting", "standing"):
        if pose in allowed:
            continue
        if pose == "lying":
            updated = re.sub(rf"\bis lying\b", "is present", updated, flags=re.IGNORECASE)
            updated = re.sub(rf"\blying\b", "", updated, flags=re.IGNORECASE)
        elif pose == "kneeling":
            updated = re.sub(rf"\bis kneeling\b", "is present", updated, flags=re.IGNORECASE)
            updated = re.sub(rf"\bkneeling\b", "", updated, flags=re.IGNORECASE)
        elif pose == "sitting":
            # Keep sitting only when sitting_on/pose verified.
            updated = re.sub(rf"\bis sitting\b", "is", updated, flags=re.IGNORECASE)
            updated = re.sub(rf"\bsitting\b", "", updated, flags=re.IGNORECASE)
        elif pose == "standing":
            updated = re.sub(rf"\bis standing\b", "is", updated, flags=re.IGNORECASE)
            updated = re.sub(rf"\bstands\b", "is", updated, flags=re.IGNORECASE)
            updated = re.sub(rf"\bstanding\b", "", updated, flags=re.IGNORECASE)

    updated = re.sub(r"\s{2,}", " ", updated).strip()
    updated = re.sub(r"\s+([,.;])", r"\1", updated)
    updated = re.sub(r"\bis is\b", "is", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bperson is present\b", "person is visible", updated, flags=re.IGNORECASE)
    # "another person is nearby" after stripping stands
    updated = re.sub(
        r"\banother person is nearby\b",
        "another person is farther back",
        updated,
        flags=re.IGNORECASE,
    )
    return updated


def missing_salient_hazard_count(
    text: str,
    *,
    understanding: SceneUnderstanding | None = None,
    verified: VerifiedSceneEvidence | None = None,
    environment_evidence: Iterable[str] = (),
) -> int:
    """How many verified salient hazards are absent from the caption text."""
    hazards = _collect_salient_hazards(
        understanding=understanding,
        verified=verified,
        environment_evidence=environment_evidence,
    )
    lower = (text or "").lower()
    missing = 0
    if hazards.get("fire", 0) >= 0.60 and not any(
        tok in lower for tok in ("fire", "flame", "fire pit", "firepit", "campfire")
    ):
        missing += 1
    if hazards.get("smoke", 0) >= 0.60 and "smoke" not in lower:
        missing += 1
    return missing
