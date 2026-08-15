"""Format pipeline analysis results for display in the active UI language."""

from __future__ import annotations

import re

from core.contracts.pipeline import PipelineResult
from language.refinement.caption_refiner import active_ui_language, localize_term

_PRIMARY_LABELS = frozenset(
    {
        "person",
        "people",
        "man",
        "woman",
        "child",
        "dog",
        "cat",
        "horse",
        "cow",
        "sheep",
        "bird",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "airplane",
        "train",
        "boat",
        "fire",
        "smoke",
    }
)

_WEAK_ACTIVITIES = frozenset(
    {
        "people present",
        "static scene",
        "waiting",
        "having a conversation",
        "transportation scene",
        "standing",
        "sitting",
        "present",
    }
)

_MEANINGFUL_RELATIONS = frozenset(
    {
        "holding",
        "sitting_on",
        "riding",
        "leading",
        "playing_with",
        "looking_at",
        "carrying",
        "using",
        "talking_to",
        "eating",
        "standing_beside",
        "inside",
        "driving",
    }
)

_WEAK_RELATION_TYPES = frozenset(
    {
        "near",
        "next_to",
        "left_of",
        "right_of",
        "above",
        "below",
        "far",
        "overlapping",
        "near_vehicle",
        "behind",
        "in_front_of",
    }
)

_PLACEHOLDER_ENV = frozenset(
    {
        "",
        "unknown",
        "general",
        "general scene",
        "photographed scene",
        "everyday environment",
        "none",
        "clear",
    }
)

_VISIBLE_ATTR_NAMES = frozenset(
    {
        "shirt_color",
        "pants_color",
        "shoes_color",
        "hair_color",
        "clothing_type",
        "dominant_color",
        "color",
    }
)

_MIN_ACTIVITY_CONFIDENCE = 0.65
_MIN_RELATION_CONFIDENCE = 0.62
_MIN_OBJECT_CONFIDENCE = 0.35


def _L(term: str) -> str:
    """Localize a verified analysis term into the active UI language."""
    return localize_term(term, language=active_ui_language())


def format_professional_analysis(result: PipelineResult) -> str:
    """Competition-ready English analysis — omit empty or low-value sections."""
    sections: list[str] = []

    scene = format_scene_description(result).strip()
    if scene:
        sections.append(f"Scene Description:\n{scene}")

    objects = format_detected_objects(result)
    if objects and objects != "None verified.":
        sections.append(f"Detected Objects:\n{objects}")

    environment = format_environment_brief(result)
    if environment:
        sections.append(f"Environment:\n{environment}")

    relationships = format_relationships(result)
    if relationships and relationships not in {"None verified.", "No verified relationships"}:
        sections.append(f"Relationships:\n{relationships}")

    activities = format_activities(result)
    if activities and activities not in {"None verified.", "No verified activities"}:
        sections.append(f"Activities:\n{activities}")

    attributes = format_visible_attributes(result)
    if attributes and attributes != "None verified.":
        sections.append(f"Attributes:\n{attributes}")

    return "\n\n".join(sections) if sections else "No verified analysis available."


def format_scene_summary(result: PipelineResult) -> str:
    verified = getattr(result, "verified_evidence", None)
    if verified is not None:
        human = verified.compose_human_scene_summary()
        if human:
            return human
    context = result.scene_context
    env = context.environment
    activities = ", ".join(
        item.activity
        for item in context.activities.activities
        if item.activity.lower() not in _WEAK_ACTIVITIES and item.confidence >= _MIN_ACTIVITY_CONFIDENCE
    ) or "None verified"
    dominant = ", ".join(context.dominant_objects) or "None"
    setting = env.setting if env.setting.lower() not in _PLACEHOLDER_ENV else "unspecified"
    return (
        f"Objects: {context.object_count}\n"
        f"Dominant: {dominant}\n"
        f"Environment: {setting} ({env.indoor_outdoor})\n"
        f"Activities: {activities}\n\n"
        f"{context.spatial_summary}"
    )


def format_scene_description(result: PipelineResult) -> str:
    caption = result.caption
    text = (caption.narrative_full or caption.text or "").strip()
    return text


def format_detected_objects(result: PipelineResult) -> str:
    """Object inventory with instance counts (authoritative graph nodes + hazards)."""
    from collections import Counter

    graph = result.scene_context.graph
    conf_by_index: dict[int, float] = {}
    for attr in result.scene_context.attributes.attributes:
        if attr.name == "confidence":
            raw = attr.value.strip().rstrip("%")
            try:
                value = float(raw)
                conf_by_index[attr.object_index] = value / 100.0 if value > 1.0 else value
            except ValueError:
                continue

    counts: Counter[str] = Counter()
    for node in graph.nodes:
        conf = conf_by_index.get(node.index)
        if conf is not None and conf < _MIN_OBJECT_CONFIDENCE:
            continue
        counts[node.label.lower()] += 1

    hazard_conf = _hazard_confidence_from_evidence(result)
    for label in result.scene_context.dominant_objects:
        key = label.lower().strip()
        if key not in {"fire", "smoke"} or key in counts:
            continue
        counts[key] += 1

    if not counts:
        return "None verified."

    lines: list[str] = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        shown = _L(label)
        if count > 1:
            lines.append(f"- {count}× {shown}")
        else:
            conf = hazard_conf.get(label)
            if conf is not None and label in {"fire", "smoke"}:
                lines.append(f"- {shown} (confidence: {conf:.0%})")
            else:
                lines.append(f"- {shown}")
    total = sum(counts.values())
    lines.append(f"\nObject instances (graph + verified hazards): {total}")
    return "\n".join(lines)


def _hazard_confidence_from_evidence(result: PipelineResult) -> dict[str, float]:
    conf: dict[str, float] = {}
    for line in result.scene_context.environment.evidence:
        lower = line.lower()
        for label in ("fire", "smoke"):
            if label not in lower:
                continue
            if "confidence:" in lower:
                try:
                    pct = lower.split("confidence:", 1)[1].strip().rstrip(").%")
                    conf[label] = float(pct) / 100.0
                except ValueError:
                    conf.setdefault(label, 0.7)
            else:
                conf.setdefault(label, 0.7)
    return conf


def format_environment_brief(result: PipelineResult) -> str:
    """Visible environment only — omit unknowns and placeholder settings."""
    env = result.scene_context.environment
    setting = (env.setting or "").replace("_", " ").strip()
    indoor = (env.indoor_outdoor or "").replace("_", " ").strip().lower()
    weather = (env.weather or "").replace("_", " ").strip().lower()
    time_of_day = (env.time_of_day or "").replace("_", " ").strip().lower()
    parts: list[str] = []
    if indoor not in _PLACEHOLDER_ENV:
        parts.append(_L(indoor).capitalize())
    if setting.lower() not in _PLACEHOLDER_ENV:
        parts.append(_L(setting))
    if weather not in _PLACEHOLDER_ENV:
        parts.append(_L(weather))
    if time_of_day not in _PLACEHOLDER_ENV and time_of_day not in {"day", "general"}:
        parts.append(_L(time_of_day))
    if not parts:
        return ""
    text = " ".join(parts)
    if not text.endswith("."):
        text = f"{text}."
    return text[0].upper() + text[1:]


def format_attributes(result: PipelineResult) -> str:
    return format_visible_attributes(result)


def format_visible_attributes(result: PipelineResult) -> str:
    """Only clearly visible appearance attributes (colors / clothing type)."""
    attrs = result.scene_context.attributes.attributes
    nodes = {node.index: node.label for node in result.scene_context.graph.nodes}
    by_object: dict[int, list[str]] = {}
    for item in attrs:
        if item.name not in _VISIBLE_ATTR_NAMES:
            continue
        value = (item.value or "").strip().lower()
        if not value or value in {"unknown", "unlikely", "none", "possible", "casual"}:
            continue
        by_object.setdefault(item.object_index, []).append(
            f"{item.name.replace('_', ' ')}: {item.value}"
        )
    lines: list[str] = []
    for index in sorted(by_object):
        label = nodes.get(index, f"object {index}")
        details = "; ".join(by_object[index][:4])
        lines.append(f"- {label}: {details}")
    return "\n".join(lines) if lines else "None verified."


def format_relationships(result: PipelineResult) -> str:
    """Meaningful, high-confidence relations only — omit spatial clutter."""
    verified = getattr(result, "verified_evidence", None)
    if verified is not None:
        rels = verified.qa_relations() or verified.narrative_relations()
        if not rels:
            return "No verified relationships"
        lines: list[str] = []
        for relation in rels:
            sub = verified.entity_by_id(relation.subject_id)
            obj = verified.entity_by_id(relation.object_id)
            subject = sub.label if sub is not None else relation.subject_id
            object_label = obj.label if obj is not None else relation.object_id
            # Keep person_N identity for multi-person role clarity.
            if sub is not None and sub.label.lower() in {"person", "man", "woman", "child"}:
                subject = relation.subject_id.replace("_", " ")
            if obj is not None and obj.label.lower() in {"person", "man", "woman", "child"}:
                object_label = relation.object_id.replace("_", " ")
            rel = relation.relation_type.lower()
            lines.append(
                f"- {_L(subject)} {_L(rel.replace('_', ' '))} {_L(object_label)} "
                f"(confidence: {relation.confidence:.0%})"
            )
        return "\n".join(lines) if lines else "No verified relationships"

    from analysis.relationships.relation_metrics import meaningful_relations

    graph = result.scene_context.graph
    relations = meaningful_relations(graph)
    if not relations:
        return "No verified relationships"
    nodes = {node.index: node.label for node in graph.nodes}
    lines: list[str] = []
    for relation in relations:
        subject = nodes.get(relation.subject_index, "?")
        obj = nodes.get(relation.object_index, "?")
        # Never expose internal instance IDs in judge-facing text.
        if re.match(r"(?i)^(?:person|horse|object)_\d+$", str(subject or "")):
            subject = str(subject).split("_", 1)[0]
        if re.match(r"(?i)^(?:person|horse|object)_\d+$", str(obj or "")):
            obj = str(obj).split("_", 1)[0]
        rel = relation.relation_type.lower()
        lines.append(
            f"- {_L(subject)} {_L(rel.replace('_', ' '))} {_L(obj)} "
            f"(confidence: {relation.confidence:.0%})"
        )
    return "\n".join(lines) if lines else "No verified relationships"


def format_activities(result: PipelineResult) -> str:
    """Strong, evidence-backed activities only — CONFIRMED / SUPPORTED tiers."""
    verified = getattr(result, "verified_evidence", None)
    if verified is not None:
        from core.contracts.verified_evidence import ActivityEvidenceLevel

        confirmed = [
            a
            for a in verified.activities
            if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.qa_safe
        ]
        supported = [
            a
            for a in verified.activities
            if a.evidence_level == ActivityEvidenceLevel.SUPPORTED and a.qa_safe
        ]
        lines: list[str] = []
        if confirmed:
            lines.append("CONFIRMED:")
            for item in confirmed:
                who = ", ".join(item.entity_ids) if item.entity_ids else "scene"
                lines.append(
                    f"- {_L(item.activity)} [{who}] (confidence: {item.confidence:.0%})"
                )
        if supported:
            lines.append("SUPPORTED:")
            for item in supported:
                who = ", ".join(item.entity_ids) if item.entity_ids else "scene"
                lines.append(
                    f"- {_L(item.activity)} [{who}] (confidence: {item.confidence:.0%})"
                )
        if not lines:
            return "No verified activities"
        return "\n".join(lines)

    items = [
        item
        for item in result.scene_context.activities.activities
        if item.activity.lower() not in _WEAK_ACTIVITIES
        and item.confidence >= _MIN_ACTIVITY_CONFIDENCE
        and "minimal interaction" not in item.rationale.lower()
    ]
    if not items:
        return "No verified activities"
    return "\n".join(
        f"- {_L(item.activity)} (confidence: {item.confidence:.0%})" for item in items
    )


def format_environment(result: PipelineResult) -> str:
    """Detailed environment for debug — still omit placeholders and enrichment fluff."""
    env = result.scene_context.environment
    lines: list[str] = []
    setting = env.setting.replace("_", " ") if env.setting else ""
    indoor = env.indoor_outdoor.replace("_", " ") if env.indoor_outdoor else ""
    if indoor and indoor.lower() not in _PLACEHOLDER_ENV:
        lines.append(f"Indoor/outdoor: {indoor}")
    if setting and setting.lower() not in _PLACEHOLDER_ENV:
        lines.append(f"Setting: {setting}")
    if env.weather and env.weather.lower() not in _PLACEHOLDER_ENV:
        lines.append(f"Weather: {env.weather}")
    if env.time_of_day and env.time_of_day.lower() not in _PLACEHOLDER_ENV:
        lines.append(f"Time of day: {env.time_of_day}")
    return "\n".join(lines) if lines else "None verified."


def format_reasoning_evidence(result: PipelineResult) -> str:
    env = result.scene_context.environment
    lines = [
        line
        for line in env.evidence[:12]
        if "minimal interaction" not in line.lower() and "photographed" not in line.lower()
    ]
    for item in result.scene_context.activities.activities[:5]:
        if item.activity.lower() in _WEAK_ACTIVITIES:
            continue
        if item.confidence < _MIN_ACTIVITY_CONFIDENCE:
            continue
        lines.append(f"Activity evidence: {item.rationale}")
    return "\n".join(f"- {line}" for line in lines) if lines else "None verified."


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.0%}"


def format_confidence_analysis(result: PipelineResult) -> str:
    report = result.quality_report
    activities = result.scene_context.activities
    return (
        f"Activity confidence: {activities.confidence:.0%}\n"
        f"Caption quality: {report.overall_quality:.0%}\n"
        f"Evidence consistency: {report.evidence_consistency:.0%}\n"
        f"Hallucination risk: {_fmt_metric(report.hallucination_risk)}\n"
        f"Object coverage: {_fmt_metric(report.object_coverage)}\n"
        f"Relationship coverage: {_fmt_metric(report.relationship_coverage)}\n"
        f"Activity coverage: {_fmt_metric(report.activity_coverage)}"
    )


def format_image_quality(result: PipelineResult) -> str:
    report = result.image_quality
    if report is None:
        return "No image quality report available."
    metrics = report.metrics
    level = (report.quality_level or "MEDIUM").upper()
    ops = list(report.enhancement_operations or ())
    status_key = getattr(report, "enhancement_status", "") or ""
    verified = bool(getattr(report, "enhancement_verified", False) or report.enhancement_applied)
    if verified and report.super_resolution_used:
        enhancement_status = "Verified (true super-resolution)"
    elif verified:
        enhancement_status = "Verified (image restoration/correction)"
    elif status_key == "ENHANCEMENT_FAILED":
        enhancement_status = "Attempted — model unavailable / failed"
    elif status_key == "ENHANCEMENT_ATTEMPTED_UNVERIFIED":
        enhancement_status = "Attempted but not verified"
    elif status_key == "ENHANCEMENT_REJECTED" or getattr(report, "enhancement_rejected", False):
        enhancement_status = "Rejected (candidate not better)"
    elif level in {"LOW", "MEDIUM"} and getattr(report, "enhancement_attempted", False):
        enhancement_status = "Attempted but not verified"
    else:
        enhancement_status = "Not required"
    operations = "\n".join(f"- {item}" for item in ops) or "None"
    reason = (
        getattr(report, "verification_reason", "")
        or getattr(report, "rejection_reason", "")
        or ""
    )
    reason_line = f"Reason: {reason}\n" if reason else ""
    ow = getattr(report, "original_width", 0) or metrics.resolution_width
    oh = getattr(report, "original_height", 0) or metrics.resolution_height
    out_w = getattr(report, "output_width", 0) or metrics.resolution_width
    out_h = getattr(report, "output_height", 0) or metrics.resolution_height
    delta = getattr(report, "quality_delta_percent", report.improvement_percent)
    return (
        f"Quality: {level}\n"
        f"Enhancement: {enhancement_status}\n"
        f"Verified: {'Yes' if verified else 'No'}\n"
        f"Original resolution: {ow}×{oh}\n"
        f"Output resolution: {out_w}×{out_h}\n"
        f"Estimated quality: {metrics.estimated_quality:.0%}\n"
        f"Brightness: {metrics.brightness:.2f} | Contrast: {metrics.contrast:.2f}\n"
        f"Blur score: {metrics.blur_score:.2f} | Motion blur: {metrics.motion_blur_score:.2f}\n"
        f"Noise: {metrics.noise_score:.2f} | Sharpness: {metrics.sharpness:.2f}\n"
        f"Dynamic range: {metrics.dynamic_range:.2f}\n"
        f"Compression / JPEG artifacts: {metrics.compression_artifact_score:.2f}\n"
        f"Exposure: {metrics.exposure_score:.2f} | White balance: {metrics.white_balance_score:.2f}\n"
        f"Enhancement applied: {'Yes' if report.enhancement_applied else 'No'}\n"
        f"True super-resolution: {'Yes' if report.super_resolution_used else 'No'}\n"
        f"SR model: {getattr(report, 'sr_model', '') or '—'}\n"
        f"SR scale: {getattr(report, 'sr_scale', 1)}× | Device: {getattr(report, 'sr_device', '') or '—'}\n"
        f"SR input→output: {getattr(report, 'sr_input_size', '') or '—'} → {getattr(report, 'sr_output_size', '') or '—'}\n"
        f"Before quality: {report.before_quality:.0%} → After: {report.after_quality:.0%}\n"
        f"Quality delta: {delta:+.1f}%\n"
        f"Reported improvement: {report.improvement_percent:.1f}%\n"
        f"Processing time: {report.processing_time_ms:.1f} ms\n"
        f"{reason_line}\n"
        f"Operations:\n{operations}"
    )


def format_object_details(result: PipelineResult) -> str:
    return format_visible_attributes(result) or ""


def format_color_palette(result: PipelineResult) -> str:
    colors: list[str] = []
    for item in result.scene_context.attributes.attributes:
        if item.name in {"dominant_color", "color", "hair_color", "shirt_color", "pants_color"}:
            value = (item.value or "").strip().lower()
            if value in {"unknown", "unlikely", "none", ""}:
                continue
            entry = f"{item.name.replace('_', ' ')}: {item.value}"
            if entry not in colors:
                colors.append(entry)
    if not colors:
        return ""
    return "\n".join(f"- {entry}" for entry in colors[:12])


def format_scene_graph(result: PipelineResult) -> str:
    graph = result.scene_context.graph
    if not graph.nodes:
        return ""
    lines = ["Nodes:"]
    for node in graph.nodes:
        lines.append(
            f"- [{node.index}] {node.label} @ {node.position_zone} "
            f"(area {node.bounding_box_area_ratio:.0%})"
        )
    meaningful = format_relationships(result)
    if meaningful:
        lines.append("\nRelations:")
        lines.append(meaningful)
    return "\n".join(lines)


def format_caption_confidence(result: PipelineResult) -> str:
    from language.validation.sentence_evidence import SentenceEvidenceAnalyzer

    caption = result.caption.narrative_full or result.caption.text
    analyzed = SentenceEvidenceAnalyzer().analyze(caption, result.scene_context)
    if not analyzed:
        return ""
    lines = []
    for item in analyzed[:12]:
        sources = ", ".join(item.sources)
        lines.append(f"- ({item.confidence:.0%}) [{sources}] {item.sentence[:120]}")
    return "\n".join(lines)


def format_quality_report(result: PipelineResult) -> str:
    report = result.quality_report
    notes = "\n".join(f"- {note}" for note in report.notes) if report.notes else "None"
    qa_status = "Passed" if result.qa_passed else "Recovered via fallback"
    return (
        f"Overall quality: {report.overall_quality:.0%}\n"
        f"Hallucination risk: {_fmt_metric(report.hallucination_risk)}\n"
        f"Evidence consistency: {report.evidence_consistency:.0%}\n"
        f"Object coverage: {_fmt_metric(report.object_coverage)}\n"
        f"Relationship coverage: {_fmt_metric(report.relationship_coverage)}\n"
        f"Activity coverage: {_fmt_metric(report.activity_coverage)}\n"
        f"Grammar: {report.grammar_score:.0%} | Fluency: {report.fluency_score:.0%}\n"
        f"QA: {qa_status}\n"
        f"Notes:\n{notes}"
    )


def format_execution_metrics(result: PipelineResult) -> str:
    metrics = result.metrics
    stage_lines = "\n".join(
        f"- {item.stage.value}: {item.duration_ms:.1f} ms" for item in metrics.stage_metrics
    ) or "None"
    return (
        f"Total duration: {metrics.total_duration_ms:.1f} ms\n"
        f"Peak RAM: {metrics.peak_ram_mb:.1f} MB | Peak VRAM: {metrics.peak_vram_mb:.1f} MB\n"
        f"Object instances detected: {metrics.objects_detected}\n"
        f"Semantic relationships inferred: {metrics.relationships_inferred}\n"
        f"Activities inferred: {metrics.activities_inferred}\n"
        f"Scene graph: {metrics.scene_graph_nodes} nodes / "
        f"{metrics.scene_graph_edges} raw edges "
        f"({metrics.relationships_inferred} semantic)\n"
        f"Caption quality score: {metrics.caption_quality_score:.0%}\n"
        f"Recovery events: {metrics.recovery_events} | Fallback events: {metrics.fallback_events}\n"
        f"Competition mode: {'Yes' if metrics.competition_mode else 'No'}\n"
        f"QA passed: {'Yes' if metrics.qa_passed else 'No'}\n\n"
        f"Stage timings:\n{stage_lines}"
    )
