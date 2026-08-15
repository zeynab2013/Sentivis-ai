"""Build compact evidence prompts for Ollama high-level semantic synthesis."""

from __future__ import annotations

from core.contracts.analysis import ActivityHints, AttributeSet, EnvironmentInfo, SceneContext, SceneGraph
from core.contracts.language import VisualObservations

# Keep instructions short — no chain-of-thought, no evidence echo.
SYSTEM_PROMPT = (
    "You synthesize a verified scene into JSON. "
    "Use ONLY the evidence provided. Do not invent objects, people, colors, or activities. "
    "Do not explain your reasoning. "
    "Write captions that are natural, coherent, and informative — never inventory lists "
    "or detector stubs like 'A person is near a table'. "
    "Scale detail to scene complexity: simple scenes stay concise; richer scenes use "
    "multiple flowing sentences covering subjects, actions, attributes, and place. "
    "Do not pad. Do not start with 'The image shows' or 'In this image'. "
    "Return JSON with keys: scene_explanation, rejected_conclusions, "
    "contradictions_resolved, caption."
)

JSON_SCHEMA = (
    '{"scene_explanation":"1-2 sentences",'
    '"rejected_conclusions":[],'
    '"contradictions_resolved":[],'
    '"caption":"fluent multi-sentence caption grounded only in verified facts '
    '(~40-120 words for moderate/complex scenes; shorter when the scene is minimal)"}'
)


def build_semantic_reasoning_prompt(
    context: SceneContext,
    observations: VisualObservations | None,
) -> str:
    """Compose a compact user prompt from verified scene context."""
    sections = [
        "Return JSON matching:",
        JSON_SCHEMA,
        "Write caption directly. Do not repeat the evidence list.",
        "Prefer natural synthesis over one sentence per relation.",
        "Include verified colors/clothing and distinct people when present.",
        "Treat spatial 'near/beside' as layout, not holding/talking/using.",
        "",
        _scene_graph_section(context.graph),
        _attributes_section(context.attributes, context.graph),
        _relationships_section(context.graph),
        _verified_activities_section(context.activities),
        _environment_section(context.environment),
        _observation_section(observations),
    ]
    return "\n".join(sections)


def _scene_graph_section(graph: SceneGraph) -> str:
    if not graph.nodes:
        return "OBJECTS:\nnone"
    # Cap object list — richest nodes first by area.
    nodes = sorted(graph.nodes, key=lambda n: n.bounding_box_area_ratio, reverse=True)[:12]
    lines = ["OBJECTS:"]
    for node in nodes:
        lines.append(f"- {node.index}:{node.label}@{node.position_zone}")
    return "\n".join(lines)


def _attributes_section(attributes: AttributeSet, graph: SceneGraph) -> str:
    if not attributes.attributes:
        return "ATTRIBUTES:\nnone"
    labels = {node.index: node.label for node in graph.nodes}
    # Keep only high-signal attributes.
    useful = {
        "clothing_type",
        "shirt_color",
        "pants_color",
        "shoes_color",
        "hair_color",
        "dominant_color",
        "color",
        "pose",
    }
    lines = ["ATTRIBUTES:"]
    count = 0
    for attr in attributes.attributes:
        if attr.name not in useful:
            continue
        value = (attr.value or "").strip().lower()
        if not value or value in {"unknown", "none", "unlikely", "possible"}:
            continue
        label = labels.get(attr.object_index, str(attr.object_index))
        lines.append(f"- {label}.{attr.name}={attr.value}")
        count += 1
        if count >= 16:
            break
    if count == 0:
        return "ATTRIBUTES:\nnone"
    return "\n".join(lines)


def _relationships_section(graph: SceneGraph) -> str:
    semantic = [
        r
        for r in graph.relations
        if r.relation_type not in {"left_of", "right_of", "above", "below", "far", "near"}
        and r.confidence >= 0.55
    ]
    relations = semantic[:12]
    if not relations:
        return "RELATIONSHIPS:\nnone"
    labels = {node.index: node.label for node in graph.nodes}
    lines = ["RELATIONSHIPS:"]
    for rel in relations:
        subj = labels.get(rel.subject_index, "?")
        obj = labels.get(rel.object_index, "?")
        lines.append(f"- {subj} {rel.relation_type.replace('_', ' ')} {obj}")
    return "\n".join(lines)


def _verified_activities_section(activities: ActivityHints) -> str:
    if not activities.activities:
        return "ACTIVITIES:\nnone"
    lines = ["ACTIVITIES:"]
    for item in activities.activities[:6]:
        if item.confidence < 0.55:
            continue
        lines.append(f"- {item.activity} ({item.confidence:.2f})")
    if len(lines) == 1:
        return "ACTIVITIES:\nnone"
    return "\n".join(lines)


def _environment_section(environment: EnvironmentInfo) -> str:
    return (
        "ENVIRONMENT:\n"
        f"- type={environment.scene_type}; setting={environment.setting}; "
        f"io={environment.indoor_outdoor}; crowd={environment.crowd_level}"
    )


def _observation_section(observations: VisualObservations | None) -> str:
    if observations is None:
        return "VLM_HINT:\nnone"
    # One short hint only — avoid duplicating the full observation dump.
    hint = (observations.raw_caption.text or "").strip()
    if not hint and observations.observations:
        hint = observations.observations[0]
    if not hint:
        return "VLM_HINT:\nnone"
    if len(hint) > 180:
        hint = hint[:177] + "..."
    return f"VLM_HINT:\n- {hint}"
