"""Build deterministic structured prompts for reasoning."""

from core.contracts.analysis import SceneContext
from core.contracts.language import Prompt, VisualObservations
from core.logging import get_logger

logger = get_logger(__name__)

_REASONING_RULES = (
    "Every sentence must be supported by the evidence sections below.",
    "If evidence is insufficient, state uncertainty instead of inventing details.",
    "Do not introduce objects, activities, or relationships that are not listed.",
    "Prefer omission over fabrication.",
    "Do not contradict the scene graph or verified relationships.",
)


class PromptBuilder:
    """Constructs deterministic Gemma prompts from structured semantic data."""

    def build(
        self,
        context: SceneContext,
        observations: VisualObservations | None = None,
    ) -> Prompt:
        system = (
            "You are Sentivis AI, a visual understanding assistant. "
            "Describe meaning, not pixels. Reason only over the structured evidence provided. "
            + " ".join(_REASONING_RULES)
        )
        sections = [
            self._scene_graph_section(context),
            self._relationship_section(context),
            self._activity_section(context),
            self._context_section(context),
            self._blip_section(observations),
            "Write one cohesive caption (4-8 detailed sentences, about 80-140 words for complex "
            "scenes) that is accurate, natural, and evidence-based. Prefer detail over brevity.",
        ]
        user = "\n\n".join(section for section in sections if section)
        logger.debug("Built deterministic reasoning prompt")
        return Prompt(system=system, user=user)

    def _scene_graph_section(self, context: SceneContext) -> str:
        if not context.graph.nodes:
            return "SCENE GRAPH:\n- No verified objects."
        lines = ["SCENE GRAPH:"]
        for node in sorted(context.graph.nodes, key=lambda item: item.index):
            lines.append(
                f"- [{node.object_id}] {node.label} "
                f"(zone={node.position_zone}, area_ratio={node.bounding_box_area_ratio:.3f})"
            )
        return "\n".join(lines)

    def _relationship_section(self, context: SceneContext) -> str:
        if not context.graph.relations:
            return "RELATIONSHIPS:\n- No verified relationships."
        lines = ["RELATIONSHIPS:"]
        nodes = {node.index: node.label for node in context.graph.nodes}
        for relation in sorted(
            context.graph.relations,
            key=lambda item: (item.subject_index, item.object_index, item.relation_type),
        ):
            subject = nodes.get(relation.subject_index, f"object_{relation.subject_index}")
            obj = nodes.get(relation.object_index, f"object_{relation.object_index}")
            lines.append(
                f"- {subject} {relation.relation_type.replace('_', ' ')} {obj} "
                f"(confidence={relation.confidence:.2f})"
            )
        return "\n".join(lines)

    def _activity_section(self, context: SceneContext) -> str:
        if not context.activities.activities:
            return "ACTIVITIES:\n- No verified activities."
        lines = ["ACTIVITIES:"]
        for item in context.activities.activities:
            nodes = ", ".join(str(index) for index in item.supporting_node_indices) or "none"
            relations = ", ".join(item.supporting_relation_types) or "none"
            lines.append(
                f"- {item.activity} (confidence={item.confidence:.2f}; "
                f"nodes={nodes}; relations={relations}; rationale={item.rationale})"
            )
        return "\n".join(lines)

    def _context_section(self, context: SceneContext) -> str:
        env = context.environment
        evidence = "; ".join(env.evidence) if env.evidence else "none"
        return (
            "CONTEXT:\n"
            f"- scene_type={env.scene_type}\n"
            f"- setting={env.setting}\n"
            f"- indoor_outdoor={env.indoor_outdoor}\n"
            f"- social_context={env.social_context}\n"
            f"- crowd_level={env.crowd_level}\n"
            f"- scene_complexity={env.scene_complexity}\n"
            f"- time_of_day={env.time_of_day}\n"
            f"- weather={env.weather}\n"
            f"- evidence={evidence}\n"
            f"- spatial_summary={context.spatial_summary}"
        )

    def _blip_section(self, observations: VisualObservations | None) -> str:
        if observations is None:
            return "BLIP OBSERVATIONS:\n- Visual description unavailable; rely on structured evidence only."
        lines = ["BLIP OBSERVATIONS (non-final visual hints):"]
        for item in observations.observations:
            lines.append(f"- observation: {item}")
        for item in observations.object_attributes:
            lines.append(f"- verified_attribute: {item}")
        for item in observations.candidate_descriptions:
            lines.append(f"- candidate_description: {item}")
        lines.append(f"- confidence={observations.confidence:.2f}")
        return "\n".join(lines)
