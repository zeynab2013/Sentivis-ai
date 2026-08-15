"""Map BLIP output and scene context into structured visual observations."""

import re

from core.contracts.analysis import SceneContext
from core.contracts.language import RawCaption, VisualObservations
from core.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class BlipObservationMapper:
    """Combines BLIP visual description with verified scene graph data."""

    def map(self, raw_caption: RawCaption, context: SceneContext) -> VisualObservations:
        """Build visual observations without making final semantic decisions."""
        blip_clauses = self._split_clauses(raw_caption.text)
        verified_objects = tuple(sorted({node.label for node in context.graph.nodes}))
        verified_attributes = self._verified_attributes(context)
        observations = tuple(
            clause
            for clause in blip_clauses
            if clause and not self._contradicts_graph(clause, verified_objects)
        )
        if not observations and raw_caption.text.strip():
            observations = (raw_caption.text.strip(),)

        context_summary = (
            f"Scene contains {context.object_count} verified objects: "
            f"{', '.join(verified_objects) if verified_objects else 'none'}."
        )
        clothing_hints = tuple(
            hint for hint in verified_attributes if any(
                key in hint for key in ("clothing_type=", "shirt_color=", "pants_color=", "crop_description=")
            )
        )[:8]
        candidate_descriptions = tuple(
            dict.fromkeys(
                item
                for item in (
                    raw_caption.text.strip(),
                    context_summary,
                    context.spatial_summary,
                    *clothing_hints,
                )
                if item
            )
        )
        logger.debug(
            "Mapped BLIP output to %d observations and %d attribute hints",
            len(observations),
            len(verified_attributes),
        )
        return VisualObservations(
            observations=observations,
            object_attributes=verified_attributes,
            candidate_descriptions=candidate_descriptions,
            confidence=raw_caption.confidence,
            raw_caption=raw_caption,
        )

    def _verified_attributes(self, context: SceneContext) -> tuple[str, ...]:
        hints: list[str] = []
        labels_by_index = {node.index: node.label for node in context.graph.nodes}
        for attribute in context.attributes.attributes:
            label = labels_by_index.get(attribute.object_index, f"object_{attribute.object_index}")
            hints.append(f"{label}: {attribute.name}={attribute.value}")
        return tuple(sorted(hints))

    def _split_clauses(self, text: str) -> tuple[str, ...]:
        cleaned = text.strip()
        if not cleaned:
            return ()
        parts = _SENTENCE_SPLIT.split(cleaned)
        return tuple(part.strip() for part in parts if part.strip())

    def _contradicts_graph(self, clause: str, verified_objects: tuple[str, ...]) -> bool:
        lower = clause.lower()
        for label in verified_objects:
            if label.lower() in lower:
                return False
        invented = self._extract_candidate_labels(clause)
        return bool(invented) and not verified_objects

    def _extract_candidate_labels(self, clause: str) -> tuple[str, ...]:
        tokens = re.findall(r"[a-zA-Z]{3,}", clause.lower())
        stopwords = {"the", "and", "with", "that", "this", "there", "their", "from", "into"}
        return tuple(token for token in tokens if token not in stopwords)
