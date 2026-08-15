"""SceneReasoner — merges multimodal evidence with multi-source confidence fusion."""

from __future__ import annotations

from collections import defaultdict

from analysis.ocr.text_extractor import OcrExtraction
from analysis.pose.pose_estimator import PoseEstimate
from core.contracts.analysis import SceneContext
from core.contracts.image_quality import ImageQualityReport
from core.contracts.language import VisualObservations
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from core.logging import get_logger

logger = get_logger(__name__)

_PERSON = {"person", "people", "man", "woman", "child"}
_WEAK_SPATIAL = {"near", "next_to", "above", "below", "left_of", "right_of", "far", "overlapping"}
_SPORT_LABELS = {
    "tennis racket",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "skis",
    "snowboard",
    "frisbee",
    "kite",
}
_FORMAL_VALUES = {"formal suit", "blazer", "formal pants", "formal shoes", "formal"}
_CLOTHING_KEYS = {
    "shirt_color",
    "pants_color",
    "shoes_color",
    "hair_color",
    "hair_length",
    "hairstyle",
    "clothing_type",
    "clothing_style",
    "sleeve_length",
    "jacket",
    "coat",
    "dress",
    "hoodie",
    "blazer",
    "sweater",
    "skirt",
    "jeans",
    "shorts",
    "footwear_type",
    "backpack",
    "handbag",
    "glasses",
    "sunglasses",
    "hat",
    "cap",
    "watch",
    "necklace",
    "earrings",
    "accessories",
    "secondary_color",
    # Promote person color attrs into evidence (not just garment type keys).
    "clothing_color",
    "dominant_color",
    "color",
}
_COLOR_KEYS = {
    "shirt_color",
    "pants_color",
    "shoes_color",
    "hair_color",
    "secondary_color",
    "dominant_color",
    "color",
    "clothing_color",
}
_ACCESSORY_KEYS = {
    "backpack",
    "handbag",
    "glasses",
    "sunglasses",
    "hat",
    "cap",
    "watch",
    "necklace",
    "earrings",
    "accessories",
}
_CLOTHING_TYPE_KEYS = {
    "clothing_type",
    "clothing_style",
    "sleeve_length",
    "footwear_type",
    "jacket",
    "coat",
    "dress",
    "hoodie",
    "blazer",
    "sweater",
    "skirt",
    "jeans",
    "shorts",
}
_OBJECT_KEYS = {
    "color",
    "dominant_color",
    "secondary_color",
    "visibility",
    "occlusion",
}
_SEMANTIC_IMPORTANCE = {
    "person": 1.0,
    "people": 1.0,
    "man": 1.0,
    "woman": 1.0,
    "child": 1.0,
    "dog": 0.88,
    "cat": 0.88,
    "horse": 0.86,
    "bird": 0.5,
    "elephant": 0.85,
    "bear": 0.84,
    "zebra": 0.82,
    "giraffe": 0.82,
    "cow": 0.78,
    "sheep": 0.75,
    "bicycle": 0.72,
    "motorcycle": 0.74,
    "car": 0.7,
    "bus": 0.72,
    "truck": 0.72,
    "train": 0.78,
    "boat": 0.7,
    "airplane": 0.95,
    "fire": 0.96,
    "smoke": 0.9,
    "tennis racket": 0.8,
    "sports ball": 0.76,
    "baseball bat": 0.76,
    "baseball glove": 0.7,
    "skateboard": 0.74,
    "surfboard": 0.74,
    "skis": 0.72,
    "snowboard": 0.72,
    "frisbee": 0.68,
    "kite": 0.65,
    "umbrella": 0.55,
    "backpack": 0.42,
    "handbag": 0.42,
    "suitcase": 0.45,
    "chair": 0.38,
    "couch": 0.48,
    "bed": 0.5,
    "dining table": 0.55,
    "bench": 0.4,
    "toilet": 0.28,
    "sink": 0.4,
    "refrigerator": 0.58,
    "oven": 0.55,
    "microwave": 0.4,
    "tv": 0.4,
    "laptop": 0.5,
    "cell phone": 0.35,
    "book": 0.3,
    "clock": 0.28,
    "vase": 0.28,
    "bottle": 0.22,
    "cup": 0.24,
    "bowl": 0.26,
    "banana": 0.3,
    "apple": 0.3,
    "orange": 0.3,
    "broccoli": 0.25,
    "carrot": 0.25,
    "pizza": 0.4,
    "donut": 0.3,
    "cake": 0.4,
    "potted plant": 0.32,
    "traffic light": 0.38,
    "fire hydrant": 0.28,
    "stop sign": 0.42,
    "parking meter": 0.25,
    "tie": 0.3,
}
# Small props that must never outrank a true scene anchor by count alone.
_PROP_LABELS = frozenset(
    {
        "bottle",
        "cup",
        "bowl",
        "book",
        "vase",
        "clock",
        "cell phone",
        "remote",
        "mouse",
        "keyboard",
        "tie",
        "toothbrush",
        "spoon",
        "fork",
        "knife",
    }
)
_SCENE_ANCHOR_LABELS = frozenset(
    {
        "airplane",
        "train",
        "bus",
        "truck",
        "car",
        "boat",
        "horse",
        "elephant",
        "fire",
        "smoke",
        "dining table",
        "refrigerator",
        "oven",
        "couch",
        "bed",
    }
)

# Fire/smoke are not COCO classes — recover from clear VLM language only.
_HAZARD_TOKENS: dict[str, tuple[str, ...]] = {
    "fire": ("fire", "flame", "flames", "burning", "blaze", "campfire", "bonfire", "wildfire"),
    "smoke": ("smoke", "smoky", "smokes", "smoldering"),
}

# Per-family calibration priors (historical weights from measured runs).
_FAMILY_PRIOR = {
    "color": 0.62,
    "clothing": 0.58,
    "accessory": 0.48,
    "relationship": 0.60,
    "activity": 0.55,
    "ocr": 0.50,
    "object": 0.55,
    "environment": 0.58,
    "pose": 0.57,
}

class SceneReasoner:
    """Brain of the system: multi-source confidence fusion, then keep correct facts."""

    def reason(
        self,
        context: SceneContext,
        *,
        observations: VisualObservations | None = None,
        poses: tuple[PoseEstimate, ...] = (),
        ocr: OcrExtraction | None = None,
        image_quality: ImageQualityReport | None = None,
    ) -> SceneUnderstanding:
        facts: list[EvidenceFact] = []
        discarded = 0
        contradictions = 0

        labels = {node.index: node.label for node in context.graph.nodes}
        attr_map: dict[int, dict[str, str]] = defaultdict(dict)
        for attribute in context.attributes.attributes:
            attr_map[attribute.object_index][attribute.name] = attribute.value

        vlm_text = ""
        vlm_conf = 0.0
        if observations is not None and observations.raw_caption.text.strip():
            vlm_text = observations.raw_caption.text.strip().lower()
            vlm_conf = float(observations.confidence)
            for hint in observations.object_attributes:
                vlm_text += " " + hint.lower()
            for hint in observations.candidate_descriptions:
                vlm_text += " " + hint.lower()

        pose_by_index = {pose.object_index: pose for pose in poses}
        has_sport = any(label.lower() in _SPORT_LABELS for label in labels.values())
        quality_scale = 1.0
        if image_quality is not None and image_quality.metrics.estimated_quality < 0.35:
            quality_scale = 0.9

        _accessory_floor = {
            "backpack": 0.78,
            "handbag": 0.75,
            "suitcase": 0.65,
            "umbrella": 0.68,
            "tie": 0.75,
            "cell phone": 0.72,
            "remote": 0.70,
            "mouse": 0.65,
        }
        for index, label in labels.items():
            conf = self._parse_confidence(attr_map.get(index, {}).get("confidence", "0.6"))
            values = attr_map.get(index, {})
            mask_score = 0.85 if values.get("segmentation") == "mask" else 0.42
            subject = self._subject_name(label, index, labels)
            label_l = label.lower()
            floor = _accessory_floor.get(label_l)
            if floor is not None and conf < floor:
                discarded += 1
                logger.info(
                    "SceneReasoner discarded accessory label=%s conf=%.3f floor=%.2f",
                    label_l,
                    conf,
                    floor,
                )
                continue

            keep_object, obj_conf, sources = self._fuse(
                family="object",
                detection=conf,
                segmentation=mask_score,
                vlm_agreement=self._vlm_mentions(label, vlm_text, vlm_conf),
                scene=0.65,
                relationship=0.55,
                spatial=self._spatial_score(values),
                prior=_FAMILY_PRIOR["object"],
            )
            if not keep_object and label_l not in _PERSON:
                discarded += 1
                continue
            facts.append(EvidenceFact(subject, "is", label, obj_conf * quality_scale, "yolo"))

            if label.lower() in _PERSON:
                for key in _CLOTHING_KEYS:
                    value = values.get(key)
                    if not value or value in {
                        "unknown",
                        "none detected",
                        "not_applicable",
                        "unlikely",
                        "possible",
                        "casual",
                    }:
                        discarded += 1
                        continue
                    family = self._family_for_key(key)
                    conflict = self._conflict_penalty(key, value, values, has_sport=has_sport, vlm_text=vlm_text)
                    vlm_agree = self._vlm_mentions(value, vlm_text, vlm_conf)
                    attr_agree = self._attribute_agreement(key, value, values)
                    rel_score = self._relationship_consistency_for_person(
                        index, key, value, context, pose_by_index.get(index)
                    )
                    scene_score = self._scene_consistency(key, value, context, has_sport=has_sport)
                    keep, fused, n_sources = self._fuse(
                        family=family,
                        detection=conf,
                        segmentation=mask_score if family in {"color", "clothing"} else 0.5,
                        vlm_agreement=vlm_agree,
                        scene=scene_score,
                        relationship=rel_score,
                        spatial=self._spatial_score(values),
                        prior=_FAMILY_PRIOR[family],
                        attribute_agreement=attr_agree,
                        conflict=conflict,
                    )
                    if not keep:
                        discarded += 1
                        continue
                    facts.append(
                        EvidenceFact(subject, key, value, min(0.96, fused * quality_scale), "attributes")
                    )
            else:
                for key in _OBJECT_KEYS:
                    value = values.get(key)
                    if not value or value in {"unknown", "not_applicable"}:
                        continue
                    family = "color" if key.endswith("color") or key == "color" else "object"
                    vlm_agree = self._vlm_mentions(value, vlm_text, vlm_conf)
                    keep, fused, _n = self._fuse(
                        family=family,
                        detection=conf,
                        segmentation=mask_score,
                        vlm_agreement=vlm_agree,
                        scene=0.6,
                        relationship=0.55,
                        spatial=self._spatial_score(values),
                        prior=_FAMILY_PRIOR[family],
                        attribute_agreement=0.55,
                    )
                    if not keep:
                        discarded += 1
                        continue
                    facts.append(
                        EvidenceFact(subject, key, value, min(0.95, fused * quality_scale), "attributes")
                    )

        for pose in poses:
            keep, fused, _n = self._fuse(
                family="pose",
                detection=pose.confidence,
                segmentation=0.5,
                vlm_agreement=self._vlm_mentions(pose.action, vlm_text, vlm_conf),
                scene=0.6,
                relationship=0.65 if pose.action != "unknown" else 0.45,
                spatial=0.6,
                prior=_FAMILY_PRIOR["pose"],
            )
            if not keep:
                discarded += 1
                continue
            subject = self._subject_name(labels.get(pose.object_index, "person"), pose.object_index, labels)
            if pose.pose != "unknown":
                facts.append(
                    EvidenceFact(subject, "pose", pose.pose, fused, pose.source, pose.processing_time_ms)
                )
            if pose.action != "unknown":
                facts.append(
                    EvidenceFact(
                        subject,
                        "action",
                        pose.action,
                        fused,
                        pose.source,
                        pose.processing_time_ms,
                    )
                )

        for relation in context.graph.relations:
            subject_label = labels.get(relation.subject_index, "object")
            object_label = labels.get(relation.object_index, "object")
            subject = self._subject_name(subject_label, relation.subject_index, labels)
            obj = self._subject_name(object_label, relation.object_index, labels)
            det_a = self._parse_confidence(attr_map.get(relation.subject_index, {}).get("confidence", "0.6"))
            det_b = self._parse_confidence(attr_map.get(relation.object_index, {}).get("confidence", "0.6"))
            detection = min(det_a, det_b, relation.confidence)
            semantic = relation.relation_type not in _WEAK_SPATIAL
            vlm_agree = self._vlm_mentions(
                f"{subject_label} {relation.relation_type.replace('_', ' ')} {object_label}",
                vlm_text,
                vlm_conf,
            )
            activity_agree = 0.7
            for activity in context.activities.activities:
                if relation.subject_index in activity.supporting_node_indices and (
                    relation.relation_type in activity.supporting_relation_types
                    or relation.relation_type.replace("_", " ") in activity.activity
                ):
                    activity_agree = max(activity_agree, 0.55 + 0.4 * activity.confidence)
            keep, fused, n_sources = self._fuse(
                family="relationship",
                detection=detection,
                segmentation=0.55,
                vlm_agreement=vlm_agree,
                scene=0.7 if semantic else 0.45,
                relationship=activity_agree,
                spatial=0.7 if semantic else 0.4,
                prior=_FAMILY_PRIOR["relationship"],
                attribute_agreement=0.65 if semantic else 0.4,
            )
            if (
                relation.relation_type in {"near", "next_to", "standing_beside"}
                and subject_label.lower() in _PERSON
                and object_label.lower() in _PERSON
                and n_sources < 2
            ):
                discarded += 1
                continue
            if not keep:
                discarded += 1
                continue
            facts.append(
                EvidenceFact(subject, relation.relation_type, obj, fused * quality_scale, "relationships")
            )

        for activity in context.activities.activities:
            rel_support = 0.78 if activity.supporting_relation_types else 0.45
            # Multi-model vote: YOLO nodes + relations + VLM mention + scene prior.
            keep, fused, n_sources = self._fuse(
                family="activity",
                detection=activity.confidence,
                segmentation=0.55 if activity.supporting_node_indices else 0.45,
                vlm_agreement=self._vlm_mentions(activity.activity, vlm_text, vlm_conf),
                scene=0.7,
                relationship=rel_support,
                spatial=0.62,
                prior=_FAMILY_PRIOR["activity"],
                attribute_agreement=0.6 if len(activity.supporting_node_indices) >= 2 else 0.5,
            )
            if not keep:
                discarded += 1
                continue
            # Thin single-source activities stay uncertain — do not promote hallucinations.
            if n_sources <= 1 and activity.confidence < 0.74:
                discarded += 1
                continue
            facts.append(EvidenceFact("scene", "activity", activity.activity, fused, "activity"))

        env = context.environment
        for key, value in (
            ("indoor_outdoor", env.indoor_outdoor),
            ("scene_type", env.scene_type),
            ("weather", env.weather),
            ("time_of_day", env.time_of_day),
            ("crowd_level", env.crowd_level),
            ("setting", env.setting),
        ):
            if not value or value in {
                "unknown",
                "general",
                "general scene",
                "none",
                "photographed scene",
                "everyday environment",
            }:
                continue
            vlm_agree = self._vlm_mentions(value, vlm_text, vlm_conf)
            keep, fused, _n = self._fuse(
                family="environment",
                detection=0.7,
                segmentation=0.5,
                vlm_agreement=vlm_agree,
                scene=0.75,
                relationship=0.55,
                spatial=0.6,
                prior=_FAMILY_PRIOR["environment"],
            )
            if keep:
                facts.append(EvidenceFact("scene", key, value, fused, "environment"))

        ocr_texts: tuple[str, ...] = ()
        if ocr is not None and ocr.texts:
            ocr_texts = ocr.texts
            for text in ocr_texts[:6]:
                keep, fused, _n = self._fuse(
                    family="ocr",
                    detection=ocr.confidence,
                    segmentation=0.5,
                    vlm_agreement=self._vlm_mentions(text, vlm_text, vlm_conf),
                    scene=0.55,
                    relationship=0.5,
                    spatial=0.5,
                    prior=_FAMILY_PRIOR["ocr"],
                )
                if keep:
                    facts.append(EvidenceFact("scene", "visible_text", text, fused, ocr.source))
                else:
                    discarded += 1

        # Fire/smoke are not COCO YOLO classes — recover them from VLM evidence when clear.
        hazard_facts = self._hazard_facts_from_vlm(vlm_text, vlm_conf)
        facts.extend(hazard_facts)
        if hazard_facts:
            logger.info(
                "SceneReasoner added hazard evidence: %s",
                ", ".join(f"{f.value}:{f.confidence:.2f}" for f in hazard_facts),
            )

        facts, contradictions = self._resolve_contradictions(facts)
        facts = self._dedupe(facts)
        facts, sport_discards = self._suppress_formal_wear_near_sports(facts, labels)
        discarded += sport_discards
        ranked = self._rank_subjects(facts, labels, context)
        if observations is not None and observations.raw_caption.text.strip():
            facts.append(
                EvidenceFact(
                    "vlm",
                    "observation",
                    observations.raw_caption.text.strip(),
                    observations.confidence,
                    observations.raw_caption.source,
                )
            )

        brief = self._build_brief(facts, ranked, ocr_texts)
        overall = float(sum(f.confidence for f in facts) / max(1, len(facts)))
        understanding = SceneUnderstanding(
            facts=tuple(facts),
            ranked_subjects=ranked,
            environment_keys=tuple(
                f"{f.predicate}={f.value}" for f in facts if f.subject == "scene" and f.predicate != "visible_text"
            ),
            activity_keys=tuple(f.value for f in facts if f.predicate == "activity"),
            ocr_text=ocr_texts,
            evidence_brief=brief,
            overall_confidence=overall,
            discarded_count=discarded,
            contradictions_resolved=contradictions,
        )
        logger.info(
            "SceneReasoner facts=%d discarded=%d contradictions=%d confidence=%.2f",
            len(facts),
            discarded,
            contradictions,
            overall,
        )
        return understanding

    def _family_for_key(self, key: str) -> str:
        if key in _COLOR_KEYS or key.endswith("_color"):
            return "color"
        if key in _ACCESSORY_KEYS:
            return "accessory"
        if key in _CLOTHING_TYPE_KEYS or key in {"hairstyle", "hair_length"}:
            return "clothing"
        return "object"

    def _fuse(
        self,
        *,
        family: str,
        detection: float,
        segmentation: float,
        vlm_agreement: float,
        scene: float,
        relationship: float,
        spatial: float,
        prior: float,
        attribute_agreement: float = 0.5,
        conflict: float = 0.0,
    ) -> tuple[bool, float, int]:
        """Weighted multi-source fusion. Returns (keep, fused_confidence, independent_sources)."""
        channels = [
            ("detection", detection, 0.22),
            ("segmentation", segmentation, 0.14),
            ("vlm", vlm_agreement, 0.16),
            ("scene", scene, 0.12),
            ("relationship", relationship, 0.12),
            ("spatial", spatial, 0.10),
            ("attribute", attribute_agreement, 0.10),
            ("prior", prior, 0.04),
        ]
        # Count independent supportive sources (not priors).
        # Crop-derived clothing/color is grounded in detection+pixels; 0.55 is enough
        # to count segmentation / attribute agreement as a real source.
        supportive = [
            name
            for name, score, _weight in channels
            if name != "prior" and score >= 0.55
        ]
        n_sources = len(supportive)
        total_w = sum(weight for _n, _s, weight in channels)
        fused = sum(score * weight for _n, score, weight in channels) / max(total_w, 1e-6)
        fused = max(0.0, min(0.98, fused - conflict))

        keep = self._accept(family, fused, n_sources, detection=detection)
        return keep, fused, n_sources

    def _accept(self, family: str, fused: float, n_sources: int, *, detection: float) -> bool:
        """Per-attribute acceptance — crop-grounded clothing/color must not vanish."""
        if n_sources >= 3:
            return fused >= 0.48
        if n_sources == 2:
            if family == "color":
                return fused >= 0.50
            if family == "clothing":
                return fused >= 0.52
            if family == "accessory":
                return fused >= 0.56
            if family == "relationship":
                return fused >= 0.52
            return fused >= 0.53
        # ROOT CAUSE FIX: clothing/color from ClothingAnalyzer+mask were discarded
        # whenever VLM did not echo them (n_sources==1 → hard reject). That deleted
        # jackets, shirt colors, and animal colors before captioning ever ran.
        if family == "clothing" and detection >= 0.55 and fused >= 0.52:
            return True
        if family == "color" and detection >= 0.55 and fused >= 0.54:
            return True
        if family == "accessory" and detection >= 0.62 and fused >= 0.56:
            return True
        if family in {"object", "environment"} and detection >= 0.8 and fused >= 0.68:
            return True
        if family == "ocr" and fused >= 0.62:
            return True
        # Strong interaction-backed activity/relationship may pass with one solid source.
        # Prefer uncertainty over hallucination when evidence is thin.
        if family == "activity" and n_sources <= 1:
            return detection >= 0.74 and fused >= 0.64
        if family == "relationship" and n_sources <= 1:
            return detection >= 0.76 and fused >= 0.66
        return fused >= 0.72 and detection >= 0.85

    def _vlm_mentions(self, value: str, vlm_text: str, vlm_conf: float) -> float:
        if not value or not vlm_text:
            return 0.5
        tokens = [tok for tok in value.lower().replace("_", " ").split() if len(tok) > 2]
        if not tokens:
            return 0.5
        hits = sum(1 for tok in tokens if tok in vlm_text)
        if hits == 0:
            return 0.35
        ratio = hits / len(tokens)
        return min(0.95, 0.55 + 0.4 * ratio * max(0.4, vlm_conf))

    def _hazard_facts_from_vlm(self, vlm_text: str, vlm_conf: float) -> list[EvidenceFact]:
        """Recover fire/smoke when VLM text clearly mentions them (not in YOLO COCO)."""
        if not vlm_text or vlm_conf < 0.30:
            return []
        facts: list[EvidenceFact] = []
        for label, tokens in _HAZARD_TOKENS.items():
            hits = sum(1 for tok in tokens if tok in vlm_text)
            if hits == 0:
                continue
            conf = min(0.93, 0.52 + 0.12 * hits + 0.28 * max(0.35, vlm_conf))
            if conf < 0.62:
                continue
            facts.append(EvidenceFact(label, "is", "present", conf, "vlm"))
            facts.append(EvidenceFact("scene", "hazard", label, conf, "vlm"))
        return facts

    def _attribute_agreement(self, key: str, value: str, values: dict[str, str]) -> float:
        score = 0.5
        if key.endswith("_color") or key in {"secondary_color", "clothing_color"}:
            palette = values.get("clothing_palette", "")
            if value and value in palette:
                score += 0.2
            if key == "shirt_color" and values.get("clothing_color") == value:
                score += 0.2
            if key == "pants_color" and values.get("secondary_color") == value:
                score += 0.15
        if key == "clothing_type":
            if values.get(value, "") == "likely" or values.get(value.replace(" ", "_"), "") == "likely":
                score += 0.25
            if value == "jeans" and values.get("jeans") == "likely":
                score += 0.15
            if value == "hoodie" and values.get("hoodie") == "likely":
                score += 0.15
        if key == "footwear_type" and values.get("shoes_color") not in {"", "unknown", None}:
            score += 0.1
        return min(0.95, score)

    def _conflict_penalty(
        self,
        key: str,
        value: str,
        values: dict[str, str],
        *,
        has_sport: bool,
        vlm_text: str,
    ) -> float:
        penalty = 0.0
        if has_sport and value in _FORMAL_VALUES:
            penalty += 0.25
        if key == "clothing_type" and value in _FORMAL_VALUES and "sport" in vlm_text:
            penalty += 0.15
        if key.endswith("_color") or key in {"color", "dominant_color", "secondary_color", "clothing_color"}:
            peers = [
                values.get("clothing_color"),
                values.get("dominant_color"),
                values.get("secondary_color"),
                values.get("shirt_color"),
                values.get("color"),
            ]
            conflicting = [
                peer
                for peer in peers
                if peer and peer not in {"unknown", "", value} and peer != value
            ]
            # Color disagreement across extractors → prefer uncertainty over hallucination.
            if conflicting and key in {"shirt_color", "dominant_color", "color", "clothing_color"}:
                if not (key == "shirt_color" and values.get("secondary_color") in conflicting):
                    penalty += 0.35
            if value and vlm_text:
                # VLM names a different color family → heavy uncertainty.
                color_tokens = (
                    "red", "blue", "green", "yellow", "black", "white", "brown",
                    "orange", "pink", "purple", "gray", "grey", "beige", "tan", "navy",
                )
                mentioned = {tok for tok in color_tokens if tok in vlm_text}
                value_l = value.lower().replace("_", " ")
                if mentioned and not any(tok in value_l for tok in mentioned):
                    penalty += 0.28
        return penalty

    def _scene_consistency(
        self,
        key: str,
        value: str,
        context: SceneContext,
        *,
        has_sport: bool,
    ) -> float:
        env = context.environment
        score = 0.6
        if has_sport and value not in _FORMAL_VALUES:
            score += 0.1
        if env.indoor_outdoor == "outdoor" and key == "clothing_type" and value == "sportswear":
            score += 0.15
        if env.setting in {"office", "classroom"} and value in {"formal suit", "blazer", "shirt"}:
            score += 0.1
        return min(0.95, score)

    def _relationship_consistency_for_person(
        self,
        index: int,
        key: str,
        value: str,
        context: SceneContext,
        pose: PoseEstimate | None,
    ) -> float:
        score = 0.55
        if pose is not None:
            if key in {"clothing_type", "footwear_type"} and pose.action in {"running", "playing", "walking"}:
                if value in {"sportswear", "sneakers", "shorts", "t-shirt"}:
                    score += 0.2
                if value in _FORMAL_VALUES:
                    score -= 0.15
            if pose.action == "sitting" and key == "clothing_type":
                score += 0.05
        for relation in context.graph.relations:
            if relation.subject_index != index:
                continue
            if relation.relation_type == "holding" and key == "accessories":
                score += 0.1
        return max(0.2, min(0.95, score))

    def _spatial_score(self, values: dict[str, str]) -> float:
        visibility = values.get("visibility", "")
        occlusion = values.get("occlusion", "")
        score = 0.55
        if visibility in {"high", "clear", "fully_visible"}:
            score += 0.2
        if occlusion in {"none", "low", "clear"}:
            score += 0.15
        if occlusion in {"high", "heavy"}:
            score -= 0.2
        return max(0.2, min(0.95, score))

    def _suppress_formal_wear_near_sports(
        self,
        facts: list[EvidenceFact],
        labels: dict[int, str],
    ) -> tuple[list[EvidenceFact], int]:
        has_sport = any(label.lower() in _SPORT_LABELS for label in labels.values())
        if not has_sport:
            return facts, 0
        kept: list[EvidenceFact] = []
        removed = 0
        for fact in facts:
            if fact.predicate in {"clothing_type", "footwear_type", "clothing_style"} and fact.value in _FORMAL_VALUES:
                removed += 1
                continue
            if fact.predicate in {"jacket", "blazer", "dress"} and fact.value == "likely":
                removed += 1
                continue
            kept.append(fact)
        return kept, removed

    def _subject_name(self, label: str, index: int, labels: dict[int, str]) -> str:
        same = [i for i, name in labels.items() if name.lower() == label.lower()]
        if len(same) <= 1:
            return label
        return f"{label} #{same.index(index) + 1}"

    def _parse_confidence(self, value: str) -> float:
        text = value.strip().replace("%", "")
        try:
            number = float(text)
        except ValueError:
            return 0.6
        if number > 1.0:
            return max(0.0, min(1.0, number / 100.0))
        return max(0.0, min(1.0, number))

    def _resolve_contradictions(self, facts: list[EvidenceFact]) -> tuple[list[EvidenceFact], int]:
        best: dict[tuple[str, str], EvidenceFact] = {}
        resolved = 0
        dropped: set[tuple[str, str]] = set()
        for fact in facts:
            key = (fact.subject, fact.predicate)
            if key in dropped:
                resolved += 1
                continue
            existing = best.get(key)
            if existing is None:
                best[key] = fact
                continue
            resolved += 1
            same_family = fact.predicate.endswith("_color") or fact.predicate in {
                "color",
                "dominant_color",
                "secondary_color",
                "clothing_color",
            }
            # Close-confidence color disagreement → drop both (uncertainty > hallucination).
            if (
                same_family
                and existing.value.lower() != fact.value.lower()
                and abs(existing.confidence - fact.confidence) <= 0.12
            ):
                best.pop(key, None)
                dropped.add(key)
                continue
            if fact.confidence >= existing.confidence:
                best[key] = fact
        return list(best.values()), resolved

    def _dedupe(self, facts: list[EvidenceFact]) -> list[EvidenceFact]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[EvidenceFact] = []
        for fact in facts:
            key = (fact.subject.lower(), fact.predicate.lower(), fact.value.lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(fact)
        return unique

    def _rank_subjects(
        self,
        facts: list[EvidenceFact],
        labels: dict[int, str],
        context: SceneContext,
    ) -> tuple[str, ...]:
        """Semantic importance: interaction, event, saliency, uniqueness — not count or confidence alone."""
        node_by_index = {node.index: node for node in context.graph.nodes}
        subject_to_index: dict[str, int] = {}
        for index, label in labels.items():
            subject_to_index[self._subject_name(label, index, labels)] = index

        label_counts: dict[str, int] = defaultdict(int)
        for label in labels.values():
            label_counts[label.lower()] += 1

        interaction: dict[str, float] = defaultdict(float)
        semantic_rels = {
            "holding",
            "sitting_on",
            "looking_at",
            "carrying",
            "using",
            "playing_with",
            "talking_to",
            "riding",
            "leading",
            "eating",
        }
        for relation in context.graph.relations:
            if relation.confidence < 0.55:
                continue
            subj = self._subject_name(labels.get(relation.subject_index, "object"), relation.subject_index, labels)
            obj = self._subject_name(labels.get(relation.object_index, "object"), relation.object_index, labels)
            boost = 0.55 if relation.relation_type in semantic_rels else 0.12
            interaction[subj] += boost * relation.confidence
            interaction[obj] += 0.62 * boost * relation.confidence

        # Event significance from activity / action evidence (firefighting, tennis, riding…).
        event: dict[str, float] = defaultdict(float)
        for fact in facts:
            if fact.confidence < 0.55 or fact.value in {"unknown", "unlikely", "none", "general"}:
                continue
            if fact.predicate in {"action", "activity", "pose"}:
                if fact.subject not in {"scene", "vlm"}:
                    event[fact.subject] = max(event[fact.subject], 0.55 + 0.35 * fact.confidence)
                else:
                    # Scene-level activity lifts sport props and interacting people.
                    for activity in context.activities.activities:
                        for index in activity.supporting_node_indices:
                            label = labels.get(index, "object")
                            subject = self._subject_name(label, index, labels)
                            event[subject] = max(event[subject], 0.45 + 0.4 * activity.confidence)

        for activity in context.activities.activities:
            if activity.confidence < 0.55:
                continue
            for index in activity.supporting_node_indices:
                label = labels.get(index, "object")
                subject = self._subject_name(label, index, labels)
                event[subject] = max(event[subject], 0.5 + 0.4 * activity.confidence)

        setting = (context.environment.setting or context.environment.scene_type or "").lower()
        landscape_setting = any(
            token in setting
            for token in ("forest", "mountain", "beach", "field", "park", "valley", "lake", "sea", "sky")
        )

        fact_mass: dict[str, float] = defaultdict(float)
        for fact in facts:
            if fact.subject in {"scene", "vlm"}:
                continue
            fact_mass[fact.subject] += fact.confidence

        # Ensure every detected node can compete even with thin attributes.
        for index, label in labels.items():
            subject = self._subject_name(label, index, labels)
            fact_mass.setdefault(subject, 0.35)

        has_major_anchor = any(
            label.lower() in {"airplane", "train", "boat", "elephant"} for label in labels.values()
        )
        scores: dict[str, float] = {}
        for subject, mass in fact_mass.items():
            index = subject_to_index.get(subject)
            node = node_by_index.get(index) if index is not None else None
            label = (node.label if node is not None else subject.split("#")[0]).lower().strip()
            size = float(node.bounding_box_area_ratio) if node is not None else 0.05
            zone = (node.position_zone if node is not None else "middle-center").lower()
            center = 1.0 if "center" in zone else 0.65 if "middle" in zone else 0.35
            semantic = _SEMANTIC_IMPORTANCE.get(label, 0.42)
            is_person = any(token in label for token in _PERSON)
            if is_person:
                semantic = 1.0
            count = label_counts.get(label, 1)
            # Uniqueness matters more for rare event props than for duplicated clutter.
            if label in _SPORT_LABELS:
                unique = 1.0
            elif label in _PROP_LABELS:
                unique = 0.35 / max(1, count)
            else:
                unique = 1.0 / max(1, count)
            # Cohort helps traffic/vehicle scenes; never boost prop clutter.
            cohort = 0.0
            if label not in _PROP_LABELS and semantic >= 0.55:
                cohort = min(0.16, 0.05 * max(0, count - 1))
            # Attention proxy: large + central, without double-counting size heavily.
            attention = min(1.0, size * 5.0) * (0.4 + 0.6 * center)
            relevance = 0.0
            if label in _SCENE_ANCHOR_LABELS:
                relevance = 0.15 if label in {"car", "bus", "truck"} else 0.22
            if landscape_setting and is_person and size < 0.12 and interaction.get(subject, 0.0) < 0.2:
                # Distant person in a landscape — forest/field owns attention.
                semantic *= 0.55
            if is_person and has_major_anchor and size < 0.08 and interaction.get(subject, 0.0) < 0.2:
                # Parked airplane / train with a tiny bystander.
                semantic *= 0.55
            if label in _PROP_LABELS:
                semantic = min(semantic, 0.22)
                # Tiny props need strong interaction before they can lead.
                if size < 0.04 and interaction.get(subject, 0.0) < 0.45:
                    semantic *= 0.55
            # Large meaningful objects outrank clutter even without interaction.
            size_term = min(1.0, size * 8.0)
            if label not in _PROP_LABELS and size >= 0.10:
                size_term = min(1.0, size_term + 0.25)
            # Interaction / saliency / depth-proxy dominate over raw detection mass.
            depth_proxy = min(1.0, size * 6.5)  # larger ≈ closer / foreground
            score = (
                0.05 * min(1.0, mass / 5.0)
                + 0.14 * size_term
                + 0.06 * center
                + 0.16 * semantic
                + 0.28 * min(1.0, interaction.get(subject, 0.0))
                + 0.15 * min(1.0, event.get(subject, 0.0))
                + 0.06 * unique
                + 0.06 * attention
                + 0.04 * depth_proxy
                + relevance
                + cohort
            )
            scores[subject] = score
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return tuple(name for name, _ in ordered)

    def _build_brief(
        self,
        facts: list[EvidenceFact],
        ranked: tuple[str, ...],
        ocr_texts: tuple[str, ...],
    ) -> str:
        """Structured high-confidence evidence package for VLM narration."""
        high = [
            fact
            for fact in facts
            if fact.subject != "vlm"
            and fact.confidence >= 0.55
            and fact.value not in {"unknown", "unlikely", "none detected", "not_applicable", "possible"}
        ]
        people = [s for s in ranked if any(t in s.lower() for t in _PERSON)]
        objects = [s for s in ranked if s not in people and s not in {"scene", "vlm"}]

        sections: list[str] = [
            "HIGH-CONFIDENCE EVIDENCE PACKAGE (authoritative; prefer over visual guesses)",
        ]

        # Semantic scene story — understanding before language (not an object inventory).
        story = self._semantic_scene_story(high, ranked, people, objects)
        if story:
            sections.append("SEMANTIC SCENE STORY\n" + story)

        if people:
            person_lines: list[str] = []
            for subject in people[:4]:
                attrs = [f for f in high if f.subject == subject]
                clothing = [
                    f"{f.predicate}={f.value}"
                    for f in attrs
                    if f.predicate
                    in {
                        "clothing_type",
                        "clothing_style",
                        "sleeve_length",
                        "footwear_type",
                        "jacket",
                        "hoodie",
                        "dress",
                        "jeans",
                        "shorts",
                        "sweater",
                        "skirt",
                        "blazer",
                    }
                    and f.value not in {"unlikely"}
                ]
                colors = [
                    f"{f.predicate}={f.value}"
                    for f in attrs
                    if f.predicate.endswith("_color") or f.predicate in {"clothing_color", "secondary_color"}
                ]
                hair = [
                    f"{f.predicate}={f.value}"
                    for f in attrs
                    if f.predicate in {"hair_color", "hair_length", "hairstyle"}
                ]
                accessories = [
                    f"{f.predicate}={f.value}"
                    for f in attrs
                    if f.predicate
                    in {
                        "accessories",
                        "backpack",
                        "handbag",
                        "glasses",
                        "sunglasses",
                        "hat",
                        "cap",
                        "watch",
                        "necklace",
                        "earrings",
                    }
                    and f.value not in {"unlikely", "unknown"}
                ]
                actions = [
                    f"{f.predicate}={f.value}"
                    for f in attrs
                    if f.predicate in {"action", "pose"}
                ]
                bits = []
                if clothing:
                    bits.append("clothing: " + ", ".join(clothing[:6]))
                if colors:
                    bits.append("colors: " + ", ".join(colors[:6]))
                if hair:
                    bits.append("hair: " + ", ".join(hair[:3]))
                if accessories:
                    bits.append("accessories: " + ", ".join(accessories[:4]))
                if actions:
                    bits.append("activity: " + ", ".join(actions[:2]))
                if bits:
                    person_lines.append(f"- {subject}: " + "; ".join(bits))
            if person_lines:
                sections.append("People\n" + "\n".join(person_lines))

        object_lines: list[str] = []
        for subject in objects[:8]:
            attrs = [f for f in high if f.subject == subject]
            label = next((f.value for f in attrs if f.predicate == "is"), subject)
            colors = [
                f.value
                for f in attrs
                if f.predicate in {"dominant_color", "color", "secondary_color"}
            ]
            color_txt = f", colors={', '.join(colors)}" if colors else ""
            object_lines.append(f"- {label}{color_txt}")
        if object_lines:
            sections.append("Objects\n" + "\n".join(object_lines))

        semantic_relations = {
            "holding",
            "sitting_on",
            "carrying",
            "using",
            "playing_with",
            "leading",
            "guiding",
            "riding",
            "wearing",
            "inside",
            "driving",
            "eating",
        }
        relations = [
            f"- {f.subject} {f.predicate.replace('_', ' ')} {f.value}"
            for f in high
            if f.source == "relationships"
            and f.predicate in semantic_relations
            and f.confidence >= 0.70
            and f.subject.split("#")[0].strip().lower() != f.value.split("#")[0].strip().lower()
        ]
        if relations:
            sections.append("Relationships\n" + "\n".join(relations[:6]))

        activities = [
            f"- {f.value}"
            for f in high
            if f.predicate == "activity" or (f.predicate == "action" and f.subject == "scene")
        ]
        if activities:
            sections.append("Activities\n" + "\n".join(list(dict.fromkeys(activities))[:6]))

        env_map = {
            f.predicate: f.value
            for f in high
            if f.subject == "scene"
            and f.predicate
            in {
                "indoor_outdoor",
                "scene_type",
                "setting",
                "weather",
                "time_of_day",
                "crowd_level",
                "lighting",
            }
        }
        if env_map:
            env_lines = [f"- {key}={value}" for key, value in env_map.items()]
            sections.append("Environment / Scene category / Weather / Lighting\n" + "\n".join(env_lines))

        ocr_lines = [f'- "{text}"' for text in ocr_texts[:4]]
        ocr_lines.extend(f'- "{f.value}"' for f in high if f.predicate == "visible_text")
        if ocr_lines:
            sections.append("OCR\n" + "\n".join(list(dict.fromkeys(ocr_lines))[:4]))

        return "\n\n".join(sections)

    def _semantic_scene_story(
        self,
        high: list[EvidenceFact],
        ranked: tuple[str, ...],
        people: list[str],
        objects: list[str],
    ) -> str:
        """Internal semantic scene: what is happening, what draws attention, what supports it."""
        interact_priority = (
            "holding",
            "carrying",
            "riding",
            "leading",
            "playing_with",
            "using",
            "sitting_on",
            "looking_at",
            "talking_to",
            "eating",
        )
        interactions = [
            f
            for f in high
            if f.predicate in interact_priority
            and f.source in {"relationships", "pose_estimator", "attributes"}
        ]
        interactions.sort(
            key=lambda f: (
                interact_priority.index(f.predicate) if f.predicate in interact_priority else 99,
                -f.confidence,
            )
        )
        defining = interactions[0] if interactions else None
        activities = [f.value.replace("_", " ") for f in high if f.predicate in {"activity", "action"}]
        env = {
            f.predicate: f.value
            for f in high
            if f.subject == "scene" and f.predicate in {"setting", "scene_type", "indoor_outdoor", "weather", "time_of_day"}
        }
        setting = (env.get("setting") or env.get("scene_type") or env.get("indoor_outdoor") or "scene").replace("_", " ")

        if defining is not None:
            agent = defining.subject.split("#")[0].strip()
            patient = defining.value.split("#")[0].strip()
            rel = defining.predicate.replace("_", " ")
            event = f"{agent} {rel} {patient}"
            attention = f"the {rel} interaction between {agent} and {patient}"
            actors = f"{defining.subject}, {defining.value}"
        elif activities:
            event = activities[0]
            attention = f"the {event} event"
            actors = ", ".join(people[:2] or ranked[:2])
        elif people:
            event = f"{people[0].split('#')[0]} present in {setting}"
            attention = people[0].split("#")[0]
            actors = ", ".join(people[:3])
        elif objects:
            event = f"{objects[0].split('#')[0]} as the scene focus in {setting}"
            attention = objects[0].split("#")[0]
            actors = objects[0]
        else:
            return ""

        actor_set = {a.strip() for a in actors.split(",")}
        # Supporting = objects involved in interactions or high-ranked; background = rest.
        interaction_subjects = {defining.subject, defining.value} if defining is not None else set()
        supporting: list[str] = []
        for candidate in ranked:
            if candidate in actor_set or candidate in interaction_subjects:
                continue
            if candidate in people[:3] or candidate in objects[:4]:
                # Prefer objects that participate in any verified interaction fact.
                involved = any(
                    f.subject == candidate or f.value == candidate
                    for f in interactions
                )
                if involved or len(supporting) < 3:
                    supporting.append(candidate)
            if len(supporting) >= 6:
                break
        background = [
            o
            for o in objects
            if o not in actor_set and o not in supporting and o not in interaction_subjects
        ][:4]
        if not background and len(ranked) > 4:
            background = [
                o
                for o in ranked[3:]
                if o not in actor_set and o not in supporting
            ][:4]

        atmosphere_bits = [setting]
        if env.get("weather") and env["weather"] not in {"unknown", "clear", "none"}:
            atmosphere_bits.append(env["weather"])
        if env.get("time_of_day") and env["time_of_day"] not in {"unknown", "day", "general"}:
            atmosphere_bits.append(env["time_of_day"].replace("_", " "))

        # Primary subject follows attention ranking, not first person detection order.
        primary_raw = ranked[0] if ranked else (people[0] if people else "subject")
        primary = primary_raw.split("#")[0]
        secondary = []
        for subject in ranked[1:]:
            bare = subject.split("#")[0]
            if bare == primary or bare in secondary:
                continue
            secondary.append(bare)
            if len(secondary) >= 3:
                break
        why = (
            f"{activities[0]} in {setting}"
            if activities
            else f"document {event} within {setting}"
        )
        lines = [
            f"- Primary subject: {primary}",
            f"- Secondary subjects: {', '.join(secondary) if secondary else 'none verified'}",
            f"- What is happening: {event}",
            f"- Why this scene: {why}",
            f"- Most important event: {activities[0] if activities else event}",
            f"- Human intention / activity: {activities[0] if activities else event}",
            f"- Object interactions: {defining.predicate.replace('_', ' ') if defining else 'spatial co-presence'}",
            f"- Attention focus: {attention}",
            f"- Primary actors: {actors}",
            f"- Supporting: {', '.join(supporting) if supporting else 'none beyond the interaction'}",
            f"- Foreground: {', '.join([primary] + [s.split('#')[0] for s in supporting[:2]])}",
            f"- Background: {', '.join(background) if background else 'minimal'}",
            f"- Environment / atmosphere: {', '.join(atmosphere_bits)}",
            f"- Scene purpose: {why}",
            f"- Story: {event} unfolds in {setting}, with attention on {attention}.",
        ]
        return "\n".join(lines)
