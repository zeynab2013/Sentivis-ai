"""Pipeline output schema validation."""

from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import SceneContext, SceneGraph
from core.contracts.detection import DetectionResult
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import CaptionQualityReport, Prompt, RefinedCaption, VisualObservations
from core.contracts.pipeline import PipelineResult
from core.exceptions.analysis import AnalysisError
from core.exceptions.service import OrchestrationError
from core.exceptions.vision import ValidationError


def validate_validated_image(image: ValidatedImage) -> None:
    if image.width <= 0 or image.height <= 0:
        raise ValidationError(
            "The image dimensions are invalid.",
            f"ValidatedImage has invalid dimensions {image.width}x{image.height}",
        )
    if image.pixels.size == 0:
        raise ValidationError(
            "The image contains no readable pixel data.",
            "ValidatedImage pixels buffer is empty",
        )


def validate_preprocessed_image(image: PreprocessedImage) -> None:
    validate_validated_image(image.source)
    if image.inference_width <= 0 or image.inference_height <= 0:
        raise ValidationError(
            "The image could not be prepared for analysis.",
            "PreprocessedImage inference dimensions invalid",
        )


def validate_detection_result(result: DetectionResult) -> None:
    if result.image_width <= 0 or result.image_height <= 0:
        raise AnalysisError(
            "Object detection produced invalid output.",
            "DetectionResult image dimensions invalid",
            stage=PipelineStage.YOLO_DETECTION,
            recoverable=False,
        )
    for index, detection in enumerate(result.detections):
        if not detection.object_id:
            raise AnalysisError(
                "Object detection produced invalid output.",
                f"Detection at index {index} missing object_id",
                stage=PipelineStage.YOLO_DETECTION,
                recoverable=False,
            )
        if not (0.0 <= detection.confidence <= 1.0):
            raise AnalysisError(
                "Object detection produced invalid output.",
                f"Detection {detection.object_id} has invalid confidence",
                stage=PipelineStage.YOLO_DETECTION,
                recoverable=False,
            )


def validate_scene_graph(graph: SceneGraph) -> None:
    node_indices = {node.index for node in graph.nodes}
    for node in graph.nodes:
        if not node.object_id:
            raise AnalysisError(
                "Scene analysis produced invalid output.",
                f"SceneNode {node.index} missing object_id",
                stage=PipelineStage.SCENE_GRAPH,
            )
    for relation in graph.relations:
        if relation.subject_index not in node_indices or relation.object_index not in node_indices:
            raise AnalysisError(
                "Scene analysis produced invalid output.",
                "SceneGraph relation references missing node",
                stage=PipelineStage.SCENE_GRAPH,
            )


def validate_scene_context(context: SceneContext) -> None:
    validate_scene_graph(context.graph)
    if context.object_count != len(context.graph.nodes):
        raise AnalysisError(
            "Scene context is inconsistent.",
            "SceneContext.object_count does not match graph node count",
            stage=PipelineStage.SCENE_CONTEXT,
        )


def validate_prompt(prompt: Prompt) -> None:
    if not prompt.system.strip() or not prompt.user.strip():
        raise AnalysisError(
            "Prompt construction failed.",
            "Prompt contains empty system or user section",
            stage=PipelineStage.PROMPT_BUILDING,
            recoverable=True,
        )


def validate_visual_observations(observations: VisualObservations) -> None:
    if not (0.0 <= observations.confidence <= 1.0):
        raise AnalysisError(
            "Visual description output is invalid.",
            "VisualObservations confidence out of range",
            stage=PipelineStage.BLIP_UNDERSTANDING,
            recoverable=True,
        )


def validate_refined_caption(caption: RefinedCaption) -> None:
    if not caption.text.strip():
        raise OrchestrationError(
            "Caption generation did not produce a usable result.",
            "RefinedCaption text is empty",
            stage=PipelineStage.CAPTION_REFINEMENT,
            recoverable=True,
        )


def validate_quality_report(report: CaptionQualityReport) -> None:
    # Coverage / hallucination may be None (N/A) when denominators are empty
    # or risk cannot be measured. Required scores must stay in [0, 1].
    required_scores = (
        report.grammar_score,
        report.fluency_score,
        report.evidence_consistency,
        report.context_coverage,
        report.overall_quality,
    )
    optional_scores = (
        report.object_coverage,
        report.relationship_coverage,
        report.activity_coverage,
        report.hallucination_risk,
    )
    for score in required_scores:
        if not 0.0 <= score <= 1.0:
            raise OrchestrationError(
                "Caption quality evaluation produced invalid metrics.",
                f"Quality score out of range: {score}",
                stage=PipelineStage.QUALITY_EVALUATION,
                recoverable=True,
            )
    for score in optional_scores:
        if score is None:
            continue
        if not 0.0 <= score <= 1.0:
            raise OrchestrationError(
                "Caption quality evaluation produced invalid metrics.",
                f"Quality score out of range: {score}",
                stage=PipelineStage.QUALITY_EVALUATION,
                recoverable=True,
            )


def validate_pipeline_result(result: PipelineResult) -> None:
    validate_scene_context(result.scene_context)
    validate_refined_caption(result.caption)
    validate_quality_report(result.quality_report)
    if result.metrics.total_duration_ms < 0.0:
        raise OrchestrationError(
            "Pipeline metrics are invalid.",
            "total_duration_ms is negative",
            recoverable=False,
        )
    if result.metrics.caption_quality_score != result.quality_report.overall_quality:
        raise OrchestrationError(
            "Pipeline metrics are inconsistent.",
            "caption_quality_score does not match quality_report.overall_quality",
            recoverable=False,
        )
