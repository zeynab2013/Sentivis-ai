"""Pipeline orchestration using interface-only dependencies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from PIL import Image

from analysis.activity.activity_reasoning_service import ActivityReasoningService
from analysis.interfaces.attribute_extractor import IAttributeExtractor
from analysis.interfaces.context_builder import ISceneContextBuilder
from analysis.interfaces.relationship_analyzer import IRelationshipAnalyzer
from analysis.interfaces.scene_graph_builder import ISceneGraphBuilder
from analysis.ocr.text_extractor import OcrExtractor
from analysis.pose.pose_estimator import PoseEstimator
from analysis.scene_reasoner.scene_reasoner import SceneReasoner
from analysis.semantic.semantic_reasoning_service import SemanticReasoningResult, SemanticReasoningService
from core.config.app_config import AppConfig
from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import SceneContext
from core.contracts.language import RawCaption, RefinedCaption, VisualObservations
from core.contracts.pipeline import PipelineRequest, PipelineResult
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.base import SentivisError
from core.exceptions.service import CancelledError, OrchestrationError
from core.logging import get_logger
from core.validation.output_validators import validate_pipeline_result
from language.interfaces.caption_refiner import ICaptionRefiner
from language.interfaces.prompt_builder import IPromptBuilder
from language.interfaces.quality_evaluator import ICaptionQualityEvaluator
from language.interfaces.reasoning import IReasoningModel
from language.interfaces.vision_language import IVisionLanguageModel
from language.prompts.context_caption import build_context_caption
from language.semantic.narrative_generator import NarrativeGenerator
from language.semantic.natural_caption_service import NaturalCaptionService
from language.validation.caption_validator import CaptionEvidenceValidator
from services.interfaces.cancellation import ICancellationToken
from services.interfaces.model_manager import IModelManager
from services.interfaces.progress import IProgressReporter
from services.interfaces.stage_runner import IStageRunner
from services.memory.memory_manager import MemoryManager
from services.pipeline.cancellation import CancellationToken
from services.pipeline.competition_context import activate, deactivate
from services.pipeline.metrics_collector import PipelineMetricsCollector
from services.pipeline.pipeline_guard import PipelineGuard
from services.pipeline.quality_assurance import PipelineQualityAssurance
from vision.interfaces.detector import IObjectDetector
from vision.interfaces.preprocessor import IImagePreprocessor
from vision.interfaces.validator import IImageValidator

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Coordinates the visual understanding pipeline via abstract interfaces."""

    def __init__(
        self,
        validator: IImageValidator,
        preprocessor: IImagePreprocessor,
        detector: IObjectDetector,
        attribute_extractor: IAttributeExtractor,
        relationship_analyzer: IRelationshipAnalyzer,
        scene_graph_builder: ISceneGraphBuilder,
        activity_reasoning: ActivityReasoningService,
        semantic_reasoning: SemanticReasoningService,
        context_builder: ISceneContextBuilder,
        vision_language: IVisionLanguageModel,
        prompt_builder: IPromptBuilder,
        reasoning_model: IReasoningModel,
        caption_refiner: ICaptionRefiner,
        quality_evaluator: ICaptionQualityEvaluator,
        stage_runner: IStageRunner,
        model_manager: IModelManager,
        memory_manager: MemoryManager,
        progress: IProgressReporter,
        app_config: AppConfig,
        metrics_collector: PipelineMetricsCollector,
        quality_assurance: PipelineQualityAssurance,
        pipeline_guard: PipelineGuard | None = None,
        scene_reasoner: SceneReasoner | None = None,
        natural_caption: NaturalCaptionService | None = None,
        pose_estimator: PoseEstimator | None = None,
        ocr_extractor: OcrExtractor | None = None,
    ) -> None:
        self._validator = validator
        self._preprocessor = preprocessor
        self._detector = detector
        self._attribute_extractor = attribute_extractor
        self._relationship_analyzer = relationship_analyzer
        self._scene_graph_builder = scene_graph_builder
        self._activity_reasoning = activity_reasoning
        self._semantic_reasoning = semantic_reasoning
        self._context_builder = context_builder
        self._vision_language = vision_language
        self._prompt_builder = prompt_builder
        self._reasoning_model = reasoning_model
        self._caption_refiner = caption_refiner
        self._quality_evaluator = quality_evaluator
        self._stage_runner = stage_runner
        self._model_manager = model_manager
        self._memory = memory_manager
        self._progress = progress
        self._app_config = app_config
        self._metrics = metrics_collector
        self._qa = quality_assurance
        self._guard = pipeline_guard
        self._scene_reasoner = scene_reasoner or SceneReasoner()
        self._natural_caption = natural_caption
        self._pose_estimator = pose_estimator or PoseEstimator()
        self._ocr_extractor = ocr_extractor or OcrExtractor()
        self._caption_validator = CaptionEvidenceValidator()
        self._narrative = NarrativeGenerator()
        self._cancel_token: ICancellationToken = CancellationToken()

    @property
    def cancellation(self) -> ICancellationToken:
        return self._cancel_token

    def cancel(self) -> None:
        """Request cooperative pipeline cancellation."""
        self._cancel_token.cancel()

    def analyze(self, request: PipelineRequest) -> PipelineResult:
        """Execute the full pipeline for one image request."""
        run_id = str(uuid4())
        warnings: list[str] = []
        completed: list[PipelineStage] = []
        competition_mode = request.options.competition_mode
        self._cancel_token = CancellationToken()
        self._stage_runner.set_cancellation(self._cancel_token)
        self._metrics.begin_run(competition_mode=competition_mode)
        if competition_mode:
            activate(self._app_config.competition.deterministic_seed)
        if self._guard:
            self._guard.begin_run()

        try:
            if hasattr(self._preprocessor, "set_analysis_options"):
                self._preprocessor.set_analysis_options(request.options)
            if hasattr(self._detector, "set_sam2_enabled"):
                self._detector.set_sam2_enabled(request.options.enable_sam2)

            validated = self._stage_runner.run(
                PipelineStage.VALIDATION,
                5,
                "Validating image",
                lambda: self._validator.validate(request.image_path),
            )
            completed.append(PipelineStage.VALIDATION)

            preprocessed = self._stage_runner.run(
                PipelineStage.PREPROCESSING,
                10,
                "Preprocessing image",
                lambda: self._preprocessor.preprocess(validated),
            )
            completed.append(PipelineStage.PREPROCESSING)

            detections = self._stage_runner.run(
                PipelineStage.YOLO_DETECTION,
                20,
                "Running object detection",
                lambda: self._detector.detect(preprocessed),
            )
            completed.append(PipelineStage.YOLO_DETECTION)

            attributes = self._stage_runner.run(
                PipelineStage.ATTRIBUTE_EXTRACTION,
                35,
                "Extracting attributes",
                lambda: self._attribute_extractor.extract(detections, preprocessed),
            )
            completed.append(PipelineStage.ATTRIBUTE_EXTRACTION)

            relations = self._stage_runner.run(
                PipelineStage.RELATIONSHIP_ANALYSIS,
                42,
                "Analyzing relationships",
                lambda: self._relationship_analyzer.analyze(detections),
            )
            completed.append(PipelineStage.RELATIONSHIP_ANALYSIS)

            graph = self._stage_runner.run(
                PipelineStage.SCENE_GRAPH,
                50,
                "Building scene graph",
                lambda: self._scene_graph_builder.build(detections, relations),
            )
            completed.append(PipelineStage.SCENE_GRAPH)

            activities = self._stage_runner.run(
                PipelineStage.ACTIVITY_ANALYSIS,
                55,
                "Detecting activities (heuristics)",
                lambda: self._activity_reasoning.detect_activities(graph),
            )
            completed.append(PipelineStage.ACTIVITY_ANALYSIS)

            scene_context = self._stage_runner.run(
                PipelineStage.SCENE_CONTEXT,
                60,
                "Building scene context",
                lambda: self._context_builder.build(graph, attributes, activities),
            )
            completed.append(PipelineStage.SCENE_CONTEXT)

            observations = self._run_vision_language(preprocessed, scene_context, completed, warnings)
            self._memory.clear_gpu_cache()

            poses = self._pose_estimator.estimate(
                detections,
                relations,
                pixels=preprocessed.display_pixels,
            )
            ocr = self._ocr_extractor.extract(preprocessed.display_pixels)
            # PART 1.75: multi-signal interaction fusion (VLM = candidate, not authority).
            from analysis.evidence.interaction_fusion import InteractionEvidenceFuser

            fuser = InteractionEvidenceFuser()
            fusion = fuser.fuse(
                scene_context,
                detections=detections,
                observations=observations,
                poses=poses,
            )
            scene_context = fuser.apply_to_context(scene_context, fusion)
            relations = fusion.relations
            self._progress.emit(PipelineStage.PROMPT_BUILDING, 68, "Scene reasoning")
            understanding = self._scene_reasoner.reason(
                scene_context,
                observations=observations,
                poses=poses,
                ocr=ocr,
                image_quality=preprocessed.quality_report,
            )
            scene_context = _merge_hazard_evidence(scene_context, understanding)
            # PART 1.5: single verified evidence object — Caption + QA must use this.
            from analysis.evidence.verified_evidence_builder import (
                build_verified_scene_evidence,
                language_understanding_from_verified,
            )

            verified_evidence = build_verified_scene_evidence(
                scene_context,
                understanding,
                ocr_snippets=tuple(getattr(understanding, "ocr_text", ()) or ()),
                vlm_caption=(
                    (observations.raw_caption.text if observations and observations.raw_caption else "")
                    or ""
                ),
                vlm_confidence=float(
                    getattr(observations, "confidence", 0.75)
                    if observations is not None
                    else 0.75
                ),
            )
            # Caption writers receive verified projection — not raw reasoner relations.
            understanding = language_understanding_from_verified(
                verified_evidence,
                base=understanding,
            )
            natural_paragraph = ""
            if self._natural_caption is not None:
                self._progress.emit(PipelineStage.PROMPT_BUILDING, 72, "Natural caption candidates")
                natural_paragraph = self._natural_caption.generate(
                    preprocessed,
                    understanding,
                    context=scene_context,
                )
                self._memory.clear_gpu_cache()

            self._progress.emit(PipelineStage.PROMPT_BUILDING, 74, "Semantic synthesis (Ollama)")
            semantic_result = self._semantic_reasoning.synthesize(scene_context, observations)

            prompt = self._stage_runner.run(
                PipelineStage.PROMPT_BUILDING,
                78,
                "Building prompt",
                lambda: self._prompt_builder.build(scene_context, observations),
            )
            completed.append(PipelineStage.PROMPT_BUILDING)

            reasoning_caption = self._run_reasoning(
                prompt,
                scene_context,
                request,
                completed,
                warnings,
                observations,
                semantic_caption=(
                    semantic_result.caption
                    if isinstance(semantic_result, SemanticReasoningResult)
                    else None
                ),
            )
            self._memory.clear_gpu_cache()

            fallback_caption = self._best_fallback_caption(observations, scene_context)
            refined = self._stage_runner.run(
                PipelineStage.CAPTION_REFINEMENT,
                90,
                "Refining caption",
                lambda: self._caption_refiner.refine(
                    reasoning_caption,
                    fallback_caption,
                    scene_context,
                ),
            )
            completed.append(PipelineStage.CAPTION_REFINEMENT)

            # Narrative → QA → Final Caption (locked). Never mutate after lock.
            # Prefer a natural paragraph only when it is not awkward template filler;
            # otherwise keep the evidence-backed Ollama/refined caption.
            from language.refinement.caption_arbitration import choose_better_caption_grounded
            from language.refinement.caption_sanity import (
                choose_better_caption,
                sanitize_caption,
            )
            from language.validation.caption_factuality import (
                clamp_caption_object_counts,
                filter_unsupported_claims_verified,
            )

            ollama_text = semantic_result.caption.text if semantic_result.caption else ""
            # Ground all writer candidates against VerifiedSceneEvidence (factuality first).
            from language.refinement.caption_coverage import ensure_salient_verified_coverage

            _env_ev = tuple(getattr(scene_context.environment, "evidence", ()) or ())
            from language.refinement.caption_coverage import expand_verified_information_density

            natural_paragraph = expand_verified_information_density(natural_paragraph or "")
            natural_paragraph = ensure_salient_verified_coverage(
                natural_paragraph or "",
                verified=verified_evidence,
                understanding=understanding,
                environment_evidence=_env_ev,
            )
            ollama_text = ensure_salient_verified_coverage(
                ollama_text or "",
                verified=verified_evidence,
                understanding=understanding,
                environment_evidence=_env_ev,
            )
            preferred_paragraph = choose_better_caption_grounded(
                verified_evidence,
                natural_paragraph,
                ollama_text,
                refined.text,
            )
            if not preferred_paragraph:
                preferred_paragraph = choose_better_caption(
                    natural_paragraph,
                    ollama_text,
                    refined.text,
                )
            natural_words = len((natural_paragraph or "").split())
            preferred_words = len((preferred_paragraph or "").split())
            natural_l = (natural_paragraph or "").lower()
            preferred_l = (preferred_paragraph or "").lower()

            def _general_entity_coverage(text_l: str) -> int:
                """Count distinct salient entity mentions — general, not horse-specific."""
                tokens = (
                    "person",
                    "people",
                    "man",
                    "woman",
                    "child",
                    "horse",
                    "dog",
                    "cat",
                    "car",
                    "bus",
                    "truck",
                    "bicycle",
                    "motorcycle",
                    "table",
                    "chair",
                    "refrigerator",
                    "kitchen",
                    "street",
                    "fire",
                    "ball",
                    "racket",
                )
                return sum(1 for tok in tokens if tok in text_l)

            def _observed_person_shirt_colors() -> set[str]:
                colors: set[str] = set()
                for attr in verified_evidence.attributes:
                    if (getattr(attr, "name", "") or "").lower() not in {
                        "shirt_color",
                        "clothing_color",
                    }:
                        continue
                    eid = (getattr(attr, "entity_id", "") or "").lower()
                    if not eid.startswith("person"):
                        continue
                    status = getattr(attr, "status", None)
                    status_s = (
                        status.value if hasattr(status, "value") else str(status or "")
                    ).upper()
                    if status_s != "OBSERVED":
                        continue
                    val = (attr.value or "").strip().lower()
                    if val and val not in {
                        "unknown",
                        "olive",
                        "khaki",
                        "tan",
                        "beige",
                        "cream",
                    }:
                        colors.add(val)
                return colors

            def _shirt_color_survival(text: str, colors: set[str]) -> int:
                low = (text or "").lower()
                return sum(1 for c in colors if c and c in low)

            natural_ok = (
                bool(natural_paragraph)
                and natural_words >= 24
                and "a person talking to a person" not in natural_l
                and not natural_l.startswith("two people are visible")
            )
            # Template / meta wording only. Do NOT treat verified clothing words
            # (olive/khaki/t-shirt) as weak — that forced thin Natural stubs over
            # evidence-grounded Ollama captions that correctly named shirt colors.
            preferred_weak = any(
                marker in preferred_l
                for marker in (
                    "observing the",
                    "a pair",
                    "individuals",
                    "casual moment",
                    "overall impression",
                    "are present at",
                    "occupying a space defined",
                    "observed activity:",
                    "the location is outdoor",
                    "the location is indoor",
                    "person, and",
                    "dominant brown appearance",
                    "dominant appearance",
                    "overall composition",
                    "adding a detail",
                )
            )
            # Setting contradiction: alternate invents restaurant while kitchen is evidenced.
            evidence_kitchen = (
                "kitchen" in " ".join(understanding.environment_keys or ()).lower()
                or any(
                    f.predicate in {"setting", "scene_type"}
                    and "kitchen" in (f.value or "").lower()
                    for f in understanding.facts
                )
                or any(
                    lab in {s.split("#")[0].strip().lower() for s in understanding.ranked_subjects}
                    for lab in ("refrigerator", "oven", "sink", "microwave")
                )
            )
            if evidence_kitchen and "restaurant" in preferred_l and "kitchen" not in preferred_l:
                preferred_weak = True
            # Prefer Natural only when it is competitive and not a detector inventory.
            natural_inventory = (
                natural_l.count(" are visible")
                + natural_l.count(" is visible")
                + natural_l.count(" nearby")
                + natural_l.count("sit within the scene")
                + natural_l.count("sits within the scene")
                + natural_l.count("is also nearby")
            ) >= 3 or natural_l.count("we have") + natural_l.count("we can find") >= 1
            forced_natural_lock = False
            shirt_colors = _observed_person_shirt_colors()
            # Prefer natural whenever it is at least as informative as the alternate,
            # but only if arbitration already found it factually competitive.
            if natural_ok and not natural_inventory and preferred_paragraph == sanitize_caption(natural_paragraph):
                forced_natural_lock = True
            elif natural_ok and not natural_inventory and preferred_weak:
                # Weak alternate wording (template / body-color fluff) must not displace a
                # solid NaturalCaption paragraph — fluency helpers can wrongly flip back.
                # Exception: never displace an alternate that uniquely keeps OBSERVED shirts.
                if _shirt_color_survival(natural_paragraph or "", shirt_colors) >= _shirt_color_survival(
                    preferred_paragraph or "", shirt_colors
                ):
                    preferred_paragraph = natural_paragraph
                    forced_natural_lock = True
                    logger.info("Preferred NaturalCaptionService over weak alternate wording")
            elif (
                shirt_colors
                and _shirt_color_survival(natural_paragraph or "", shirt_colors)
                > _shirt_color_survival(preferred_paragraph or "", shirt_colors)
            ):
                # OBSERVED shirt/clothing colors must survive caption arbitration.
                preferred_paragraph = natural_paragraph or preferred_paragraph
                if natural_ok and not natural_inventory:
                    forced_natural_lock = True
                logger.info(
                    "Preferred caption retaining OBSERVED person shirt colors %s",
                    sorted(shirt_colors),
                )
            elif (
                shirt_colors
                and _shirt_color_survival(preferred_paragraph or "", shirt_colors)
                > _shirt_color_survival(natural_paragraph or "", shirt_colors)
            ):
                # Keep the alternate when it alone retains OBSERVED shirt colors.
                forced_natural_lock = False
                logger.info(
                    "Kept alternate caption retaining OBSERVED person shirt colors %s",
                    sorted(shirt_colors),
                )
            elif natural_ok and not natural_inventory and (
                preferred_words < max(30, int(natural_words * 0.90))
                or (
                    natural_words >= preferred_words
                    and _general_entity_coverage(natural_l)
                    >= _general_entity_coverage(preferred_l)
                )
            ):
                # Re-check natural against verified before overriding arbitration.
                grounded_natural = choose_better_caption_grounded(
                    verified_evidence,
                    natural_paragraph,
                    preferred_paragraph,
                )
                candidate = grounded_natural or preferred_paragraph
                # If grounded flip drops OBSERVED shirt colors that natural kept, keep natural.
                if (
                    shirt_colors
                    and _shirt_color_survival(natural_paragraph or "", shirt_colors)
                    > _shirt_color_survival(candidate or "", shirt_colors)
                ):
                    preferred_paragraph = natural_paragraph
                    forced_natural_lock = True
                elif (
                    shirt_colors
                    and _shirt_color_survival(preferred_paragraph or "", shirt_colors)
                    > _shirt_color_survival(candidate or "", shirt_colors)
                ):
                    # Keep prior preferred when it uniquely has shirts.
                    forced_natural_lock = False
                else:
                    preferred_paragraph = candidate
                if preferred_paragraph == sanitize_caption(natural_paragraph) or (
                    natural_paragraph
                    and preferred_paragraph
                    and preferred_paragraph.strip() == natural_paragraph.strip()
                ):
                    forced_natural_lock = True
                logger.info(
                    "Kept NaturalCaptionService paragraph as final "
                    "(natural=%d words, alternate=%d)",
                    natural_words,
                    preferred_words,
                )
            before_preferred = preferred_paragraph or ""
            preferred_paragraph = filter_unsupported_claims_verified(
                preferred_paragraph or "",
                verified_evidence,
            ) or preferred_paragraph
            # Never let claim filtering collapse a rich preferred caption to a stub.
            if (
                len(before_preferred.split()) >= 24
                and len((preferred_paragraph or "").split())
                < max(20, int(len(before_preferred.split()) * 0.55))
            ):
                preferred_paragraph = before_preferred
            preferred_paragraph = ensure_salient_verified_coverage(
                preferred_paragraph or "",
                verified=verified_evidence,
                understanding=understanding,
                environment_evidence=_env_ev,
            )
            # Authoritative verified entity counts win over VLM/arbitration quantity claims.
            preferred_paragraph = clamp_caption_object_counts(
                preferred_paragraph or "",
                understanding,
                verified=verified_evidence,
            )
            if forced_natural_lock and preferred_paragraph:
                # Narrative must not remix a forced natural caption with weak Ollama text.
                narrative = self._narrative.from_natural_paragraph(preferred_paragraph)
            else:
                narrative = self._narrative.generate(
                    scene_context,
                    observations=observations,
                    semantic_summary=semantic_result.scene_explanation,
                    quality_report=None,
                    ollama_caption=ollama_text,
                    natural_paragraph=preferred_paragraph or natural_paragraph,
                )
            draft_text = sanitize_caption(
                (narrative.full_caption or refined.text).strip()
            )
            canonical_en = sanitize_caption(
                (preferred_paragraph or natural_paragraph or draft_text).strip()
            )
            if not draft_text:
                draft_text = canonical_en
            if not canonical_en:
                canonical_en = draft_text
            caption_sources = tuple(
                dict.fromkeys((*refined.sources, "scene_reasoner", "natural_caption"))
            )
            draft_caption = RefinedCaption(
                text=draft_text,
                sources=caption_sources,
                narrative_full=draft_text,
                narrative_short=(narrative.short_caption or draft_text).strip(),
                executive_summary=(narrative.executive_summary or draft_text).strip(),
                canonical_caption_en=canonical_en,
            )

            quality_report = self._stage_runner.run(
                PipelineStage.QUALITY_EVALUATION,
                97,
                "Evaluating caption quality",
                lambda: self._quality_evaluator.evaluate(draft_caption.text, scene_context),
            )
            completed.append(PipelineStage.QUALITY_EVALUATION)

            qa_result = self._qa.evaluate(
                draft_caption,
                scene_context,
                quality_report,
                strict=competition_mode,
            )
            qa_count = 1
            if qa_result.rejected_caption:
                self._metrics.record_fallback()
                # Repair narrative/evidence package BEFORE final caption lock.
                # Never re-run VLM perception. Never collapse a rich evidence caption
                # into a one-liner when verified scene evidence still supports detail.
                source_text = (natural_paragraph or fallback_caption.text).strip()
                recovered_text = self._caption_validator.filter_unsupported_sentences(
                    source_text,
                    scene_context,
                )
                if not recovered_text.strip():
                    recovered_text = self._caption_validator.filter_unsupported_sentences(
                        fallback_caption.text,
                        scene_context,
                    )
                source_words = len(source_text.split())
                recovered_words = len(recovered_text.split())
                if source_text and source_words >= 24 and recovered_words < max(20, source_words // 2):
                    unsupported = self._caption_validator.unsupported_object_tokens(
                        source_text,
                        scene_context,
                    )
                    # Keep the evidence-rich narrative unless it is heavily unsupported.
                    if len(unsupported) <= 6:
                        recovered_text = source_text
                        warnings.append(
                            "QA soft-recovery kept evidence narrative after over-pruning."
                        )
                recovered_text = sanitize_caption(recovered_text)
                recovered_text = ensure_salient_verified_coverage(
                    recovered_text,
                    verified=verified_evidence,
                    understanding=understanding,
                    environment_evidence=_env_ev,
                )
                # Prefer a coherent alternate if recovery left robotic fragments.
                recovered_text = choose_better_caption_grounded(
                    verified_evidence,
                    recovered_text,
                    preferred_paragraph,
                    ollama_text,
                    fallback_caption.text,
                ) or choose_better_caption(
                    recovered_text,
                    preferred_paragraph,
                    ollama_text,
                    fallback_caption.text,
                ) or recovered_text
                # After arbitration, never let a weak fragment displace rich preferred text.
                if (
                    preferred_paragraph
                    and len(preferred_paragraph.split()) >= 24
                    and len((recovered_text or "").split())
                    < max(20, int(len(preferred_paragraph.split()) * 0.55))
                ):
                    recovered_text = preferred_paragraph
                    warnings.append(
                        "QA recovery restored preferred evidence caption over thin fragment."
                    )
                recovered_text = filter_unsupported_claims_verified(
                    recovered_text,
                    verified_evidence,
                ) or recovered_text
                if forced_natural_lock and preferred_paragraph:
                    recovered_text = preferred_paragraph
                    warnings.append(
                        "QA recovery kept forced NaturalCaption paragraph."
                    )
                narrative = self._narrative.from_natural_paragraph(recovered_text)
                draft_text = sanitize_caption(narrative.full_caption.strip())
                canonical_en = sanitize_caption(recovered_text.strip() or draft_text)
                if not draft_text:
                    draft_text = canonical_en
                caption_sources = ("recovery", fallback_caption.source, "natural_caption")
                warnings.append(
                    "Caption rejected by quality assurance; repaired narrative before final lock."
                )
                for issue in qa_result.issues:
                    warnings.append(f"QA: {issue}")
                quality_report = self._quality_evaluator.evaluate(draft_text, scene_context)
                draft_caption = RefinedCaption(
                    text=draft_text,
                    sources=caption_sources,
                    narrative_full=draft_text,
                    narrative_short=narrative.short_caption,
                    executive_summary=narrative.executive_summary,
                    canonical_caption_en=canonical_en,
                )
                # Soft re-check only — do not prune again after evidence-preserving repair.
                qa_result = self._qa.evaluate(
                    draft_caption,
                    scene_context,
                    quality_report,
                    strict=False,
                )
                qa_count = 2

            # FINAL CAPTION LOCK — immutable after this point (translation only later).
            locked_text = sanitize_caption(draft_text) or draft_text
            locked_canonical = sanitize_caption(canonical_en or locked_text) or locked_text
            if forced_natural_lock and preferred_paragraph:
                locked_text = preferred_paragraph
                locked_canonical = preferred_paragraph
            else:
                # Never let narrative/sanitize collapse coverage below the preferred paragraph.
                locked_text = choose_better_caption_grounded(
                    verified_evidence,
                    locked_canonical,
                    locked_text,
                    preferred_paragraph,
                ) or choose_better_caption(locked_canonical, locked_text) or locked_text
                locked_canonical = choose_better_caption_grounded(
                    verified_evidence,
                    locked_text,
                    locked_canonical,
                    preferred_paragraph,
                ) or choose_better_caption(locked_text, locked_canonical) or locked_text
            locked_text = filter_unsupported_claims_verified(
                locked_text, verified_evidence
            ) or locked_text
            locked_canonical = filter_unsupported_claims_verified(
                locked_canonical, verified_evidence
            ) or locked_canonical
            # Salient verified scene elements (e.g. fire) must not vanish after lock.
            locked_text = ensure_salient_verified_coverage(
                locked_text,
                verified=verified_evidence,
                understanding=understanding,
                environment_evidence=tuple(
                    getattr(scene_context.environment, "evidence", ()) or ()
                ),
            )
            locked_text = expand_verified_information_density(locked_text)
            locked_text = ensure_salient_verified_coverage(
                locked_text,
                verified=verified_evidence,
                understanding=understanding,
                environment_evidence=_env_ev,
            )
            locked_canonical = ensure_salient_verified_coverage(
                locked_canonical,
                verified=verified_evidence,
                understanding=understanding,
                environment_evidence=tuple(
                    getattr(scene_context.environment, "evidence", ()) or ()
                ),
            )
            locked_canonical = expand_verified_information_density(locked_canonical)
            locked_canonical = ensure_salient_verified_coverage(
                locked_canonical,
                verified=verified_evidence,
                understanding=understanding,
                environment_evidence=_env_ev,
            )
            if forced_natural_lock and preferred_paragraph:
                # Re-apply after filters so weak fragments cannot re-enter.
                locked_text = expand_verified_information_density(preferred_paragraph)
                locked_text = ensure_salient_verified_coverage(
                    locked_text,
                    verified=verified_evidence,
                    understanding=understanding,
                    environment_evidence=_env_ev,
                )
                locked_canonical = locked_text
            # FINAL object-count gate: never ship counts above distinct verified entities.
            locked_text = clamp_caption_object_counts(
                locked_text or "",
                understanding,
                verified=verified_evidence,
            )
            locked_canonical = clamp_caption_object_counts(
                locked_canonical or locked_text or "",
                understanding,
                verified=verified_evidence,
            )
            # Final polish after coverage/clamp reinjection — drop fragments/dupes only.
            locked_text = sanitize_caption(locked_text or "") or locked_text
            locked_canonical = sanitize_caption(locked_canonical or "") or locked_canonical or locked_text
            from language.semantic.narrative_generator import (
                executive_summary_from_paragraph,
                short_caption_from_paragraph,
            )

            final_caption = RefinedCaption(
                text=locked_text,
                sources=caption_sources,
                narrative_full=locked_text,
                narrative_short=short_caption_from_paragraph(locked_text),
                executive_summary=executive_summary_from_paragraph(locked_text),
                canonical_caption_en=locked_canonical,
            )
            logger.info("Final caption locked: %s", final_caption.text[:220])
            vlm_executions = 0
            if hasattr(self._vision_language, "execution_count"):
                vlm_executions = int(getattr(self._vision_language, "execution_count") or 0)
                logger.info("VLM execution count: %s", vlm_executions)

            self._progress.emit(PipelineStage.EXPORT, 100, "Analysis complete")
            self._memory.log_peak(run_id)
            metrics = self._metrics.finalize(
                scene_context,
                quality_report,
                qa_passed=qa_result.passed or not qa_result.rejected_caption,
                vlm_executions=vlm_executions,
                caption_generation_count=1,
                qa_count=qa_count,
            )
            result = PipelineResult(
                request=request,
                scene_context=scene_context,
                caption=final_caption,
                quality_report=quality_report,
                metrics=metrics,
                qa_passed=metrics.qa_passed,
                stages_completed=tuple(completed),
                warnings=tuple(warnings),
                image_quality=preprocessed.quality_report,
                enhanced_preview_path=_write_enhanced_preview(preprocessed, request.image_path),
                caption_translations=(("en", final_caption.canonical_caption_en or final_caption.text),),
                evidence_brief=verified_evidence.as_evidence_brief(),
                ocr_snippets=tuple(verified_evidence.ocr_text),
                initial_vlm_calls=vlm_executions,
                verified_evidence=verified_evidence,
            )
            validate_pipeline_result(result)
            return result
        except CancelledError:
            self._model_manager.release_active()
            raise
        except SentivisError:
            self._model_manager.release_active()
            raise
        except Exception as exc:
            self._model_manager.release_active()
            raise OrchestrationError(
                "Analysis could not be completed.",
                f"Unexpected pipeline error: {exc}",
                recoverable=False,
            ) from exc
        finally:
            if competition_mode:
                deactivate()

    def _run_vision_language(
        self,
        preprocessed: object,
        scene_context: object,
        completed: list[PipelineStage],
        warnings: list[str],
    ) -> VisualObservations | None:
        from core.contracts.analysis import SceneContext
        from core.contracts.image import PreprocessedImage

        if not isinstance(preprocessed, PreprocessedImage) or not isinstance(scene_context, SceneContext):
            warnings.append("Visual description skipped due to invalid pipeline state.")
            return None

        if hasattr(self._vision_language, "reset_execution_count"):
            self._vision_language.reset_execution_count()

        def action() -> VisualObservations:
            return self._vision_language.understand(preprocessed, scene_context)

        try:
            result = self._stage_runner.run(
                PipelineStage.BLIP_UNDERSTANDING,
                65,
                "Visual description",
                action,
            )
            completed.append(PipelineStage.BLIP_UNDERSTANDING)
            if hasattr(self._vision_language, "execution_count"):
                logger.info(
                    "VLM execution count: %s",
                    getattr(self._vision_language, "execution_count"),
                )
            return result
        except SentivisError as exc:
            self._metrics.record_fallback()
            warnings.append(exc.user_message)
            logger.warning("BLIP failed; continuing with scene graph and context only.")
            if hasattr(self._vision_language, "execution_count"):
                logger.info(
                    "VLM execution count: %s (perception failed)",
                    getattr(self._vision_language, "execution_count"),
                )
            return None

    def _run_reasoning(
        self,
        prompt: object,
        scene_context: object,
        request: PipelineRequest,
        completed: list[PipelineStage],
        warnings: list[str],
        observations: VisualObservations | None,
        *,
        semantic_caption: RawCaption | None = None,
    ) -> RawCaption:
        from core.contracts.analysis import SceneContext
        from core.contracts.language import Prompt

        fallback = self._best_fallback_caption(observations, scene_context)

        if (
            semantic_caption
            and semantic_caption.text.strip()
            and self._semantic_reasoning.enabled
            and self._semantic_reasoning.prefer_over_gemma
        ):
            if PipelineStage.GEMMA_REASONING not in completed:
                completed.append(PipelineStage.GEMMA_REASONING)
            logger.info("Using Ollama semantic synthesis caption.")
            return semantic_caption

        if not isinstance(prompt, Prompt) or not isinstance(scene_context, SceneContext):
            return fallback

        if not request.options.enable_gemma:
            return fallback

        def action() -> RawCaption:
            return self._reasoning_model.reason(prompt, scene_context)

        try:
            caption = self._stage_runner.run(
                PipelineStage.GEMMA_REASONING,
                82,
                "Generating caption",
                action,
            )
            completed.append(PipelineStage.GEMMA_REASONING)
            return caption
        except SentivisError as exc:
            self._metrics.record_fallback()
            warnings.append(exc.user_message)
            logger.warning("Gemma failed; using best validated caption from previous stages.")
            completed.append(PipelineStage.GEMMA_REASONING)
            return fallback

    def _best_fallback_caption(
        self,
        observations: VisualObservations | None,
        scene_context: object,
    ) -> RawCaption:
        from core.contracts.analysis import SceneContext

        if observations and observations.raw_caption.text.strip():
            return observations.raw_caption
        if isinstance(scene_context, SceneContext):
            return build_context_caption(scene_context)
        return RawCaption(
            text="The scene content remains uncertain based on available evidence.",
            source="template",
            confidence=0.4,
        )


def _merge_hazard_evidence(context: SceneContext, understanding: SceneUnderstanding) -> SceneContext:
    """Surface VLM-recovered fire/smoke into scene context for English object lists."""
    hazards: list[tuple[str, float]] = []
    seen: set[str] = set()
    for fact in understanding.facts:
        label = ""
        if fact.predicate == "hazard" and fact.value.lower() in {"fire", "smoke"}:
            label = fact.value.lower()
        elif fact.subject.lower() in {"fire", "smoke"} and fact.predicate == "is":
            label = fact.subject.lower()
        if not label or label in seen or fact.confidence < 0.62:
            continue
        seen.add(label)
        hazards.append((label, fact.confidence))
    if not hazards:
        return context
    evidence = list(context.environment.evidence)
    for label, conf in hazards:
        evidence.append(f"Hazard detected: {label} (confidence: {conf:.0%})")
    dominant = tuple(dict.fromkeys((*(label for label, _ in hazards), *context.dominant_objects)))
    return replace(
        context,
        environment=replace(context.environment, evidence=tuple(evidence)),
        dominant_objects=dominant,
    )


def _write_enhanced_preview(preprocessed: object, source_path: Path) -> Path | None:
    from core.contracts.image import PreprocessedImage

    if not isinstance(preprocessed, PreprocessedImage):
        return None
    if not preprocessed.enhancement_applied:
        return None
    cache_dir = source_path.parent / ".sentivis_cache"
    cache_dir.mkdir(exist_ok=True)
    preview_path = cache_dir / f"{source_path.stem}_enhanced.png"
    Image.fromarray(preprocessed.display_pixels).save(preview_path)
    return preview_path
