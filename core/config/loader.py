"""TOML configuration loader."""

from pathlib import Path

from core.config._toml import TOMLDecodeError, load_toml
from core.config.analysis_config import (
    ActivityHeuristicsConfig,
    ActivityReasoningConfig,
    AnalysisConfig,
    AttributeHeuristicsConfig,
    ContextHeuristicsConfig,
    RelationshipHeuristicsConfig,
    SemanticReasoningConfig,
)
from core.config.app_config import (
    AppConfig,
    CompetitionConfig,
    HardwareConfig,
    ImageConfig,
    LoggingConfig,
    PathsConfig,
    WorkerConfig,
)
from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from core.config.enhancement_config import EnhancementConfig
from core.config.model_config import (
    BlipModelConfig,
    FlorenceModelConfig,
    GemmaModelConfig,
    ModelConfig,
    PluginConfig,
    YoloModelConfig,
)
from core.config.schema_validator import (
    validate_analysis_config,
    validate_app_config,
    validate_model_config,
    validate_theme_config,
)
from core.config.theme_config import ThemeConfig
from core.config.toml_helpers import (
    get_bool,
    get_bool_default,
    get_float,
    get_float_default,
    get_int,
    get_int_default,
    get_optional_str,
    get_str,
    optional_section,
    section,
)
from core.config.vlm_config import VlmModelIds, VlmSelectionConfig
from core.exceptions.config import ConfigurationError
from core.utils.paths import normalize_optional_path, project_root, resolve_user_path


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            return load_toml(handle)
    except (OSError, ValueError, TOMLDecodeError) as exc:
        raise ConfigurationError(f"Failed to parse configuration file {path}: {exc}") from exc


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        items: tuple[object, ...] = tuple(value)
    elif isinstance(value, tuple):
        items = value
    else:
        return ()
    return tuple(str(item) for item in items if str(item).strip())


def _build_app_config(data: dict[str, object]) -> AppConfig:
    app = section(data, "app")
    logging_data = section(data, "logging")
    image = section(data, "image")
    enhancement = optional_section(image, "enhancement")
    hardware = section(data, "hardware")
    paths = section(data, "paths")
    workers = section(data, "workers")
    competition = section(data, "competition")
    root = project_root()
    raw_search = paths.get("model_search_paths", [])
    model_search_paths: tuple[Path, ...] = ()
    if isinstance(raw_search, list):
        model_search_paths = tuple(
            resolve_user_path(str(item), root)
            for item in raw_search
            if str(item).strip()
        )

    return AppConfig(
        app_name=get_str(app, "name"),
        app_version=get_str(app, "version"),
        logging=LoggingConfig(
            level=get_str(logging_data, "level"),
            console_enabled=get_bool(logging_data, "console_enabled"),
            max_file_bytes=get_int(logging_data, "max_file_bytes"),
            backup_count=get_int(logging_data, "backup_count"),
        ),
        image=ImageConfig(
            max_dimension=get_int(image, "max_dimension"),
            max_file_size_bytes=get_int(image, "max_file_size_bytes"),
            yolo_inference_size=get_int(image, "yolo_inference_size"),
            enhancement=EnhancementConfig(
                enabled=get_bool_default(enhancement, "enabled", DEFAULT_ENHANCEMENT_CONFIG.enabled),
                adaptive_mode=get_bool_default(enhancement, "adaptive_mode", DEFAULT_ENHANCEMENT_CONFIG.adaptive_mode),
                quality_threshold=get_float_default(
                    enhancement, "quality_threshold", DEFAULT_ENHANCEMENT_CONFIG.quality_threshold
                ),
                min_brightness=get_float_default(
                    enhancement, "min_brightness", DEFAULT_ENHANCEMENT_CONFIG.min_brightness
                ),
                max_brightness=get_float_default(
                    enhancement, "max_brightness", DEFAULT_ENHANCEMENT_CONFIG.max_brightness
                ),
                min_contrast=get_float_default(enhancement, "min_contrast", DEFAULT_ENHANCEMENT_CONFIG.min_contrast),
                min_sharpness=get_float_default(
                    enhancement, "min_sharpness", DEFAULT_ENHANCEMENT_CONFIG.min_sharpness
                ),
                max_blur_score=get_float_default(
                    enhancement, "max_blur_score", DEFAULT_ENHANCEMENT_CONFIG.max_blur_score
                ),
                max_noise_score=get_float_default(
                    enhancement, "max_noise_score", DEFAULT_ENHANCEMENT_CONFIG.max_noise_score
                ),
                enable_clahe=get_bool_default(enhancement, "enable_clahe", DEFAULT_ENHANCEMENT_CONFIG.enable_clahe),
                enable_gamma=get_bool_default(enhancement, "enable_gamma", DEFAULT_ENHANCEMENT_CONFIG.enable_gamma),
                enable_denoise=get_bool_default(
                    enhancement, "enable_denoise", DEFAULT_ENHANCEMENT_CONFIG.enable_denoise
                ),
                enable_sharpen=get_bool_default(
                    enhancement, "enable_sharpen", DEFAULT_ENHANCEMENT_CONFIG.enable_sharpen
                ),
                enable_color_correction=get_bool_default(
                    enhancement,
                    "enable_color_correction",
                    DEFAULT_ENHANCEMENT_CONFIG.enable_color_correction,
                ),
                enable_super_resolution=get_bool_default(
                    enhancement,
                    "enable_super_resolution",
                    DEFAULT_ENHANCEMENT_CONFIG.enable_super_resolution,
                ),
                super_resolution_min_dimension=get_int_default(
                    enhancement,
                    "super_resolution_min_dimension",
                    DEFAULT_ENHANCEMENT_CONFIG.super_resolution_min_dimension,
                ),
                competition_always_enhance=get_bool_default(
                    enhancement,
                    "competition_always_enhance",
                    DEFAULT_ENHANCEMENT_CONFIG.competition_always_enhance,
                ),
                sr_model=get_optional_str(
                    enhancement, "sr_model", DEFAULT_ENHANCEMENT_CONFIG.sr_model
                ),
                sr_scale=get_int_default(
                    enhancement, "sr_scale", DEFAULT_ENHANCEMENT_CONFIG.sr_scale
                ),
                sr_tile_size=get_int_default(
                    enhancement, "sr_tile_size", DEFAULT_ENHANCEMENT_CONFIG.sr_tile_size
                ),
                sr_tile_overlap=get_int_default(
                    enhancement, "sr_tile_overlap", DEFAULT_ENHANCEMENT_CONFIG.sr_tile_overlap
                ),
                sr_device=get_optional_str(
                    enhancement, "sr_device", DEFAULT_ENHANCEMENT_CONFIG.sr_device
                ),
                sr_max_output_side=get_int_default(
                    enhancement,
                    "sr_max_output_side",
                    DEFAULT_ENHANCEMENT_CONFIG.sr_max_output_side,
                ),
                sr_allow_download=get_bool_default(
                    enhancement,
                    "sr_allow_download",
                    DEFAULT_ENHANCEMENT_CONFIG.sr_allow_download,
                ),
                quality_low_threshold=get_float_default(
                    enhancement,
                    "quality_low_threshold",
                    DEFAULT_ENHANCEMENT_CONFIG.quality_low_threshold,
                ),
                quality_medium_threshold=get_float_default(
                    enhancement,
                    "quality_medium_threshold",
                    DEFAULT_ENHANCEMENT_CONFIG.quality_medium_threshold,
                ),
            ),
        ),
        hardware=HardwareConfig(
            vram_warning_ratio=get_float(hardware, "vram_warning_ratio"),
            ram_warning_ratio=get_float(hardware, "ram_warning_ratio"),
            cpu_fallback_enabled=get_bool(hardware, "cpu_fallback_enabled"),
            pipeline_timeout_seconds=get_int(hardware, "pipeline_timeout_seconds"),
        ),
        paths=PathsConfig(
            cache_dir=resolve_user_path(get_str(paths, "cache_dir"), root),
            exports_dir=resolve_user_path(get_str(paths, "exports_dir"), root),
            logs_dir=resolve_user_path(get_str(paths, "logs_dir"), root),
            models_dir=resolve_user_path(get_str(paths, "models_dir"), root),
            model_search_paths=model_search_paths,
        ),
        workers=WorkerConfig(
            export_thread_pool_size=get_int(workers, "export_thread_pool_size"),
        ),
        competition=CompetitionConfig(
            quality_threshold=get_float(competition, "quality_threshold"),
            max_hallucination_risk=get_float(competition, "max_hallucination_risk"),
            deterministic_seed=get_int(competition, "deterministic_seed"),
            gemma_temperature=get_float(competition, "gemma_temperature"),
            vram_release_threshold_mb=get_float(competition, "vram_release_threshold_mb"),
        ),
    )


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load application configuration from TOML.

    Args:
        config_path: Optional override path; defaults to ``config/app.default.toml``.

    Returns:
        Parsed AppConfig instance.
    """
    path = config_path or project_root() / "config" / "app.default.toml"
    data = _read_toml(path)
    validate_app_config(data)
    return _build_app_config(data)


def _build_model_config(data: dict[str, object]) -> ModelConfig:
    yolo = section(data, "yolo")
    blip = section(data, "blip")
    florence = optional_section(data, "florence") or {}
    gemma = section(data, "gemma")
    plugins_raw = data.get("plugins")
    plugins: dict[str, object] = plugins_raw if isinstance(plugins_raw, dict) else {}
    weights_path = normalize_optional_path(yolo.get("weights_path"))

    return ModelConfig(
        yolo=YoloModelConfig(
            variant=get_str(yolo, "variant"),
            weights_path=weights_path,
            confidence_threshold=get_float(yolo, "confidence_threshold"),
            iou_threshold=get_float(yolo, "iou_threshold"),
            preferred_device=get_str(yolo, "preferred_device"),
        ),
        blip=BlipModelConfig(
            model_id=get_str(blip, "model_id"),
            preferred_device=get_str(blip, "preferred_device"),
            max_length=get_int(blip, "max_length"),
        ),
        florence=FlorenceModelConfig(
            model_id=get_optional_str(florence, "model_id", "microsoft/Florence-2-base-ft"),
            preferred_device=get_optional_str(florence, "preferred_device", get_str(blip, "preferred_device")),
            max_new_tokens=get_int_default(florence, "max_new_tokens", 128),
            fallback_to_blip=get_bool_default(florence, "fallback_to_blip", True),
        ),
        gemma=GemmaModelConfig(
            model_id=get_str(gemma, "model_id"),
            preferred_device=get_str(gemma, "preferred_device"),
            quantization=get_str(gemma, "quantization"),
            max_new_tokens=get_int(gemma, "max_new_tokens"),
            temperature=get_float(gemma, "temperature"),
        ),
        plugins=PluginConfig(
            detection_plugin=get_optional_str(plugins, "detection", "vision.yolo_v8n"),
            vision_language_plugin=get_optional_str(plugins, "vision_language", "language.florence2"),
            reasoning_plugin=get_optional_str(plugins, "reasoning", "language.gemma_2b"),
        ),
        vlm=_build_vlm_config(data, blip, florence),
    )


def _build_vlm_config(
    data: dict[str, object],
    blip: dict[str, object],
    florence: dict[str, object],
) -> VlmSelectionConfig:
    vlm = optional_section(data, "vlm")
    return VlmSelectionConfig(
        auto_select=get_bool_default(vlm, "auto_select", True),
        preferred_adapter=get_optional_str(vlm, "preferred_adapter", ""),
        model_ids=VlmModelIds(
            gemma_vision=get_optional_str(vlm, "gemma_vision", "gemma3:4b"),
            florence_base=get_optional_str(
                vlm,
                "florence_base",
                get_optional_str(florence, "model_id", "microsoft/Florence-2-base-ft"),
            ),
            florence_plain=get_optional_str(vlm, "florence_plain", "microsoft/Florence-2-base"),
            florence_large=get_optional_str(vlm, "florence_large", "microsoft/Florence-2-large-ft"),
            moondream=get_optional_str(vlm, "moondream", "vikhyatk/moondream2"),
            blip2=get_optional_str(vlm, "blip2", "Salesforce/blip2-opt-2.7b"),
            blip=get_optional_str(vlm, "blip", get_str(blip, "model_id")),
            qwen=get_optional_str(vlm, "qwen", "Qwen/Qwen2.5-VL-3B-Instruct"),
            internvl=get_optional_str(vlm, "internvl", "OpenGVLab/InternVL2-8B"),
        ),
        min_vram_gemma_vision_gb=get_float_default(vlm, "min_vram_gemma_vision_gb", 0.0),
        min_vram_florence_base_gb=get_float_default(vlm, "min_vram_florence_base_gb", 1.5),
        min_vram_florence_plain_gb=get_float_default(vlm, "min_vram_florence_plain_gb", 1.5),
        min_vram_moondream_gb=get_float_default(vlm, "min_vram_moondream_gb", 1.7),
        min_vram_blip2_gb=get_float_default(vlm, "min_vram_blip2_gb", 3.5),
        min_vram_florence_large_gb=get_float_default(vlm, "min_vram_florence_large_gb", 3.5),
        min_vram_qwen_gb=get_float_default(vlm, "min_vram_qwen_gb", 6.0),
        min_vram_internvl_gb=get_float_default(vlm, "min_vram_internvl_gb", 12.0),
    )


def load_model_config(config_path: Path | None = None) -> ModelConfig:
    """Load model configuration from TOML."""
    path = config_path or project_root() / "config" / "models.default.toml"
    data = _read_toml(path)
    validate_model_config(data)
    return _build_model_config(data)


def _build_analysis_config(data: dict[str, object]) -> AnalysisConfig:
    analysis = section(data, "analysis")
    attributes = section(analysis, "attributes")
    relationships = section(analysis, "relationships")
    activity = section(analysis, "activity")
    activity_reasoning = section(analysis, "activity_reasoning")
    semantic_reasoning = section(analysis, "semantic_reasoning")
    context = section(analysis, "context")
    return AnalysisConfig(
        attributes=AttributeHeuristicsConfig(
            size_small_max_ratio=get_float(attributes, "size_small_max_ratio"),
            size_medium_max_ratio=get_float(attributes, "size_medium_max_ratio"),
            zone_split_low=get_float(attributes, "zone_split_low"),
            zone_split_high=get_float(attributes, "zone_split_high"),
            distance_near_ratio=get_float(attributes, "distance_near_ratio"),
            distance_medium_ratio=get_float(attributes, "distance_medium_ratio"),
            pose_standing_ratio=get_float(attributes, "pose_standing_ratio"),
            pose_lying_ratio=get_float(attributes, "pose_lying_ratio"),
            visibility_high_threshold=get_float(attributes, "visibility_high_threshold"),
            visibility_medium_threshold=get_float(attributes, "visibility_medium_threshold"),
        ),
        relationships=RelationshipHeuristicsConfig(
            overlap_distance=get_float(relationships, "overlap_distance"),
            overlap_confidence=get_float(relationships, "overlap_confidence"),
            distance_confidence_factor=get_float(relationships, "distance_confidence_factor"),
            max_confidence=get_float(relationships, "max_confidence"),
            near_distance_ratio=get_float(relationships, "near_distance_ratio"),
            far_distance_ratio=get_float(relationships, "far_distance_ratio"),
        ),
        activity=ActivityHeuristicsConfig(
            confidence_with_nodes=float(get_float(activity, "confidence_with_nodes")),
            confidence_empty=float(get_float(activity, "confidence_empty")),
        ),
        activity_reasoning=ActivityReasoningConfig(
            enabled=get_bool(activity_reasoning, "enabled"),
            mode=get_str(activity_reasoning, "mode"),
            model=get_str(activity_reasoning, "model"),
            base_url=get_str(activity_reasoning, "base_url"),
            timeout_seconds=get_float(activity_reasoning, "timeout_seconds"),
            fallback_to_minimal=get_bool(activity_reasoning, "fallback_to_minimal"),
            prefer_ollama_caption=get_bool(activity_reasoning, "prefer_ollama_caption"),
            models=_string_tuple(activity_reasoning.get("models", ())),
        ),
        semantic_reasoning=SemanticReasoningConfig(
            enabled=get_bool(semantic_reasoning, "enabled"),
            mode=get_str(semantic_reasoning, "mode"),
            model=get_str(semantic_reasoning, "model"),
            base_url=get_str(semantic_reasoning, "base_url"),
            timeout_seconds=get_float(semantic_reasoning, "timeout_seconds"),
            fallback_to_context_caption=get_bool(semantic_reasoning, "fallback_to_context_caption"),
            prefer_over_gemma=get_bool(semantic_reasoning, "prefer_over_gemma"),
            models=_string_tuple(semantic_reasoning.get("models", ())),
        ),
        context=ContextHeuristicsConfig(
            crowd_threshold=get_int(context, "crowd_threshold"),
            complexity_high_relations=get_int(context, "complexity_high_relations"),
            complexity_medium_relations=get_int(context, "complexity_medium_relations"),
        ),
    )


def load_analysis_config(config_path: Path | None = None) -> AnalysisConfig:
    """Load analysis heuristics configuration from TOML."""
    path = config_path or project_root() / "config" / "analysis.default.toml"
    data = _read_toml(path)
    validate_analysis_config(data)
    return _build_analysis_config(data)


def _build_theme_config(data: dict[str, object]) -> ThemeConfig:
    theme = section(data, "theme")
    root = project_root()
    return ThemeConfig(
        name=get_str(theme, "name"),
        stylesheet_path=root / get_str(theme, "stylesheet_path"),
        font_family=get_str(theme, "font_family"),
        font_size=get_int(theme, "font_size"),
        accent_color=get_str(theme, "accent_color"),
        background_color=get_str(theme, "background_color"),
    )


def load_theme_config(config_path: Path | None = None) -> ThemeConfig:
    """Load theme configuration from TOML."""
    path = config_path or project_root() / "config" / "themes.default.toml"
    data = _read_toml(path)
    validate_theme_config(data)
    return _build_theme_config(data)
