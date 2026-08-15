"""Configuration schema validation."""

from core.config.toml_helpers import as_float, as_int, require_section, section
from core.exceptions.config import ConfigurationError

BUILTIN_PLUGIN_IDS = frozenset(
    {
        "vision.yolo_v8n",
        "language.blip_base",
        "language.florence2",
        "language.gemma_2b",
    }
)


def validate_app_config(raw: dict[str, object]) -> None:
    """Validate parsed application configuration."""
    require_section(raw, "app")
    require_section(raw, "logging")
    require_section(raw, "image")
    require_section(raw, "hardware")
    require_section(raw, "paths")
    require_section(raw, "workers")
    require_section(raw, "competition")

    image = section(raw, "image")
    max_dimension = as_int(image["max_dimension"], "image.max_dimension")
    yolo_size = as_int(image["yolo_inference_size"], "image.yolo_inference_size")
    if max_dimension <= 0:
        raise ConfigurationError("image.max_dimension must be positive")
    if yolo_size <= 0:
        raise ConfigurationError("image.yolo_inference_size must be positive")
    if yolo_size > max_dimension:
        raise ConfigurationError("image.yolo_inference_size must be <= image.max_dimension")

    max_file_size = as_int(image["max_file_size_bytes"], "image.max_file_size_bytes")
    if max_file_size <= 0:
        raise ConfigurationError("image.max_file_size_bytes must be positive")

    hardware = section(raw, "hardware")
    vram_ratio = as_float(hardware["vram_warning_ratio"], "hardware.vram_warning_ratio")
    if not 0.0 < vram_ratio <= 1.0:
        raise ConfigurationError("hardware.vram_warning_ratio must be in (0, 1]")
    ram_ratio = as_float(hardware["ram_warning_ratio"], "hardware.ram_warning_ratio")
    if not 0.0 < ram_ratio <= 1.0:
        raise ConfigurationError("hardware.ram_warning_ratio must be in (0, 1]")
    timeout = as_int(hardware["pipeline_timeout_seconds"], "hardware.pipeline_timeout_seconds")
    if timeout <= 0:
        raise ConfigurationError("hardware.pipeline_timeout_seconds must be positive")

    competition = section(raw, "competition")
    quality_threshold = as_float(competition["quality_threshold"], "competition.quality_threshold")
    if not 0.0 <= quality_threshold <= 1.0:
        raise ConfigurationError("competition.quality_threshold must be in [0, 1]")
    max_hallucination = as_float(competition["max_hallucination_risk"], "competition.max_hallucination_risk")
    if not 0.0 <= max_hallucination <= 1.0:
        raise ConfigurationError("competition.max_hallucination_risk must be in [0, 1]")
    if as_int(competition["deterministic_seed"], "competition.deterministic_seed") < 0:
        raise ConfigurationError("competition.deterministic_seed must be >= 0")
    temperature = as_float(competition["gemma_temperature"], "competition.gemma_temperature")
    if temperature < 0.0:
        raise ConfigurationError("competition.gemma_temperature must be >= 0")
    release_threshold = as_float(competition["vram_release_threshold_mb"], "competition.vram_release_threshold_mb")
    if release_threshold < 0.0:
        raise ConfigurationError("competition.vram_release_threshold_mb must be >= 0")


def validate_model_config(raw: dict[str, object]) -> None:
    """Validate parsed model configuration."""
    for section_name in ("yolo", "blip", "gemma"):
        require_section(raw, section_name)

    yolo = section(raw, "yolo")
    _validate_ratio("yolo.confidence_threshold", as_float(yolo["confidence_threshold"], "yolo.confidence_threshold"))
    _validate_ratio("yolo.iou_threshold", as_float(yolo["iou_threshold"], "yolo.iou_threshold"))

    blip = section(raw, "blip")
    if as_int(blip["max_length"], "blip.max_length") <= 0:
        raise ConfigurationError("blip.max_length must be positive")

    gemma = section(raw, "gemma")
    if as_int(gemma["max_new_tokens"], "gemma.max_new_tokens") <= 0:
        raise ConfigurationError("gemma.max_new_tokens must be positive")
    temperature = as_float(gemma["temperature"], "gemma.temperature")
    if temperature < 0.0:
        raise ConfigurationError("gemma.temperature must be >= 0")

    plugins = raw.get("plugins")
    if plugins is None:
        return
    if not isinstance(plugins, dict):
        raise ConfigurationError("Section [plugins] must be a table")
    for key in ("detection", "vision_language", "reasoning"):
        if key not in plugins:
            continue
        plugin_id = str(plugins[key])
        if plugin_id not in BUILTIN_PLUGIN_IDS:
            available = ", ".join(sorted(BUILTIN_PLUGIN_IDS))
            raise ConfigurationError(
                f"plugins.{key} = {plugin_id!r} is not registered. Available: {available}"
            )


def validate_analysis_config(raw: dict[str, object]) -> None:
    """Validate parsed analysis heuristics configuration."""
    require_section(raw, "analysis")
    analysis = section(raw, "analysis")
    subsections = (
        "attributes",
        "relationships",
        "activity",
        "activity_reasoning",
        "semantic_reasoning",
        "context",
    )
    for subsection in subsections:
        require_section(analysis, subsection)

    attributes = section(analysis, "attributes")
    small = as_float(attributes["size_small_max_ratio"], "analysis.attributes.size_small_max_ratio")
    medium = as_float(attributes["size_medium_max_ratio"], "analysis.attributes.size_medium_max_ratio")
    if not 0.0 < small < medium <= 1.0:
        raise ConfigurationError("analysis.attributes size ratios must satisfy 0 < small < medium <= 1")

    zone_low = as_float(attributes["zone_split_low"], "analysis.attributes.zone_split_low")
    zone_high = as_float(attributes["zone_split_high"], "analysis.attributes.zone_split_high")
    if not 0.0 < zone_low < zone_high < 1.0:
        raise ConfigurationError("analysis.attributes zone splits must satisfy 0 < low < high < 1")


def validate_theme_config(raw: dict[str, object]) -> None:
    """Validate parsed theme configuration."""
    require_section(raw, "theme")
    theme = section(raw, "theme")
    for key in ("name", "stylesheet_path", "font_family", "accent_color", "background_color"):
        if key not in theme:
            raise ConfigurationError(f"theme.{key} is required")
    if as_int(theme["font_size"], "theme.font_size") <= 0:
        raise ConfigurationError("theme.font_size must be positive")


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1]")
