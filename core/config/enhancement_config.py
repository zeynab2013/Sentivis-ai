"""Image enhancement configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementConfig:
    """Adaptive image enhancement settings."""

    enabled: bool
    adaptive_mode: bool
    quality_threshold: float
    min_brightness: float
    max_brightness: float
    min_contrast: float
    min_sharpness: float
    max_blur_score: float
    max_noise_score: float
    enable_clahe: bool
    enable_gamma: bool
    enable_denoise: bool
    enable_sharpen: bool
    enable_color_correction: bool
    enable_super_resolution: bool
    super_resolution_min_dimension: int
    competition_always_enhance: bool
    # Centralized SR knobs (hardware-aware defaults for ~2 GB VRAM / 16 GB RAM).
    sr_model: str = "realesr-general-x4v3"
    sr_scale: int = 2
    sr_tile_size: int = 128
    sr_tile_overlap: int = 8
    sr_device: str = "auto"
    sr_max_output_side: int = 2048
    sr_allow_download: bool = True
    quality_low_threshold: float = 0.55
    quality_medium_threshold: float = 0.78
