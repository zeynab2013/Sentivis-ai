"""Default enhancement configuration for tests and fallbacks."""

from core.config.enhancement_config import EnhancementConfig

DEFAULT_ENHANCEMENT_CONFIG = EnhancementConfig(
    enabled=True,
    adaptive_mode=True,
    quality_threshold=0.72,
    min_brightness=0.28,
    max_brightness=0.82,
    min_contrast=0.22,
    min_sharpness=0.18,
    max_blur_score=0.65,
    max_noise_score=0.55,
    enable_clahe=True,
    enable_gamma=True,
    enable_denoise=True,
    enable_sharpen=True,
    enable_color_correction=True,
    # Real SR is the intended path for MEDIUM/LOW on competition demos.
    enable_super_resolution=True,
    # Attempt SR when the shorter side is below this (500×333 farm image qualifies).
    super_resolution_min_dimension=720,
    competition_always_enhance=False,
    sr_model="realesr-general-x4v3",
    sr_scale=2,
    sr_tile_size=128,
    sr_tile_overlap=8,
    sr_device="auto",
    sr_max_output_side=2048,
    sr_allow_download=True,
    quality_low_threshold=0.55,
    quality_medium_threshold=0.78,
)
