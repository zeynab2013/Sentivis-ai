"""UI theme configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThemeConfig:
    """Theme and appearance settings."""

    name: str
    stylesheet_path: Path
    font_family: str
    font_size: int
    accent_color: str
    background_color: str
