"""Centralized design tokens for the Sentivis AI design system."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DesignTokens:
    """Visual design tokens — no widget may hardcode values outside this system."""

    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    card: str
    border: str
    success: str
    warning: str
    error: str
    text_primary: str
    text_secondary: str
    radius_sm: int
    radius_md: int
    radius_lg: int
    spacing_xs: int
    spacing_sm: int
    spacing_md: int
    spacing_lg: int
    animation_ms: int
    icon_sm: int
    icon_md: int
    icon_lg: int
    font_family: str
    font_size_sm: int
    font_size_md: int
    font_size_lg: int
    font_size_xl: int
    focus_ring: str

    def spacing(self, size: str) -> int:
        mapping = {
            "xs": self.spacing_xs,
            "sm": self.spacing_sm,
            "md": self.spacing_md,
            "lg": self.spacing_lg,
        }
        return mapping[size]

    def radius(self, size: str) -> int:
        mapping = {"sm": self.radius_sm, "md": self.radius_md, "lg": self.radius_lg}
        return mapping[size]

    def font_size(self, scale: str) -> int:
        mapping = {
            "sm": self.font_size_sm,
            "md": self.font_size_md,
            "lg": self.font_size_lg,
            "xl": self.font_size_xl,
        }
        return mapping[scale]

    def icon_size(self, size: str) -> int:
        mapping = {"sm": self.icon_sm, "md": self.icon_md, "lg": self.icon_lg}
        return mapping[size]
