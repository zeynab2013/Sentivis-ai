"""Activity analysis entry point — delegates to minimal fallback for legacy imports."""

from analysis.activity.minimal_activity_fallback import MinimalActivityFallback

ActivityAnalyzer = MinimalActivityFallback

__all__ = ["ActivityAnalyzer", "MinimalActivityFallback"]
