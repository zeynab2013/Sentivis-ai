"""Caption refiner interface."""

from typing import Protocol

from core.contracts.analysis import SceneContext
from core.contracts.language import RawCaption, RefinedCaption


class ICaptionRefiner(Protocol):
    """Refine raw captions for presentation while preserving evidence."""

    def refine(
        self,
        primary: RawCaption,
        fallback: RawCaption | None,
        context: SceneContext,
    ) -> RefinedCaption:
        """Return polished caption."""
        ...
