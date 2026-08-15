"""Heavy AI model identifiers."""

from enum import Enum, auto


class ModelKind(Enum):
    """Registered heavy model types managed by ModelManager."""

    YOLO = auto()
    BLIP = auto()
    GEMMA = auto()
