"""Production model catalog — single source of truth for required models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.constants.model_kinds import ModelKind


class DownloadSource(str, Enum):  # noqa: UP042
    """Official download channels supported by Sentivis AI."""

    ULTRALYTICS = "ultralytics"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    DIRECT = "direct"


@dataclass(frozen=True)
class ProductionModelSpec:
    """Immutable specification for one production model."""

    kind: ModelKind
    model_id: str
    display_name: str
    version: str
    provider: str
    download_source: DownloadSource
    local_filename: str
    expected_size_bytes: int | None
    checksum_sha256: str | None
    license_name: str
    mandatory: bool
    quantization: str | None = None
    ollama_tag: str | None = None
    hf_repo_id: str | None = None


PRODUCTION_MODELS: tuple[ProductionModelSpec, ...] = (
    ProductionModelSpec(
        kind=ModelKind.YOLO,
        model_id="yolo11x",
        display_name="Ultralytics YOLO11x",
        version="11.x",
        provider="Ultralytics",
        download_source=DownloadSource.ULTRALYTICS,
        local_filename="yolo11x.pt",
        expected_size_bytes=114_000_000,
        checksum_sha256=None,
        license_name="AGPL-3.0",
        mandatory=True,
    ),
    ProductionModelSpec(
        kind=ModelKind.BLIP,
        model_id="Salesforce/blip-image-captioning-large",
        display_name="BLIP Image Captioning Large",
        version="large",
        provider="Salesforce / Hugging Face",
        download_source=DownloadSource.HUGGINGFACE,
        local_filename="",
        expected_size_bytes=1_900_000_000,
        checksum_sha256=None,
        license_name="BSD-3-Clause",
        mandatory=True,
        hf_repo_id="Salesforce/blip-image-captioning-large",
    ),
    ProductionModelSpec(
        kind=ModelKind.GEMMA,
        model_id="google/gemma-2-2b-it",
        display_name="Gemma 2 2B Instruct",
        version="2b-it",
        provider="Google / Hugging Face",
        download_source=DownloadSource.HUGGINGFACE,
        local_filename="",
        expected_size_bytes=5_000_000_000,
        checksum_sha256=None,
        license_name="Gemma Terms of Use",
        mandatory=True,
        quantization="int4",
        hf_repo_id="google/gemma-2-2b-it",
        ollama_tag="gemma2:2b",
    ),
)


def spec_for_kind(kind: ModelKind) -> ProductionModelSpec:
    """Return the production spec for a model kind."""
    for spec in PRODUCTION_MODELS:
        if spec.kind == kind:
            return spec
    raise KeyError(f"No production spec for {kind}")


def total_expected_download_bytes() -> int:
    """Sum expected sizes for mandatory models missing local copies."""
    return sum(spec.expected_size_bytes or 0 for spec in PRODUCTION_MODELS if spec.mandatory)
