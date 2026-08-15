"""Multi-path model discovery for runtime asset management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiscoveredFile:
    """One file discovered in a model search path."""

    path: Path
    search_root: Path
    size_bytes: int


@dataclass(frozen=True)
class ModelDiscoveryResult:
    """Files discovered across configured search paths."""

    search_paths: tuple[Path, ...]
    files: tuple[DiscoveredFile, ...]

    @property
    def weight_files(self) -> tuple[DiscoveredFile, ...]:
        return tuple(item for item in self.files if item.path.suffix.lower() in {".pt", ".onnx"})


def resolve_model_search_paths(models_dir: Path, extra_paths: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    """Build de-duplicated search paths without hardcoded absolute locations."""
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in (models_dir, *extra_paths):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return tuple(ordered)


def discover_model_files(search_paths: tuple[Path, ...]) -> ModelDiscoveryResult:
    """Scan configured directories for model-related files."""
    discovered: list[DiscoveredFile] = []
    for root in search_paths:
        root.mkdir(parents=True, exist_ok=True)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pt", ".onnx", ".bin", ".safetensors"}:
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError:
                continue
            discovered.append(DiscoveredFile(path=path, search_root=root, size_bytes=size_bytes))
    return ModelDiscoveryResult(search_paths=search_paths, files=tuple(discovered))
