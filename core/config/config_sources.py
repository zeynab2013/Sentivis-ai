"""Configuration source metadata for diagnostics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigSource:
    """One loaded configuration file and its precedence layer."""

    name: str
    path: str
    layer: str


@dataclass(frozen=True)
class LoadedConfiguration:
    """Bundle of configuration sources loaded during startup."""

    sources: tuple[ConfigSource, ...]

    def summary(self) -> str:
        lines = [f"{source.name}: {source.path} ({source.layer})" for source in self.sources]
        return "\n".join(lines)
