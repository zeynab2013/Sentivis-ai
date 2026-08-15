"""TOML parser backend (stdlib tomllib or tomli fallback)."""

import importlib
import sys
from typing import BinaryIO, cast

_tomllib = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")
TOMLDecodeError: type[Exception] = _tomllib.TOMLDecodeError


def load_toml(handle: BinaryIO) -> dict[str, object]:
    return cast(dict[str, object], _tomllib.load(handle))
