"""Typed accessors for parsed TOML configuration tables."""

from core.exceptions.config import ConfigurationError


def require_section(raw: dict[str, object], name: str) -> None:
    if name not in raw:
        raise ConfigurationError(f"Missing configuration section: [{name}]")


def section(raw: dict[str, object], name: str) -> dict[str, object]:
    require_section(raw, name)
    value = raw[name]
    if not isinstance(value, dict):
        raise ConfigurationError(f"Section [{name}] must be a table")
    return value


def get_str(table: dict[str, object], key: str) -> str:
    if key not in table:
        raise ConfigurationError(f"Missing configuration key: {key}")
    value = table[key]
    if not isinstance(value, str):
        raise ConfigurationError(f"{key} must be a string")
    return value


def get_int(table: dict[str, object], key: str) -> int:
    if key not in table:
        raise ConfigurationError(f"Missing configuration key: {key}")
    return as_int(table[key], key)


def get_float(table: dict[str, object], key: str) -> float:
    if key not in table:
        raise ConfigurationError(f"Missing configuration key: {key}")
    return as_float(table[key], key)


def get_bool(table: dict[str, object], key: str) -> bool:
    if key not in table:
        raise ConfigurationError(f"Missing configuration key: {key}")
    value = table[key]
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be a boolean")
    return value


def get_optional_str(table: dict[str, object], key: str, default: str) -> str:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, str):
        raise ConfigurationError(f"{key} must be a string")
    return value


def get_bool_default(table: dict[str, object], key: str, default: bool) -> bool:
    if key not in table:
        return default
    return get_bool(table, key)


def get_int_default(table: dict[str, object], key: str, default: int) -> int:
    if key not in table:
        return default
    return get_int(table, key)


def get_float_default(table: dict[str, object], key: str, default: float) -> float:
    if key not in table:
        return default
    return get_float(table, key)


def optional_section(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"Section [{name}] must be a table")
    return value


def as_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ConfigurationError(f"{field} must be an integer")


def as_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    raise ConfigurationError(f"{field} must be a number")
