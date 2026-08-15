# Python 3.10 Compatibility Report

**Project:** Sentivis AI v1.0.0  
**Migration date:** 2026-07-31  
**Official runtime:** Python 3.10.11 (3.10.x line)

## Executive Summary

Sentivis AI has been migrated from a Python 3.11-only target to **native Python 3.10.11** support on Windows 11. The migration covers source code, runtime guards, dependency pins, tooling configuration, CLI entry points, and documentation.

## Compatibility Matrix

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| `requires-python` | `>=3.11,<3.12` | `>=3.10,<3.11` | ✓ |
| Ruff target | `py311` | `py310` | ✓ |
| Mypy target | `3.11` | `3.10` | ✓ |
| Environment probe | Rejects Python 3.10 | Accepts 3.10.x, rejects 3.11+ | ✓ |
| TOML parsing | `tomllib` / `tomli` fallback | `tomli` required on 3.10 | ✓ |
| Installer manifest | `minimum_python: 3.11` | `minimum_python: 3.10.11` | ✓ |

## Language Feature Audit

| Feature | Found in project? | Action |
|---------|-------------------|--------|
| `typing.Self` | No | N/A |
| `typing.override` | No | N/A |
| `typing.Required` / `NotRequired` | No | N/A |
| `StrEnum` | No — uses `(str, Enum)` pattern | Compatible |
| `ExceptionGroup` | No | N/A |
| `match` / `case` | No (only XML testcase iteration) | Compatible |
| PEP 695 type parameters | No | N/A |
| `tomllib` stdlib | Conditional import with `tomli` fallback | Compatible |
| `X \| None` union syntax | Yes — deferred via `from __future__ import annotations` | Compatible |
| `datetime.UTC` | No — uses `timezone.utc` | Compatible |

## Runtime Validation (Python 3.10.11)

| Check | Result |
|-------|--------|
| `pip install -e ".[dev]"` | PASS |
| `ruff check .` | PASS |
| `mypy .` | PASS (312 files) |
| `pytest` | 130 passed, 10 skipped |
| `python -m acceptance` | 38/38 passed |
| Desktop app launch | PASS (startup, UI, DI, registry, plugins) |
| `sentivis-ai` entry point | PASS |
| `sentivis-models` | PASS |
| `sentivis-build --help` | PASS |
| `sentivis-certify --help` | PASS |

## Code Changes Summary

- **`app/startup/environment_probe.py`** — Accept Python 3.10.x; reject 3.9 and 3.11+
- **`app/startup/recovery.py`** — Recovery message references Python 3.10.11
- **`release/validator.py`** — Build validation requires 3.10.x
- **`certification/system_verification.py`** — Removed runtime dependency on `tests` package
- **`certification/pipeline_stubs.py`** / **`pipeline_harness.py`** — CLI-safe stub pipeline for certification
- **`acceptance/report.py`** — Fixed variable shadowing (mypy)
- **`language/gemma/gemma_engine.py`** — Typed transformer calls via `Any` indirection
- **Test typing fixes** — Annotations and casts in acceptance/integration tests

## Not Supported

- Python 3.9 and earlier
- Python 3.11 and later (explicitly rejected by environment probe)
- macOS / Linux (Windows target for v1.0)

## Recommendations

1. Use **Python 3.10.11** from [python.org](https://www.python.org/downloads/release/python-31011/) for production and CI.
2. Pin dependencies via `requirements.txt` or `pip install -e .` — do not upgrade NumPy to 2.x or PyTorch beyond pinned ranges without re-validation.
3. Rebuild `dist/` artifacts after migration (`sentivis-build production`).
