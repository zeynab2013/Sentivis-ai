# Migration Report — Python 3.10.11

**Project:** Sentivis AI v1.0.0  
**Migration:** Python 3.11 target → Python 3.10.11 native  
**Date:** 2026-07-31  
**Status:** COMPLETE

## Objective

Stop feature development and perform a full compatibility migration so Sentivis AI runs natively on:

- Windows 11
- Python 3.10.11
- NVIDIA GPU (2 GB VRAM)
- 8–16 GB RAM

## What Changed

### 1. Project configuration

| File | Change |
|------|--------|
| `pyproject.toml` | `requires-python = ">=3.10,<3.11"`, ruff `py310`, mypy `3.10`, dependency pins |
| `requirements.txt` | New — runtime pins for Python 3.10 |
| `requirements-dev.txt` | New — dev dependency bundle |

### 2. Runtime guards

| File | Change |
|------|--------|
| `app/startup/environment_probe.py` | Requires 3.10.x; rejects 3.11+ |
| `app/startup/recovery.py` | User-facing message for Python 3.10.11 |
| `release/validator.py` | Build host must be 3.10.x |
| `release/resources/installer_manifest.json` | `minimum_python: 3.10.11` |
| `certification/health_report.py` | Health report text updated |

### 3. CLI fix

`sentivis-certify` previously imported from `tests.integration`, which is not installed as a package. Stubs and pipeline harness moved to:

- `certification/pipeline_stubs.py`
- `certification/pipeline_harness.py`

### 4. Dependency stability

During validation, unpinned `pip install -e .` upgraded NumPy to 2.x and PyTorch to 2.13, causing test collection failures. Upper bounds added — see [DEPENDENCY_AUDIT.md](DEPENDENCY_AUDIT.md).

### 5. Documentation

Updated: README, SYSTEM_REQUIREMENTS, INSTALLATION_GUIDE, QUICK_START, TROUBLESHOOTING, DEVELOPER_GUIDE, TEST_CHECKLIST.

Generated:

- [PYTHON_3_10_COMPATIBILITY_REPORT.md](PYTHON_3_10_COMPATIBILITY_REPORT.md)
- [DEPENDENCY_AUDIT.md](DEPENDENCY_AUDIT.md)
- This report

## What Did NOT Change

- Frozen architecture (UI, pipeline, startup subsystems per Part 5 freeze)
- Application functionality and API surface
- Model catalog and pipeline stage order
- Plugin registry behavior

## Validation Results

```
Python 3.10.11
pip install -e ".[dev]"     ✓
ruff check .                ✓
mypy .                      ✓ (312 files)
pytest                      ✓ 130 passed, 10 skipped
python -m acceptance        ✓ 38/38 passed
Desktop app launch          ✓ (DI, registry, plugins, UI window)
sentivis-models status      ✓
sentivis-build --help       ✓
sentivis-certify --help     ✓
```

## Remaining Manual Steps

1. Rebuild release artifacts in `dist/` (`sentivis-build production`) — existing dist copies still reference Python 3.11.
2. Install production models on deployment hardware: `sentivis-models download`.
3. Historical validation reports in `docs/` may still mention Python 3.11 as the prior target; new installs should follow updated guides.

## Rollback

To revert (not recommended): restore `requires-python = ">=3.11,<3.12"` and environment probe check `(3, 11)`. The codebase contains no Python 3.11-only syntax, so either line works at the language level — runtime guards enforce the official version.
