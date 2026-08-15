# Release Engineering Validation Report — Part 5 (3/4)

**Project:** Sentivis AI  
**Architecture:** v2.3 FROZEN  
**AI Pipeline:** FROZEN  
**Presentation Layer:** FROZEN  
**Production Infrastructure:** FROZEN  
**Runtime Asset Management:** FROZEN  
**Date:** 2026-07-30  
**Scope:** Release engineering, packaging, deployment readiness

---

## Executive Summary

| Gate | Result |
|------|--------|
| **Part 5 (3/4) validation** | **PASS** |
| ruff | PASS — 0 issues |
| mypy (strict, 254 files) | PASS — 0 errors |
| pytest | PASS — 77/77 tests |
| Centralized version management | PASS |
| Build profiles (4) | PASS |
| Pre-build validation | PASS |
| Installer resource bundle | PASS |
| Production documentation set | PASS |
| About dialog (F1) | PASS |
| Development build artifact | PASS |
| Frozen layer modification | NONE (UI widgets, pipeline, runtime services) |

Part 5 (3/4) is complete. Sentivis AI is packaged for reliable Windows distribution with reproducible builds, centralized versioning, installer-ready resources, production documentation, and an About dialog — without modifying frozen architecture, pipeline, presentation widgets, or prior infrastructure layers.

---

## Static Analysis

```
python -m ruff check .                    → All checks passed
python -m mypy .                          → Success: 254 source files
python -m pytest tests/ -q                → 77 passed
python -m release development --validate-only → Build validation passed
python -m release development             → Build complete (dist/development/...)
```

---

## Centralized Version Management

Single source of truth: `release/version.py`

| Constant | Value |
|----------|-------|
| Application | 1.0.0 |
| Architecture | 2.3 |
| AI Pipeline | 1.0.0 |
| Model Registry | 1.0.0 |
| Configuration | 1.0.0 |

`ReleaseInfo` (`release/metadata.py`) adds build number, git commit, build timestamp, and profile. Loaded at container build and injected into:

- `ApplicationContext.release_info`
- Settings General tab via `app_version` overlay (`1.0.0 (build 0)`)
- Bootstrap startup log
- About dialog (F1)

Environment overrides supported: `SENTIVIS_APP_VERSION`, `SENTIVIS_BUILD_NUMBER`, `SENTIVIS_GIT_COMMIT`, `SOURCE_DATE_EPOCH`.

---

## Build Profiles

| Profile | Dev Tools | Docs | Samples | Git Required | Output |
|---------|-----------|------|---------|--------------|--------|
| development | Yes | Yes | Yes | No | `dist/development/` |
| production | No | Yes | No | No | `dist/production/` |
| portable | No | Yes | Yes | No | `dist/portable/` |
| release | No | Yes | Yes | Yes | `dist/release/` |

CLI: `python -m release <profile>` or `sentivis-build <profile>`

Each build writes `build_manifest.json` with release metadata and validation results.

---

## Build Validation

`BuildValidator` (`release/validator.py`) checks before every build:

- Configuration files (`config/*.default.toml`)
- Application entry point (`app/main.py`)
- Dependencies (warnings if missing in build env)
- Runtime assets (`assets/icons`, samples, templates)
- Model configuration completeness
- Icons and theme engine
- Required documentation set
- Release resources (LICENSE, THIRD_PARTY_NOTICES, installer manifest)

Build fails gracefully with enumerated errors when mandatory components are missing.

---

## Installer Readiness (Manifest-Only)

Prepared under `release/resources/` and copied to `installer/` in build output:

| Asset | Path |
|-------|------|
| Application metadata | `installer_manifest.json` |
| Icon | `assets/icons/app_icon.svg` |
| License | `LICENSE` (root + release/resources) |
| Third-party notices | `THIRD_PARTY_NOTICES.md` |
| Default config bundle | `default_config/*.default.toml` |
| Sample assets placeholder | `assets/samples/` |
| Export templates placeholder | `assets/export_templates/` |

Platform-specific installers (MSI/EXE) deferred per spec.

---

## Production Documentation

| Document | Status |
|----------|--------|
| Installation Guide | Updated (`docs/INSTALLATION_GUIDE.md`) |
| Quick Start Guide | **New** |
| User Manual | **New** |
| Troubleshooting Guide | **New** |
| Release Notes | **New** |
| System Requirements | **New** |
| Known Limitations | Existing |
| Directory Structure | **New** |

---

## About Dialog

`release/about_dialog.py` — production QDialog in the release package (not frozen `ui/`).

Displays: application name, version, build number, architecture/pipeline/registry/config versions, git commit, build timestamp, license, credits, website placeholder.

Triggered via **F1** shortcut wired in `release/hooks.py` from `app/bootstrap.py` without modifying frozen UI widgets.

---

## Deployment Audit

| Check | Result |
|-------|--------|
| Clean directory structure | PASS — documented in DIRECTORY_STRUCTURE.md |
| Temporary files in repo | PASS — logs/cache/dist gitignored |
| Debug artifacts | PASS — no debug scripts in source |
| Obsolete resources | PASS — previously removed panels confirmed absent |
| Unused assets | PASS — placeholders documented with README |
| Broken references | PASS — assets/icons populated |
| MANIFEST.in | **Added** for sdist completeness |

---

## Files Added / Modified

### Added

- `release/` — version, metadata, profiles, validator, builder, about dialog, hooks, CLI
- `release/resources/` — LICENSE, THIRD_PARTY_NOTICES, installer_manifest, default_config
- `assets/icons/app_icon.svg`, `assets/samples/`, `assets/export_templates/`
- `LICENSE`, `MANIFEST.in`
- Production docs (Quick Start, User Manual, Troubleshooting, Release Notes, System Requirements, Directory Structure)
- `tests/unit/release_engineering/` — 5 test modules

### Modified

- `app/container.py` — `release_info` on ApplicationContext, version overlay
- `app/bootstrap.py` — About shortcut, release log line
- `pyproject.toml` — release package, sentivis-build script, package-data, mypy exclude
- `README.md` — build commands
- `docs/INSTALLATION_GUIDE.md` — version alignment

### Not Modified (Frozen)

- `ui/` widgets and themes
- `services/pipeline/`
- `app/startup/`
- `services/runtime/`

---

## Known Limitations / Follow-up (Part 5 4/4)

| Item | Status |
|------|--------|
| Windows MSI/EXE installer | Deferred per spec |
| `.ico` conversion from SVG | SVG provided; ICO generation in Part 5 (4/4) |
| Python 3.11 build host | Required for production; 3.10 dev host shows warning only |
| Settings Help menu entry for About | F1 shortcut provided; menu wiring when UI unfreezes |

---

## Conclusion

Part 5 (3/4) release engineering objectives are met. Sentivis AI can be built reproducibly across four profiles, validates mandatory components before packaging, ships installer-ready resources and documentation, and displays complete version information in Settings and the About dialog — ready for final deployment hardening in Part 5 (4/4).
