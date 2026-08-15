# Runtime Asset Validation Report — Part 5 (2/4)

**Project:** Sentivis AI  
**Architecture:** v2.3 FROZEN  
**AI Pipeline:** FROZEN  
**Presentation Layer:** FROZEN  
**Production Infrastructure:** FROZEN  
**Date:** 2026-07-30  
**Scope:** Model management, runtime assets, deployment reliability

---

## Executive Summary

| Gate | Result |
|------|--------|
| **Part 5 (2/4) validation** | **PASS** |
| ruff | PASS — 0 issues |
| mypy (strict, 239 files) | PASS — 0 errors |
| pytest | PASS — 72/72 tests |
| Central Model Registry | PASS |
| Model validation before inference | PASS |
| Model status lifecycle | PASS |
| Multi-path model discovery | PASS |
| Runtime asset managers (9 categories) | PASS |
| Cache maintenance | PASS |
| Runtime self-test + health score | PASS |
| Frozen layer modification | NONE (UI, pipeline, feature engines) |

Part 5 (2/4) is complete. Sentivis AI now tracks every configured model in a centralized registry, validates models before inference, manages nine runtime asset categories, maintains cache hygiene, and runs a startup self-test with an overall health score — without requiring users to inspect internal folders.

---

## Static Analysis

```
python -m ruff check .                    → All checks passed
python -m mypy .                          → Success: 239 source files
python -m pytest tests/ -q                → 72 passed
```

---

## Central Model Registry

`CentralModelRegistry` (`services/runtime/model_registry.py`) tracks for every model:

| Field | Source |
|-------|--------|
| Identifier | Model configuration |
| Display name | Derived (e.g. `YOLO yolov8n`, `BLIP (blip-image-captioning-base)`) |
| Version | Plugin descriptor |
| Provider | Ultralytics / Hugging Face / Google |
| Supported tasks | Plugin pipeline stages |
| File location | Discovered weights or configured path |
| Device compatibility | Configured preferred device + CPU |
| Runtime status | Lifecycle enum (see below) |
| Integrity status | Validation outcome |
| Last validation time | UTC timestamp on each validation |

Wired into `ApplicationContext.model_registry` and connected to `ModelManager` for automatic status updates during load/unload.

---

## Model Validation

`ModelValidationService` (`services/runtime/model_validation.py`) verifies before inference:

- Model exists and configuration is present
- Required files exist and are readable (YOLO weights)
- Configuration matches registry identifier
- Plugin version is supported (`1.0.0`)
- SHA-256 checksum when local weight files are available
- Device availability (CUDA fallback reported as non-blocking warning)

`ModelManager.acquire()` calls `validate_before_inference()` and raises `ModelLoadError` with precise failure summaries when validation fails.

---

## Model Status Lifecycle

`ModelRuntimeStatus` enum (`services/runtime/model_status.py`):

| Status | When set |
|--------|----------|
| `installed` | Model configured and available |
| `missing` | Required configuration or files absent |
| `loading` | `ModelManager.acquire()` started |
| `ready` | Validation passed, not yet loaded |
| `in_use` | Model loaded and active |
| `released` | Model unloaded after use |
| `unavailable` | Reserved for future runtime failures |
| `validation_failed` | Pre-inference validation failed |

Exposed to UI adapters via `RuntimeStatusProvider.model_statuses()` → `ModelStatusView` DTOs on `ApplicationContext.runtime_status` (no presentation widget changes required).

---

## Model Discovery

`services/runtime/model_discovery.py`:

- `resolve_model_search_paths()` — de-duplicated search paths from `models_dir` + optional `model_search_paths` in config
- `discover_model_files()` — recursive scan for `.pt`, `.onnx`, `.bin`, `.safetensors`
- No hardcoded absolute paths; all paths resolved via `resolve_user_path()`

Configuration extension in `config/app.default.toml`:

```toml
[paths]
model_search_paths = []
```

---

## Runtime Asset Managers

Nine dedicated managers in `services/runtime/assets.py`:

| Manager | Category | Root |
|---------|----------|------|
| `ModelsAssetManager` | models | `paths.models_dir` |
| `IconsAssetManager` | icons | `assets/icons` |
| `ThemesAssetManager` | themes | `ui/themes` |
| `ConfigurationAssetManager` | configuration | user + default config |
| `SamplesAssetManager` | samples | `assets/samples` |
| `ExportTemplatesAssetManager` | export_templates | `assets/export_templates` |
| `LogsAssetManager` | logs | `paths.logs_dir` |
| `CacheAssetManager` | cache | `paths.cache_dir` |
| `TemporaryAssetManager` | temporary | `{project}/tmp` |

Each manager provides `ensure_directory()`, `inventory()`, and `verify()` with category-specific warnings.

---

## Cache Maintenance

`CacheMaintenanceService` (`services/runtime/cache_maintenance.py`):

| Capability | Method |
|------------|--------|
| Cache size reporting | `report_size()` → bytes, file counts, orphans |
| Safe cache cleanup | `safe_cleanup()` → removes corrupt JSON, orphans, stale temp files |
| Orphaned file detection | `detect_orphans()` → non-`.json` files in cache dir |
| Temporary file cleanup | Included in `safe_cleanup()` |

---

## Runtime Self-Test

`SelfTestRunner` (`services/runtime/self_test.py`) verifies at container build:

| Check | Weight |
|-------|--------|
| Configuration files | 20 |
| Each model (YOLO, BLIP, Gemma) | 15 each |
| Each asset category | 5 each |
| Logging handlers | 10 |
| Plugin registration | 10 |
| Write permissions | 10 |

Returns `SelfTestReport` with weighted **health score** (0–100). Accessible via `RuntimeStatusProvider.health_score`.

---

## Integration

```
DependencyContainer.build()
  → CentralModelRegistry (discovery + validation)
  → ModelManager(model_catalog=central_registry)
  → build_runtime_assets() → 9 asset managers
  → CacheMaintenanceService
  → SelfTestRunner → RuntimeStatusProvider
  → ApplicationContext.runtime_status + .model_registry
```

---

## Test Coverage (New)

| Test file | Coverage |
|-----------|----------|
| `tests/unit/services/runtime/test_model_discovery.py` | Multi-path discovery |
| `tests/unit/services/runtime/test_model_registry.py` | Registry + status provider |
| `tests/unit/services/runtime/test_model_validation.py` | Remote model validation |
| `tests/unit/services/runtime/test_assets.py` | All 9 asset managers |
| `tests/unit/services/runtime/test_cache_maintenance.py` | Orphan detection + cleanup |
| `tests/unit/services/runtime/test_self_test.py` | Health score generation |

---

## Files Added / Modified

### Added

- `services/runtime/` — model registry, validation, discovery, assets, cache maintenance, self-test, status provider

### Modified

- `app/container.py` — wires runtime services into `ApplicationContext`
- `services/models/model_manager.py` — pre-inference validation + status updates
- `core/config/app_config.py` — `model_search_paths` on `PathsConfig`
- `core/config/loader.py` — loads optional search paths
- `config/app.default.toml` — `model_search_paths = []`

### Not Modified (Frozen)

- `ui/` — presentation layer
- `services/pipeline/` — AI pipeline orchestration
- `app/startup/` — production infrastructure startup sequence
- `vision/`, `language/`, `analysis/` — feature engines

---

## Known Limitations / Follow-up

| Item | Status |
|------|--------|
| UI widget binding for model status | Deferred until presentation unfreeze; DTOs ready on `ApplicationContext.runtime_status` |
| Hugging Face model file integrity | Skipped for remote models (download-on-first-use) |
| Benchmark log channel caller migration | Unchanged from Part 5 (1/4) |

---

## Conclusion

Part 5 (2/4) runtime asset management objectives are met. Users receive precise model validation messages, automatic status tracking, multi-path discovery, dedicated asset managers, cache maintenance, and a startup health score — without modifying frozen architecture, pipeline, presentation, or production infrastructure layers.
