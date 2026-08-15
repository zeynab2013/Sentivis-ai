# YOLO Path Fix Report

**Project:** Sentivis AI v1.0.0  
**Date:** 2026-07-31  
**Issue:** YOLO downloaded successfully but validation reported `"YOLO weights path is not a file: ."`

## Root Cause

`config/models.default.toml` sets `weights_path = ""` (empty string). The config loader converted any non-`None` value with:

```python
weights_path = Path(str(weights)) if weights is not None else None
```

An empty string became `Path("")`, which Python resolves to the **current working directory** (`.`). That caused two failures:

1. **`ModelValidationService._validate_yolo`** treated `weights_path` as configured and validated `.` as the weights file — producing the error `"YOLO weights path is not a file: ."`.
2. **`YoloEngine.load`** could pass `"."` to Ultralytics instead of the downloaded file in `models/yolo11x.pt`.

Discovery and download worked correctly; the bug was in **path normalization and validation precedence**, not in `DownloadManager`.

## Fix

Introduced centralized YOLO weight resolution in `services/runtime/yolo_weights.py`:

- `normalize_optional_path()` — treats `""`, `"."`, and whitespace as unset (`None`)
- `resolve_yolo_weights_path()` — searches configured path, registry search paths, and discovered `.pt` files (variant prefix, then any weight file)

Applied across the discovery pipeline:

| Component | Change |
|-----------|--------|
| `core/config/loader.py` | Empty `weights_path` → `None` |
| `services/runtime/model_registry.py` | Resolve absolute path on refresh |
| `services/runtime/model_validation.py` | Validate resolved file; store absolute path on record |
| `model_management/validation.py` | Resolve via search paths; store `.resolve()` |
| `model_management/registry.py` | Enrich YOLO records with resolved path |
| `vision/detection/yolo_engine.py` | Load from resolved path via search paths |
| `app/container.py` | Pass model search paths to `YoloEngine` |
| `app/startup/model_discovery.py` | Discover via resolver, not raw config |
| `services/models/model_validator.py` | Skip invalid configured paths |

## Files Modified

- `core/utils/paths.py`
- `core/config/loader.py`
- `services/runtime/yolo_weights.py` *(new)*
- `services/runtime/model_registry.py`
- `services/runtime/model_validation.py`
- `model_management/validation.py`
- `model_management/registry.py`
- `vision/detection/yolo_engine.py`
- `app/container.py`
- `app/startup/model_discovery.py`
- `services/models/model_validator.py`
- `tests/unit/services/runtime/test_yolo_weights.py` *(new)*
- `tests/unit/services/runtime/test_model_validation_yolo.py` *(new)*

## Path Resolution

| Stage | Old path | New resolved path |
|-------|----------|-------------------|
| Config `weights_path` | `Path("")` → `.` | `None` (auto-discover) |
| Registry `file_location` | `None` or unresolved | `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\models\yolo11x.pt` |
| Validation target | `.` (invalid) | Absolute path to existing 109 MB file |

**On-disk file:** `models/yolo11x.pt` — 114,636,239 bytes (~109 MB)

## Validation Output

```
weights_path config: None
models_dir: D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\models
registry file_location: D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\models\yolo11x.pt
runtime_status: ModelRuntimeStatus.READY
validate_before_inference passed: True
validate summary: Validation passed
resolved path: D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\models\yolo11x.pt
```

Non-blocking warning remains: `CUDA requested but unavailable; CPU fallback will be used` (expected on dev machine without GPU).

## Startup Verification

```
Startup complete with 1 warnings
startup errors: 0
Runtime self-test health score: 100 (was 89 before fix)
```

No Python version or YOLO path errors in startup diagnostics.

## First Inference

```
yolo loaded: True
yolo device: cpu
detections: 0
INFERENCE_OK
```

YOLO loads from the resolved absolute path and completes inference on CPU.

## Test Results

```
pytest: 145 passed (includes 4 new YOLO path tests)
ruff: PASS
mypy: PASS (315 files)
```

## Relocation Behavior

If `yolo11x.pt` is moved within configured search paths (`models/`, `model_search_paths`, project-relative config paths), `registry.refresh()` and `resolve_yolo_weights_path()` re-discover the file by:

1. Explicit configured absolute/relative path (if set)
2. Recursive scan of search paths for `.pt` files matching variant prefix (`yolo11x`)
3. Fallback to any `.pt` in search paths
4. Direct check for `{search_root}/yolo11x.pt`

The registry always stores the **absolute resolved path** after validation.
