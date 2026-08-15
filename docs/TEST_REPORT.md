# Sentivis AI — Acceptance Test Report

**Generated:** 2026-07-31 08:20:43 UTC
**Status:** PASSED

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 38 |
| Passed | 38 |
| Failed | 0 |
| Skipped | 0 |
| Duration | 32.27s |

## Results by Category

### test_e2e_desktop (4/4 passed)

| Test | Result | Time (s) |
|------|--------|----------|
| `test_desktop_load_image_and_run_analysis` | PASS | 3.066 |
| `test_desktop_export_all_formats_from_ui` | PASS | 1.709 |
| `test_desktop_graceful_shutdown` | PASS | 0.582 |
| `test_desktop_analysis_updates_results_panel` | PASS | 1.740 |

### test_e2e_pipeline (3/3 passed)

| Test | Result | Time (s) |
|------|--------|----------|
| `test_complete_pipeline_produces_result` | PASS | 1.277 |
| `test_pipeline_objects_and_relationships` | PASS | 1.295 |
| `test_all_export_formats` | PASS | 1.356 |

### test_performance (5/5 passed)

| Test | Result | Time (s) |
|------|--------|----------|
| `test_startup_time_under_threshold` | PASS | 0.033 |
| `test_inference_time_under_threshold` | PASS | 1.357 |
| `test_ram_usage_recorded` | PASS | 1.276 |
| `test_vram_usage_recorded` | PASS | 1.309 |
| `test_gpu_memory_released_after_inference` | PASS | 1.449 |

### test_smoke (8/8 passed)

| Test | Result | Time (s) |
|------|--------|----------|
| `test_application_startup_completes_all_stages` | PASS | 0.033 |
| `test_dependency_container_creation` | PASS | 0.030 |
| `test_configuration_loading` | PASS | 0.033 |
| `test_plugin_loading` | PASS | 0.026 |
| `test_model_discovery_registers_three_models` | PASS | 0.030 |
| `test_asset_loading_inventory` | PASS | 0.017 |
| `test_startup_diagnostics_export` | PASS | 0.061 |
| `test_graceful_shutdown` | PASS | 0.201 |

### test_stress (9/9 passed)

| Test | Result | Time (s) |
|------|--------|----------|
| `test_multiple_sequential_analyses` | PASS | 6.343 |
| `test_large_image_analysis` | PASS | 1.391 |
| `test_missing_model_graceful_registry` | PASS | 0.005 |
| `test_invalid_image_rejected` | PASS | 0.009 |
| `test_corrupted_image_rejected` | PASS | 0.094 |
| `test_empty_file_rejected` | PASS | 0.009 |
| `test_empty_models_folder_discovery` | PASS | 0.003 |
| `test_gpu_unavailable_cpu_fallback` | PASS | 0.007 |
| `test_pipeline_cpu_fallback_completes` | PASS | 1.294 |

### test_ui (9/9 passed)

| Test | Result | Time (s) |
|------|--------|----------|
| `test_sidebar_buttons_exist_and_respond` | PASS | 0.262 |
| `test_image_viewer_zoom_controls` | PASS | 0.279 |
| `test_drag_and_drop_loads_image` | PASS | 0.272 |
| `test_presentation_mode_toggle` | PASS | 0.432 |
| `test_settings_dialog_opens_and_cancels` | PASS | 0.525 |
| `test_export_panel_buttons_present` | PASS | 1.759 |
| `test_results_panel_copy_and_expand` | PASS | 1.726 |
| `test_keyboard_shortcuts` | PASS | 0.410 |
| `test_about_dialog_via_release_info` | PASS | 0.384 |

## Manual Checklist

See [TEST_CHECKLIST.md](../TEST_CHECKLIST.md) for manual verification steps not covered by automation.
