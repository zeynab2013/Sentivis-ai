# Sentivis AI — Runtime Execution Report

**Generated:** 2026-07-31 09:43:23 UTC
**Entry point:** `sentivis-ai (app.main:main → app.bootstrap.bootstrap)`

## Startup

- Status: SUCCESS
- Window title: Sentivis AI — SEE. UNDERSTAND. INSPIRE.
- Window visible: True

## Models

| Kind | Status | Location | Detail |
|------|--------|----------|--------|
| YOLO | ModelRuntimeStatus.READY | D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\models\yolo11x.pt | CUDA requested but unavailable; CPU fallback will be used |
| BLIP | ModelRuntimeStatus.READY | Salesforce/blip-image-captioning-large | CUDA requested but unavailable; CPU fallback will be used |
| GEMMA | ModelRuntimeStatus.READY | google/gemma-2-2b-it | CUDA requested but unavailable; CPU fallback will be used |

## Analysis

- Image: `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\runtime_verify_sample.png`
- Started: True
- Completed: True
- Duration: 29.3s
- Caption: The scene content remains uncertain based on available evidence.
- Objects detected: 0
- Stages: VALIDATION, PREPROCESSING, YOLO_DETECTION, ATTRIBUTE_EXTRACTION, RELATIONSHIP_ANALYSIS, SCENE_GRAPH, ACTIVITY_ANALYSIS, SCENE_CONTEXT, BLIP_UNDERSTANDING, PROMPT_BUILDING, GEMMA_REASONING, CAPTION_REFINEMENT, QUALITY_EVALUATION

## Exports

| Format | Status | Bytes | Path |
|--------|--------|-------|------|
| json | ok | 1928 | `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\runtime_verify_exports\runtime_verify_sample_sentivis.json` |
| txt | ok | 1671 | `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\runtime_verify_exports\runtime_verify_sample_sentivis.txt` |
| md | ok | 1620 | `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\runtime_verify_exports\runtime_verify_sample_sentivis.md` |
| pdf | ok | 3050 | `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\runtime_verify_exports\runtime_verify_sample_sentivis.pdf` |

## UI Verification

- analyze_button_enabled: PASS
- caption_displayed: PASS
- copy_all_button: PASS
- export_panel_enabled: PASS
- history_widget: PASS
- image_viewer_loaded: PASS
- image_viewer_zoom: PASS
- notifications: PASS
- objects_displayed: PASS
- presentation_mode_hide_sidebar: PASS
- presentation_mode_restore_sidebar: PASS
- progress_timeline: PASS
- results_panel_caption: PASS
- search_shortcut: PASS
- settings_dialog: PASS

## Failures

- None

## Remaining Runtime Issues

- Gemma inference used fallback caption — authenticate with Hugging Face (HF_TOKEN) and accept the google/gemma-2-2b-it license
