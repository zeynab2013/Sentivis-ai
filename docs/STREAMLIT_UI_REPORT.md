# Sentivis AI — Streamlit UI Report

**Updated:** 2026-07-31

## Overview

The desktop presentation layer has been replaced with a **premium Streamlit application** (`streamlit_app/`) while preserving the frozen backend: DI container, pipeline orchestrator, vision/analysis/language services, startup infrastructure, and Python 3.10.11 compatibility.

The legacy PySide6 desktop remains available via `sentivis-desktop`.

## Launch

| Command | UI |
|---------|-----|
| `sentivis-ai` | Streamlit (default) |
| `sentivis-streamlit` | Streamlit (alias) |
| `sentivis-desktop` | Legacy PySide6 |

## Architecture

```
Streamlit UI (streamlit_app/)
    ↓
StreamlitBackend (bootstrap.py)
    ↓
StartupOrchestrator → DependencyContainer (unchanged)
    ↓
PipelineOrchestrator.analyze() / ExportManager.export()
    ↓
ui/formatters/result_formatters.py (display text)
```

No modifications were made to `services/pipeline/`, `vision/`, `analysis/`, `language/`, or `core/contracts/`.

## UI Features

| Feature | Implementation |
|---------|----------------|
| Premium dark theme | Deep purple, soft gold, glass cards, Inter font (`theme.py`) |
| Three-column layout | Sidebar nav + center viewer (~70%) + results panel |
| Image viewer | Overlays, zoom slider, mini map, before/after (`streamlit-image-comparison`) |
| Results panel | Executive summary, narrative, short caption, objects, relationships, activities, environment, image quality, metrics |
| Dashboard | Plotly stage timing chart, metric cards |
| Exports | PDF, HTML, Markdown, TXT, JSON via `ExportManager` |
| i18n | Hot language switch (en, fa, es, zh, fr) via existing `ui/i18n/translator` |
| Settings | Enhancement, SAM2, super resolution, comparison mode, competition mode, accessibility |
| Branding | `assets/branding/logo/logo.svg` with PNG fallback |

## Validation

| Gate | Status |
|------|--------|
| pytest (unit + acceptance) | Run after install |
| ruff / mypy | Run after install |
| Backend benchmark | See `docs/SEMANTIC_REPORT.md` for measured metrics |

## Measured Semantic Metrics (COCO val2017, 20 images)

| Metric | Measured |
|--------|----------|
| Caption quality | 87.1% |
| Hallucination rate | 3.2% |
| Overall semantic score | 83.4% |

Competition targets (>96% overall) are **not met** on the current benchmark. Re-run after Ollama/Gemma is available locally.

## Dependencies Added

- streamlit, streamlit-extras, streamlit-option-menu
- plotly, streamlit-aggrid, streamlit-image-comparison, streamlit-lottie

## Files Added

- `streamlit_app/main.py` — main application (entry point; do not name `app.py`)
- `streamlit_app/bootstrap.py` — backend adapter
- `streamlit_app/theme.py`, `i18n.py`, `preferences.py`, `branding.py`
- `streamlit_app/components/` — viewer, results, dashboard, exports
- `assets/branding/logo/logo.svg`
- `app/desktop_main.py` — legacy desktop entry
