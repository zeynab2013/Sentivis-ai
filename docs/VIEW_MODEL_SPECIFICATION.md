# SENTIVIS AI — ViewModel Specification

**Version:** 2.1  
**Part:** 2 / 4 (section 2)

---

## 1. Principle

UI widgets render **ViewModels only**. Service objects never exposed to widgets.

```
Widget → ViewModel (ui/view_models/) → Controller → Service Interface
```

---

## 2. Package Structure

```
ui/
  interfaces/           ← Protocol definitions
  view_models/
    pipeline_view_model.py
    export_view_model.py
    history_view_model.py
    application_state.py
  controllers/          ← Translate ViewModel commands to services
  widgets/              ← Pure rendering (future split from app_window)
```

---

## 3. `PipelineViewModel`

### Mission

Present pipeline state to UI in formatted, bindable properties.

### Responsibilities

| Area | Detail |
|------|--------|
| State management | `idle | running | completed | failed | cancelled` |
| Progress | `progress_percent`, `stage_label`, `device_label` |
| Status | `status_message` for status bar |
| Notifications | `warnings`, `error_message` |
| Formatting | Caption text, scene summary markdown |
| Validation | Disable analyze when no image loaded |

### Properties (Qt bindable via signals)

| Property | Type | Source |
|----------|------|--------|
| `state` | `PipelineUiState` | Internal FSM |
| `progress_percent` | `float` | `StageProgress` |
| `stage_label` | `str` | `StageProgress.stage.display_name` |
| `caption_text` | `str` | `RefinedCaption.text` |
| `scene_summary` | `str` | Formatted `SceneContext` |
| `image_path` | `Path \| None` | User selection |
| `is_analyze_enabled` | `bool` | Derived |

### Commands

| Command | Delegates to |
|---------|--------------|
| `load_image(path)` | Controller stores path + preview signal |
| `start_analysis()` | `PipelineController.analyze_image()` |
| `cancel_analysis()` | `PipelineController.cancel()` |

### Dependencies

- `ui/interfaces/IPipelineViewModel`
- Receives updates via controller signals — **not** direct orchestrator access

### Thread Model

- Properties updated on main thread via Qt signals from controller
- ViewModel holds no locks; main thread only

### Acceptance Criteria

- [ ] Zero imports from `services.pipeline.orchestrator`
- [ ] Zero imports from `app.container`
- [ ] All widget bindings go through ViewModel properties

---

## 4. `ExportViewModel`

| Property | Purpose |
|----------|---------|
| `is_export_enabled` | True when result available |
| `last_export_path` | Last successful path |
| `export_status` | idle / exporting / done / error |

Commands: `export_json()`, `export_txt()`, `export_pdf()` → `ExportController`

---

## 5. `HistoryViewModel`

| Property | Purpose |
|----------|---------|
| `entries` | `tuple[HistoryEntry, ...]` |
| `selected_index` | Current selection |

`HistoryEntry`: `{ image_name, caption_preview, timestamp, result_id }`

---

## 6. `IApplicationFacade`

Replaces direct `ApplicationContext` import in UI.

```python
class IApplicationFacade(Protocol):
    @property
    def pipeline_view_model(self) -> IPipelineViewModel: ...
    @property
    def export_view_model(self) -> IExportViewModel: ...
    @property
    def history_view_model(self) -> IHistoryViewModel: ...
    @property
    def window_title(self) -> str: ...
```

`AppWindow` receives `IApplicationFacade` only — constructed in `app/bootstrap.py`.

---

## 7. Presentation Models

DTO → display string formatters live in ViewModels, not widgets:

| Formatter | Input | Output |
|-----------|-------|--------|
| `format_scene_context` | `SceneContext` | Multi-line summary |
| `format_progress` | `StageProgress` | `"Detection — 35%"` |
| `format_history_entry` | `PipelineResult` | One-line preview |

---

*Architecture Phase — implementation pending*
