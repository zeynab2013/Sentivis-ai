# UI Extension Points

**Date:** 2026-07-30  
**Purpose:** Document safe extension surfaces for post-freeze UI work (Part 5+)

## Architecture Boundaries

```
┌─────────────────────────────────────────┐
│           Presentation Layer (ui/)       │  ← Extension points below
├─────────────────────────────────────────┤
│     Controllers (ui/controllers/)        │
├─────────────────────────────────────────┤
│     Services / Pipeline (FROZEN)         │
└─────────────────────────────────────────┘
```

Do not modify pipeline services or architecture when extending UI.

## Safe Extension Points

### 1. New widgets

Add to `ui/widgets/` and wire through `AppWindow._build_layout()`. Follow existing patterns:
- Import components from `ui/components/`
- Bind to ViewModel properties, never services
- Use design tokens via QSS classes

### 2. New components

Add to `ui/components/` and register in `ui/components/__init__.py`. Add QSS rules to `ui/themes/theme_engine.py`.

### 3. New ViewModels

Add to `ui/view_models/`, expose via `ApplicationFacade` and `IApplicationFacade` protocol. Wire in `app/container.py`.

### 4. Result formatters

Extend `ui/formatters/result_formatters.py` for new display sections. Update `ResultsPanelWidget` and `CollapsibleSection` instances.

### 5. Export formats

Add writer in `services/export/export_manager.py` and section builder in `services/export/report_builder.py`. Wire button in `ExportPanelWidget` and method in `ExportViewModel`.

### 6. Theme tokens

Add fields to `ui/design/tokens.py`, update `dark_tokens.py` / `light_tokens.py`, and add QSS selectors in `theme_engine.py`.

### 7. Presentation mode sections

Extend `ResultsPanelWidget.set_presentation_mode()` and `AnalysisDashboardWidget.set_presentation_mode()` to show/hide additional sections.

### 8. Notifications

Use `NotificationCenter.show_info/success/warning/error()` from `AppWindow`.

### 9. Empty states

Use `EmptyStateWidget(title, message, action_hint=...)` for new panels.

### 10. Settings categories

Add tab in `SettingsDialog` with `SettingField` tuples. Extend `SettingsViewModel` for new values.

## Extension Checklist

When adding UI features:

- [ ] No hardcoded colors or spacing — use tokens
- [ ] No direct service imports in widgets
- [ ] Keyboard shortcut or tooltip documented
- [ ] Empty state for new panels
- [ ] Presentation mode behavior considered
- [ ] pytest / ruff / mypy pass

## Do Not Extend (Frozen)

- `services/pipeline/` — AI pipeline orchestration
- `core/contracts/` — domain contracts
- `analysis/`, `vision/`, `language/` — analysis engines
- Architecture v2.3 module boundaries

## Recommended Next Extensions (Part 5+)

1. History click-to-restore via `HistoryViewModel.selected_index`
2. Save-as export dialog in `ExportController`
3. System theme auto-detection in `ThemeManager`
4. Custom competition export template
5. Installer splash screen using existing design tokens
