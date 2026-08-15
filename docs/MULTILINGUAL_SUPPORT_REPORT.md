# Multilingual Support Report

## Catalogs

| Code | File |
|------|------|
| en | `translations/en.json` |
| fa | `translations/fa.json` |
| es | `translations/es.json` |
| zh | `translations/zh.json` |
| fr | `translations/fr.json` |

## Runtime

- **Translator:** `ui/i18n/translator.py` — `tr(key, **params)`, `language_changed` signal
- **Persistence:** `ui/preferences/ui_preferences.py` via QSettings
- **Hot reload:** Changing language in Settings updates sidebar, results, export panel, image viewer immediately (no restart)

## Covered UI surfaces

- Sidebar actions and tooltips
- Results panel sections and toolbar
- Export panel
- Settings dialog tabs and controls
- Image viewer toolbar and overlay toggles
- Splash screen
- About dialog title (via `tr`)

## Settings integration

Settings → Appearance → Language selector (5 languages). Save applies theme, language, competition mode, and accessibility options together.

## Fallback

Missing keys fall back to English, then to the key string.
