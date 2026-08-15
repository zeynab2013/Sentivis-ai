# Sentivis AI — Branding Guide

## Asset layout

```
assets/branding/
  logo/logo.png      ← Primary logo (user-provided)
  icons/             ← Optional icon variants
  splash/            ← Optional splash assets
```

If `assets/branding/logo/logo.png` is absent, the application loads the bundled default from `assets/icons/app_icon.svg`.

## Runtime loading

- **Module:** `ui/branding/logo_provider.py`
- **API:** `load_logo_pixmap(size)`, `load_window_icon()`, `branding_logo_path()`
- **Never hardcode** logo paths in UI or export code — always use `logo_provider`.

## Where the logo appears

| Surface | Implementation |
|---------|----------------|
| Splash screen | `ui/widgets/splash_screen.py` |
| Sidebar | `ui/widgets/sidebar.py` |
| About dialog | `release/about_dialog.py` |
| Window / taskbar icon | `app/bootstrap.py` |
| PDF exports | `services/export/export_manager.py` |
| HTML exports | `HtmlExportWriter` in export manager |

## Custom logo

1. Place a PNG at `assets/branding/logo/logo.png` (recommended ≥256×256, transparent background).
2. Restart the app — no configuration change required.

## Presentation mode

Presentation mode hides developer chrome but retains narrative results; the sidebar logo is hidden with the sidebar itself.
