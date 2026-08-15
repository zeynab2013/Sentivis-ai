# Sentivis AI — Quick Start Guide

**Version:** 1.0.0

## 1. Install

Follow [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) to install Python 3.10.11 and project dependencies.

## 2. Launch

```bash
sentivis-ai
```

Or:

```bash
python -m app.main
```

## 3. First Analysis

1. Press **Ctrl+O** or click **Open Image** in the sidebar.
2. Select a JPG or PNG image.
3. Press **Ctrl+R** or click **Run Analysis**.
4. Review detections, scene summary, and caption in the results panel.

## 4. Export Results

1. After analysis completes, open the **Export** panel.
2. Choose JSON, TXT, Markdown, or PDF.
3. Exports are saved to the configured `exports/` directory.

## 5. Settings & About

- **Ctrl+,** — open Settings (General, AI Models, Diagnostics)
- **F1** — About dialog with version and build information
- **F11** — Presentation mode for demos

## 6. Diagnostics

Startup diagnostics are written automatically to `logs/startup-diagnostics.json` and `.txt`.

For issues, see [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md).
