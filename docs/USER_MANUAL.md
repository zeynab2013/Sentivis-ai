# Sentivis AI — User Manual

**Version:** 1.0.0

## Overview

Sentivis AI is a desktop visual understanding application. It detects objects, builds a scene graph, generates observations with BLIP, and produces refined captions with Gemma.

## Main Window

| Area | Purpose |
|------|---------|
| Sidebar | Open image, run/cancel analysis, export, settings |
| Image viewer | Zoom, pan, fit-to-window |
| Dashboard | Stage progress, detections, caption, scene summary |
| Notifications | Success, warning, and error toasts |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open image |
| Ctrl+R | Run analysis |
| Escape | Cancel analysis |
| Ctrl+, | Settings |
| F1 | About |
| F11 | Presentation mode |
| Ctrl+F | Focus results search |

## Analysis Pipeline

The pipeline runs thirteen stages: validation, preprocessing, YOLO detection, attribute extraction, relationships, scene graph, activity analysis, context building, BLIP understanding, prompt building, Gemma reasoning, caption refinement, and quality evaluation.

Models load one at a time to conserve VRAM. CPU fallback is automatic when CUDA is unavailable.

## Export Formats

- **JSON** — structured pipeline result
- **TXT** — plain-text summary
- **Markdown** — formatted report
- **PDF** — printable report

## Settings

- **General** — application name, version, paths
- **AI Models** — models directory (read-only)
- **Diagnostics** — competition and diagnostics toggles
- **Appearance** — theme and font settings

## Model Status

Model availability and validation are tracked by the runtime model registry. Check startup diagnostics or logs if a model fails to load.

## Further Reading

- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
- [MODEL_GUIDE.md](MODEL_GUIDE.md)
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
