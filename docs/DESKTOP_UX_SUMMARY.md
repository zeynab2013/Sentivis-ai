# Desktop UX Summary

**Product:** Sentivis AI  
**Phase:** Part 4 Complete  
**Date:** 2026-07-30

## Overview

Sentivis AI presents a professional three-panel desktop experience optimized for visual understanding workflows: load an image, run analysis, inspect structured results, and export professional reports.

## Layout

```
┌─────────────┬──────────────────────────┬─────────────────────┐
│  Sidebar    │     Image Viewer         │    Dashboard        │
│             │                          │                     │
│  Brand      │  Zoom / pan / overlays   │  Stage progress     │
│  Session    │  Drag-and-drop           │  Results panel      │
│  Navigation │                          │  Export panel       │
│  History    │                          │                     │
│  Status     │                          │                     │
└─────────────┴──────────────────────────┴─────────────────────┘
```

## User Journey

1. **Start** — Application opens with guided empty states
2. **Load** — Open or drag-drop an image (Ctrl+O)
3. **Analyze** — Run pipeline (Ctrl+R); skeleton loading + stage progress
4. **Inspect** — Caption, scene summary, objects, relationships, activities, environment, quality, metrics
5. **Export** — TXT, Markdown, JSON, or PDF with full report sections
6. **Repeat** — Session history tracks recent analyses locally

## Presentation Mode (F11)

For live demos and judging:

- Hides sidebar entirely — maximizes image workspace
- Shows only **Final Caption** and **Scene Summary**
- Hides stage timings, metrics, quality report, developer exports
- Hides overlay toggles and zoom toolbar
- Preserves analyze, export (TXT/PDF), and all core functionality
- Reversible instantly via F11 or sidebar Presentation button

## Key Interactions

| Action | Shortcut |
|--------|----------|
| Open image | Ctrl+O |
| Run analysis | Ctrl+R |
| Cancel | Esc |
| Settings | Ctrl+, |
| Search results | Ctrl+F |
| Zoom in/out/fit | Ctrl++ / Ctrl+- / Ctrl+0 |
| Presentation mode | F11 |

## Feedback Systems

- **Status badge** — unified operation status with text prefix
- **Notifications** — non-blocking toasts for load, complete, export, warnings
- **Modals** — errors and overwrite confirmation only
- **Empty states** — guided next actions in every major panel

## Design Language

- Dark theme default (Sentivis brand palette)
- Token-driven spacing, typography, colors, and focus rings
- Card-based sections with collapsible detail
- Subtle hover states (<250ms perceived)

## Export Formats

| Format | Use case |
|--------|----------|
| TXT | Plain-text full report |
| Markdown | Structured shareable report |
| JSON | Machine-readable full payload |
| PDF | Printable summary for judges |

All formats include caption, scene summary, objects, relationships, activities, context, quality metrics, and pipeline metrics.
