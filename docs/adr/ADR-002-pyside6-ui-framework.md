# ADR-002: PySide6 Desktop UI Framework

**Status:** Accepted  
**Date:** 2026-07-30  
**Version:** 1.0

## Decision

Use **PySide6** (Qt 6 bindings) for the desktop UI on Windows 11.

## Context

Sentivis AI requires a premium commercial desktop experience: responsive layout, native file dialogs, theming, image viewer, progress indicators, and background workers with signal/slot integration. Target platform is Windows 11 with Python 3.11.

## Alternatives

1. **Tkinter** — stdlib, limited styling.
2. **CustomTkinter** — improved Tkinter, still limited for complex layouts.
3. **Electron + Python backend** — heavy memory footprint on 8 GB RAM.
4. **PyQt6** — GPL/commercial licensing complexity.
5. **Dear PyGui** — immediate mode; less suitable for complex multi-panel apps.

## Advantages

- Native look on Windows; mature widget set.
- QThread / QThreadPool integrate with worker policy.
- QSS theming for premium appearance.
- LGPL licensing via PySide6.
- Large ecosystem and documentation.

## Disadvantages

- Qt dependency size (~100 MB).
- Learning curve for signals/slots and thread rules.
- Packaging complexity (PyInstaller/cx_Freeze).

## Justification

PySide6 provides the best balance of professional UI capability, threading integration, and licensing for a commercial-grade Python desktop app. Memory overhead is acceptable compared to Electron. Threading model directly supports the multithreading policy (Part 1 §2).
