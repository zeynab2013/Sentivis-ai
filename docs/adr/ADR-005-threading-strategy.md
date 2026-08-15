# ADR-005: Threading and Worker Strategy

**Status:** Accepted  
**Date:** 2026-07-30  
**Version:** 1.0

## Decision

- **UI thread:** interaction and rendering only.
- **`PipelineWorker` (QThread):** runs full pipeline or stage sequence; owns `ModelManager` access during run.
- **`ExportWorker` (QThreadPool):** PDF/JSON/TXT/image export.
- Progress and errors cross thread boundary via **Qt signals** carrying immutable DTOs or primitive summaries.
- **`CancellationToken`:** checked between stages; cooperative abort.

## Context

Part 1 (2/4) requires the UI to remain interactive during inference. Heavy stages (preprocessing, YOLO, BLIP, Gemma, export) must not block the main thread. User must never believe the app has frozen.

## Alternatives

1. **asyncio event loop** — poor integration with Qt main loop on Windows.
2. **multiprocessing** — duplicate model memory; exceeds 8 GB RAM.
3. **Synchronous pipeline on UI thread** — violates performance objectives.
4. **One thread per model** — violates sequential GPU policy.

## Advantages

- Native Qt integration.
- Clear thread ownership rules.
- Cancellation without killing process.
- Progress bars update via signals.

## Disadvantages

- Developer must respect Qt thread affinity rules.
- Python GIL limits CPU parallelism (acceptable; GPU stages are sequential).
- Debugging cross-thread issues requires discipline.

## Justification

QThread aligns with PySide6 (ADR-002) and enforces the multithreading policy without the memory duplication of multiprocessing on RAM-constrained hardware.
