# ADR-003: GPU Memory Strategy (2 GB VRAM)

**Status:** Accepted  
**Date:** 2026-07-30  
**Version:** 1.0

## Decision

Enforce a **single active heavy model slot** in VRAM. Models load lazily per pipeline stage, unload immediately after inference, and `MemoryManager` clears CUDA cache after every release. No permanent GPU reservation between runs.

## Context

Target hardware provides only **2 GB VRAM**. Three heavy models (YOLO, BLIP, Gemma) cannot coexist. Stability is prioritized over speed. Part 1 (2/4) mandates sequential execution and immediate release.

## Alternatives

1. **Keep all models loaded** — faster re-inference, impossible on 2 GB.
2. **Model swapping with partial weights** — complex, framework-specific.
3. **GPU-only, fail on OOM** — violates stability requirement.
4. **Cloud inference** — network dependency, out of v1 scope.

## Advantages

- Fits 2 GB VRAM budget.
- Predictable peak memory.
- Simple mental model for developers.
- Testable: assert VRAM drops after each stage.

## Disadvantages

- Model load latency between stages (mitigated by progress UI).
- Cannot parallelize heavy inference.
- Repeated load/unload wears SSD if weights re-read (mitigated by OS cache).

## Justification

Sequential single-slot loading is the only reliable strategy for 2 GB VRAM with three transformer/CNN models. `ModelManager` centralizes this so no module can violate the policy.
