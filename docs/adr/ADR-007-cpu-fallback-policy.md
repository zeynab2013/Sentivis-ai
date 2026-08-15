# ADR-007: CPU Fallback on GPU OOM

**Status:** Accepted  
**Date:** 2026-07-30  
**Version:** 1.0

## Decision

When GPU memory allocation fails, `ModelManager` automatically:

1. Logs WARNING with `MemoryManager` snapshot.
2. Releases GPU resources and clears CUDA cache.
3. Retries inference on **CPU** with the same model weights.
4. Notifies UI via progress event (device switch, not error).

If CPU execution also fails, apply stage-level recovery (degraded caption or abort stage) without crashing the application.

## Context

2 GB VRAM is a hard ceiling. BLIP and Gemma peak usage varies with input size and driver overhead. Application stability is more important than inference speed (Part 1 §2).

## Alternatives

1. **Fail hard on OOM** — poor UX; violates error recovery policy.
2. **Reduce batch/size and retry on GPU** — limited headroom; may still OOM.
3. **Skip model entirely** — loses capability without trying CPU.
4. **Prompt user to choose device** — adds friction; automatic is preferred.

## Advantages

- Pipeline completes on marginal hardware.
- No user intervention required.
- Logged for developer diagnosis.
- Consistent policy across all three heavy models.

## Disadvantages

- CPU inference significantly slower.
- May cause long runtimes on 8 GB RAM systems.
- Some CUDA ops may not have CPU equivalents (handled per engine).

## Justification

Automatic CPU fallback maximizes completion rate on the mandated 2 GB VRAM target while keeping device logic centralized in `ModelManager`, preventing ad-hoc handling in feature modules.
