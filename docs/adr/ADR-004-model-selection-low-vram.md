# ADR-004: AI Model Selection for Low VRAM

**Status:** Accepted  
**Date:** 2026-07-30  
**Version:** 1.0

## Decision

Default model variants optimized for 2 GB VRAM:

| Role | Model | Rationale |
|------|-------|-----------|
| Detection | **YOLOv8n** | ~6 MB weights; ~300–500 MB peak VRAM at 640 px |
| Vision-Language | **BLIP base** (`Salesforce/blip-image-captioning-base`) | ~990 MB; fits alone after YOLO release |
| Reasoning | **Gemma-2B** (INT4 quantized) | ~900–1200 MB; tight fit; CPU fallback expected on some runs |

Image preprocessing caps YOLO input at **640 px**; validator rejects images above **4096 px** on longest edge.

## Context

Hardware contract fixes 2 GB VRAM and 8 GB RAM minimum. Larger models (YOLOv8x, BLIP-2, Gemma-7B) exceed budget. Model selection must not assume high-end GPUs.

## Alternatives

1. **YOLOv8s/m** — better accuracy, higher VRAM.
2. **BLIP-2 / LLaVA** — exceed 2 GB alone.
3. **Gemma-7B** — requires >4 GB VRAM even quantized.
4. **Single multimodal model** — simplifies slot policy but no model fits 2 GB with full pipeline quality.

## Advantages

- Pipeline completes on target hardware.
- Each model fits individually after prior release.
- Quantized Gemma preserves reasoning capability at lower footprint.
- Swappable via `ModelRegistry` without architecture change.

## Disadvantages

- Lower accuracy vs larger variants.
- Gemma-2B limited reasoning depth vs 7B+.
- INT4 quantization may reduce caption nuance.

## Justification

Commercial demonstration on judge hardware requires reliable completion over state-of-the-art accuracy. Model variants are configuration-driven (`config/models.default.toml`) so users with more VRAM can upgrade variants without code changes.
