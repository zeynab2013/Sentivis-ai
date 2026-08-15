# Sentivis AI — Model Guide

**Version:** 1.0

## Default Models

| Stage | Model | VRAM (approx) |
|-------|-------|---------------|
| Detection | YOLOv8n | 300–500 MB |
| Visual semantics | BLIP base | 900–1100 MB |
| Reasoning | Gemma-2-2b-it (INT4) | 900–1200 MB |

Configured in `config/models.default.toml`.

## Lifecycle

Models load only when needed and unload immediately after each stage. Only one heavy model occupies GPU memory at a time.

## CPU Fallback

If GPU memory is insufficient, models automatically retry on CPU. Analysis takes longer but completes reliably.

## Customization

Edit `config/models.default.toml`:

```toml
[yolo]
variant = "yolov8n"
preferred_device = "cuda"

[blip]
model_id = "Salesforce/blip-image-captioning-base"

[gemma]
model_id = "google/gemma-2-2b-it"
quantization = "int4"
```

## Hugging Face Authentication

Gemma may require accepting license terms and setting `HF_TOKEN` for download.

## Hardware Target

Designed for **2 GB VRAM**. Larger models are not recommended on target hardware.
