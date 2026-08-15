# Model Installation Guide

## Requirements

- Windows 11, Python 3.10.11
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with 2 GB+ VRAM recommended
- Internet for first-time model download

## Automatic Installation

1. Launch Sentivis AI: `sentivis-ai`
2. If models are missing, the **Model Setup** dialog appears
3. Click **Download All** (or select individual models)
4. Provide `HF_TOKEN` if prompted for Gemma access
5. Wait for validation to complete — analysis is enabled when all mandatory models are ready

## Manual Installation

```bash
sentivis-models download
sentivis-models validate
sentivis-models status
```

### YOLO11x

Automatically downloaded to `models/yolo11x.pt` via Ultralytics on first download.

### BLIP Large

Cached via Hugging Face Hub (`Salesforce/blip-image-captioning-large`).

### Gemma 2 2B

Cached via Hugging Face Hub (`google/gemma-2-2b-it`). INT4 quantization is used on CUDA when available.

## Offline Use

If models are already installed, Sentivis AI runs offline. Missing models are listed explicitly without crashing.

## Troubleshooting

See [MODEL_RECOVERY_GUIDE.md](MODEL_RECOVERY_GUIDE.md).
