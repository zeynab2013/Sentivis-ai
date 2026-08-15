# Sentivis AI — Installation Guide

**Version:** 1.0.0

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 11 |
| Python | 3.10.11 (3.10.x) |
| GPU | NVIDIA CUDA-capable (2 GB VRAM) |
| RAM | 8 GB min · 16 GB recommended |
| Disk | SSD recommended · ~5 GB for models |

## Steps

### 1. Install Python 3.10.11

Download Python 3.10.11 from [python.org](https://www.python.org/downloads/release/python-31011/). Enable "Add Python to PATH".

### 2. Clone or extract project

```bash
cd "D:\path\to\SENTIVIS AI"
```

### 3. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 5. CUDA (optional but recommended)

Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) matching your PyTorch build.

### 6. Model weights

Models download automatically on first run via Hugging Face and Ultralytics:

- YOLOv8n
- BLIP base
- Gemma-2-2b-it (may require Hugging Face account token for gated models)

Set `HF_TOKEN` environment variable if required:

```bash
set HF_TOKEN=your_token_here
```

### 7. Launch

```bash
sentivis-ai
```

## Verification

```bash
pytest tests/unit/ -q
ruff check .
mypy app core vision language analysis services ui --strict
```

## Troubleshooting

| Issue | Action |
|-------|--------|
| CUDA OOM | CPU fallback activates automatically; close other GPU apps |
| Gemma load fails | Set HF_TOKEN; or disable Gemma in UI (BLIP fallback) |
| Slow first run | Models downloading; check `models/` and cache |
