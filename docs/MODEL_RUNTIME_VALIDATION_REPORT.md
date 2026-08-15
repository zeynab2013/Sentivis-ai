# Model Runtime Validation Report

**Date:** 2026-07-30

## Validation Gates

| Gate | Automated | Real Models |
|------|-----------|-------------|
| Catalog IDs (yolo11x, BLIP large, Gemma 2B) | PASS | — |
| Download manager unit tests | PASS | — |
| Install validation | PASS | — |
| Offline/auth handling | PASS | — |
| Registry enrichment | PASS | — |
| Full pytest suite | PASS (131 tests) | — |
| Real pipeline (10 images) | — | Run on target hardware |

## Commands

```bash
pytest tests/ -q
ruff check .
mypy .
sentivis-models status
```

## Runtime Checks

After installing production models on target hardware:

1. `sentivis-ai` launches without errors
2. Model Setup completes or all models show `validated`
3. Analysis runs on a sample image using real models (no stubs)
4. Exports succeed (JSON, TXT, MD, PDF)
5. `nvidia-smi` / Task Manager shows GPU memory released after analysis

## Hardware Profile

Target: Windows 11 · Python 3.11 · NVIDIA 2 GB VRAM · 8–16 GB RAM

The hardware advisor warns when BLIP Large or Gemma may exceed available RAM/VRAM.
