# Real Model Validation Report

**Date:** 2026-07-30  
**Scope:** Production models (YOLO11x, BLIP Large, Gemma 2 2B)

## Status

Real-model end-to-end validation requires:

- Python 3.11
- ~7 GB+ free disk for model cache
- Network access for first download
- HF_TOKEN for Gemma (if gated)

Run validation:

```bash
set SENTIVIS_TEST_MODE=
python -m model_management download
python -m model_management validate
python -m pytest tests/integration/test_real_models.py -v -m real_models
```

## Test Images (10 scenarios)

The integration suite covers: people, vehicles, indoor, outdoor, animals, crowded, landscape, food, low-light, multi-object.

## Environment Note

Automated CI runs use `SENTIVIS_TEST_MODE=1` with stub pipeline tests. Real model validation is executed on deployment hardware with models installed.

## Expected Results

- YOLO11x loads and detects objects
- BLIP Large generates visual observations
- Gemma 2 2B produces captions
- Full pipeline completes without exceptions
- GPU memory released after inference

See [MODEL_RUNTIME_VALIDATION_REPORT.md](MODEL_RUNTIME_VALIDATION_REPORT.md) for runtime metrics.
