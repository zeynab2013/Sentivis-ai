# Model Management

Automatic download and runtime management for production AI models.

## Overview

Part 5.5 adds the `model_management` package responsible for:

- Detecting missing production models at startup
- Background/resumable downloads from official sources
- Post-download validation and registry synchronization
- First-launch setup dialog
- Offline mode and cache maintenance

## Production Models

| Role | Model | Source |
|------|-------|--------|
| Detection | Ultralytics YOLO11x (`yolo11x.pt`) | Ultralytics |
| Vision-Language | Salesforce/blip-image-captioning-large | Hugging Face |
| Reasoning | google/gemma-2-2b-it | Hugging Face |

## CLI

```bash
python -m model_management status
python -m model_management download
python -m model_management validate
python -m model_management cache
```

Or: `sentivis-models status`

## Authentication

Set `HF_TOKEN` for gated Hugging Face models. Tokens can also be entered in the first-launch dialog and are stored in the user config directory (never logged).

## Testing Mode

Set `SENTIVIS_TEST_MODE=1` to skip first-launch dialogs during automated tests.

See also: [MODEL_INSTALLATION_GUIDE.md](MODEL_INSTALLATION_GUIDE.md)
