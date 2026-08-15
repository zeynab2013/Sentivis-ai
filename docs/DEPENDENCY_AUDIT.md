# Dependency Audit — Python 3.10.11

**Project:** Sentivis AI v1.0.0  
**Audit date:** 2026-07-31  
**Target:** Windows 11 · Python 3.10.11 · 8–16 GB RAM · NVIDIA 2 GB VRAM

## Policy

All runtime dependencies must declare Python 3.10 support. Upper bounds were added where unpinned upgrades caused breakage during migration validation (NumPy 2.x ABI, Hugging Face Hub 1.x API, PyTorch DLL conflicts).

## Runtime Dependencies

| Package | Constraint | Python 3.10 | Notes |
|---------|------------|-------------|-------|
| PySide6 | `>=6.6.0` | ✓ | Qt 6.6 desktop shell |
| torch | `>=2.1.0,<2.3.0` | ✓ | Pinned upper bound after 2.13 upgrade caused DLL issues with PySide6 |
| torchvision | `>=0.16.0,<0.18.0` | ✓ | Matched to torch range |
| ultralytics | `>=8.1.0` | ✓ | YOLO11 integration |
| transformers | `>=4.38.0,<5.0.0` | ✓ | BLIP + Gemma; v5 API differs |
| accelerate | `>=0.27.0` | ✓ | Model loading helpers |
| bitsandbytes | `>=0.42.0` (Windows) | ✓ | INT4 quantization on CUDA |
| Pillow | `>=10.2.0` | ✓ | Image I/O |
| psutil | `>=5.9.0` | ✓ | Hardware probing |
| reportlab | `>=4.1.0` | ✓ | PDF export |
| numpy | `>=1.26.0,<2.0.0` | ✓ | **Critical:** NumPy 2.x breaks compiled extensions |
| huggingface_hub | `>=0.21.0,<1.0.0` | ✓ | v1.x removed `resume_download` kwarg |
| tomli | `>=2.0.0` | ✓ | Required TOML parser on Python 3.10 |

## Development Dependencies

| Package | Constraint | Python 3.10 |
|---------|------------|-------------|
| pytest | `>=8.0.0` | ✓ |
| pytest-cov | `>=4.1.0` | ✓ |
| ruff | `>=0.3.0` | ✓ |
| mypy | `>=1.8.0` | ✓ |
| types-Pillow | latest | ✓ |

## Packages Evaluated — No Python 3.11 Requirement

All listed packages support Python 3.10. No dependency was removed; constraints were tightened for stability.

## Installation Verification

```bash
python --version          # Python 3.10.11
pip install -e ".[dev]"   # PASS — no version conflicts
```

## Lock Files

- **`requirements.txt`** — Runtime pins for pip users
- **`requirements-dev.txt`** — Includes `-r requirements.txt` plus dev tools
- **`pyproject.toml`** — Canonical source of truth for setuptools/pip editable installs

## Upgrade Guidance

Before bumping major versions of torch, numpy, transformers, or huggingface_hub:

1. Run full test suite on Python 3.10.11
2. Launch desktop app and run `python -m acceptance`
3. Verify model download (`sentivis-models download`) on target hardware
