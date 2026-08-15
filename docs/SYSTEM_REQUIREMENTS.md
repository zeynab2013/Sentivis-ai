# Sentivis AI — System Requirements

**Version:** 1.0.0

## Minimum

| Component | Requirement |
|-----------|-------------|
| Operating System | Windows 11 (64-bit) |
| Python | 3.10.11 (3.10.x) |
| CPU | 64-bit x86 processor |
| RAM | 8 GB |
| Disk | 10 GB free (includes model cache) |
| Display | 1100×720 minimum resolution |

## Recommended

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA CUDA-capable, 2 GB+ VRAM |
| RAM | 16 GB |
| Disk | SSD with 15 GB free |
| Network | Required for first-time Hugging Face model download |

## Software Dependencies

Installed automatically via `pip install -e .`:

- PySide6 6.6+
- PyTorch 2.1+
- Ultralytics 8.1+
- Transformers 4.38+
- Pillow, psutil, reportlab, numpy

## Optional

- CUDA toolkit matching PyTorch build for GPU acceleration
- Git (for release builds with commit metadata)

## Compatibility Matrix

| Python | Status |
|--------|--------|
| 3.10.11 | **Official supported version** |
| 3.10.x | Supported |
| 3.11+ | Not supported |
| 3.9 and earlier | Not supported |

See [PYTHON_3_10_COMPATIBILITY_REPORT.md](PYTHON_3_10_COMPATIBILITY_REPORT.md) for full audit details.
- macOS / Linux (Windows target for v1.0)
- Integrated GPU-only systems may run with CPU fallback (slower)
