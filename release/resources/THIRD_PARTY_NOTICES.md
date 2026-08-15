# Third-Party Notices

Sentivis AI includes or depends on the following third-party components.

## Runtime Dependencies

| Component | License | Notes |
|-----------|---------|-------|
| PySide6 | LGPL / Commercial | Qt for Python desktop UI |
| PyTorch | BSD-style | Deep learning runtime |
| torchvision | BSD-style | Vision utilities |
| Ultralytics YOLO | AGPL-3.0 | Object detection |
| Hugging Face Transformers | Apache 2.0 | BLIP and Gemma model loading |
| Accelerate | Apache 2.0 | Model acceleration |
| bitsandbytes | MIT | Quantization (Windows) |
| Pillow | HPND | Image I/O |
| psutil | BSD-3-Clause | System metrics |
| ReportLab | BSD-style | PDF export |
| NumPy | BSD-3-Clause | Numerical operations |

## Models (Downloaded at Runtime)

| Model | Provider | License / Terms |
|-------|----------|-----------------|
| YOLOv8 | Ultralytics | AGPL-3.0 |
| BLIP | Salesforce | BSD-3-Clause |
| Gemma 2 | Google | Gemma Terms of Use |

## Development Tools (Not Shipped in Production Builds)

| Component | License |
|-----------|---------|
| pytest | MIT |
| ruff | MIT |
| mypy | MIT |

Users are responsible for complying with all third-party licenses when deploying
Sentivis AI in production or competition environments.
