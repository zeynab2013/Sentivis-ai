# Portability Report

**Validated:** 2026-07-31 16:25:42 UTC
**Project root:** `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI`

## Runtime directories

Auto-created on launch: `models/`, `cache/`, `logs/`, `exports/`, `tmp/`, `assets/user/`, `.sentivis/`

## Path resolution

- Project root detected via `pyproject.toml` / `config/app.default.toml` markers
- Override with environment variable `SENTIVIS_PROJECT_ROOT`
- All configured paths resolved relative to project root

## Readiness

- **System Ready** — Core requirements satisfied. Review optional notes below.
- Offline mode: False
- GPU available: False

## Dependencies

| Package | Available | Version | Required |
|---------|-----------|---------|----------|
| Python | Yes | 3.10.11 | Yes |
| PyTorch | Yes | 2.2.2+cpu | Yes |
| Transformers | Yes | 4.57.6 | Yes |
| OpenCV | Yes | 4.11.0 | Yes |
| Streamlit | Yes | 1.60.0 | Yes |
| Ultralytics (YOLO) | Yes | 8.4.94 | Yes |
| Pillow | Yes | 12.3.0 | Yes |
| NumPy | Yes | 1.26.4 | Yes |
| BLIP | Yes | 4.57.6 | No |
| Ollama | Yes | cli | No |
| SAM2 | No | optional | No |