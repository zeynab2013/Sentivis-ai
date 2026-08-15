# Sentivis AI — Troubleshooting Guide

**Version:** 1.0.0

## Startup Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Python version error | Python ≠ 3.10.x | Install Python 3.10.11 and recreate venv |
| Missing config file | Incomplete install | Verify `config/*.default.toml` exist |
| Models directory not writable | Permissions | Run from a writable project folder |
| CUDA warning | No NVIDIA GPU / driver | Expected on CPU-only systems; fallback enabled |

Check `logs/startup-diagnostics.txt` for full environment details.

## Model Loading

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| YOLO weights not found | First run | Allow automatic download or place `.pt` in `models/` |
| BLIP / Gemma download slow | Hugging Face fetch | Ensure network access; wait for first download |
| Out of memory | VRAM exhausted | Close other GPU apps; CPU fallback will activate |
| Validation failed | Config mismatch | Review `config/models.default.toml` |

## Analysis Failures

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Image rejected | Size/format limits | Use JPG/PNG under configured max size |
| Pipeline timeout | Long-running analysis | Increase `pipeline_timeout_seconds` in config |
| Partial results | Model recovery | Check logs; pipeline continues when possible |

## Export Issues

| Symptom | Fix |
|---------|-----|
| Export directory missing | Created automatically on startup |
| PDF export fails | Verify `reportlab` is installed |

## Logs

| File | Contents |
|------|----------|
| `logs/application.log` | General application events |
| `logs/pipeline.log` | Pipeline stage events |
| `logs/error.log` | Errors only |
| `logs/startup-diagnostics.json` | Startup health report |

## Getting Help

1. Run the runtime self-test via application startup (health score in diagnostics).
2. Review [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).
3. Include `startup-diagnostics.txt` when reporting issues.
