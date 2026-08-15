# Sentivis AI — Maintenance Guide

**Version:** 1.0.0  
**Audience:** Operators and maintainers

---

## Routine Maintenance

### Logs

Rotating logs in `logs/`:

| File | Retention | Action |
|------|-----------|--------|
| `application.log` | 5 backups × 5 MB | Auto-rotated |
| `pipeline.log` | 5 backups × 5 MB | Auto-rotated |
| `error.log` | 5 backups × 5 MB | Auto-rotated |
| `startup-diagnostics.json` | Per startup | Review after updates |

Archive or delete old logs when disk space is low.

### Cache

```python
# Via runtime services (when integrated) or manually:
# Delete contents of cache/ and tmp/ directories
```

Use `CacheMaintenanceService.safe_cleanup()` for orphan and corrupt entry removal.

### Models

- YOLO weights: place `.pt` files in `models/` or allow auto-download
- BLIP/Gemma: cached by Hugging Face in user cache after first download
- Check model status via startup diagnostics or runtime registry

### Configuration

User overrides: `%APPDATA%/SentivisAI/config/*.toml`

Never edit `config/*.default.toml` in production deployments — use user overrides instead.

---

## Version Upgrades

1. Back up `%APPDATA%/SentivisAI/config/` and `exports/`
2. Deploy new build artifact or `pip install -e .`
3. Run `python -m certification --skip-build` to validate
4. Launch and verify `logs/startup-diagnostics.txt`
5. Review [RELEASE_NOTES.md](RELEASE_NOTES.md)

Centralized version constants: `release/version.py`

---

## Health Monitoring

Run periodic certification:

```bash
sentivis-certify --skip-build
```

Review `docs/PROJECT_HEALTH_REPORT.md` output.

Self-test health score ≥ 70 indicates operational readiness.

---

## Backup Recommendations

| Data | Location | Priority |
|------|----------|------------|
| User config | `%APPDATA%/SentivisAI/config/` | High |
| Exports | `exports/` | Medium |
| Models | `models/` + HF cache | Medium |
| Logs | `logs/` | Low |

---

## Re-certification Triggers

Re-run full certification after:

- Python or dependency upgrades
- Configuration schema changes
- New model plugin registration
- OS or hardware changes on deployment targets

```bash
python -m certification
```

---

## Frozen Subsystems

Do not modify without explicit unfreeze:

- Architecture v2.3 (`core/contracts/`, layer boundaries)
- AI Pipeline (`services/pipeline/`)
- Presentation (`ui/`)
- Production Infrastructure (`app/startup/`)
- Runtime Assets (`services/runtime/`)
- Release Engineering (`release/`)

Maintenance changes should target `certification/`, documentation, and deployment scripts only.
