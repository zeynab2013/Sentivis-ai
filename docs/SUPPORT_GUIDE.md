# Sentivis AI — Support Guide

**Version:** 1.0.0

---

## Getting Help

### Self-Service Diagnostics

1. Check `logs/startup-diagnostics.txt` — full environment and model status
2. Press **F1** in the application for version and build information
3. Open Settings (**Ctrl+,**) → Diagnostics tab
4. Run `sentivis-certify --skip-build` for health score

### Common Issues

| Issue | First Step |
|-------|------------|
| Won't start | Verify Python 3.10.11; read startup diagnostics |
| Model load failure | Check `config/models.default.toml`; verify network for HF models |
| Slow analysis | Expected on CPU; verify GPU drivers for CUDA |
| Export fails | Confirm `exports/` directory is writable |
| Config not applied | Check `%APPDATA%/SentivisAI/config/` overrides |

Full troubleshooting: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

---

## Information to Include in Support Requests

- Application version (F1 About dialog)
- `logs/startup-diagnostics.txt`
- `logs/error.log` (last 50 lines)
- Windows version and Python version
- GPU model (if applicable)
- Steps to reproduce

---

## Escalation Path

1. **User documentation** — Quick Start, User Manual, Troubleshooting
2. **Diagnostics export** — `startup-diagnostics.json`
3. **Certification report** — `docs/production_certification.json`
4. **Developer guide** — [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for engineering issues

---

## Known Limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and [KNOWN_AI_LIMITATIONS.md](KNOWN_AI_LIMITATIONS.md).

Key points for support staff:

- First model run requires internet for BLIP/Gemma download
- YOLO weights may auto-download on first detection
- Python 3.10 and earlier are not supported
- Presentation layer enhancements require Part 6 unfreeze

---

## Maintenance Contacts

- Documentation: `docs/` directory
- License inquiries: see root `LICENSE`
- Third-party compliance: `release/resources/THIRD_PARTY_NOTICES.md`
