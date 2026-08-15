# Sentivis AI — Deployment Summary

**Version:** 1.0.0  
**Certification:** PASSED (99/100)  
**Date:** 2026-07-30

---

## What Was Delivered in Part 5

| Phase | Scope | Status |
|-------|-------|--------|
| 5 (1/4) | Production infrastructure — startup, logging, diagnostics | FROZEN |
| 5 (2/4) | Runtime asset management — model registry, self-test | FROZEN |
| 5 (3/4) | Release engineering — builds, versioning, docs, About dialog | FROZEN |
| 5 (4/4) | Production certification — E2E validation, deployment readiness | COMPLETE |

---

## Deploying on a Clean Windows System

### 1. Prerequisites

- Windows 11 (64-bit)
- Python 3.10.11
- 16 GB RAM recommended
- NVIDIA GPU optional (CPU fallback supported)
- Network for first-time model download

See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md).

### 2. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Or deploy a pre-built artifact from `dist/production/sentivis-ai-1.0.0-build0/`.

### 3. First Launch

```bash
sentivis-ai
```

Startup automatically:

- Validates environment
- Creates runtime directories
- Loads layered configuration
- Discovers models
- Runs self-test (health score)
- Exports diagnostics to `logs/startup-diagnostics.json`

### 4. Verify

- Press **F1** for About dialog (version, build, architecture)
- Press **Ctrl+,** for Settings
- Check `logs/startup-diagnostics.txt` for health report

### 5. Build Profiles

| Command | Purpose |
|---------|---------|
| `sentivis-build development` | Dev artifact with all docs |
| `sentivis-build production` | Production deployment bundle |
| `sentivis-build portable` | Portable with samples |
| `sentivis-build release` | Release candidate with git metadata |
| `sentivis-certify` | Full production certification |

---

## Directory Layout After Deployment

```
sentivis-ai-1.0.0-build0/
├── app/ core/ services/ ui/ ...    # Application code
├── config/                          # Default TOML configuration
├── assets/                          # Icons, samples, templates
├── docs/                            # User and admin documentation
├── installer/                       # LICENSE, notices, manifest, config
└── build_manifest.json              # Build provenance
```

User data: `%APPDATA%/SentivisAI/config/`

---

## Support Resources

- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
- [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- [SUPPORT_GUIDE.md](SUPPORT_GUIDE.md)
- [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md)

---

## Next Phase

Part 6 focuses on long-term maintainability, CI/CD, developer tooling, and future evolution. All Part 5 subsystems remain frozen.
