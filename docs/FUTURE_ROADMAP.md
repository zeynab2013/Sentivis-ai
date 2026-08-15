# Sentivis AI — Future Roadmap (Part 6+)

**Version:** 1.0.0  
**Date:** 2026-07-30  
**Status:** Part 5 complete; Part 6 planning

---

## Part 6 — Maintainability & Evolution

The next development phase focuses on long-term sustainability without modifying frozen Part 5 subsystems.

### CI/CD & Automation

- GitHub Actions pipeline: pytest, ruff, mypy, certification on every PR
- Automated build matrix (development, production, portable, release)
- Nightly full certification with artifact archival
- Dependency vulnerability scanning

### Developer Tooling

- Pre-commit hooks aligned with certification checks
- Local dev container / reproducible environment spec
- Debug profiling mode (separate from production builds)
- Model download and cache pre-warming scripts

### Quality Assurance

- GPU benchmark gate in certification
- Visual regression tests for presentation layer (post-unfreeze)
- Expanded integration tests with real model smoke tests
- Performance baseline tracking

### Deployment

- Windows MSI/EXE installer (Inno Setup or WiX)
- `.ico` icon generation from SVG
- Offline deployment bundle with pre-cached models
- Auto-update channel infrastructure

### Product Evolution (Requires Presentation Unfreeze)

- Settings UI model status panel (DTOs ready on `ApplicationContext.runtime_status`)
- Help menu with About, documentation links
- Runtime config overrides (CLI/env)
- Internationalization and accessibility audit

### Architecture (Requires Architecture Unfreeze)

- Plugin marketplace / third-party engine loading
- Multi-image batch analysis
- Cloud inference backend option

---

## Prioritized Backlog

| Priority | Item | Phase |
|----------|------|-------|
| P0 | CI/CD with certification gate | Part 6 |
| P0 | Python 3.11 deployment documentation | Part 6 |
| P1 | Windows installer | Part 6 |
| P1 | Settings model status UI | Part 6 (post-unfreeze) |
| P2 | Offline model bundle | Part 7 |
| P2 | Batch analysis mode | Part 7 |

---

## References

- [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) — engineering improvement backlog
- [PROJECT_HEALTH_REPORT.md](PROJECT_HEALTH_REPORT.md) — current health baseline
- [PRODUCTION_CERTIFICATION_REPORT.md](PRODUCTION_CERTIFICATION_REPORT.md) — certification evidence
