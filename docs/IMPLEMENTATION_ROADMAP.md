# SENTIVIS AI — Implementation Roadmap

**Version:** 1.0  
**Architecture:** 2.3 FROZEN  
**Date:** 2026-07-30  
**Authorization:** Awaiting Master Prompt Book Part 3

---

## 1. Module Build Order

Build and harden modules in dependency order. Each milestone requires passing unit tests for that module before proceeding.

```
Phase A: Foundation (complete — scaffold)
  1. core/          — config, contracts, logging, exceptions
  2. config/        — all TOML defaults + validation

Phase B: Features (scaffold complete — harden in Part 3)
  3. vision/        — validator, preprocessor, YOLO engine
  4. analysis/      — full heuristic pipeline
  5. language/      — BLIP, Gemma, prompts, refinement

Phase C: Services (scaffold complete — harden in Part 3)
  6. services/models/     — ModelManager, DeviceSelector
  7. services/memory/     — MemoryManager, ManagedResources
  8. services/pipeline/   — Orchestrator, StageRunner
  9. services/export/     — ExportManager
  10. services/plugins/   — PluginRegistry + builtins

Phase D: Application (scaffold complete)
  11. app/          — container, bootstrap, lifecycle

Phase E: Presentation (scaffold complete — polish in Part 3)
  12. ui/view_models/
  13. ui/controllers/
  14. ui/widgets/
  15. ui/themes/

Phase F: Quality & Release (Part 3–4)
  16. tests/        — unit, integration, e2e
  17. packaging/    — installer, model download
  18. docs/         — user-facing guides update
```

---

## 2. Development Milestones

### M1 — Model Inference Hardening (Week 1–2)

| Task | Module | Deliverable |
|------|--------|-------------|
| YOLO weights download + verify | vision/ | Working detection on sample images |
| BLIP load + caption | language/blip | Real RawCaption output |
| Gemma INT4 load + reason | language/gemma | Real refined narrative |
| GPU/CPU fallback validation | services/models | DeviceSelector tests |
| VRAM profiling | services/memory | ≤2 GB peak on target hardware |

**Exit criteria:** End-to-end pipeline on real image with all three models; progress UI responsive; cancellation works.

### M2 — UI Completion (Week 2–3)

| Task | Module | Deliverable |
|------|--------|-------------|
| Detection overlay on image viewer | ui/widgets | Bounding box render |
| Progress stage labels | ui/view_models | Stage-accurate progress |
| Export flow (TXT, JSON, PDF) | ui/controllers | Working export dialogs |
| Settings panel | ui/controllers | Theme + option toggles |
| History persistence | ui/view_models | Session history |

**Exit criteria:** Full user workflow without CLI; no UI freeze during inference.

### M3 — Test & Quality Gate (Week 3–4)

| Task | Module | Deliverable |
|------|--------|-------------|
| Unit coverage ≥80% | tests/unit | pytest-cov report |
| Integration tests with mocks | tests/integration | Pipeline stages isolated |
| ruff + mypy clean | all | CI gate |
| Performance benchmarks | tests/perf | ≤30s typical image on 2GB GPU |

**Exit criteria:** All quality gates in ENGINEERING_CONTRACT pass.

### M4 — Packaging & Release (Week 4–5)

| Task | Module | Deliverable |
|------|--------|-------------|
| Model download wizard | app/ | First-run setup |
| Windows installer | packaging/ | MSI or equivalent |
| User documentation | docs/ | INSTALLATION_GUIDE validated |
| Release checklist | docs/ | CHANGELOG, version tag |

**Exit criteria:** Clean install on Windows 11 target machine.

---

## 3. Parallel Workstreams

| Stream | Owner Focus | Can Parallelize With |
|--------|-------------|---------------------|
| AI engines | vision/, language/ | UI widgets (mock data) |
| Pipeline | services/pipeline | Analysis heuristics tuning |
| UI/UX | ui/ | Export formats |
| QA | tests/ | All streams (continuous) |

---

## 4. Out of Scope (Part 3 unless specified)

- Video pipeline (architecture stub only)
- Cloud inference plugins
- Multi-language UI localization
- Object tracking (noop tracker placeholder)
- Batch folder processing UI

These are architecturally supported but deferred per FUTURE_IMPROVEMENTS.md.

---

## 5. Definition of Done (Part 3)

Per `DEVELOPER_GUIDE.md` and `ENGINEERING_CONTRACT.md`:

- [ ] All public APIs typed and documented
- [ ] Unit + integration tests pass
- [ ] No architecture dependency violations
- [ ] Config validated at startup
- [ ] Memory policy enforced under load
- [ ] User-facing error messages actionable
- [ ] ADR for any deviation from v2.3

---

*Roadmap effective upon Part 3 authorization*
