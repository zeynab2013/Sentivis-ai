# SENTIVIS AI — Architecture Freeze Report

**Version:** 2.3 FROZEN  
**Date:** 2026-07-30  
**Phase:** Architecture Phase — **CLOSED**  
**Next Phase:** Implementation (Master Prompt Book Part 3)

---

## 1. Freeze Declaration

Architecture **Version 2.3** is hereby **frozen**. All specifications, interface contracts, dependency rules, configuration schemas, and module boundaries defined in Part 2 are authoritative.

No implementation may violate this specification. Changes require:
1. New ADR in `docs/adr/`
2. Architecture version bump (2.4+)
3. Updated compliance report

Draft source code has been realigned to v2.3. **No new production/feature code** shall be written until Part 3 authorizes implementation.

---

## 2. Frozen Artifacts

| Document | Version | Role |
|----------|---------|------|
| `ARCHITECTURE.md` | 2.3 FROZEN | Layer model, pipeline, module catalog |
| `MODULE_SPECIFICATIONS.md` | 2.0 | Per-module responsibilities |
| `INTERFACE_SPECIFICATION.md` | 2.1 | Protocol design |
| `INTERFACE_CONTRACTS.md` | 2.1 | DTO/field reference |
| `STAGE_RUNNER_SPECIFICATION.md` | 2.1 | Stage execution lifecycle |
| `MANAGED_RESOURCES_SPECIFICATION.md` | 2.1 | GPU/resource lifecycle |
| `VIEW_MODEL_SPECIFICATION.md` | 2.1 | UI binding layer |
| `PLUGIN_ARCHITECTURE.md` | 2.1 | Plugin registry |
| `CONFIGURATION_CONTRACT.md` | 2.1 | Config schemas |
| `ARCHITECTURE_COMPLIANCE_REPORT.md` | 2.3 | Final validation |
| `ENGINEERING_CONTRACT.md` | 1.2 | Quality foundation |
| `HARDWARE_PERFORMANCE_POLICY.md` | 1.0 | 2 GB VRAM contract |
| ADR-001 … ADR-007 | — | Architecture decisions |

---

## 3. Frozen Package Structure

```
sentivis-ai/
├── app/                 # Composition root, bootstrap, lifecycle
├── core/                # Config, contracts, exceptions, logging, utils
├── vision/              # Validation, preprocessing, detection, tracking
├── analysis/            # Attributes, relations, graph, activity, context
├── language/            # BLIP, Gemma, prompts, refinement
├── services/            # Pipeline, models, memory, export, cache, plugins
├── ui/                  # PySide6 app, widgets, view_models, controllers
├── config/              # TOML defaults (app, models, themes, analysis)
├── tests/               # unit/, integration/
└── docs/                # Specifications, ADRs, guides
```

---

## 4. Frozen Dependency Rules

1. **ui/** → ViewModels + `ui/interfaces` + `core/contracts` only
2. **services/pipeline** → feature **interfaces** only (model-agnostic)
3. **vision/**, **analysis/**, **language/** → **core/** only
4. **app/container.py** → sole composition root
5. Cross-feature communication via **core/contracts DTOs** only
6. **PluginRegistry** → swappable AI engines without orchestrator changes

---

## 5. Frozen Pipeline

```
Validation → Preprocessing → YOLO Detection
  → Attributes → Relations → Scene Graph → Activity → Context
  → BLIP Understanding → Prompt Build → Gemma Reasoning → Refinement
  → Export / UI
```

Sequential GPU slot enforced by **ModelManager** + **MemoryManager** + **StageRunner**.

---

## 6. Frozen Configuration Files

| File | Purpose |
|------|---------|
| `config/app.default.toml` | App, hardware, paths, workers |
| `config/models.default.toml` | YOLO, BLIP, Gemma, plugin IDs |
| `config/themes.default.toml` | UI theme tokens |
| `config/analysis.default.toml` | Analysis heuristics thresholds |

Validation: `core/config/schema_validator.py` (fail-fast).

---

## 7. Final Audit Results

| Audit Area | Result |
|------------|--------|
| Folder naming consistency | PASS |
| Package naming consistency | PASS |
| Class naming consistency | PASS |
| Module placement | PASS |
| Import structure | PASS |
| Configuration layout | PASS |
| Documentation layout | PASS |
| Resource organization | PASS |
| Circular dependencies | NONE |
| Hidden dependencies | NONE |
| Service locator abuse | NONE |

---

## 8. Extension Readiness (Design Approved)

Architecture supports future extension **without redesign**:

| Extension | Mechanism |
|-----------|-----------|
| New AI models | PluginRegistry + ModelManager |
| New export formats | IExportManager + ExportManager |
| New languages | PromptBuilder + Gemma plugin swap |
| New UI themes | ThemeConfig + QSS |
| New plugins | PluginDescriptor registration |
| Batch processing | PipelineWorker pool (config-driven) |
| Video processing | New pipeline stages + vision tracking stub |
| Cloud inference | Remote engine plugins implementing IModelEngine |

---

## 9. Performance Readiness (2 GB VRAM)

| Requirement | Architecture Support |
|-------------|-------------------|
| Sequential model execution | ModelManager single-slot |
| GPU memory release | ManagedResources + MemoryManager |
| Lazy loading | ModelManager on-demand acquire |
| Background workers | PipelineWorker, export thread pool |
| CPU fallback | DeviceSelector + engine fallback |
| Cache policy | CacheManager |
| Cancellation | ICancellationToken + StageRunner |
| Resource cleanup | ApplicationLifecycle shutdown |

---

## 10. Phase Transition

| From | To |
|------|-----|
| Architecture Phase (Part 2) | **CLOSED** |
| Implementation Phase (Part 3) | **AWAITING AUTHORIZATION** |

Deliverables for Part 3 entry:
- `IMPLEMENTATION_READINESS_REPORT.md`
- `IMPLEMENTATION_ROADMAP.md`
- `RISK_REGISTER.md`

---

*Architecture v2.3 — frozen 2026-07-30*
