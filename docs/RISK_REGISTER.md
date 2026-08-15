# SENTIVIS AI — Risk Register

**Version:** 1.0  
**Architecture:** 2.3 FROZEN  
**Date:** 2026-07-30  
**Review Cycle:** Per milestone (M1–M4)

---

## Risk Matrix Legend

| Likelihood | Impact | Priority |
|------------|--------|----------|
| High | High | **P1 — Critical** |
| High | Medium | **P2 — High** |
| Medium | High | **P2 — High** |
| Medium | Medium | **P3 — Medium** |
| Low | Any | **P4 — Low** |

---

## Active Risks

### R-001 — 2 GB VRAM Insufficient for Concurrent Loads

| Field | Value |
|-------|-------|
| **ID** | R-001 |
| **Category** | Performance / Hardware |
| **Priority** | P2 |
| **Likelihood** | Medium |
| **Impact** | High |
| **Description** | BLIP + Gemma sequential load may still peak above 2 GB during transition if release is delayed. |
| **Mitigation** | ModelManager single-slot; explicit `release()` in StageRunner post-stage; MemoryManager VRAM warnings; CPU fallback via DeviceSelector. |
| **Owner** | services/models, services/memory |
| **Status** | Mitigated (architecture); verify in M1 profiling |

---

### R-002 — Model Download / Offline Failure

| Field | Value |
|-------|-------|
| **ID** | R-002 |
| **Category** | Operational |
| **Priority** | P2 |
| **Likelihood** | High |
| **Impact** | Medium |
| **Description** | First-run requires Hugging Face / Ultralytics weight downloads; network failures block core functionality. |
| **Mitigation** | Model download wizard (M4); clear error messages; offline mode with cached weights; MODEL_GUIDE documents manual placement. |
| **Owner** | app/, docs/ |
| **Status** | Open — implement in Part 3 |

---

### R-003 — PySide6 + CUDA Driver Conflicts

| Field | Value |
|-------|-------|
| **ID** | R-003 |
| **Category** | Platform |
| **Priority** | P3 |
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Description** | Windows GPU drivers + PyTorch CUDA builds may mismatch on user machines. |
| **Mitigation** | Document supported CUDA version in INSTALLATION_GUIDE; CPU fallback always available; test matrix on target hardware. |
| **Owner** | docs/, services/models |
| **Status** | Open — document in M4 |

---

### R-004 — Gemma INT4 Quantization on Windows

| Field | Value |
|-------|-------|
| **ID** | R-004 |
| **Category** | AI / Platform |
| **Priority** | P2 |
| **Likelihood** | Medium |
| **Impact** | High |
| **Description** | bitsandbytes INT4 may fail on some Windows GPU configurations. |
| **Mitigation** | Fallback to FP16/CPU; plugin descriptor documents resource requirements; test early in M1. |
| **Owner** | language/gemma |
| **Status** | Open — validate in M1 |

---

### R-005 — Insufficient Test Coverage

| Field | Value |
|-------|-------|
| **ID** | R-005 |
| **Category** | Quality |
| **Priority** | P3 |
| **Likelihood** | High |
| **Impact** | Medium |
| **Description** | Current 10 tests insufficient for production; UI untested; real model paths untested. |
| **Mitigation** | M3 milestone: ≥80% coverage; integration tests with stubs (baseline exists); mock engine tests. |
| **Owner** | tests/ |
| **Status** | In progress — baseline integration tests added |

---

### R-006 — UI Thread Blocking

| Field | Value |
|-------|-------|
| **ID** | R-006 |
| **Category** | UX |
| **Priority** | P3 |
| **Likelihood** | Low |
| **Impact** | High |
| **Description** | Accidental synchronous model call on main thread would freeze UI. |
| **Mitigation** | PipelineWorker enforced; code review gate; no orchestrator imports in ui/ except via controller. |
| **Owner** | ui/, services/pipeline |
| **Status** | Mitigated (architecture) |

---

### R-007 — Plugin Misconfiguration

| Field | Value |
|-------|-------|
| **ID** | R-007 |
| **Category** | Configuration |
| **Priority** | P4 |
| **Likelihood** | Low |
| **Impact** | Medium |
| **Description** | User overrides plugin IDs to unregistered engines. |
| **Mitigation** | schema_validator validates against BUILTIN_PLUGIN_IDS; fail-fast at startup. |
| **Owner** | core/config |
| **Status** | Closed |

---

### R-008 — Scope Creep Before Part 3

| Field | Value |
|-------|-------|
| **ID** | R-008 |
| **Category** | Process |
| **Priority** | P2 |
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Description** | Feature implementation before Part 3 authorization violates Master Prompt Book contract. |
| **Mitigation** | Architecture v2.3 frozen; EXECUTION_CONTRACT suspended; compliance report gates. |
| **Owner** | Process |
| **Status** | Closed (Architecture Phase complete) |

---

## Risk Summary

| Priority | Count | Open |
|----------|-------|------|
| P1 | 0 | 0 |
| P2 | 4 | 3 |
| P3 | 3 | 2 |
| P4 | 1 | 0 |
| **Total** | **8** | **5** |

---

## Review Schedule

| Milestone | Review Focus |
|-----------|--------------|
| M1 complete | R-001, R-004 (VRAM, Gemma) |
| M3 complete | R-005 (coverage) |
| M4 complete | R-002, R-003 (deployment) |

---

*Register maintained through Implementation Phase*
