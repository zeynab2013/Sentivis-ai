# SENTIVIS AI — Pre-Implementation Engineering Review

**Version:** 1.0  
**Date:** 2026-07-30  
**Gate:** Required before first source file is written  
**Spec basis:** Parts 1 (1/4, 2/4) · Part 2 Architecture

---

## Review Summary

| Review | Result | Reviewer Role |
|--------|--------|---------------|
| Architecture Review | **PASS** | Principal Software Architect |
| Dependency Review | **PASS** | Senior Python Engineer |
| Performance Review | **PASS** (with 2 GB constraints) | Performance Engineer |
| Memory Review | **PASS** (with CPU fallback) | Senior ML Engineer |
| Security Review | **PASS** (desktop scope) | Security Engineer |
| Maintainability Review | **PASS** | Principal Software Architect |

**Verdict:** Design-phase reviews passed. Implementation remains gated on Parts 3–4. Re-run this checklist after Parts 3–4 may alter UI, models, or pipeline scope.

---

## 1. Architecture Review

### Scope

Validate Feature-Based Clean Architecture, pipeline ordering, UI isolation, extensibility.

### Findings

| Check | Status | Notes |
|-------|--------|-------|
| Acyclic dependency graph | ✓ | ui → services → features → core |
| Single responsibility per module | ✓ | Module catalog in ARCHITECTURE §9 |
| Pipeline non-bypassable | ✓ | Orchestrator enforces 13 stages |
| Business logic isolated from UI | ✓ | Controllers + worker threads |
| Model isolation | ✓ | ModelManager sole model authority |
| DTO-based cross-feature comms | ✓ | core/contracts |
| Extension without rewrite | ✓ | Interface + registry pattern |

### Actions

None. Architecture v1.2 incorporates hardware constraints.

---

## 2. Dependency Review

### Scope

Verify no circular imports, injectable dependencies, single composition root.

### Findings

| Check | Status | Notes |
|-------|--------|-------|
| No feature-to-feature imports | ✓ | Enforced by layer rules |
| DI composition root in app/bootstrap | ✓ | Single wiring location |
| No global mutable service state | ✓ | Container-scoped beans |
| Config externalized | ✓ | config/*.toml + app-data |
| Test override path | ✓ | Container accepts fake bindings |

### Actions

Define `pyproject.toml` dependency pins during implementation (Part 3 may specify versions).

---

## 3. Performance Review

### Scope

Validate responsiveness, threading, progress, cancellation against 2 GB VRAM target.

### Findings

| Check | Status | Notes |
|-------|--------|-------|
| UI thread protected | ✓ | PipelineWorker + ExportWorker |
| Sequential heavy inference | ✓ | One model slot |
| Lazy model loading | ✓ | Per-stage acquire |
| Progress reporting | ✓ | IProgressReporter on every stage |
| Cancellation | ✓ | CancellationToken between stages |
| Image size limits | ✓ | core/constants/limits.py (max 4096 px, preprocess to 640 for YOLO) |
| No premature optimization | ✓ | Measure in performance/ tests post-impl |

### Risks

| Risk | Mitigation |
|------|------------|
| Gemma 2B tight on 2 GB VRAM | CPU fallback; INT4 quantization |
| Large image RAM pressure on 8 GB | Stream decode; cap dimensions in validator |
| Model load perceived as hang | Progress heartbeat + loading overlay |

### Actions

Add performance tests in `tests/performance/` measuring stage latency on reference hardware profile.

---

## 4. Memory Review

### Scope

Validate VRAM policy, ModelManager/MemoryManager split, resource cleanup, OOM recovery.

### Findings

| Check | Status | Notes |
|-------|--------|-------|
| One heavy model in VRAM | ✓ | ModelManager active_slot |
| Immediate unload post-inference | ✓ | release() in stage finally block |
| CUDA cache clear | ✓ | MemoryManager.clear_gpu_cache() |
| CPU fallback on OOM | ✓ | ModelManager device retry |
| No permanent GPU reservation | ✓ | IDLE state between runs |
| Tensor/buffer cleanup | ✓ | Context managers + explicit del |
| Peak memory logging | ✓ | Per pipeline run |
| Memory warnings | ✓ | 85% VRAM / 90% RAM thresholds |

### VRAM Sequence (2 GB target)

```
Idle (0 MB models)
  → YOLO load (~400 MB) → infer → release → clear → Idle
  → BLIP load (~1000 MB) → infer → release → clear → Idle
  → Gemma load (~1000 MB) → infer → release → clear → Idle
```

Analysis stages 4–8: CPU only, ~0 MB VRAM.

### Actions

Implement `MemoryManager.snapshot()` before integration tests. Assert VRAM drop after each `release_active()`.

---

## 5. Security Review

### Scope

Desktop application threat model: local files, model weights, exports, logs.

### Findings

| Check | Status | Notes |
|-------|--------|-------|
| No secrets in repo | ✓ | Config in app-data; .gitignore for cache/exports |
| Path traversal on image load | ✓ | paths.py validates and resolves |
| Export path validation | ✓ | User-selected dir; sanitize filename |
| Log redaction | ✓ | No absolute user paths in INFO; full in DEBUG |
| Model weight integrity | ✓ | Checksum optional in model_config |
| No network in v1 pipeline | ✓ | Offline inference; future cloud via new interface |
| Stack traces not in UI | ✓ | Error recovery policy |

### Actions

Add security notes to DEVELOPER_GUIDE when written.

---

## 6. Maintainability Review

### Scope

Code quality gates, documentation, ADRs, testability, senior-engineer navigability.

### Findings

| Check | Status | Notes |
|-------|--------|-------|
| Quality gates defined | ✓ | HARDWARE_PERFORMANCE_POLICY §12 |
| ADRs for major decisions | ✓ | docs/adr/ (7 records) |
| Module documentation policy | ✓ | Contract + module docstrings |
| Independent module evolution | ✓ | Interface boundaries |
| Test pyramid | ✓ | unit / integration / performance |
| No prohibited artifacts | ✓ | Engineering contract §8 |
| Versioned documents | ✓ | All docs carry version |

### Actions

Publish `DEVELOPER_GUIDE.md` before first feature PR.

---

## Sign-Off

Pre-implementation design reviews **passed** for accumulated specification (Parts 1.1, 1.2, Part 2).

Implementation start requires:

1. Parts 3–4 integrated
2. This review re-validated if Parts 3–4 change architecture scope
3. DEVELOPER_GUIDE published

---

*Reviewed against HARDWARE_PERFORMANCE_POLICY v1.0 and ARCHITECTURE v1.2.*
