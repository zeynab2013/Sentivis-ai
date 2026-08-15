# SENTIVIS AI — Hardware & Performance Policy

**Version:** 1.0  
**Part:** 1 / 4 (section 2) — Hardware Contract · Performance · Memory · Governance  
**Status:** Active — mandatory engineering constraints  
**Extends:** `docs/ENGINEERING_CONTRACT.md` · `docs/ARCHITECTURE.md`

---

## 1. Target Platform (Fixed)

All architectural and implementation decisions are bound to this hardware profile.

| Resource | Specification |
|----------|---------------|
| Operating System | Windows 11 |
| Python | 3.10.11 |
| GPU | NVIDIA (CUDA-capable) |
| VRAM | **2 GB** (hard ceiling for design) |
| System RAM | 8 GB minimum · 16 GB recommended |
| CPU | Mid-range Intel or AMD |
| Storage | SSD preferred |

### Design Mandates

- Never assume high-end hardware.
- Never optimize exclusively for powerful GPUs.
- Stability outweighs inference speed when resources are constrained.
- Every model variant, image limit, and batch size must be justified against 2 GB VRAM.

---

## 2. Performance Objectives

| Objective | Implementation Requirement |
|-----------|---------------------------|
| Responsive at every stage | UI event loop never blocked |
| Interactive during inference | All heavy work on worker threads |
| Large images do not freeze UI | Preprocessing + inference off main thread; streaming progress |
| Load models only when required | Lazy load per pipeline stage via `ModelManager` |
| Background workers for heavy compute | `PipelineWorker` (QThread) + optional `QThreadPool` for export |
| Progress always available | `IProgressReporter` emits stage, percent, message on every transition |
| Cancellation supported | `CancellationToken` checked between stages; cooperative abort |
| Application never appears hung | Indeterminate progress during model load; heartbeat log every 5 s |

---

## 3. GPU Memory Policy

VRAM is the scarcest resource. Treat it as a shared, ephemeral slot.

### Rules

1. Load heavy models **only** when their pipeline stage executes.
2. Unload every heavy model **immediately** after inference completes.
3. Call `torch.cuda.empty_cache()` (via `MemoryManager`) after every release.
4. **Never** keep multiple large models in VRAM simultaneously.
5. Avoid unnecessary tensor duplication — prefer views and in-place ops where safe.
6. Avoid unnecessary image duplication — single canonical `PreprocessedImage` buffer per request.
7. Reuse pre-allocated buffers within a stage when dimensions are stable.
8. On insufficient VRAM → **automatic CPU fallback** (see §5).
9. Application stability > inference speed.

### VRAM Budget (Design Target)

| Model | Variant | Est. Peak VRAM (GPU) | Notes |
|-------|---------|----------------------|-------|
| YOLO | YOLOv8n | ~300–500 MB | FP16; input capped at 640 px |
| BLIP | blip-base-capfilt | ~900–1100 MB | Load only after YOLO released |
| Gemma | Gemma-2B INT4 | ~900–1200 MB | Load only after BLIP released; CPU fallback common on 2 GB |

Analysis stages (4–8) run on CPU using detection DTOs — zero additional VRAM.

Reserve ~200 MB for CUDA context and transient tensors.

---

## 4. Model Execution Policy

Heavy models execute **sequentially**. No parallel heavy inference.

```
Initialize → Load → Inference → Collect Results → Release Resources → Clear Device Cache → Idle
```

The software must **never** permanently reserve GPU memory between pipeline runs.

---

## 5. CPU Fallback Policy

When GPU allocation fails or projected usage exceeds available VRAM:

1. `ModelManager` catches `CudaOutOfMemoryError` / allocation failure.
2. Log WARNING with memory snapshot.
3. Release GPU resources via `MemoryManager`.
4. Retry same model on CPU with configured CPU-optimized settings.
5. Emit progress event: device switched to CPU (user sees slightly longer estimate, not an error).
6. If CPU also fails → recoverable pipeline degradation per stage (see `ARCHITECTURE.md` §10).

Device selection logic lives **exclusively** in `ModelManager`. No other module calls `model.to(device)` directly.

---

## 6. Model Manager Responsibilities

`services/models/model_manager.py` is the **sole** authority for AI model lifecycle.

| Responsibility | Detail |
|----------------|--------|
| Model registration | `ModelRegistry` maps `ModelKind` → factory |
| Model loading | Lazy load on `acquire(kind)` |
| Model unloading | `release_active()` before next acquire |
| Device selection | GPU first; CPU fallback on failure |
| CPU fallback | Automatic retry with release + cache clear |
| Memory monitoring | Pre/post load snapshots via `MemoryManager` |
| Model health | Detect corrupt weights, version mismatch; raise `ModelLoadError` |
| Inference lifecycle | State machine: LOADING → READY → INFERRING → RELEASING |

**No other module** may load, unload, or select device for AI models.

---

## 7. Memory Manager Responsibilities

`services/memory/memory_manager.py` owns all memory observability and cleanup.

| Responsibility | Detail |
|----------------|--------|
| Memory statistics | Process RSS, available system RAM |
| VRAM monitoring | `torch.cuda.memory_allocated`, `memory_reserved` |
| RAM monitoring | `psutil` process and system metrics |
| Cache cleanup | `gc.collect()` after model release |
| GPU cache cleanup | `torch.cuda.empty_cache()` after model release |
| Temporary resource cleanup | Context manager `managed_tensor()`, `managed_buffer()` |
| Peak memory logging | Record peak VRAM/RAM per pipeline run |
| Memory warnings | WARNING when VRAM > 85% or RAM > 90% of available |
| Recovery actions | Force release + gc + empty_cache on OOM |

---

## 8. Resource Management Rules

Every module must:

- Close temporary resources in `finally` blocks or context managers.
- Release file handles after read/write.
- Release image buffers when stage output is persisted to DTO.
- Release tensors immediately after conversion to CPU numpy/DTO.
- Release model references on `release()` — set to `None`.
- Join or cancel worker threads on shutdown.
- Avoid memory leaks, orphan threads, and unmanaged resources.

---

## 9. Multithreading Policy

### UI Thread (Main Thread)

- User input, rendering, animations, signal/slot dispatch only.
- **Forbidden:** model inference, image preprocessing, export I/O, file parsing.

### Worker Threads

| Task | Worker |
|------|--------|
| Image preprocessing | `PipelineWorker` |
| YOLO inference | `PipelineWorker` |
| BLIP inference | `PipelineWorker` |
| Gemma inference | `PipelineWorker` |
| Export operations | `ExportWorker` (`QThreadPool`) |
| Large file processing | `ExportWorker` or `PipelineWorker` |

### Thread Safety

- DTOs passed UI ← worker are immutable (`frozen dataclass`).
- Progress/cancellation via Qt signals or thread-safe queue.
- `ModelManager` is accessed only from the active pipeline worker during inference (one pipeline at a time in v1).

---

## 10. Logging Policy

Every important operation is logged. Categories:

| Category | Level | Examples |
|----------|-------|----------|
| Application Startup | INFO | Version, config path, hardware probe |
| Application Shutdown | INFO | Cleanup duration, final memory |
| Configuration | INFO | Loaded keys (secrets redacted) |
| Model Loading | INFO | Model kind, device, duration, VRAM before/after |
| Model Unloading | INFO | Model kind, VRAM reclaimed |
| Inference | INFO | Stage, duration_ms, device |
| Warnings | WARNING | CPU fallback, memory threshold, degraded caption |
| Errors | ERROR | Load failure, unrecoverable stage (with developer_detail) |
| Exports | INFO | Format, path, size |
| Performance | DEBUG | Stage timing breakdown, throughput |
| Memory | DEBUG | Snapshots at transitions; peak per run |

Multiple severity levels: DEBUG, INFO, WARNING, ERROR. Logs must enable rapid failure diagnosis.

---

## 11. Error Recovery Policy

- Unexpected failures **never** terminate the application.
- Recover automatically when possible (CPU fallback, caption degradation, skip optional stage).
- When automatic recovery is impossible → user-friendly message in UI.
- Technical details and stack traces → developer logs only.
- Stack traces **never** shown to end users.

---

## 12. Quality Gates

A module is **not complete** until all gates pass:

| Gate | Tool / Method |
|------|---------------|
| Responsibility defined | Documented in module docstring + `docs/` |
| Public interfaces documented | Docstrings on all public methods |
| Follows architecture | No forbidden imports; acyclic deps |
| Static analysis | `ruff check` |
| Formatting | `ruff format --check` |
| Type checking | `mypy --strict` (project packages) |
| Unit tests | `pytest tests/unit/` for module |
| Integration | Dependent modules pass integration tests |

If any gate fails → module is corrected before downstream development continues.

Full Definition of Done: `docs/DEVELOPER_GUIDE.md` §11. Self-review checklist: §9.

---

## 13. Architecture Decision Records

Major decisions require an ADR in `docs/adr/`. See index: `docs/adr/README.md`.

Required when choosing: frameworks, AI models, structure, patterns, caching, memory, threading, UI, DI.

---

## 14. Versioning Policy

Every engineering document carries a version header. Increment on significant change.

| Document | Current Version |
|----------|-----------------|
| ENGINEERING_CONTRACT | 1.2 |
| DEVELOPER_GUIDE | 1.0 |
| HARDWARE_PERFORMANCE_POLICY | 1.0 |
| ARCHITECTURE | 1.2 |
| ENGINEERING_REVIEW | 1.0 |
| MASTER_SPEC_INDEX | — (living index) |

---

## 15. Documentation Policy

Documentation is part of the product. Every important module documents:

- Purpose
- Responsibilities
- Dependencies
- Public Interfaces
- Execution Flow
- Extension Points
- Known Limitations
- Future Improvements

Module-level docstrings satisfy this for code modules; feature-level summaries live in `docs/`.

---

## 16. Pre-Implementation Engineering Review

Required before source implementation. See `docs/ENGINEERING_REVIEW.md`.

Reviews: Architecture · Dependency · Performance · Memory · Security · Maintainability.

**Status:** Reviews performed at design phase (Part 1 §2 integration). Re-run before first merge to main.

---

*Extends all prior Master Prompt Book requirements. Does not replace them.*
