# SENTIVIS AI — Software Architecture

**Version:** 2.3 FROZEN  
**Status:** Architecture Phase **CLOSED** — v2.3 frozen (Part 2 §4)  
**Style:** Feature-Based Clean Architecture  
**Module Specs:** `docs/MODULE_SPECIFICATIONS.md` v2.0  
**Interfaces:** `docs/INTERFACE_SPECIFICATION.md` v2.1  
**Compliance:** `docs/ARCHITECTURE_COMPLIANCE_REPORT.md` v2.3  
**Freeze Report:** `docs/ARCHITECTURE_FREEZE_REPORT.md` v2.3  
**Contract:** `docs/ENGINEERING_CONTRACT.md` v1.2  
**Hardware:** `docs/HARDWARE_PERFORMANCE_POLICY.md` v1.0  
**Developer Guide:** `docs/DEVELOPER_GUIDE.md` v1.0  
**Index:** `docs/MASTER_SPEC_INDEX.md`

> **Architecture Freeze (Part 2 §4):** Architecture v2.3 is frozen. Draft source aligned to specification. Feature implementation awaits Master Prompt Book Part 3 authorization.

---

## 1. Executive Summary

Sentivis AI is a **visual understanding platform** — not a detector, not a caption toy. It analyzes images through staged comprehension (objects → attributes → relationships → spatial layout → activities → environment → scene context → visual semantics → intent → narrative) before generating natural language.

Technically: a **PySide6** desktop application (Windows 11 · Python 3.10.11 · **2 GB VRAM**) running a canonical AI pipeline (YOLOv8n → scene analysis → BLIP base → Gemma-2B INT4 → caption refinement). Business logic, AI models, and UI are strictly separated. **Pipeline Orchestrator** enforces stage order on a **PipelineWorker** thread; **Model Manager** (sole model authority) and **Memory Manager** enforce sequential single-slot GPU usage with automatic CPU fallback.

All implementation must conform to the Engineering Contract and Hardware & Performance Policy.

---

## 2. Acceptance Criteria Validation

| Criterion | Status | Evidence |
|-----------|--------|----------|
| No circular dependencies | ✓ | Dependency graph is acyclic (§4) |
| Every module has clear responsibility | ✓ | Module catalog (§5) |
| New AI models without rewriting modules | ✓ | Engine interfaces + Model Manager registry (§6, §8) |
| Business logic isolated from UI | ✓ | UI → Services only; no PySide6 in pipeline (§3, §7) |
| Supports future expansion | ✓ | Tracking stub, plugin engines, export formats (§5, §8, §13) |
| Runs on 2 GB VRAM target | ✓ | Sequential model slot, ADR-003/004, CPU fallback (§8, §19) |
| UI responsive during inference | ✓ | PipelineWorker, progress, cancellation (§7, §19) |

**Architecture specification v2.3 FROZEN** (Part 2 §4). Draft implementation aligned to architecture. Gate: **Architecture Phase CLOSED**; await Part 3 for feature implementation.

---

## 3. Layer Model & Dependency Rules

```
┌─────────────────────────────────────────────────────────────┐
│  PRESENTATION          ui/                                   │
│  (PySide6 widgets, themes, dialogs — no AI, no business)    │
└──────────────────────────────┬──────────────────────────────┘
                               │ calls controllers / view-models
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION           app/ + services/                    │
│  (orchestration, DI, lifecycle, export, cache)              │
└──────────────────────────────┬──────────────────────────────┘
                               │ uses interfaces
                               ▼
┌──────────────┬───────────────┴───────────────┬──────────────┐
│  FEATURES    │                               │              │
│  vision/     │  analysis/                    │  language/   │
│  (YOLO…)     │  (scene graph, relations…)    │  (BLIP,Gemma)│
└──────────────┴───────────────────────────────┴──────────────┘
                               │ all depend only on
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  FOUNDATION            core/                                 │
│  (config, logging, exceptions, contracts/DTOs, utils)       │
└─────────────────────────────────────────────────────────────┘
```

### Hard Rules

1. **ui/** may import **services/** and **core/contracts** only (types for display). Never import `vision/`, `language/`, or `analysis/` directly.
2. **vision/**, **language/**, **analysis/** may import **core/** only. They must not import each other or **ui/**.
3. **services/** may import **core/** and feature modules via their **public interfaces** (abstract base classes in each feature’s `interfaces/` package).
4. **app/** wires concrete implementations; it is the only composition root.
5. Feature modules communicate **only through DTOs** in **core/contracts/** and **behaviour through `*/interfaces/`** — never concrete cross-feature imports.
6. **`PipelineOrchestrator`** depends on **interfaces only** — model-agnostic; see `docs/INTERFACE_SPECIFICATION.md` §8.
7. **UI widgets** bind **ViewModels** only — see `docs/VIEW_MODEL_SPECIFICATION.md`.

---

## 4. Dependency Graph (Acyclic)

```
app
 ├── services/interfaces + implementations
 │    ├── vision.interfaces
 │    ├── analysis.interfaces
 │    ├── language.interfaces
 │    └── core
 ├── ui/interfaces + view_models + controllers
 └── core

vision ──► core
analysis ──► core
language ──► core
services/pipeline ──► */interfaces (never concrete features)
ui/widgets ──► ui/interfaces (never services, never app)
```

No back-edges. No feature-to-feature edges. No ui → app.

---

## 5. Complete Project Structure

```
SentivisAI/
├── pyproject.toml                 # Package metadata, entry point: sentivis_ai.app.main
├── README.md
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # Entry point; minimal — delegates to bootstrap
│   ├── bootstrap.py               # DI container build, service registration
│   ├── container.py               # DependencyContainer (typed registry)
│   ├── lifecycle.py               # Startup / shutdown hooks, signal handling
│   └── settings_loader.py         # Loads config/ into core Config objects
│
├── core/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── app_config.py          # AppConfig dataclass
│   │   ├── model_config.py        # Per-model paths, device, precision, quant mode
│   │   ├── hardware_config.py     # VRAM/RAM thresholds, CPU fallback flags
│   │   └── theme_config.py        # Theme tokens ( consumed by ui/themes )
│   ├── constants/
│   │   ├── __init__.py
│   │   ├── pipeline_stages.py     # Stage enum, ordering
│   │   └── limits.py              # Max image size, batch limits
│   ├── contracts/                 # Cross-module DTOs (no business logic)
│   │   ├── __init__.py
│   │   ├── image.py               # ImagePayload, ValidatedImage, PreprocessedImage
│   │   ├── detection.py           # Detection, BoundingBox, DetectionResult
│   │   ├── analysis.py            # AttributeSet, Relation, SceneGraph, SceneContext
│   │   ├── language.py            # Prompt, RawCaption, RefinedCaption
│   │   └── pipeline.py            # PipelineRequest, PipelineResult, StageOutcome
│   ├── exceptions/
│   │   ├── __init__.py
│   │   ├── base.py                # SentivisError
│   │   ├── vision.py              # ValidationError, DetectionError
│   │   ├── language.py            # ModelLoadError, InferenceError
│   │   ├── analysis.py            # AnalysisError
│   │   └── service.py             # OrchestrationError, ExportError
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── logger_factory.py      # get_logger(name), configure_logging()
│   │   └── formatters.py          # Structured + human-readable formatters
│   └── utils/
│       ├── __init__.py
│       ├── paths.py               # Safe path resolution
│       ├── timing.py              # @timed decorator, Stopwatch
│       ├── memory.py              # RAM/GPU snapshot helpers (read-only probes)
│       └── images.py              # Format-agnostic helpers (no ML)
│
├── vision/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── validator.py           # IImageValidator
│   │   ├── preprocessor.py        # IImagePreprocessor
│   │   ├── detector.py            # IObjectDetector
│   │   └── tracker.py             # IObjectTracker (future; stub now)
│   ├── validation/
│   │   ├── __init__.py
│   │   └── image_validator.py     # Size, format, corruption checks
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── standard_preprocessor.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_engine.py         # YOLO load/infer/release
│   │   └── yolo_detector.py       # IObjectDetector implementation
│   └── tracking/                  # Future-ready placeholder
│       ├── __init__.py
│       └── noop_tracker.py        # IObjectTracker no-op for v1
│
├── language/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── vision_language.py     # IVisionLanguageModel (BLIP)
│   │   ├── reasoning.py           # IReasoningModel (Gemma)
│   │   ├── prompt_builder.py      # IPromptBuilder
│   │   └── caption_refiner.py     # ICaptionRefiner
│   ├── blip/
│   │   ├── __init__.py
│   │   ├── blip_engine.py
│   │   └── blip_model.py          # IVisionLanguageModel
│   ├── gemma/
│   │   ├── __init__.py
│   │   ├── gemma_engine.py
│   │   └── gemma_model.py         # IReasoningModel
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py      # SceneContext → Prompt
│   │   └── templates/             # Jinja or string templates (data, not code logic)
│   └── refinement/
│       ├── __init__.py
│       └── caption_refiner.py     # Post-process Gemma output
│
├── analysis/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── attribute_extractor.py # IAttributeExtractor
│   │   ├── relationship_analyzer.py
│   │   ├── scene_graph_builder.py
│   │   ├── context_builder.py     # ISceneContextBuilder
│   │   └── activity_analyzer.py   # IActivityAnalyzer
│   ├── attributes/
│   │   └── attribute_extractor.py # DetectionResult → AttributeSet
│   ├── relationships/
│   │   └── relationship_analyzer.py
│   ├── scene_graph/
│   │   └── scene_graph_builder.py
│   ├── context/
│   │   └── context_builder.py     # Aggregates graph + attributes → SceneContext
│   └── activity/
│       └── activity_analyzer.py     # Optional enrichment from spatial layout
│
├── services/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # PipelineOrchestrator — enforces stage order
│   │   ├── stage_runner.py        # Single-stage exec + error boundary
│   │   ├── progress_reporter.py   # Callback protocol for UI progress
│   │   ├── pipeline_worker.py     # QThread — runs pipeline off UI thread
│   │   └── cancellation.py        # CancellationToken — cooperative abort
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_manager.py       # Sole model authority: load/unload/device/fallback
│   │   ├── model_registry.py      # Maps ModelKind → factory
│   │   └── device_selector.py     # GPU probe, VRAM estimate, CPU fallback decision
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── memory_manager.py      # Stats, cleanup, warnings, peak logging
│   │   └── managed_resources.py   # Context managers for tensors/buffers
│   ├── cache/
│   │   ├── __init__.py
│   │   └── cache_manager.py       # Disk cache for intermediate pipeline artifacts
│   └── export/
│       ├── __init__.py
│       ├── export_manager.py      # Facade
│       └── writers/
│           ├── pdf_writer.py
│           ├── json_writer.py
│           ├── txt_writer.py
│           └── image_writer.py
│
├── ui/
│   ├── __init__.py
│   ├── app_window.py              # QMainWindow shell
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── main_controller.py     # Bridges UI events → services
│   │   ├── pipeline_controller.py
│   │   ├── export_controller.py
│   │   └── settings_controller.py
│   ├── view_models/
│   │   ├── __init__.py
│   │   ├── pipeline_state.py      # Observable state for widgets
│   │   └── history_model.py
│   ├── dashboard/
│   ├── sidebar/
│   ├── image_viewer/
│   ├── caption_panel/
│   ├── history/
│   ├── settings/
│   ├── dialogs/
│   └── themes/
│       ├── __init__.py
│       ├── theme_manager.py
│       └── styles/                # QSS files
│
├── assets/
│   ├── icons/
│   ├── fonts/
│   ├── logo/
│   └── animations/
│
├── config/
│   ├── app.default.toml
│   ├── models.default.toml
│   └── themes.default.toml
│
├── exports/                       # Runtime output directory (gitignored)
├── cache/                         # Runtime cache (gitignored)
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
│
└── docs/
    ├── ARCHITECTURE.md            # This document
    ├── DEVELOPER_GUIDE.md
    └── USER_GUIDE.md
```

---

## 6. Canonical Pipeline Flow

Stages are **ordered**, **mandatory** for full analysis, and **non-bypassable** except where noted for degraded/recovery paths (§10). Final caption generation occurs **only after** all understanding stages complete (Part 1 §1).

```
Stage 0   Load Image              → ImagePayload
Stage 1   Image Validation        → ValidatedImage
Stage 2   Image Preprocessing     → PreprocessedImage
Stage 3   YOLO Detection          → DetectionResult          [Objects]
Stage 4   Attribute Extraction    → AttributeSet             [Object Attributes]
Stage 5   Relationship Analysis   → Relation[]               [Object Relationships]
Stage 6   Scene Graph Build       → SceneGraph               [Spatial Layout]
Stage 7   Activity Analysis       → ActivityHints            [Human Activities]
Stage 8   Scene Context Build     → SceneContext             [Environment, Scene Context]
Stage 9   BLIP Understanding      → RawCaption               [Visual Semantics]
Stage 10  Prompt Builder          → Prompt
Stage 11  Gemma Reasoning         → RawCaption               [Image Intent, Possible Story]
Stage 12  Caption Refinement      → RefinedCaption           [Final natural language]
Stage 13  Persist / Export / UI   → PipelineResult
```

### Part 1 Understanding Traceability

| Understanding Dimension | Primary Module | DTO |
|-------------------------|----------------|-----|
| Objects | `vision/detection` | `DetectionResult` |
| Object Attributes | `analysis/attributes` | `AttributeSet` |
| Object Relationships | `analysis/relationships` | `Relation[]` |
| Spatial Layout | `analysis/scene_graph` | `SceneGraph` |
| Human Activities | `analysis/activity` | `ActivityHints` |
| Environment | `analysis/context` | `SceneContext.environment` |
| Scene Context | `analysis/context` | `SceneContext` |
| Visual Semantics | `language/blip` | `RawCaption` (blip) |
| Image Intent | `language/gemma` | `RawCaption` (gemma) |
| Possible Story | `language/gemma` + `language/refinement` | `RefinedCaption` |

### Orchestrator Responsibilities

- Execute stages sequentially via `IStageRunner`.
- Pass **immutable DTOs** between stages; each stage returns a new object.
- Emit progress events (`StageStarted`, `StageCompleted`, `StageFailed`) through `IProgressReporter`.
- On stage failure: apply recovery policy (§10); never propagate unhandled exceptions to UI.
- Request **ModelManager** to load the engine required for the upcoming stage and **release** the previous heavy model before loading the next (§8).

### Model Slot Schedule (GPU Memory)

| Stages | Active Heavy Model |
|--------|-------------------|
| 3 | YOLO |
| 9 | BLIP |
| 11 | Gemma |

Analysis stages (4–8) are CPU/light-GPU logic on `DetectionResult` DTOs — no additional heavy model.

---

## 7. UI ↔ Application Boundary

### Controllers (services layer façade for UI)

| Controller | Responsibility |
|------------|----------------|
| `MainController` | Window lifecycle, navigation, global error toast |
| `PipelineController` | `analyze(image_path) → PipelineResult`, cancel, progress subscription |
| `ExportController` | `export(result, format, path)` |
| `SettingsController` | Read/write user preferences via config |

### UI Rules

- Widgets bind to **view models** (`PipelineState`, `HistoryModel`), not raw DTOs with 20 fields.
- **`PipelineWorker` (QThread)** runs the full pipeline; **`ExportWorker` (QThreadPool)** runs exports.
- Controllers emit Qt signals to main thread; DTOs crossing threads are immutable.
- **Progress always visible:** stage name, percent, device, elapsed time; heartbeat during model load.
- **Cancellation:** UI cancel button → `CancellationToken.cancel()` → checked between stages.
- UI displays **user-friendly** messages from `SentivisError.user_message`; logs retain `developer_detail`.
- Stack traces **never** shown to users.
- **No** `import torch`, `ultralytics`, or model code in `ui/`.

---

## 8. Model Lifecycle

Every engine (`YoloEngine`, `BlipEngine`, `GemmaEngine`) implements:

```text
IModelEngine
  initialize(config: ModelConfig) -> None
  load() -> None
  infer(input: TIn) -> TOut
  release() -> None          # drop weights, set to None
  clear_device_cache() -> None
  @property is_loaded -> bool
  @property model_kind -> ModelKind
```

### State Machine

```text
UNINITIALIZED → INITIALIZED → LOADING → READY → INFERRING → READY
                    ↓                      ↓
                 FAILED                  RELEASING → IDLE (GPU free)
```

### ModelManager Policy

**No other module** may load, unload, or select device for AI models.

1. Maintain at most **one** `READY` heavy model in VRAM (`active_slot`).
2. Before `load()` of model B: `release_active()` + `MemoryManager.clear_gpu_cache()` on model A.
3. `MemoryManager` logs before/after snapshots on every transition; records peak per run.
4. On GPU OOM: release → clear cache → retry on CPU (ADR-007); emit WARNING + progress event.
5. `DeviceSelector` probes CUDA availability and estimates headroom before load.
6. Engines are **not** singletons holding permanent weights; session-scoped via `ModelManager`.
7. Model health: validate weight files exist and match config before load.

### MemoryManager Policy

1. `snapshot()` → RAM RSS, system available, VRAM allocated/reserved.
2. `clear_gpu_cache()` → `gc.collect()` + `torch.cuda.empty_cache()`.
3. `managed_tensor()` / `managed_buffer()` context managers for stage temporaries.
4. WARNING at 85% VRAM or 90% RAM; trigger preemptive release if safe.
5. On OOM recovery: force release all model refs, clear caches, log peak.

### Adding a New Model (e.g., LLaVA)

1. Add `ModelKind.LLAVA` to `core/constants`.
2. Implement `ILlavaModel` under `language/llava/` (or new feature package).
3. Register factory in `ModelRegistry`.
4. Insert stage(s) in `pipeline_stages.py` and orchestrator — **no changes** to existing engine code.

---

## 9. Module Catalog (Single Responsibility)

| Module / Class | Single Responsibility |
|----------------|----------------------|
| `ImageValidator` | Reject invalid/corrupt/oversized images |
| `StandardPreprocessor` | Normalize resolution, color space, tensor-ready arrays |
| `YoloEngine` | Own YOLO weights I/O and raw inference |
| `YoloDetector` | Map engine output → `DetectionResult` DTO |
| `AttributeExtractor` | Derive per-object attributes from detections |
| `RelationshipAnalyzer` | Spatial/semantic relations between objects |
| `SceneGraphBuilder` | Nodes + edges graph structure |
| `ActivityAnalyzer` | Infer coarse activities from layout (heuristic/ML-ready) |
| `ContextBuilder` | Merge graph, attributes, activities → `SceneContext` |
| `BlipEngine` / `BlipModel` | BLIP load/infer/release |
| `GemmaEngine` / `GemmaModel` | Gemma load/infer/release |
| `PromptBuilder` | Deterministic prompt from `SceneContext` + templates |
| `CaptionRefiner` | Grammar, length, safety pass on final text |
| `PipelineOrchestrator` | Stage ordering, DTO threading, recovery |
| `StageRunner` | Invoke one stage; catch & wrap exceptions |
| `ModelManager` | Registration, load/unload, device, CPU fallback, health, lifecycle |
| `MemoryManager` | Stats, VRAM/RAM monitoring, cleanup, warnings, peak logging, recovery |
| `PipelineWorker` | Off-UI-thread pipeline execution |
| `CancellationToken` | Cooperative cancel between stages |
| `DeviceSelector` | GPU probe and fallback decision |
| `CacheManager` | Optional disk persistence of stage outputs |
| `ExportManager` | Route to format-specific writers |
| `*Controller` | Translate UI intents to service calls |
| `ThemeManager` | Apply QSS / palette from theme config |

---

## 10. Error Handling Strategy

### Exception Hierarchy

All inherit from `SentivisError`:

- `user_message: str` — safe for UI
- `developer_detail: str` — full context for logs
- `recoverable: bool`
- `stage: PipelineStage | None`

### Policies by Layer

| Layer | Behavior |
|-------|----------|
| Feature (vision/language/analysis) | Raise specific subclasses; never catch-and-ignore |
| StageRunner | Catch `SentivisError`; wrap unknown as `OrchestrationError` |
| Orchestrator | If `recoverable`: skip optional stages or use fallback DTO; else abort pipeline with partial result |
| Controller | Map to signal `error_occurred(user_message)` |
| UI | QMessageBox or inline banner; app continues running |

### Recovery Examples

- **Validation fails** → abort early; no model load.
- **YOLO fails** → abort; no downstream stages (insufficient data).
- **BLIP fails** → fallback: caption from `SceneContext` template only (`recoverable=True`).
- **Gemma fails** → fallback: use BLIP caption (`recoverable=True`).
- **Export fails** → show error; keep `PipelineResult` in memory.

**No uncaught exception may terminate the Qt event loop.**

---

## 11. Logging Strategy

### Configuration

- Levels: DEBUG, INFO, WARNING, ERROR (via `config/app.default.toml`).
- Dual handlers: rotating file (`logs/sentivis.log`) + stderr (dev mode).
- Structured fields: `timestamp`, `level`, `logger`, `stage`, `model`, `duration_ms`, `memory_mb`, `gpu_mb`.

### Required Log Points

| Event | Level |
|-------|-------|
| Application startup / shutdown | INFO |
| Config loaded | INFO |
| Model load start / complete / fail | INFO / ERROR |
| Inference start / complete | INFO |
| Memory snapshot (before/after model transition) | DEBUG |
| Peak memory per pipeline run | INFO |
| CPU fallback triggered | WARNING |
| Memory threshold exceeded | WARNING |
| Stage start / complete | DEBUG |
| Performance timing per stage | DEBUG |
| Export operation | INFO |
| Recoverable degradation | WARNING |
| Unrecoverable pipeline abort | ERROR |

Use `core.logging.logger_factory.get_logger(__name__)` — no `print()`, no ad-hoc loggers.

---

## 12. Dependency Injection

### Composition Root: `app/bootstrap.py`

Registration order:

1. Load configuration → `AppConfig`, `ModelConfig`, `ThemeConfig`
2. Configure logging
3. Register core utilities (singletons)
4. Register feature implementations bound to interfaces
5. Register services (`ModelManager`, `MemoryManager`, `CacheManager`, `PipelineOrchestrator`, `ExportManager`)
6. Register UI controllers with service dependencies
7. Return `ApplicationContext` consumed by `main.py`

### Container Pattern

- Constructor injection only; no service locator in feature code.
- **No global mutable state**; optional module-level constants in `core/constants` only.
- Tests override container bindings with fakes (`FakeDetector`, `FakeBlipModel`).

---

## 13. Scalability & Future Expansion

| Future Need | Extension Point |
|-------------|-----------------|
| Object tracking | Implement `IObjectTracker`; insert stage after detection |
| Video input | New `media/` feature + orchestrator branch for frame iterator |
| New detector | `IObjectDetector` alternative; swap in DI |
| Cloud inference | New `infrastructure/remote_*` engine implementing same interfaces |
| Plugin export formats | Register writer in `ExportManager` |
| Batch processing | `PipelineOrchestrator.analyze_batch()` with queue + same stage runner |
| Additional analysis | New `I*Analyzer` in `analysis/`; new stage in ordered enum |

---

## 14. Testing Strategy (Architecture-Level)

| Layer | Test Type | Approach |
|-------|-----------|----------|
| core/contracts | unit | DTO validation, immutability |
| vision/language/analysis | unit | Mock engines; test DTO mapping |
| services/orchestrator | integration | Fake models; full pipeline with fixture images |
| model lifecycle | integration | Assert GPU memory released between stages (performance/) |
| ui/controllers | unit | Mock services; signal/slot behavior |
| end-to-end | integration | Headless pipeline without UI |

Every public function carries a docstring per `docs/DEVELOPER_GUIDE.md` §4; test modules mirror source tree under `tests/`.

---

## 15. Configuration & Assets

- **config/*.toml** — defaults committed; user overrides in OS app-data path (loaded by `settings_loader`).
- **assets/** — read-only resources located via `core.utils.paths.resource_path()`.
- **cache/** and **exports/** — runtime-only; created on startup if missing; gitignored.

---

## 16. Key Interface Sketches (Contracts Only)

These are architectural contracts, not implementation.

```python
# core/contracts/pipeline.py
@dataclass(frozen=True)
class PipelineRequest:
    image_path: Path
    options: AnalysisOptions

@dataclass(frozen=True)
class PipelineResult:
    request: PipelineRequest
    scene_context: SceneContext
    caption: RefinedCaption
    stages_completed: tuple[PipelineStage, ...]
    warnings: tuple[str, ...]
```

```python
# services/pipeline/orchestrator.py
class PipelineOrchestrator:
    def analyze(self, request: PipelineRequest) -> PipelineResult: ...
    def cancel(self) -> None: ...
```

```python
# vision/interfaces/detector.py
class IObjectDetector(Protocol):
    def detect(self, image: PreprocessedImage) -> DetectionResult: ...
```

```python
# services/models/model_manager.py
class ModelManager:
    def acquire(self, kind: ModelKind) -> IModelEngine: ...
    def release_active(self) -> None: ...
    def handle_oom(self, kind: ModelKind) -> Device: ...  # returns CPU after cleanup
```

```python
# services/memory/memory_manager.py
class MemoryManager:
    def snapshot(self) -> MemorySnapshot: ...
    def clear_gpu_cache(self) -> None: ...
    def log_peak(self, run_id: str) -> None: ...
```

```python
# services/pipeline/cancellation.py
class CancellationToken:
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...
    def raise_if_cancelled(self) -> None: ...
```

---

## 17. Implementation Order (Post-Approval)

When implementation begins, build bottom-up:

1. `core/` — contracts, exceptions, logging, config
2. `vision/validation` + `preprocessing` (no GPU)
3. `services/memory` + `services/models` + stub engines
4. `analysis/*` (pure logic on fixture DTOs)
5. `language/prompts` + `refinement` (no GPU)
6. Real `yolo_engine`, `blip_engine`, `gemma_engine`
7. `services/pipeline/orchestrator`
8. `services/export`
9. `ui/` shell + controllers
10. Tests at each layer before proceeding

---

## 18. Sign-Off

Architecture satisfies Part 2 acceptance criteria and Part 1 §2 hardware/performance requirements. Pre-implementation reviews passed (`docs/ENGINEERING_REVIEW.md` v1.0). ADRs 001–007 accepted.

No source code until implementation gate opens (`docs/ENGINEERING_CONTRACT.md` §16).

---

## 19. Hardware & Performance Constraints (Part 1 §2)

### Target Platform

Windows 11 · Python 3.10.11 · NVIDIA GPU · **2 GB VRAM** · 8 GB RAM min · SSD preferred.

### Default Models (ADR-004)

| Stage | Model | Config Key |
|-------|-------|------------|
| 3 | YOLOv8n | `models.yolo.variant = "yolov8n"` |
| 9 | BLIP base | `models.blip.model_id = "Salesforce/blip-image-captioning-base"` |
| 11 | Gemma-2B INT4 | `models.gemma.model_id = "google/gemma-2-2b-it"` + `quantization = "int4"` |

### Image Limits

| Limit | Value | Enforced By |
|-------|-------|-------------|
| Max input dimension | 4096 px | `ImageValidator` |
| YOLO inference size | 640 px | `StandardPreprocessor` |
| Max file size | 32 MB | `ImageValidator` |

### VRAM Timeline (Single Request)

```
Idle → YOLO (~400 MB) → release → Idle → BLIP (~1000 MB) → release → Idle → Gemma (~1000 MB) → release → Idle
```

Stages 4–8: CPU-only analysis, 0 MB model VRAM.

### Responsiveness Checklist

- [ ] UI thread never calls inference or heavy preprocessing
- [ ] Progress bar updates on every stage transition
- [ ] Cancel stops pipeline before next stage
- [ ] Loading overlay during model load with heartbeat
- [ ] CPU fallback shows informative status, not error dialog

See `docs/HARDWARE_PERFORMANCE_POLICY.md` for full policy.
