# SENTIVIS AI — Module Specifications

**Version:** 2.0  
**Part:** 2 / 4 (section 1) — Architecture Specification · Module Design  
**Status:** Architecture Phase — authoritative over draft implementation  
**Supersedes:** Any assumption that draft code is accepted

Draft source code is **provisional**. Where code and this document disagree, **this document wins**.

---

## Index

| Module | Section |
|--------|---------|
| `app/` | §1 |
| `core/` | §2 |
| `vision/` | §3 |
| `analysis/` | §4 |
| `language/` | §5 |
| `services/` | §6 |
| `ui/` | §7 |

---

## §1 — `app/` Application Layer

### Purpose

Composition root: startup, dependency injection, application lifecycle, configuration loading.

### Responsibilities

- Load TOML configuration into typed config objects
- Configure logging before any feature module runs
- Wire all concrete implementations into `DependencyContainer`
- Create Qt application and main window
- Coordinate clean shutdown (model release, cache clear)

### Public Interfaces

| Symbol | Contract |
|--------|----------|
| `main()` | Entry point; boots and runs lifecycle |
| `bootstrap()` | Returns configured `ApplicationLifecycle` |
| `DependencyContainer.build()` | Returns `ApplicationContext` |
| `ApplicationLifecycle.run()` | Starts Qt event loop |
| `ApplicationLifecycle.shutdown()` | Releases GPU and logs shutdown |

### Dependencies

- `core/` — config, logging
- `services/` — all service implementations
- `ui/` — window and controllers (wiring only)
- **Must not** contain business logic or AI inference

### Extension Points

- Additional config loaders for user app-data overrides
- Plugin registration hook (future) in `container.py`

### Internal Components

| Component | Role |
|-----------|------|
| `main.py` | Thin entry |
| `bootstrap.py` | Startup sequence |
| `container.py` | DI registry |
| `lifecycle.py` | Run/shutdown |

### Data Contracts

- Consumes: TOML files → `AppConfig`, `ModelConfig`, `ThemeConfig`
- Produces: `ApplicationContext` (controller references + config)

### Lifecycle

```
main → bootstrap → configure_logging → container.build → lifecycle.run → shutdown
```

### Configuration

- `config/app.default.toml`
- `config/models.default.toml`
- `config/themes.default.toml`

### Error Strategy

- Config load failure → log ERROR, show dialog, exit non-zero
- No silent fallback to hardcoded defaults

### Performance Considerations

- Bootstrap must complete in < 2 s excluding model preload (models are lazy)

### Testing Strategy

- Integration test: container builds without error
- Mock window for headless lifecycle test

### Known Limitations

- Single composition root; no multi-profile containers

### Future Expansion

- User config path override via CLI flag
- Plugin manifest loading

### Module Contract

| Field | Specification |
|-------|---------------|
| **Inputs** | Config file paths (optional overrides) |
| **Outputs** | Running `ApplicationLifecycle` |
| **Expected Behaviour** | All directories created; logging active; controllers wired |
| **Failure Modes** | Missing config, invalid TOML, import error |
| **Exceptions** | Propagate to main; never uncaught |
| **Recovery** | None at bootstrap; fail fast with message |
| **Performance** | No model loading at bootstrap |

---

## §2 — `core/` Foundation Layer

### Purpose

Shared infrastructure: configuration types, DTOs, exceptions, logging, constants, utilities.

### Responsibilities

- Define **all cross-module DTOs** in `contracts/`
- Define exception hierarchy with user/developer messages
- Provide logging factory
- Provide path resolution without hardcoded paths
- Hold pipeline stage and model kind enumerations

### Public Interfaces

- All types in `core/contracts/`
- `SentivisError` hierarchy
- `get_logger()`, `configure_logging()`
- `load_app_config()`, `load_model_config()`, `load_theme_config()`
- `project_root()`, `resource_path()`

### Dependencies

- **None** from application feature modules
- Third-party: `tomllib`/`tomli`, `PIL` (utils only), stdlib

### Extension Points

- New DTOs added to `contracts/` for new pipeline stages
- New exception subclasses per feature domain

### Internal Components

| Package | Role |
|---------|------|
| `config/` | Dataclasses + TOML loader |
| `contracts/` | Frozen dataclass DTOs |
| `constants/` | Enums and default limits |
| `exceptions/` | Error hierarchy |
| `logging/` | Logger factory |
| `utils/` | Paths, timing, image helpers |

### Data Contracts

All inter-module payloads live here. Features **must not** define cross-boundary DTOs locally.

### Lifecycle

Static; initialized at import. Logging configured once at bootstrap.

### Configuration

Loader reads from `config/`; exposes typed dataclasses only.

### Error Strategy

Base `SentivisError` carries `user_message`, `developer_detail`, `recoverable`, `stage`.

### Performance Considerations

- DTOs are `frozen` dataclasses for cheap passing across threads
- No heavy imports at `core` import time (lazy torch prohibited in core)

### Testing Strategy

- Unit tests for DTO immutability, config loader, path resolution

### Known Limitations

- `hardware_config` should be distinct section in `AppConfig` (spec); merge acceptable if fields present

### Future Expansion

- JSON schema validation for config files

### Module Contract

| Field | Specification |
|-------|---------------|
| **Inputs** | TOML bytes, path strings |
| **Outputs** | Typed config objects, DTOs, log records |
| **Failure Modes** | Invalid TOML, missing keys |
| **Exceptions** | `KeyError`, `ValueError` wrapped at loader boundary |
| **Recovery** | None |
| **Performance** | O(1) DTO access; loader runs once |

---

## §3 — `vision/` Computer Vision Feature

### Purpose

Image validation, preprocessing, object detection, future tracking.

### Responsibilities

- Reject invalid images before any model load
- Produce `PreprocessedImage` with separate display/inference buffers
- Run YOLO detection → `DetectionResult`
- Provide tracking interface (no-op in v1)

### Public Interfaces

| Interface | Location |
|-----------|----------|
| `IImageValidator` | `vision/interfaces/validator.py` |
| `IImagePreprocessor` | `vision/interfaces/preprocessor.py` |
| `IObjectDetector` | `vision/interfaces/detector.py` |
| `IObjectTracker` | `vision/interfaces/tracker.py` |

Implementations: `ImageValidator`, `StandardPreprocessor`, `YoloEngine`/`YoloDetector`, `NoOpTracker`.

### Dependencies

- `core/` only
- **Must not** import `analysis/`, `language/`, `ui/`, `services/`

### Extension Points

- Alternate validators (e.g. RAW support)
- Replace `YoloEngine` via `ModelRegistry`
- Real tracker implementing `IObjectTracker`

### Internal Components

```
vision/
  interfaces/
  validation/image_validator.py
  preprocessing/standard_preprocessor.py
  detection/yolo_engine.py
  tracking/noop_tracker.py
```

### Data Contracts

| Direction | Type |
|-----------|------|
| In | `Path` → `ValidatedImage` |
| Mid | `ValidatedImage` → `PreprocessedImage` |
| Out | `PreprocessedImage` → `DetectionResult` |

### Lifecycle

`YoloEngine`: UNINITIALIZED → LOADED → INFERRING → RELEASED. Managed exclusively by `ModelManager`.

### Configuration

- `AppConfig.image.*` — dimensions, file size
- `ModelConfig.yolo.*` — variant, thresholds, device

### Error Strategy

- `ValidationError` — non-recoverable, early abort
- `DetectionError` / `InferenceError` — may trigger CPU fallback via ModelManager

### Performance Considerations

- Inference buffer capped at 640 px
- No duplicate full-resolution tensors passed to YOLO

### Testing Strategy

- Unit: validator with fixture images
- Integration: YOLO with mocked weights or CPU smoke test

### Known Limitations

- Single-frame only; tracker is no-op

### Future Expansion

- ByteTrack/OC-SORT tracker
- Additional detector backends

### Module Contract — `ImageValidator`

| Field | Specification |
|-------|---------------|
| **Inputs** | `Path` to image file |
| **Outputs** | `ValidatedImage` with RGB `uint8` array |
| **Expected Behaviour** | Reject corrupt, oversize, unsupported format |
| **Failure Modes** | Missing file, corrupt decode, dimension limit |
| **Exceptions** | `ValidationError` |
| **Recovery** | None |
| **Performance** | < 500 ms for 4K image on target CPU |

### Module Contract — `YoloEngine`

| Field | Specification |
|-------|---------------|
| **Inputs** | `PreprocessedImage.inference_pixels` |
| **Outputs** | `DetectionResult` in original image coordinates |
| **Expected Behaviour** | Lazy load; infer; scale boxes to source dimensions |
| **Failure Modes** | OOM, missing weights, runtime error |
| **Exceptions** | `ModelLoadError`, `InferenceError` |
| **Recovery** | CPU fallback via ModelManager |
| **Performance** | < 3 s GPU / < 15 s CPU per image on target hardware |

---

## §4 — `analysis/` Scene Analysis Feature

### Purpose

Transform `DetectionResult` into structured scene understanding without heavy ML models.

### Responsibilities

- Extract per-object attributes
- Infer spatial relationships
- Build scene graph
- Infer coarse activities
- Aggregate `SceneContext`

### Public Interfaces

**Required** (architecture): `analysis/interfaces/` with:

| Interface | Implementation |
|-----------|----------------|
| `IAttributeExtractor` | `AttributeExtractor` |
| `IRelationshipAnalyzer` | `RelationshipAnalyzer` |
| `ISceneGraphBuilder` | `SceneGraphBuilder` |
| `IActivityAnalyzer` | `ActivityAnalyzer` |
| `ISceneContextBuilder` | `ContextBuilder` |

> **Draft gap:** `analysis/interfaces/` package not yet present in draft code.

### Dependencies

- `core/` only (DTOs from `core/contracts/analysis.py`, `detection.py`)

### Extension Points

- ML-based activity classifier
- Semantic relation model replacing heuristics

### Internal Components

Five pure-logic analyzers; no GPU, no model weights.

### Data Contracts

```
DetectionResult → AttributeSet
DetectionResult → Relation[]
DetectionResult + Relation[] → SceneGraph
SceneGraph → ActivityHints
All → SceneContext
```

### Lifecycle

Stateless per request; no persistent state between pipeline runs.

### Configuration

- Thresholds for relation distance (future: `config/analysis.default.toml`)
- v1: heuristic constants in module (must externalize in implementation pass)

### Error Strategy

- `AnalysisError` — recoverable where optional; context build failure aborts caption quality

### Performance Considerations

- O(n²) relation pairs acceptable for n < 50 detections
- CPU-only; must complete in < 1 s for typical scenes

### Testing Strategy

- Pure unit tests with fixture `DetectionResult`
- No GPU required

### Known Limitations

- Heuristic activities; no learned environment classifier
- Time/weather always "unknown" in v1

### Future Expansion

- Learned scene classifiers
- 3D spatial reasoning

### Module Contract — `ContextBuilder`

| Field | Specification |
|-------|---------------|
| **Inputs** | Detections, attributes, relations, graph, activities |
| **Outputs** | `SceneContext` |
| **Expected Behaviour** | Deterministic merge; dominant object ranking |
| **Failure Modes** | Empty detections (still produces valid context) |
| **Exceptions** | `AnalysisError` on internal inconsistency |
| **Recovery** | Minimal context with zero objects |
| **Performance** | < 100 ms |

---

## §5 — `language/` Natural Language Feature

### Purpose

Vision-language understanding, reasoning, prompt construction, caption refinement.

### Responsibilities

- BLIP: image → visual semantics caption
- PromptBuilder: `SceneContext` → structured `Prompt`
- Gemma: prompt → reasoned caption
- CaptionRefiner: polish final text

### Public Interfaces

**Required** (architecture): `language/interfaces/` with:

| Interface | Implementation |
|-----------|----------------|
| `IVisionLanguageModel` | `BlipEngine` / `BlipModel` |
| `IReasoningModel` | `GemmaEngine` / `GemmaModel` |
| `IPromptBuilder` | `PromptBuilder` |
| `ICaptionRefiner` | `CaptionRefiner` |

> **Draft gap:** `language/interfaces/` package not yet present in draft code.

### Dependencies

- `core/` only for DTOs and exceptions
- transformers/torch isolated inside engines

### Extension Points

- Replace BLIP with LLaVA via registry
- Template library in `prompts/templates/`

### Internal Components

```
language/blip/blip_engine.py
language/gemma/gemma_engine.py
language/prompts/prompt_builder.py
language/refinement/caption_refiner.py
```

### Data Contracts

| Stage | In | Out |
|-------|-----|-----|
| BLIP | `PreprocessedImage` | `RawCaption` |
| Prompt | `SceneContext` | `Prompt` |
| Gemma | `Prompt` | `RawCaption` |
| Refine | `RawCaption` ×2 | `RefinedCaption` |

### Lifecycle

`BlipEngine`, `GemmaEngine`: same IModelEngine lifecycle via ModelManager.

### Configuration

- `ModelConfig.blip.*`, `ModelConfig.gemma.*`

### Error Strategy

- BLIP fail → recoverable; template caption from context
- Gemma fail → recoverable; fall back to BLIP caption
- `ModelLoadError` if weights unavailable

### Performance Considerations

- Sequential load; never coexist with YOLO in VRAM
- Gemma INT4 on GPU; CPU fallback expected on 2 GB

### Testing Strategy

- Unit: PromptBuilder, CaptionRefiner (no GPU)
- Integration: engines with mocked generate()

### Known Limitations

- Gemma may require HF token
- Prompt templates are code-resident (should move to data files)

### Future Expansion

- Multi-language output
- User-editable prompt templates in Settings

### Module Contract — `GemmaEngine`

| Field | Specification |
|-------|---------------|
| **Inputs** | `Prompt` (system + user) |
| **Outputs** | `RawCaption` |
| **Expected Behaviour** | Load lazy; generate; release |
| **Failure Modes** | OOM, auth failure, timeout |
| **Exceptions** | `ModelLoadError`, `InferenceError` |
| **Recovery** | CPU fallback; BLIP caption as pipeline fallback |
| **Performance** | < 30 s GPU / < 120 s CPU on target hardware |

---

## §6 — `services/` Application Services

### Purpose

Orchestrate pipeline, manage models/memory, cache, export. Sole bridge between features and UI controllers.

### Responsibilities

- **PipelineOrchestrator** — enforce 13-stage order
- **ModelManager** — sole AI model authority
- **MemoryManager** — stats, cleanup, warnings
- **CacheManager** — optional disk cache
- **ExportManager** — JSON/TXT/PDF/image writers
- **ProgressReporter** / **CancellationToken**

### Public Interfaces

| Service | Consumer |
|---------|----------|
| `PipelineOrchestrator.analyze()` | `PipelineController` |
| `ModelManager.acquire/release` | Orchestrator only |
| `ExportManager.export()` | `ExportController` |
| `ProgressReporter.subscribe()` | UI progress |

**Architecture requirement:** Orchestrator depends on **feature interfaces**, not concrete classes.

> **Draft gap:** `orchestrator.py` imports concrete implementations directly.

### Dependencies

- `core/`
- Feature **interfaces** (not concrete modules, except at composition root wiring)
- **Must not** import `ui/`

### Extension Points

- New export writer registration
- Batch orchestrator method
- Remote inference adapter

### Internal Components

```
services/pipeline/orchestrator.py
services/pipeline/stage_runner.py      ← required; missing in draft
services/pipeline/progress_reporter.py
services/pipeline/cancellation.py
services/models/model_manager.py
services/models/model_registry.py
services/models/device_selector.py
services/memory/memory_manager.py
services/memory/managed_resources.py   ← required; missing in draft
services/cache/cache_manager.py
services/export/export_manager.py
```

### Data Contracts

- Pipeline: `PipelineRequest` → `PipelineResult`
- Progress: `StageProgress` events
- Export: `PipelineResult` + format → file

### Lifecycle

ModelManager maintains at most one active heavy model. Orchestrator creates new CancellationToken per run.

### Configuration

- All via injected `AppConfig`, `ModelConfig`

### Error Strategy

- StageRunner catches per-stage errors
- Orchestrator applies recovery policy
- Never leak exceptions to UI thread uncaught

### Performance Considerations

- Pipeline on worker thread only
- Model slot schedule: YOLO → BLIP → Gemma

### Testing Strategy

- Integration with fake engines implementing interfaces
- Memory release assertions between stages

### Known Limitations

- Single concurrent pipeline in v1

### Future Expansion

- Queue-based batch processing
- Pipeline checkpoint/resume via CacheManager

### Module Contract — `ModelManager`

| Field | Specification |
|-------|---------------|
| **Inputs** | `ModelKind`, preferred device string |
| **Outputs** | `IModelEngine` ready for inference |
| **Expected Behaviour** | Release prior model; load requested; log memory |
| **Failure Modes** | GPU OOM, corrupt weights |
| **Exceptions** | Propagate `ModelLoadError`; trigger CPU retry |
| **Recovery** | Release → clear cache → CPU load |
| **Performance** | Load/unload cycle < 10 s each model |

---

## §7 — `ui/` Presentation Layer

### Purpose

PySide6 desktop interface; user interaction only.

### Responsibilities

- Display image, progress, caption, scene context, history
- Dispatch user actions to controllers
- Apply theme
- Run **no** inference on main thread

### Public Interfaces

Controllers (for widgets):

| Controller | Methods |
|------------|---------|
| `MainController` | Service accessors |
| `PipelineController` | `analyze_image()`, `cancel()` |
| `ExportController` | `export_result()` |
| `SettingsController` | Config/theme access |

Signals: `progress_changed`, `analysis_completed`, `analysis_failed`, export signals.

### Dependencies

- `services/` via controllers
- `core/contracts/` for display types
- **Must not** import `vision/`, `language/`, `analysis/`, `torch`, `ultralytics`, `transformers`
- **Must not** import `app/` (draft violation: `AppWindow` imports `ApplicationContext` from `app.container`)

> **Draft gap:** UI imports `app.container`; should receive a UI-scoped facade from bootstrap injection.

### Extension Points

- Additional panels (settings, about)
- View models for complex state (`PipelineState`, `HistoryModel` — specified, missing in draft)

### Internal Components

```
ui/app_window.py
ui/controllers/
ui/workers/pipeline_worker.py
ui/themes/
ui/view_models/     ← required; missing in draft
```

### Data Contracts

- Displays: `PipelineResult`, `StageProgress`, `RefinedCaption`
- Never mutates DTOs

### Lifecycle

Tied to Qt event loop. Worker threads joined on shutdown.

### Configuration

- `ThemeConfig` via SettingsController

### Error Strategy

- Show `user_message` only in dialogs
- Never display stack traces

### Performance Considerations

- Main thread never blocks > 16 ms
- Image preview uses scaled pixmap

### Testing Strategy

- Controller unit tests with mocked services
- QTest for navigation (future)

### Known Limitations

- Monolithic `AppWindow` (should split into dashboard/sidebar/viewer panels per structure doc)

### Future Expansion

- Separate widget modules per architecture folder layout
- Settings dialog

### Module Contract — `PipelineController`

| Field | Specification |
|-------|---------------|
| **Inputs** | Image path from UI |
| **Outputs** | Qt signals with DTOs |
| **Expected Behaviour** | Spawn worker; forward progress; emit result |
| **Failure Modes** | Pipeline error, cancel |
| **Exceptions** | Caught on worker; emitted as signal |
| **Recovery** | UI remains interactive |
| **Performance** | Signal latency < 50 ms |

---

## Dependency Graph (Validated)

```
app → core, services, ui
ui → services, core.contracts          [DRAFT: also app — NON-COMPLIANT]
services → core, feature.interfaces    [DRAFT: concrete features — NON-COMPLIANT]
vision → core
analysis → core
language → core
```

**Circular dependencies:** None detected.  
**Feature-to-feature imports:** None detected.  
**UI → AI engines:** None detected.  
**Hidden dependencies:** Orchestrator concrete imports; UI → app import.

---

*Architecture Phase — Part 2 §1. Implementation not accepted until compliance report passes and remaining spec sections processed.*
