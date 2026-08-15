# SENTIVIS AI — Interface Specification

**Version:** 2.1  
**Part:** 2 / 4 (section 2) — Interface Design · Dependency Contracts  
**Status:** Architecture Phase — authoritative  
**Extends:** `docs/MODULE_SPECIFICATIONS.md` v2.0

---

## 1. Architecture Principle

Every module communicates **only** through stable interfaces. Feature modules never import each other. Dependencies flow through `core/contracts` and `*/interfaces`. Implementation details never cross boundaries.

---

## 2. Interface Package Index

| Package | Role |
|---------|------|
| `core/contracts/` | Immutable DTOs — data only, no behaviour |
| `vision/interfaces/` | Validation, preprocessing, detection, tracking |
| `analysis/interfaces/` | Attributes, relations, graph, activity, context |
| `language/interfaces/` | VLM, reasoning, prompts, refinement |
| `services/interfaces/` | Pipeline, stages, models, memory, export, plugins |
| `ui/interfaces/` | ViewModel contracts exposed to widgets |

**Rule:** No implementation module may bypass these packages. Composition root (`app/container.py`) is the **only** location that binds interface → implementation.

---

## 3. `core/contracts/` — Data Transfer Objects

Not behavioural interfaces. Frozen dataclasses only.

| DTO | Module consumers |
|-----|------------------|
| `ImagePayload`, `ValidatedImage`, `PreprocessedImage` | vision, services |
| `Detection`, `DetectionResult` | vision, analysis, services |
| `AttributeSet`, `Relation`, `SceneGraph`, `SceneContext`, `ActivityHints` | analysis, language, services |
| `RawCaption`, `Prompt`, `RefinedCaption` | language, services |
| `PipelineRequest`, `PipelineResult`, `StageProgress`, `AnalysisOptions` | services, ui/view_models |

See `docs/INTERFACE_CONTRACTS.md` for field-level contracts.

---

## 4. `vision/interfaces/`

### `IImageValidator`

| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Validate and decode image files |
| **Input** | `Path` |
| **Output** | `ValidatedImage` |
| **Exceptions** | `ValidationError` |
| **Lifecycle** | Stateless; no resources held |
| **Thread safety** | Safe on pipeline worker thread |
| **Performance** | < 500 ms @ 4K on target CPU |
| **Memory** | One RGB buffer ≤ max configured dimension |
| **Extension** | Alternate format handlers |

```python
class IImageValidator(Protocol):
    def validate(self, path: Path) -> ValidatedImage: ...
```

### `IImagePreprocessor`

| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Prepare display + inference buffers |
| **Input** | `ValidatedImage` |
| **Output** | `PreprocessedImage` |
| **Exceptions** | `ValidationError` on buffer failure |
| **Lifecycle** | Stateless |
| **Thread safety** | Pipeline worker |
| **Performance** | < 200 ms |
| **Memory** | Display buffer + one inference buffer (no duplicates) |
| **Extension** | Alternate resize/colour policies |

### `IObjectDetector`

| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Object detection abstracted from YOLO/YOLO-World/etc. |
| **Input** | `PreprocessedImage` |
| **Output** | `DetectionResult` |
| **Exceptions** | `DetectionError`, `InferenceError` |
| **Lifecycle** | Delegates to `IModelEngine` via ModelManager |
| **Thread safety** | Pipeline worker only |
| **Performance** | < 3 s GPU / < 15 s CPU |
| **Memory** | No permanent weight retention in detector facade |
| **Extension** | Plugin registry binding |

```python
class IObjectDetector(Protocol):
    def detect(self, image: PreprocessedImage) -> DetectionResult: ...
```

### `IObjectTracker`

| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Temporal ID assignment (future video) |
| **Input/Output** | `DetectionResult` → `DetectionResult` |
| **Exceptions** | `DetectionError` |
| **Lifecycle** | Stateless in v1 (NoOp) |
| **Extension** | ByteTrack, OC-SORT plugins |

---

## 5. `analysis/interfaces/`

All analysers are **stateless**, CPU-only, no GPU.

### `IAttributeExtractor`

| Input | Output | Exceptions | Performance |
|-------|--------|------------|-------------|
| `DetectionResult` | `AttributeSet` | `AnalysisError` | < 50 ms |

### `IRelationshipAnalyzer`

| Input | Output | Exceptions | Performance |
|-------|--------|------------|-------------|
| `DetectionResult` | `tuple[Relation, ...]` | `AnalysisError` | < 100 ms (n² heuristic) |

### `ISceneGraphBuilder`

| Input | Output | Exceptions | Performance |
|-------|--------|------------|-------------|
| `DetectionResult`, `tuple[Relation, ...]` | `SceneGraph` | `AnalysisError` | < 50 ms |

### `IActivityAnalyzer`

| Input | Output | Exceptions | Performance |
|-------|--------|------------|-------------|
| `SceneGraph` | `ActivityHints` | `AnalysisError` | < 50 ms |

### `ISceneContextBuilder`

| Input | Output | Exceptions | Performance |
|-------|--------|------------|-------------|
| All analysis DTOs | `SceneContext` | `AnalysisError` | < 100 ms |

**Thread safety:** All analysis interfaces — pipeline worker, immutable outputs.

---

## 6. `language/interfaces/`

### `IVisionLanguageModel`

| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Visual semantics caption (BLIP, Florence, LLaVA, …) |
| **Input** | `PreprocessedImage` |
| **Output** | `RawCaption` |
| **Exceptions** | `InferenceError`, `ModelLoadError` |
| **Lifecycle** | Via `IModelEngine`; no direct load in caller |
| **Performance** | < 10 s GPU / < 60 s CPU |
| **Memory** | Managed by ModelManager slot |
| **Extension** | Plugin: `capabilities=["vision_language"]` |

Orchestrator references **`IVisionLanguageModel` only** — never `BlipEngine`.

### `IReasoningModel`

| Input | Output | Model-agnostic |
|-------|--------|----------------|
| `Prompt` | `RawCaption` | Gemma, Llama, Mistral, … |

### `IPromptBuilder`

| Input | Output |
|-------|--------|
| `SceneContext` | `Prompt` |

Stateless. Template-driven. Configurable template path.

### `ICaptionRefiner`

| Input | Output |
|-------|--------|
| `RawCaption`, optional `RawCaption` | `RefinedCaption` |

Stateless. No model weights.

---

## 7. `services/interfaces/`

### `IModelEngine`

| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Common heavy-model lifecycle |
| **Lifecycle** | initialize → load → infer → release → clear_device_cache |
| **Thread safety** | One active caller (pipeline worker) |
| **Memory** | Must release all weights on `release()` |
| **Extension** | Plugin registry |

```python
class IModelEngine(Protocol):
    @property
    def model_kind(self) -> ModelKind: ...
    @property
    def is_loaded(self) -> bool: ...
    def load(self) -> None: ...
    def release(self) -> None: ...
    def clear_device_cache(self) -> None: ...
```

### `IModelManager`

| Purpose | Sole model authority |
| Input | `ModelKind`, device preference |
| Output | `IModelEngine` |
| Exceptions | `ModelLoadError`; triggers CPU fallback internally |

### `IStageRunner`

See `docs/STAGE_RUNNER_SPECIFICATION.md`.

### `IPipelineOrchestrator`

| Input | `PipelineRequest` |
| Output | `PipelineResult` |
| Dependencies | **Interfaces only** — see §8 |

```python
class IPipelineOrchestrator(Protocol):
    def analyze(self, request: PipelineRequest) -> PipelineResult: ...
    def cancel(self) -> None: ...
```

### `IExportService`

| Input | `PipelineResult`, format, `Path` |
| Output | None (writes file) |
| Exceptions | `ExportError` |

### `IManagedResource`

See `docs/MANAGED_RESOURCES_SPECIFICATION.md`.

### `IPluginRegistry`

See `docs/PLUGIN_ARCHITECTURE.md`.

---

## 8. Pipeline Orchestration — Interface-Only Rule

`PipelineOrchestrator` constructor **must** accept only interfaces:

```python
class PipelineOrchestrator:
    def __init__(
        self,
        validator: IImageValidator,
        preprocessor: IImagePreprocessor,
        detector: IObjectDetector,
        attribute_extractor: IAttributeExtractor,
        relationship_analyzer: IRelationshipAnalyzer,
        scene_graph_builder: ISceneGraphBuilder,
        activity_analyzer: IActivityAnalyzer,
        context_builder: ISceneContextBuilder,
        vision_language: IVisionLanguageModel,
        prompt_builder: IPromptBuilder,
        reasoning_model: IReasoningModel,
        caption_refiner: ICaptionRefiner,
        stage_runner: IStageRunner,
        model_manager: IModelManager,
        progress: IProgressReporter,
        cancellation: ICancellationToken,
    ) -> None: ...
```

### Prohibited in Orchestrator

- `from vision.detection.yolo_engine import YoloEngine`
- `from language.blip import BlipEngine`
- `isinstance(engine, YoloEngine)`
- Any reference to YOLO, BLIP, Gemma, Florence by name

### Model Replacement

Changing YOLO → YOLO-World or BLIP → Florence requires **only**:

1. Plugin registration in `PluginRegistry`
2. `config/models.default.toml` update

**Zero** orchestrator source changes.

---

## 9. `ui/interfaces/`

UI widgets depend on ViewModel interfaces, not services.

### `IPipelineViewModel`

| Method / Property | Purpose |
|-------------------|---------|
| `progress_percent: float` | 0–100 |
| `stage_label: str` | Current stage display name |
| `status_message: str` | User-facing status |
| `is_running: bool` | Analysis in progress |
| `caption_text: str` | Formatted caption |
| `scene_summary: str` | Formatted scene context |
| `warnings: tuple[str, ...]` | Degradation notices |
| `open_image(path)` | Command |
| `start_analysis()` | Command |
| `cancel_analysis()` | Command |

### `IExportViewModel`

| Method | Purpose |
|--------|---------|
| `export_json()`, `export_txt()`, `export_pdf()` | Commands |
| `last_export_path: str` | Notification state |

### `IHistoryViewModel`

| Property | Purpose |
|----------|---------|
| `entries: tuple[HistoryEntry, ...]` | Session history |

See `docs/VIEW_MODEL_SPECIFICATION.md`.

### Prohibited in UI

- Import `app.container.ApplicationContext`
- Import any `services/` concrete class into widgets
- Hold reference to `PipelineOrchestrator` in widgets

**Allowed:** Controllers translate ViewModel commands → services (controllers live in `ui/controllers/`, injected at bootstrap).

---

## 10. Dependency Flow (Updated)

```
ui/widgets → ui/interfaces (ViewModels)
ui/controllers → services/interfaces
services/pipeline → vision/interfaces
                  → analysis/interfaces
                  → language/interfaces
                  → services/interfaces
app/container → all implementations (binding only)
vision|analysis|language implementations → core/contracts + core/exceptions
```

---

## 11. P0 Violation Resolution (Architecture)

| P0 Issue | Architectural Resolution |
|----------|-------------------------|
| Missing `analysis/interfaces/` | Specified §5; files defined in `INTERFACE_CONTRACTS.md` |
| Missing `language/interfaces/` | Specified §6 |
| Orchestrator concrete imports | Constructor contract §8; binds interfaces only |
| Missing `StageRunner` | `docs/STAGE_RUNNER_SPECIFICATION.md` |
| UI → `app.container` | `IApplicationFacade` injected; widgets use ViewModels §9 |
| Missing `view_models/` | `docs/VIEW_MODEL_SPECIFICATION.md` |
| Missing `managed_resources` | `docs/MANAGED_RESOURCES_SPECIFICATION.md` |
| Missing `services/interfaces/` | Specified §7 |

All P0 items **resolved at architecture level**. Draft code remains non-compliant until implementation phase aligns to this spec.

---

*Architecture Phase — Part 2 §2. No implementation.*
