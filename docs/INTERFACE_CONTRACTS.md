# SENTIVIS AI — Interface Contracts (Field Reference)

**Version:** 2.1  
**Companion to:** `docs/INTERFACE_SPECIFICATION.md`

---

## Pipeline Stage Interface Map

| Stage | Interface | Input DTO | Output DTO |
|-------|-----------|-----------|------------|
| Validation | `IImageValidator` | `Path` | `ValidatedImage` |
| Preprocessing | `IImagePreprocessor` | `ValidatedImage` | `PreprocessedImage` |
| Detection | `IObjectDetector` | `PreprocessedImage` | `DetectionResult` |
| Attributes | `IAttributeExtractor` | `DetectionResult` | `AttributeSet` |
| Relations | `IRelationshipAnalyzer` | `DetectionResult` | `tuple[Relation, ...]` |
| Scene Graph | `ISceneGraphBuilder` | `DetectionResult`, relations | `SceneGraph` |
| Activity | `IActivityAnalyzer` | `SceneGraph` | `ActivityHints` |
| Context | `ISceneContextBuilder` | All analysis DTOs | `SceneContext` |
| VLM | `IVisionLanguageModel` | `PreprocessedImage` | `RawCaption` |
| Prompt | `IPromptBuilder` | `SceneContext` | `Prompt` |
| Reasoning | `IReasoningModel` | `Prompt` | `RawCaption` |
| Refinement | `ICaptionRefiner` | `RawCaption` ×2 | `RefinedCaption` |

---

## `analysis/interfaces/` File Manifest

```
analysis/interfaces/
  __init__.py
  attribute_extractor.py    → IAttributeExtractor
  relationship_analyzer.py  → IRelationshipAnalyzer
  scene_graph_builder.py  → ISceneGraphBuilder
  activity_analyzer.py      → IActivityAnalyzer
  context_builder.py        → ISceneContextBuilder
```

## `language/interfaces/` File Manifest

```
language/interfaces/
  __init__.py
  vision_language.py        → IVisionLanguageModel
  reasoning.py              → IReasoningModel
  prompt_builder.py         → IPromptBuilder
  caption_refiner.py        → ICaptionRefiner
```

## `services/interfaces/` File Manifest

```
services/interfaces/
  __init__.py
  model_engine.py           → IModelEngine
  model_manager.py          → IModelManager
  stage_runner.py           → IStageRunner
  pipeline.py               → IPipelineOrchestrator
  progress.py               → IProgressReporter
  cancellation.py           → ICancellationToken
  export.py                 → IExportService
  managed_resource.py       → IManagedResource, IResourceScope
  plugin_registry.py        → IPluginRegistry, IEnginePlugin
```

## `ui/interfaces/` File Manifest

```
ui/interfaces/
  __init__.py
  pipeline_view_model.py    → IPipelineViewModel
  export_view_model.py      → IExportViewModel
  history_view_model.py     → IHistoryViewModel
  application_facade.py     → IApplicationFacade
```

---

## Exception Contract by Interface

| Interface | Exceptions | Recoverable |
|-----------|------------|-------------|
| `IImageValidator` | `ValidationError` | No |
| `IObjectDetector` | `DetectionError`, `InferenceError` | Inference yes |
| `IVisionLanguageModel` | `InferenceError`, `ModelLoadError` | Inference yes |
| `IReasoningModel` | `InferenceError`, `ModelLoadError` | Inference yes |
| `I*Analyzer` (analysis) | `AnalysisError` | Usually yes |
| `IStageRunner` | `SentivisError` wrapped | Per stage policy |
| `IExportService` | `ExportError` | Yes |

---

## Thread Safety Matrix

| Interface | UI Thread | Worker Thread |
|-----------|-----------|---------------|
| ViewModels | Read/write state | Receive signals only |
| Controllers | Slot handlers | Emit from worker via Qt signals |
| `IStageRunner` | Never | Execute |
| `IModelManager` | Never | acquire/release |
| Analysis interfaces | Never | Execute |
| `IImageValidator` | Never | Execute |

---

## Memory Contract by Interface

| Interface | Peak Memory | Release Point |
|-----------|-------------|---------------|
| `IModelEngine` | ≤ 1.1 GB VRAM slot | `release()` after stage |
| `IImagePreprocessor` | 2× image buffers | Return from method |
| `IObjectDetector` | Transient tensors | Before return |
| Analysis interfaces | O(n) DTOs | GC after stage |
| `IManagedResource` | Declared in `memory_budget_mb` | `dispose()` in finally |

---

*Part 2 §2 — dependency contracts*
