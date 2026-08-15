# SENTIVIS AI — Plugin Architecture

**Version:** 2.1  
**Part:** 2 / 4 (section 2)

---

## 1. Objective

Install new AI engines without modifying orchestrator, stage runner, or existing feature modules.

---

## 2. Interface — `IEnginePlugin`

```python
@dataclass(frozen=True)
class PluginDescriptor:
    identifier: str           # e.g. "vision.yolo_v8n"
    version: str
    capabilities: tuple[str, ...]  # "object_detection", "vision_language", "reasoning"
    required_resources: ResourceRequirements
    supported_tasks: tuple[PipelineStage, ...]
    config_schema: dict[str, object]  # JSON Schema subset

@dataclass(frozen=True)
class ResourceRequirements:
    min_vram_mb: float
    min_ram_mb: float
    preferred_device: str

class IEnginePlugin(Protocol):
    @property
    def descriptor(self) -> PluginDescriptor: ...
    def create_engine(self, config: dict[str, object]) -> IModelEngine: ...
    def create_detector(self, engine: IModelEngine) -> IObjectDetector | None: ...
    def create_vlm(self, engine: IModelEngine) -> IVisionLanguageModel | None: ...
    def create_reasoner(self, engine: IModelEngine) -> IReasoningModel | None: ...
```

---

## 3. `IPluginRegistry`

```python
class IPluginRegistry(Protocol):
    def register(self, plugin: IEnginePlugin) -> None: ...
    def get(self, identifier: str) -> IEnginePlugin: ...
    def resolve(
        self,
        capability: str,
        config: ModelConfig,
    ) -> IEnginePlugin: ...
    def list_plugins(self) -> tuple[PluginDescriptor, ...]: ...
```

---

## 4. Built-in Plugins (v1)

| Identifier | Capability | Default |
|------------|------------|---------|
| `vision.yolo_v8n` | `object_detection` | ✓ |
| `language.blip_base` | `vision_language` | ✓ |
| `language.gemma_2b` | `reasoning` | ✓ |

Future: `vision.yolo_world`, `language.florence2`, `language.llama_3` — register without code changes to orchestrator.

---

## 5. Registration Flow

```
app/bootstrap.py
  → PluginRegistry.register(YoloPlugin())
  → PluginRegistry.register(BlipPlugin())
  → PluginRegistry.register(GemmaPlugin())
  → ModelManager uses registry.resolve(capability, config)
```

---

## 6. Configuration Binding

`config/models.default.toml`:

```toml
[plugins.detection]
identifier = "vision.yolo_v8n"

[plugins.vision_language]
identifier = "language.blip_base"

[plugins.reasoning]
identifier = "language.gemma_2b"
```

Changing identifier swaps engine — orchestrator unchanged.

---

## 7. Validation

Plugin config validated against `config_schema` at bootstrap. Invalid schema → fail fast with diagnostic listing missing keys.

---

## 8. Acceptance Criteria

- [ ] Orchestrator has zero plugin identifier strings
- [ ] New plugin = new package + registry line in bootstrap only
- [ ] Plugin descriptor declares VRAM budget for ModelManager scheduling

---

*Architecture Phase — implementation pending*
