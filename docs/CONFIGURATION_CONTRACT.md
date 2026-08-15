# SENTIVIS AI — Configuration Contract

**Version:** 2.1  
**Part:** 2 / 4 (section 2)

---

## 1. Principle

Every configurable module exposes a schema. Configuration validated **before runtime**. Invalid config → fail fast with actionable diagnostics.

---

## 2. Configuration Files

| File | Modules served |
|------|----------------|
| `config/app.default.toml` | app, core, services/memory, ui/workers |
| `config/models.default.toml` | vision/detection, language engines, plugins |
| `config/themes.default.toml` | ui/themes |
| `config/analysis.default.toml` | analysis heuristics (new) |

User overrides: `%APPDATA%/SentivisAI/config/` (Windows 11).

---

## 3. Schema Validation

### Validator Location

`core/config/schema_validator.py` — runs after TOML parse, before dataclass construction.

### Rules

| Rule | Example |
|------|---------|
| Required keys present | `[image].max_dimension` must exist |
| Type correctness | `max_dimension` is int > 0 |
| Range validation | `0.0 < vram_warning_ratio <= 1.0` |
| Path existence (optional) | stylesheet path warns if missing |
| Plugin identifier registered | `plugins.detection.identifier` ∈ registry |
| Cross-field | `yolo_inference_size <= max_dimension` |

### Failure Output

```
ConfigurationError: config/models.default.toml
  [plugins.reasoning].identifier = "language.unknown"
  → Plugin not registered. Available: language.gemma_2b, ...
```

User message: *"Configuration is invalid. See log for details."*

---

## 4. Per-Module Schema Exposure

| Module | Schema export |
|--------|---------------|
| `vision/detection` | `[yolo]` section + plugin descriptor schema |
| `language/blip` | `[blip]` + plugin schema |
| `language/gemma` | `[gemma]` + plugin schema |
| `analysis/*` | `[analysis]` thresholds |
| `services/export` | `[export]` default formats |
| `ui/themes` | `[theme]` tokens |

Each plugin publishes `config_schema` in `PluginDescriptor`.

---

## 5. ConfigLoader Pipeline

```
Read TOML → schema_validator.validate(raw) → dataclass construction → freeze/immutable config objects
```

No module reads TOML directly except `core/config/loader.py`.

---

## 6. Acceptance Criteria

- [ ] Bootstrap aborts before UI on invalid config
- [ ] All magic numbers in analysis/vision removed to config
- [ ] Plugin identifier validated against registry at load time

---

*Architecture Phase — implementation pending*
