# SENTIVIS AI — Developer Guide

**Version:** 1.0  
**Part:** 1 / 4 (section 3) — Coding Standards · Implementation Rules · Quality Assurance  
**Status:** Active — binding for all implementation work  
**Extends:** `docs/ENGINEERING_CONTRACT.md` · `docs/HARDWARE_PERFORMANCE_POLICY.md` · `docs/ARCHITECTURE.md`

---

## 1. Implementation Philosophy

Every implementation prioritizes, in order of trade-off resolution:

1. **Correctness**
2. **Maintainability**
3. **Readability**
4. **Performance**
5. **Reliability**
6. **Scalability**
7. **Consistency**

Never write code merely because it works. Write code a senior engineer would maintain with confidence.

### Non-Negotiable Rules

- Never sacrifice **architecture** for speed.
- Never sacrifice **readability** for cleverness.
- Never sacrifice **maintainability** for short-term convenience.
- Every line reflects **production-quality** engineering.

---

## 2. Python & Style Standards

| Rule | Requirement |
|------|-------------|
| Python version | 3.10.11 (3.10.x only) |
| Style | PEP 8 via `ruff format` |
| Line length | 100 characters |
| Imports | isort rules via `ruff`; stdlib → third-party → local |
| Functions | One responsibility; prefer ≤ 40 lines |
| Classes | One responsibility; prefer ≤ 300 lines per file |
| Files | Split when exceeding ~400 lines |
| Side effects | Explicit; never hidden in property getters or `__init__` beyond wiring |
| Complexity | No nested logic > 3 levels; extract helpers |

### Prohibited (from Engineering Contract §8)

Placeholders · fake logic · `TODO` comments · magic numbers · hardcoded paths · bare `except:` · silent exception swallowing · empty classes/methods · unused imports/variables · duplicate/dead code

---

## 3. Type Safety

```python
# Required on every public function and method
def detect(self, image: PreprocessedImage) -> DetectionResult:
    ...
```

| Rule | Detail |
|------|--------|
| Type hints | All parameters and return types |
| Public APIs | Explicit types always; no untyped public surface |
| `Any` | Avoid; justify in docstring if unavoidable |
| Structured data | `@dataclass(frozen=True)` in `core/contracts/` |
| Protocols | `typing.Protocol` for interfaces in `*/interfaces/` |
| Optional | `Optional[X]` or `X \| None` (requires `from __future__ import annotations` on 3.10) |
| Collections | `tuple[T, ...]`, `list[T]` — prefer immutable tuples for outputs |

### Static Analysis

```bash
mypy app core vision language analysis services --strict
```

Configuration lives in `pyproject.toml` `[tool.mypy]` when implementation begins.

---

## 4. Documentation Standards

### Public Class Docstring

```python
class ImageValidator:
    """Validates image files before pipeline processing.

    Responsibilities:
        - Verify format, dimensions, and file integrity.
        - Enforce size limits from ``AppConfig``.

    Dependencies:
        - ``AppConfig`` (injected)

    Extension Points:
        - Implement ``IImageValidator`` for alternate validation rules.
    """
```

### Public Function Docstring

Every public function documents:

- **Purpose** — what and why
- **Parameters** — name, type, meaning
- **Return value** — type and semantics
- **Exceptions** — raised types and conditions
- **Notes** — non-obvious behavior (thread affinity, memory, device)

```python
def validate(self, path: Path) -> ValidatedImage:
    """Load and validate an image from disk.

    Args:
        path: Absolute or project-relative path to the image file.

    Returns:
        ValidatedImage containing decoded pixels and metadata.

    Raises:
        ValidationError: If format is unsupported, file is corrupt,
            or dimensions exceed configured limits.

    Note:
        Runs on the pipeline worker thread, not the UI thread.
    """
```

Private functions (`_prefix`) require docstrings only when behavior is non-obvious.

---

## 5. Naming Conventions

### Required Patterns

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase, role noun | `RelationshipAnalyzer` |
| Functions / methods | snake_case, verb phrase | `build_scene_context` |
| Constants | UPPER_SNAKE | `MAX_IMAGE_DIMENSION` |
| Modules | snake_case, singular feature | `caption_refiner.py` |
| Interfaces | `I` prefix or Protocol name | `IObjectDetector` |
| Private | `_leading_underscore` | `_parse_detections` |
| DTOs | PascalCase noun | `DetectionResult` |

### Forbidden Names

`Manager2` · `Helper` · `Utils2` · `Temp` · `FinalVersion` · `NewCode` · `Test123` · single-letter names except loop indices · abbreviations (`img_val`, `rel_anlz`)

### Approved Module Names (from Architecture)

`ImageValidator` · `RelationshipAnalyzer` · `CaptionRefiner` · `ModelManager` · `PipelineOrchestrator` · `MemoryManager` · `ContextBuilder` · `StageRunner`

---

## 6. Configuration

**No hardcoded values** in source. All tunables live in `config/*.toml` with user overrides in OS app-data.

| Category | Config File | Examples |
|----------|-------------|----------|
| Application | `config/app.default.toml` | log level, paths, worker limits, timeouts |
| Models | `config/models.default.toml` | model IDs, weights paths, device, quantization |
| Themes | `config/themes.default.toml` | palette, QSS path, font sizes |
| Hardware | `config/app.default.toml` `[hardware]` | VRAM warning threshold, CPU fallback enabled |

### Access Pattern

```python
# Correct — injected config object
class ImageValidator:
    def __init__(self, config: AppConfig) -> None:
        self._max_dimension = config.image.max_dimension

# Forbidden
MAX_DIM = 4096  # magic number in module scope
model_path = "C:/models/yolo.pt"  # hardcoded path
```

### Required Configurable Items

Model paths · detection thresholds · theme settings · export default directory · cache size limits · worker thread limits · pipeline timeouts · memory warning thresholds · image size limits · log level

---

## 7. Error Handling

### Rules

| Rule | Implementation |
|------|----------------|
| Recoverable errors | Catch `SentivisError`; apply recovery policy |
| Never silent ignore | Log at minimum WARNING before recovery |
| No bare `except:` | Catch specific types or `Exception` with log + re-raise/wrap |
| Developer logs | `developer_detail` + exception chain via `logger.exception` |
| User messages | `user_message` only in UI layer via controllers |
| Stack traces | Logs only; never in QMessageBox or UI labels |

### Pattern

```python
try:
    result = self._engine.infer(tensor)
except torch.cuda.OutOfMemoryError as exc:
    raise InferenceError(
        user_message="Analysis paused — retrying with reduced memory.",
        developer_detail=f"CUDA OOM during YOLO infer: {exc}",
        recoverable=True,
        stage=PipelineStage.YOLO_DETECTION,
    ) from exc
```

---

## 8. Dependency Injection & Testability

### Rules

- Constructor injection only; dependencies stored as `self._dep`.
- No `import` + instantiate inside business methods.
- No global singletons for services.
- Business logic testable without Qt event loop.
- UI testable with mocked controllers/services.

### Test Double Pattern

```python
# tests/unit/vision/test_image_validator.py
def test_rejects_oversized_image(fake_config: AppConfig) -> None:
    validator = ImageValidator(config=fake_config)
    with pytest.raises(ValidationError):
        validator.validate(oversized_image_path)
```

### Layer Test Scope

| Layer | Requires UI | Requires GPU |
|-------|-------------|--------------|
| `core/` | No | No |
| `analysis/` | No | No |
| `vision/` validation | No | No |
| `vision/` detection | No | Optional (CI uses CPU/fixtures) |
| `language/` | No | Optional |
| `services/` | No | Optional |
| `ui/controllers/` | Mock services | No |

---

## 9. Self-Review Checklist

Before marking any module complete, verify:

- [ ] Architecture compliance — correct layer, no forbidden imports
- [ ] Naming consistency — follows §5
- [ ] Memory safety — tensors released, no leaks, context managers used
- [ ] Thread safety — UI thread rules respected; immutable cross-thread DTOs
- [ ] Error handling — specific exceptions, no bare except, user/dev messages split
- [ ] Documentation — class + public method docstrings complete
- [ ] Type safety — mypy strict passes
- [ ] Configuration — no hardcoded values; all tunables in config
- [ ] Logging — startup, inference, errors, memory transitions logged
- [ ] No prohibited artifacts — contract §8

---

## 10. Quality Assurance Workflow

### Before Starting the Next Module

Run on all existing packages:

```bash
ruff format --check .
ruff check .
mypy app core vision language analysis services --strict
pytest tests/unit/ -q
```

Resolve every issue. Do not ignore warnings without documented justification in code comment (rare) or ADR.

### Tool Configuration (implementation phase)

| Tool | Purpose |
|------|---------|
| `ruff` | Format + lint (replaces flake8, isort, pyupgrade) |
| `mypy` | Static type checking (`--strict`) |
| `pytest` | Unit and integration tests |
| `pytest-cov` | Coverage reporting (target ≥ 80% on `core`, `analysis`, `services`) |

### Warning Suppression

Suppressions require inline comment with reason:

```python
result = legacy_call()  # noqa: SLF001 — required by ultralytics internal API; tracked in #issue
```

Prefer fixing root cause over suppression.

---

## 11. Definition of Done

A **feature** (module, stage, or UI panel) is complete only when **all** conditions hold:

| # | Criterion |
|---|-----------|
| 1 | Implementation finished — no placeholders |
| 2 | Documentation complete — docstrings + module header |
| 3 | Logging implemented — per `HARDWARE_PERFORMANCE_POLICY` §10 |
| 4 | Configuration supported — all tunables externalized |
| 5 | Errors handled — recoverable path + user/dev messages |
| 6 | Unit tests pass |
| 7 | Integration tests pass with dependent modules |
| 8 | Architecture rules satisfied — import lint clean |
| 9 | Performance acceptable — no UI thread blocking; memory policy upheld |
| 10 | QA workflow (§10) passes clean |

---

## 12. Module Completion Gate

Aligns with `HARDWARE_PERFORMANCE_POLICY` §12 and Engineering Contract §14.

```
Implement → Self-Review (§9) → Unit Tests → QA Workflow (§10) → Integration Tests → Done (§11)
```

Failure at any step blocks downstream modules that depend on the failed module.

---

## 13. Package Layout Reference

```
app/          → bootstrap, DI, lifecycle only
core/         → config, contracts, exceptions, logging, utils
vision/       → validation, preprocessing, detection, tracking
language/     → blip, gemma, prompts, refinement
analysis/     → attributes, relationships, scene_graph, context, activity
services/     → pipeline, models, memory, cache, export
ui/           → widgets, controllers, view_models, themes
tests/        → mirrors source tree
config/       → default TOML; no secrets committed
```

Import rules: see `docs/ARCHITECTURE.md` §3.

---

## 14. Logging Implementation

Use `core.logging.logger_factory.get_logger(__name__)`.

Every module logs at minimum:

- Module-level INFO on first use (optional, prefer app startup summary)
- ERROR on unrecoverable failure
- WARNING on recoverable degradation
- DEBUG on stage timing and memory snapshots (services only)

Never use `print()`.

---

## 15. Git & Review Discipline

- One module or logical feature per commit during implementation phase.
- Commit message: imperative mood, explains *why*.
- No `.env`, model weights, or `cache/` in version control.
- Pre-commit hooks (when configured): ruff + mypy on staged Python files.

---

## 16. Related Documents

| Document | Version |
|----------|---------|
| `docs/ENGINEERING_CONTRACT.md` | 1.2 |
| `docs/HARDWARE_PERFORMANCE_POLICY.md` | 1.0 |
| `docs/ARCHITECTURE.md` | 1.2 |
| `docs/ENGINEERING_REVIEW.md` | 1.0 |
| `docs/adr/` | ADR-001 … 007 |

---

*Extends all prior Master Prompt Book requirements. Does not replace them.*
