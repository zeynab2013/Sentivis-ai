# AI Extension Points

**Frozen architecture — extension via documented hooks only**

---

## Plugin Registry

Replace detection, vision-language, or reasoning engines without changing orchestrator code:

- `vision.yolo_v8n`
- `language.blip_base`
- `language.gemma_2b`

Configure via `[plugins]` in `config/models.default.toml`. Register new plugins in `app/plugin_bootstrap.py`.

---

## Analysis Heuristics

Tune relationship, activity, attribute, and context behavior via `config/analysis.default.toml` without code changes.

---

## Competition Configuration

`config/app.default.toml` `[competition]` section:

| Key | Purpose |
|-----|---------|
| `quality_threshold` | Minimum overall caption quality in strict QA |
| `max_hallucination_risk` | Maximum allowed hallucination score |
| `deterministic_seed` | Gemma seed in competition mode |
| `gemma_temperature` | Reference temperature (0 in competition) |
| `vram_release_threshold_mb` | Post-release VRAM ceiling |

Enable per run: `AnalysisOptions(competition_mode=True)`.

---

## Benchmarking

```python
from services.benchmark import BenchmarkRunner

runner = BenchmarkRunner(orchestrator)
report = runner.run((image_path,), iterations=5, competition_mode=True)
BenchmarkRunner.export_report(report, Path("exports/benchmark.json"))
```

---

## Metrics Consumption

`PipelineResult.metrics` is available to UI, export, and logging layers:

- Stage timings for progress diagnostics
- Peak memory for hardware warnings
- Fallback/recovery counts for reliability dashboards

Part 4 should surface metrics in the desktop UI without modifying pipeline internals.

---

## Quality Assurance

`PipelineQualityAssurance` thresholds follow `CompetitionConfig`. Non-competition runs use relaxed QA (recovery only on severe failures).

Extend contradiction checks in `services/pipeline/quality_assurance.py` without changing stage order.

---

## Export

`ExportManager` JSON export includes `metrics` and `qa_passed` fields. Additional export formats can read `PipelineResult` directly.

---

## Out of Scope (requires architecture revision)

- Parallel GPU model loading
- Multi-image batch pipeline
- Video / temporal analysis
- New pipeline stages without ADR
