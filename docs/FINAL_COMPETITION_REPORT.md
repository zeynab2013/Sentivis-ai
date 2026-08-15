# Sentivis AI — Final Competition Report

**Evaluated:** 2026-07-31 15:04:05 UTC

## Validation Gate

- Python 3.10.11 compatibility: maintained
- Architecture v2.3 / DI / frozen subsystems: unchanged
- `ruff`, `mypy`, `pytest`: passing at report generation time

## Measured Performance Summary

| Area | Metric | Measured |
|------|--------|----------|
| Captions | Quality | 87.1% |
| Safety | Hallucination rate | 3.2% |
| Scene | Environment accuracy | 88.5% |
| Scene | Activity accuracy | 93.3% |
| Scene | Relationship accuracy | 55.1% |
| Evidence | Consistency | 67.3% |
| Language | Narrative fluency | 70.8% |
| Overall | Semantic score | 83.4% |
| Imaging | Avg enhancement improvement | 0.0% (1/10 images) |

## Pipeline Capabilities Delivered

- Adaptive image enhancement with measured quality report
- YOLO detection with SAM2/bbox segmentation refinement
- Segmentation-aware relationship and activity reasoning
- Evidence-only Ollama semantic synthesis with mandatory caption validation
- Executive summary, narrative (120–250 words), and short caption outputs
- Premium UI with i18n, comparison viewer, and branded exports

## Related Reports

- [REAL_WORLD_EVALUATION.md](REAL_WORLD_EVALUATION.md)
- [SEMANTIC_REPORT.md](SEMANTIC_REPORT.md)
- [IMAGE_ENHANCEMENT_REPORT.md](IMAGE_ENHANCEMENT_REPORT.md)
