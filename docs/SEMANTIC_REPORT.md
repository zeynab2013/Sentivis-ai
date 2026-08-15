# Sentivis AI — Semantic Evaluation Report

**Evaluated:** 2026-07-31 15:04:05 UTC
**Images:** 20

## Measured Semantic Metrics

| Metric | Measured | Target | Status |
|--------|----------|--------|--------|
| Caption quality | 87.1% | >97% | Not met |
| Hallucination rate | 3.2% | <1% | Not met |
| Environment accuracy | 88.5% | >96% | Not met |
| Activity accuracy | 93.3% | >98% | Not met |
| Relationship accuracy | 55.1% | >93% | Not met |
| Evidence consistency | 67.3% | >98% | Not met |
| Narrative fluency | 70.8% | >98% | Not met |
| Overall semantic score | 83.4% | >96% | Not met |

## Notes

- All values are measured on the COCO val2017 real-world validation set (20 images).
- Ollama/Gemma synthesis receives verified evidence only; detection and activity remain heuristic.
- Segmentation-aware relationship analysis rejects impossible containment when mask overlap does not support it.

Raw results: `validation/real_world/results.json`
