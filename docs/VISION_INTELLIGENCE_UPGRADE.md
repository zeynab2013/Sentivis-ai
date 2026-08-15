# Vision Intelligence Upgrade (Architecture v2.3 preserved)

## Caption pipeline (competition path)

1. SceneReasoner builds internal `SceneUnderstanding`
2. `NaturalCaptionService` builds **multiple candidates** (evidence human / atmospheric / people-first / compact / VLM / blend)
3. `CompetitionCaptionScorer` ranks by object, clothing, color, relationship, OCR, hallucination, fluency, readability
4. `CaptionRefiner.polish` humanizes grammar without stripping SceneReasoner facts
5. `AntiHallucinationFilter` final check
6. Streamlit shows **one paragraph** only (+ Enhanced Image if applied)

## Other intelligence modules

- Quality-gated enhancement (never touch good images)
- Mask-aware clothing/color when SAM2 available
- LAB/HSV natural color names with confidence gating
- Optional MediaPipe pose + relation fallback
- Adapter-specific VLM prompts via DI (`ManagedVisionModel`)

## Benchmark

```bash
python scripts/run_vision_intelligence_benchmark.py --limit 20 --no-expand
python scripts/run_vision_intelligence_benchmark.py --limit 100
```

Outputs: `validation/vision_intelligence/` (HTML, CSV, JSON) and `docs/VISION_INTELLIGENCE_BENCHMARK.md`.
