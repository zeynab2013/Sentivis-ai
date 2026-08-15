# Competition Judge Self-Review (Architecture v2.3)

## Verdict

The system now prioritizes **image understanding** over detection dumps: SceneReasoner evidence + enhanced image → VLM narration → anti-hallucination → one paragraph. Architecture v2.3, DI, offline paths, and Streamlit entry (`streamlit_app/main.py`) are preserved.

## Strengths

- Quality-gated enhancement (no ops on already-good images; `competition_always_enhance=false`)
- Mask-aware clothing/color when SAM2 masks exist
- LAB/HSV natural color names with confidence omission
- Optional MediaPipe pose with geometry/relation fallback
- Adapter-specific VLM narrate prompts requiring evidence briefs
- UX: description-first + Developer Debug Panel

## Measured benchmark (unique local validation set)

Dataset available locally: **20** COCO-style images under `validation/real_world/images` (not 100 unique yet).

Run (2026-08-01, before evidence-priority caption fix):

| Metric | Value |
|--------|------:|
| Caption quality | 50.3% |
| Hallucination rate | 61.3% |
| Object coverage | 24.6% |
| Detail richness | 8.8% |
| Human readability | 56.3% |

After preferring SceneReasoner evidence paragraphs over thin BLIP stubs, smoke tests show ~45–85 word evidence-grounded paragraphs with clothing colors, relations, and setting. Re-run:

```bash
python scripts/run_vision_intelligence_benchmark.py --limit 20 --no-expand
```

## Remaining competition deltas (honest)

- Clothing taxonomy is still heuristic without a dedicated garment classifier
- Accessories like glasses/watch/necklace often stay `unknown` without specialized detectors
- Need ≥100 unique public validation images for a full competition-grade scorecard
- Strongest captions still benefit from a strong local VLM (Qwen/Florence/InternVL) on GPU

## How to measure

```bash
python scripts/run_vision_intelligence_benchmark.py
```

Reports land under `validation/vision_intelligence/` (HTML, Markdown, CSV) when images are available.
