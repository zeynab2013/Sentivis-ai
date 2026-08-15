# Sentivis AI — Pipeline Specification

**Version:** 1.0

## Canonical Flow

```
Validation → Preprocessing → YOLO → Attributes → Relations → Scene Graph
→ Activities → Scene Context → BLIP → Prompt → Gemma → Refinement → Export/UI
```

## Stage Details

| # | Stage | Input | Output | Device |
|---|-------|-------|--------|--------|
| 1 | Validation | Path | ValidatedImage | CPU |
| 2 | Preprocessing | ValidatedImage | PreprocessedImage | CPU |
| 3 | YOLO | PreprocessedImage | DetectionResult | GPU/CPU |
| 4 | Attributes | DetectionResult | AttributeSet | CPU |
| 5 | Relations | DetectionResult | Relation[] | CPU |
| 6 | Scene Graph | Detections + Relations | SceneGraph | CPU |
| 7 | Activities | SceneGraph | ActivityHints | CPU |
| 8 | Scene Context | All analysis DTOs | SceneContext | CPU |
| 9 | BLIP | PreprocessedImage | RawCaption | GPU/CPU |
| 10 | Prompt | SceneContext | Prompt | CPU |
| 11 | Gemma | Prompt | RawCaption | GPU/CPU |
| 12 | Refinement | RawCaptions | RefinedCaption | CPU |

## Recovery

- BLIP failure → continue with template from scene context
- Gemma failure → fall back to BLIP caption
- YOLO failure → abort pipeline

## Cancellation

Cooperative cancel checked between stages via `CancellationToken`.
