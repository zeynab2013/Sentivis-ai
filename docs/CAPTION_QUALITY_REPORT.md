# Caption Quality Report

Measured narrative caption quality on 20 COCO val2017 real photographs.  
Source: `validation/real_world/results.json` (2026-07-31 11:02:04 UTC).

## Evaluation Method

Captions scored are **`narrative_full`** (competition-facing output), not the internal technical caption.

| Scorer | Purpose |
|--------|---------|
| `CaptionQualityEvaluator` | Grammar, fluency, object/relationship/activity/context coverage |
| GT hallucination estimator | Extra detections + unsupported caption tokens vs COCO labels |
| Activity / environment scorers | Heuristic activity and indoor/outdoor GT alignment |

## Aggregate Scores

| Criterion | Average | Target | Status |
|-----------|--------:|-------:|--------|
| Caption quality | 0.892 | > 0.90 | Near miss |
| Narrative fluency | 0.735 | > 0.95 | Below target |
| Evidence consistency | 0.761 | > 0.95 | Below target |
| Hallucination rate | 0.034 | < 0.03 | Near miss |
| Activity reasoning | 0.950 | > 0.90 | Met |
| Environment reasoning | 0.885 | > 0.92 | Below target |
| Object detection | 0.698 | — | — |
| Relationship correctness | 0.576 | — | — |
| Attribute accuracy | 0.930 | — | — |
| Overall semantic score | 0.842 | — | — |

## Quality Improvements vs Template Captions

Prior template-style output (pre-narrative benchmark):

```
This appears to be an outdoor setting (individual presence, single person crowd level).
Objects include person (top-left), tennis racket (top-left)...
```

Post-narrative output (same image):

```
The photograph captures an outdoor tennis court, with one person visible.
A person appears to be playing tennis...
Short: A person is playing tennis in tennis court.
```

Measured impact:

- Caption quality: **+19.2 percentage points** (70.0% → 89.2%)
- Hallucination rate: **−50.0 percentage points** (53.4% → 3.4%)

## Per-Scene Caption Quality

| Scene type | Images | Avg caption quality | Avg narrative fluency | Avg hallucination |
|------------|-------:|--------------------:|----------------------:|------------------:|
| animals | 2 | 0.941 | 0.880 | 0.084 |
| classrooms | 2 | 0.883 | 0.635 | 0.018 |
| indoor | 2 | 0.870 | 0.900 | 0.021 |
| kitchens | 2 | 0.880 | 0.920 | 0.025 |
| offices | 2 | 0.899 | 0.920 | 0.031 |
| outdoor | 2 | 0.882 | 0.635 | 0.000 |
| people | 2 | 0.903 | 0.615 | 0.000 |
| sports | 2 | 0.908 | 0.615 | 0.050 |
| streets | 2 | 0.908 | 0.615 | 0.021 |
| vehicles | 2 | 0.848 | 0.615 | 0.094 |

Computed from `validation/real_world/results.json` evaluations grouped by `scene_type`.

## Remaining Quality Gaps (measured root causes)

From benchmark confusion summary:

| Root cause | Count |
|------------|------:|
| Detection precision gap (false positive) | 17 |
| Detection recall gap (YOLO threshold or object scale) | 15 |
| Attribute zone boundary edge case | 11 |
| Relationship proximity/threshold gap | 8 |
| Environment label inference gap | 2 |
| Activity rule coverage gap | 2 |

Caption quality is capped by detection recall (missing GT objects) and relation coverage, not narrative formatting alone.

## Caption Rules Enforced

The narrative generator and validator reject or strip:

- Raw coordinates, percentages, confidence values, object IDs
- Internal terminology (`Objects include`, `Supported activity`, `crowd level`)
- Unsupported object or activity tokens

Ollama rewrites verified evidence only; activities remain heuristic-derived.

## Validation Commands

```powershell
python -m pytest tests/unit -q --ignore=tests/integration --ignore=tests/acceptance
python -m ruff check language services/pipeline ui/widgets ui/view_models analysis/activity analysis/context core/contracts
python scripts/run_real_world_evaluation.py
```

Measured test run: **92 passed** (unit tests).
