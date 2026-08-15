# FINAL OBJECT COUNT ACCURACY PATCH — Report

## Root cause

`clamp_caption_object_counts` failed to repair inflated quantities like
`"4 brown refrigerators"` because:

1. `_verified_label_counts` treated `"refrigerator #1"` as the **label**
   (the `#id` suffix was not stripped), so counts were
   `{'refrigerator #1': 1}` instead of `{'refrigerator': 1}`.
2. Quantity phrases with color adjectives (`brown refrigerators`) looked up
   `"brown refrigerator"` and found **0**, so the clamp left the inflated
   text unchanged.
3. Final pipeline lock (`orchestrator`) sanitized captions but did **not**
   re-apply count clamping against `VerifiedSceneEvidence`, so VLM/arbitration
   quantities could survive into the locked caption.

Evidence flow failure point: after NaturalCaption / refine produced or retained
`"4 beige refrigerators"` (seen in mid-pipeline logs), the final lock lacked an
authoritative verified-entity count gate.

## Exact files / functions changed

| File | Functions / change |
|------|--------------------|
| `language/validation/caption_factuality.py` | `_canonical_count_label`, `_verified_label_counts`, `label_counts_from_verified`, `clamp_caption_object_counts` — strip `#id`, strip colors, special plurals (`people`), verified counts, vague quantifiers |
| `language/semantic/natural_caption_service.py` | `_aggregate_verified_objects` UID dedupe; `_format_object_count_phrase` singleton check on bare noun; re-clamp after robotic repair |
| `services/pipeline/orchestrator.py` | Preferred + locked captions clamped with `verified=verified_evidence` |
| `tests/unit/language/test_object_count_accuracy.py` | New regression suite |
| `tmp/run_kitchen_object_count_validation.py` | Kitchen validation harness |

## Before / after (production kitchen)

Image: `tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png`

| Metric | Value |
|--------|------:|
| Verified refrigerator entities | **1** |
| Verified chair entities | **4** |
| Inflated `"4 … refrigerators"` in locked caption | **No** |
| Clamp demo on failure phrase | `a brown refrigerator and 4 brown chairs appear farther back.` |

Locked caption uses singular refrigerator language (`a beige refrigerator`). Mid-pipeline still briefly showed `"4 beige refrigerators"` before the final verified count gate repaired it.

## Tests

| Suite | Result |
|-------|--------|
| Object-count unit tests | **8 passed** |
| Full unit suite | **608 passed** |
| Production kitchen count validation | **PASS** (fridge_n=1, bad_inflated=false) |
| Critical freeze (horse/soccer/moto/bike) | **critical_fails=0** |

## Metric integrity

Coverage percentages unchanged. Count repair only rewrites quantity phrases to match distinct verified entities. Missing coverage inputs still show Unavailable.

## STOP

Object-count accuracy patch complete. No further speculative work.
