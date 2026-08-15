# FINAL GENERAL EVIDENCE COUNT AUDIT

## 1. Root cause

Object quantities in locked captions could diverge from distinct verified entities because:

1. **Count lookup ignored entity IDs** — subjects like `class#1` were treated as unique *labels*, so clamp lookups for `"brown refrigerators"` found no match and left inflated VLM/support phrases unchanged.
2. **Multiple count sources** — NaturalCaption aggregation, VLM/Ollama candidates, and refine/support sentences could each emit quantities; final lock did not always re-apply verified counts.
3. **Cross-sentence recount** — a class already mentioned (`a refrigerator`) could be counted again later (`4 refrigerators`) because support/enrichment did not budget entities across the full caption.
4. **Singleton hard-cap** (removed) — forcing large fixtures to count `1` was class-heuristic and wrong for genuine multiples; verified entity IDs must decide.

Detector IoU/IoA same-label dedupe already exists in `yolo_engine._suppress_same_label_duplicates`. The remaining failure was **caption-side quantity realization**, not detector geometry alone.

## 2. General solution

Single authoritative count path:

```
verified entity IDs → count(class) → caption quantity reconcile at lock
```

- `label_counts_from_verified` / `_verified_label_counts` canonicalize labels and count **distinct entity IDs**.
- `clamp_caption_object_counts` (left-to-right):
  - forces explicit digit/word/vague quantities to verified counts
  - uses remaining budget so later phrases cannot re-count already covered entities
  - never promotes `"a person is riding"` into a plural census head
  - drops duplicate indefinite singulars for non-person classes already covered
- Generation: `_label_already_mentioned` skips classes already present (singular/plural/synonym).
- Metrics: object coverage credits **instances** with quantity awareness (not “class name once ⇒ 100%”).

## 3. Files changed

| File | Change |
|------|--------|
| `language/validation/caption_factuality.py` | Canonical counts + general reconcile / cross-sentence budget |
| `language/semantic/natural_caption_service.py` | Entity UID aggregation; label-already-mentioned; remove singleton hard-cap; re-clamp |
| `language/evaluation/caption_quality_evaluator.py` | Instance-aware object coverage |
| `services/pipeline/orchestrator.py` | Final lock clamp with `verified=` (prior pass, retained) |
| `tests/unit/language/test_object_count_accuracy.py` | General A–M regressions |
| `README.md` | Object counting + coverage metric definitions |

## 4. Files removed

- `tmp/_count_reconcile_snippet.py` (temporary splice helper)
- Removed unused `_SINGLETON_FIXTURES` production constant (class hard-cap)

No other production modules deleted.

## 5. Why not image-specific

- No filenames, UUIDs, or kitchen/refrigerator/chair branches in production logic.
- Plural maps and color adjective stripping are linguistic helpers for **any** class.
- Person handling is role-vs-census linguistics (activity head ≠ headcount), not scene-specific.
- Tests may use kitchen/horse fixtures; production code does not.

## 6. Object-count validation

| Case | Result |
|------|--------|
| Production kitchen (`5e4dc9b3-…png`) | refrigerator **1**, chairs **4**; caption has singular fridge; `bad_inflated_refrigerator=false` |
| Clamp demo | `A brown refrigerator and 4 brown chairs appear farther back.` |
| Unit A–M | **passed** |

## 7. Color validation

Entity-bound color pipeline unchanged. Count reconcile preserves color adjectives when rewriting quantities. Does not invent colors.

## 8. Relationship validation

Critical freeze relationship/activity checks: **PASS** (horse leading/holding, soccer playing, moto/bike riding). No relationship verifier changes.

## 9. Environment validation

Kitchen setting remains kitchen on production kitchen image. No environment heuristic changes in this pass.

## 10. Metric validation

Object coverage = instance accounting with optional stated quantities; `None` → Unavailable. No fabricated 100%. README updated.

## 11. Caption quality validation

Dense/support generation still expands from verified evidence; quantity reconcile prevents inflation without deleting unrelated details (clock retained when fridge recount dropped).

## 12. Regression results

| Suite | Result |
|-------|--------|
| Full unit | **613 passed** |
| Object-count / kitchen generate | **passed** |
| Critical freeze horse/soccer/moto/bike | **critical_fails=0** |
| Kitchen real image count | **PASS** |
| Streamlit import (`streamlit_app.main` / bootstrap / runtime) | **OK** (after correcting module path) |

## 13. Remaining genuine limitations

1. Caption richness still cannot exceed verified detections.
2. Mid-pipeline drafts may briefly contain wrong quantities before final clamp (lock is authoritative).
3. Color wrongness (e.g. beige vs white fridge) is separate from count reconcile if attributes report that color.
4. Grammar after phrase removal can still leave slightly awkward joins in rare cases.
5. Object coverage credits class mentions without quantity as covering all instances of that class (natural language); explicit wrong quantities reduce credit via `min(stated, verified)`.

## STOP

General entity-count consistency is implemented and validated. No further speculative passes.
