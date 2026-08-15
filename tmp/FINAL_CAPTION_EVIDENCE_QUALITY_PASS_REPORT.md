# FINAL CAPTION + EVIDENCE QUALITY PASS — Report

## Exact files changed

| File | Reason |
|------|--------|
| `language/semantic/natural_caption_service.py` | Stop late assemble/gate from dropping newly introduced verified fixtures; continue expansion while salient verified labels remain uncovered; richness from distinct verified classes; natural multi-object phrasing |
| `language/evaluation/caption_quality_evaluator.py` | Object coverage uses **unique** verified graph labels; activity/relation phrase match treats a/an/the as equivalent so coverage is not falsely 0% |
| `language/refinement/caption_sanity.py` | Rewrite leftover “share the frame” phrasing |
| `tests/unit/language/test_dynamic_caption_evidence_coverage.py` | End-to-end rich-kitchen generate must retain fixtures |
| `tests/unit/language/test_quality_evaluator.py` | Article-variant activity coverage unit |
| `README.md` | Caption metric definitions, exports, limitations |
| `tmp/run_final_evidence_quality_validation.py` | Critical + dense validation harness (+ RAM release between runs) |
| `tmp/FINAL_EVIDENCE_QUALITY_VALIDATION.json` | Validation artifact |

## Concrete failure fixed

Rich kitchen understanding (table, chairs, refrigerator, vase, cup, …) produced ~51 words mid-pipeline, then **collapsed to ~30–34 words** because:

1. Dining-table dedupe deleted whole sentences that also introduced vase/cup  
2. Final naturalness gate treated multi-object foreground lines as inventory once any prior sentence existed  
3. Enrichment stopped even when uncovered salient verified labels remained  

## Tests executed

- Full unit suite: **600 passed** (reconfirmed 2026-08-13)
- Critical real images (horse / soccer / motorcycle / bicycle): **fails=0** (reconfirmed)
- Dense scenes: kitchen (`coco_kitchen.jpg`), tennis crowd (`random_385406.jpg`)
- Rich-kitchen unit generate: retains kitchen + people + table + chair + refrigerator + vase/cup when fixtures are verified
- Activity coverage article-variant unit: `"leading the horse"` covers `"leading a horse"` → 100%

## Critical-image results (latest reconfirm)

From `tmp/FINAL_EVIDENCE_QUALITY_VALIDATION.json` (`fails=0`):

| Case | Status | Words | Notes |
|------|--------|------:|-------|
| HORSE | PASS | 25 | leading / rope / fire; activity_coverage **1.0**; horse brown QA OK |
| SOCCER | PASS | 23 | playing football; OCR “21” in QA; relationship/activity coverage Unavailable when SceneContext lacks those inputs |
| MOTORCYCLE | PASS | 16 | riding + red jersey; QA riding OK |
| BICYCLE | PASS | 30 | riding preserved |

## Dense-scene results

- **DENSE_TENNIS:** 56 words — richer multi-person / tennis training description; verified labels limited to `person`, `tennis racket`; setting recreational area  
- **Verified-rich kitchen (unit fixtures):** multi-sentence caption retaining fridge/table/chairs/vase/cup when those labels are verified  
- **`tmp/coco_kitchen.jpg` (real YOLO):** labels = bowl, person, sink only → **25-word** caption is evidence-correct (no table/fridge to invent)

## Coverage metric definitions

- **Object coverage** = unique verified scene-graph labels mentioned / unique labels (`None` → Unavailable)  
- **Relationship coverage** = semantic relations mentioned / semantic relations (`None` → Unavailable)  
- **Activity coverage** = non-weak activities mentioned / those activities (`None` → Unavailable); a/an/the treated as equivalent  
- **Hallucination risk** = `min(1, 0.25 × unsupported_object_tokens)`  
- **100%** means full coverage of that verified set — not “caption non-empty”

## Cleanup performed

Conservative unused-module scan of `language/`, `analysis/`, `vision/`, `services/`, `core/`: **no demonstrably unused production modules**. Nothing deleted.

## README verification

README documents purpose, architecture, pipeline, detection, evidence, relationships, activities, environment, colors, OCR, caption generation + quality metrics, QA, TTS, enhancement, languages, exports (PDF/HTML/Markdown/TXT/JSON), hardware/CPU fallback, install/run, structure, testing, limitations. No invented benchmarks or 100% accuracy claims.

## Remaining limitations

- Caption cannot invent fixtures YOLO did not verify  
- Optional Ollama/Florence alternate text may compete with NaturalCaptionService; locked caption prioritizes verified activity survival over Florence verbosity  
- Sparse real kitchen image stays concise by design  
- Sequential CPU batch validation can hit RAM guards unless cache is cleared between runs (harness only)

## STOP

Critical tests pass, unit suite green (600), dense verified-evidence captions retain fixtures when evidence exists, coverage percentages are evidence-honest (Unavailable when no inputs), README updated.

**FINAL PRODUCTION FREEZE — no further speculative work.**
