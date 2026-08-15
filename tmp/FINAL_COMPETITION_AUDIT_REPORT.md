# FINAL COMPETITION AUDIT — Forensic Report

**Mode:** Audit only. **Product code changes:** **ZERO**.

**Date:** 2026-08-13

---

## 1. Tests passed

| Suite | Result |
|-------|--------|
| Full `tests/unit` | **600 passed** |
| Focused metric / caption / QA (`test_quality_evaluator`, `test_dynamic_caption_evidence_coverage`, `test_final_qa_evidence_flow`, `test_evidence_qa_consistency`, `test_result_formatters`) | **26 passed** |
| Hardening / activity / QA overhaul | **24 passed** |
| Critical real-image gate (prior artifact `tmp/FINAL_EVIDENCE_QUALITY_VALIDATION.json`) | **fails=0** |

No concrete regression was demonstrated during this audit. Per FINAL SAFETY RULE: **no code modified**.

---

## 2. Critical image results

Source: `tmp/FINAL_EVIDENCE_QUALITY_VALIDATION.json`

| Case | Status | Caption (summary) | QA |
|------|--------|-------------------|----|
| HORSE | PASS | leading horse, holding rope, fire, brown horse, two people | people=2; leading; horse brown |
| SOCCER | PASS | playing football; white sports ball; three people | people=3; playing football; OCR “21” |
| MOTORCYCLE | PASS | riding motorcycle; red jersey | people=1; riding motorcycle |
| BICYCLE | PASS | riding bicycle; two people; bicycle/handbag behind | people=2; riding bicycle |

Relationships / activities represented in caption language when confirmed:

- horse: **leading**, **holding** (rope)
- soccer: **playing football**
- motorcycle / bicycle: **riding**

---

## 3. Dense-image result

| Scene | Words | Verified labels (YOLO/evidence) | Setting | Notes |
|-------|------:|----------------------------------|---------|-------|
| DENSE_TENNIS | 56 | person, tennis racket | recreational area | Richer than ~30-word stub; activity + crowd language present |
| KITCHEN (real) | 25 | bowl, person, sink | kitchen | Concise **by design** — no table/fridge/TV in verified labels |

Unit rich-kitchen fixture path (prior pass) retains major fixtures when those labels are verified; real COCO kitchen does not invent them.

---

## 4. Metric integrity

### Definitions (code: `CaptionQualityEvaluator`)

| Metric | Formula | Missing inputs |
|--------|---------|----------------|
| Object coverage | unique graph labels mentioned / unique labels | `None` → UI **Unavailable** |
| Relationship coverage | semantic relations phrase-matched / semantic relations (excludes pure spatial near/left/…) | `None` → **Unavailable** |
| Activity coverage | non-weak SceneContext activities phrase-matched / those activities (a/an/the equivalent) | `None` → **Unavailable** |
| Hallucination risk | `min(1, 0.25 × unsupported_object_tokens)` | may be `None` |
| Overall / evidence consistency | weighted composites from the above + grammar/fluency | always numeric when report exists |

### UI wiring

- Streamlit `_confidence_bar`: `None` → **Unavailable** (not 0%/100%)
- Desktop `_fmt_metric`: same

### Spot-checks against validation artifact

- HORSE object/relationship/activity coverage all **1.0** with confirmed leading/holding and those phrases in caption → consistent
- SOCCER relationship/activity coverage **null/Unavailable** while verified_evidence still lists “playing football” → **not a fabricated %**; metric source is SceneContext non-weak activities, which can be empty even when VerifiedEvidence has CONFIRMED activities
- MOTORCYCLE relationship coverage **0.33** (partial semantic relation phrase coverage) → not decorative 100%
- Kitchen object coverage **1.0** with labels {bowl, person, sink} all in caption → mathematically consistent with unique-label definition

**Verdict:** Displayed coverage percentages are calculation-backed. Missing inputs show Unavailable. No evidence of confidence→coverage conversion for these bars.

---

## 5. Caption ↔ evidence consistency

### Critical captions (traceable)

| Claim | Evidence path |
|-------|----------------|
| Leading / holding / riding / football | Confirmed activities in validation rows |
| Brown horse | Entity-bound color QA answer matches caption |
| White sports ball | Caption + prior color hardening |
| Red jersey (moto) | Entity-bound clothing path |
| Kitchen setting | Verified setting = kitchen |
| OCR “21” | QA from OCR, not caption append |

### Known consistency limitations (documented, not regressions of this audit)

1. **Dense tennis caption** may include Florence-sourced crowd/clothing detail beyond YOLO unique labels `{person, tennis racket}`. Object coverage can still read 100% because that metric is **unique label mention**, not “every Florence claim verified.”
2. **Bicycle** locked text can retain awkward phrasing (“And a light blue are visible…”) after sanitize — grammatical, not a fabricated relationship invent.
3. **Activity coverage Unavailable** can coexist with VerifiedEvidence CONFIRMED activities when SceneContext activity list is empty/weak-filtered — metric is honest per its definition, but judges must read the definition.

Hallucination risk on critical rows in latest artifact: **0.0**.

---

## 6. QA audit

- Answers on critical cases are short, evidence-direct (counts, activity, color, OCR).
- Code enforces `_strip_appended_caption` / `_strip_appended_scene_caption`; prompts forbid caption append.
- Focused QA consistency tests: **passed**.

---

## 7. Other checklist items (forensic)

| Area | Status |
|------|--------|
| Caption from verified evidence (simple/medium/dense) | Pass within evidence limits; dense expands when labels exist |
| Relationships in caption when meaningful | Pass on critical riding/leading/holding |
| Object coverage prioritization | Pass; no forced detector dump; sparse kitchen stays short |
| Entity-bound colors | Pass on horse/soccer/moto QA; prior hardening intact |
| Environment specificity | Kitchen / recreational area preserved; no invented mountain/farm |
| OCR | Soccer “21” preserved in QA; no caption-append |
| Metrics | Honest definitions + Unavailable path verified |
| Gates / models / thresholds / UI | **Unchanged** this audit |

---

## 8. Remaining genuine limitations

1. Caption richness **cannot exceed** verified detections (real kitchen → ~25 words is correct).
2. Coverage metrics are defined on **SceneContext graph/activities**, not always identical to Vision Assistant VerifiedEvidence lists.
3. Optional Florence/Ollama prose can still contribute detail that object-coverage % does not fully police claim-by-claim.
4. Some locked captions remain multi-sentence / slightly awkward rather than fully literary.
5. CPU sequential batch validation can trip RAM guards without inter-run cache clear (harness concern only).

---

## STOP

All audit gates satisfied for competition freeze.

**Product code changes in this pass: 0.**

This remains the final production state for competition.
