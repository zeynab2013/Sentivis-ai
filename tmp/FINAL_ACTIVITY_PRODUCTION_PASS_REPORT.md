# FINAL SURGICAL ACTIVITY + PRODUCTION CLEANUP PASS — FORENSIC REPORT

**Date:** 2026-08-13  
**Status:** FINAL PRODUCTION FREEZE (activity coverage only; frozen subsystems untouched)

---

## 1. Exact root cause of the missing activity

Verified multi-person activities were **collected** into `VerifiedSceneEvidence` and projected as multiple `activity` facts, but caption planning **collapsed** them:

1. **Story builder / `_infer_rich_activity`** kept only the **first** activity fact as `story.action`.
2. **`_secondary_person_clause`** ignored remaining activities and emitted spatial filler (`another person is farther back`) instead of the second person’s verified action.
3. **`ensure_salient_verified_coverage`** used an overly aggressive skip: if *any* of the first two activity tokens appeared (e.g. shared verb `holding`), the **second distinct activity** was dropped.

Net effect: Person A’s activity survived; Person B’s verified activity was lost before lock.

---

## 2. Exact files changed

| File | Role |
|------|------|
| `language/semantic/natural_caption_service.py` | Preserve all person→activity pairs; secondary clause uses distinct activity; coverage cues |
| `language/refinement/caption_coverage.py` | Fix skip logic; “One person / Another person” for multi-actor repair |
| `tests/unit/language/test_multi_person_activity_coverage.py` | New multi-scene activity regressions |
| `README.md` | Activity multi-person behavior, languages, Unavailable / metric definitions |
| `tmp/FINAL_ACTIVITY_PRODUCTION_PASS_REPORT.md` | This report |

**Not modified:** `analysis/common/color_utils.py`, YOLO/detection, counting, OCR, environment calibration, enhancement, QA architecture, translation, TTS, export, UI layout.

---

## 3. Exact functions changed

- `_StoryFacts` — added `person_activities`
- Story construction in `NaturalCaptionService` — collect all verified activity facts; prefer lead person’s activity for primary `action`
- `_infer_rich_activity` — keep already-selected lead activity; prefer lead-subject fact over “first fact wins”
- `_secondary_person_clause` — emit `another person is …` from a **different** verified activity when present
- `_coverage_ratio` — cue uncovered secondary activities
- `ensure_salient_verified_coverage` — strict coverage match; multi-person “One/Another person” phrasing

---

## 4. Why the fix is general

- Operates on **verified activity facts + entity IDs**, not filenames, kitchen fixtures, or hard-coded activity strings.
- Any CONFIRMED narrative-safe activity (cooking, riding, holding X, looking at a phone, …) is eligible.
- No image-specific branches; no forced activity list.
- Uncertain / non-narrative activities remain omitted.

---

## 5. Tests added/updated

`tests/unit/language/test_multi_person_activity_coverage.py`:

1. Two people, different activities (cooking + phone path)
2. Shared verb must not drop second activity (holding rope + holding phone)
3. Secondary clause uses distinct activity (unit on `_secondary_person_clause`)
4. Single person / single activity unchanged
5. Uncertain activity not invented
6. Multi-person with only one verified activity — no invention for the other

---

## 6. Full unit test result

```
623 passed, 12 warnings
```

(617 baseline + 6 new activity tests)

---

## 7. Critical image results

`tmp/run_competition_freeze_validation.py` → **critical_fails=0**

| Case | Caption (locked) | Result |
|------|------------------|--------|
| HORSE | `One person is leading a horse. Another person is holding a rope. … A fire is burning nearby.` | PASS — **both** activities preserved |
| SOCCER | playing football + white sports ball; OCR path PASS | PASS |
| MOTORCYCLE | riding preserved | PASS |
| BICYCLE | riding; second person visible **without** invented activity | PASS |

---

## 8. Diverse real-image results

Exercised via critical matrix (animal interaction, sports, motorcycle, bicycle) plus unit fixtures for indoor multi-activity, outdoor multi-activity, sparse/uncertain, and single-activity scenes.  
**Not claimed:** “works on all images.”

---

## 9. Color detection unmodified

**Confirmed.** No edits to `analysis/common/color_utils.py`, clothing color paths, appliance/sports-ball/vegetation/ground logic.

---

## 10. Object detection / counting unmodified

**Confirmed.** No YOLO / IoA / dedupe / `clamp_caption_object_counts` changes in this pass.

---

## 11. OCR unmodified

**Confirmed.**

---

## 12. Environment unmodified

**Confirmed** (no environment calibration changes).

---

## 13. Translation / TTS / export untouched

**Confirmed.** Writers remain json/txt/md/pdf/html. Languages present: en, fa, de, zh, es (`translations/` + `core/resources/translations/`).

---

## 14. README status

Updated for:

- multi-person verified activity preservation
- supported languages (EN / FA / DE / ZH / ES)
- metric definitions + **Unavailable** semantics
- activity coverage meaning under multi-person scenes

Describes current implementation only.

---

## 15. Files deleted

**None.** Conservative audit found no production module that was *provably* unused with zero import/runtime/test risk. Uncertain candidates were kept.

---

## 16. Metric integrity verification

- Evaluator still returns `None` → UI **Unavailable** (`streamlit_app/components/results.py`, `ui/formatters/result_formatters.py`, radar omits N/A rather than faking 0%/100%).
- No decorative percentage injection added.
- Activity coverage denominator remains verified non-weak activities; this pass only improves whether those activities appear in the caption text used for coverage.

---

## 17. Honest remaining limitations

1. Activities must still be **CONFIRMED + narrative_safe** to enter the caption path; SUPPORTED-only stays QA-oriented by design.
2. If the detector/heuristics never produce a second person’s activity, the caption correctly will not invent one.
3. Alternate VLM/Ollama drafts can still contain awkward gender/pronoun phrasing until sanitize/factuality; activity preservation does not rewrite unrelated style issues.
4. Very dense scenes may still prioritize lead + one secondary activity in the opening clause; additional CONFIRMED activities are repaired via coverage sentences when missing.
5. No production dead-code deletions this pass (conservative policy).

---

## Freeze decision

Activity loss root cause fixed and validated. Frozen subsystems unchanged. **STOP — FINAL PRODUCTION FREEZE.**
