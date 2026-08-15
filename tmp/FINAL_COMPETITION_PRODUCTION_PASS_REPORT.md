# FINAL COMPETITION PRODUCTION PASS — FORENSIC REPORT

**Date:** 2026-08-13  
**Status:** Production freeze after validated systemic fixes (no speculative follow-on patches).

---

## A. Problems found

1. **Caption naturalness regression** after count reconcile: locked captions like `A person and person are present in an indoor kitchen… 2 people are visible in the scene.`
2. **Refrigerator color contamination**: entity color reported **beige/brown** from warm cabinet/wood bleed in the appliance bbox.
3. **Gender-strip leftovers**: `man/woman` → `person and person`; pronouns `her/his` survived.
4. **Robotic census filler**: coverage appended `N people are visible in the scene` even when plurality was already implied.
5. **Support-clause grammar**: singular props emitted as `A handbag are visible` / orphan `A light blue are visible`.
6. **Arbitration preference**: Ollama “present in an indoor…” stubs could win over stronger NaturalCaption paragraphs.
7. Mid-pipeline still briefly drafted inflated fridge counts (final clamp already corrected quantities).

---

## B. Root causes

1. `_neutralize_unverified_gender` replaced gendered nouns with `person` without collapsing dual mentions into natural plurals; pronouns were not neutralized.
2. Appliance labels used default bbox sampling; `_high_luminance_neutral_name` already recovered **white** for the kitchen fridge but was only applied to sports balls — never appliances.
3. `ensure_salient_verified_coverage` treated only explicit `people` tokens as plurality, so `person and person` still triggered a trailing census sentence.
4. Evidence support templates hardcoded plural `are visible` and could emit color-only fragments.
5. Caption arbitration robotic patterns did not reject `person and person` / `are present in an indoor`.

---

## C. Files changed

| File | Change |
|------|--------|
| `analysis/common/color_utils.py` | Appliance-aware inset + wood/cabinet bleed rejection + white recovery / unknown over beige-brown |
| `language/validation/caption_factuality.py` | Natural gender neutralize (`two people`, pronoun cleanup) |
| `language/refinement/caption_coverage.py` | Skip/soften people-census filler when plurality already present |
| `language/refinement/caption_arbitration.py` | Reject `person and person` / indoor “present in” stubs |
| `language/refinement/caption_sanity.py` | Humanize `person and person`, agreement fixes, orphan color drop, census trim |
| `language/semantic/natural_caption_service.py` | Subject–verb agreement; reject color-only phrases |
| `README.md` | Appliance color + naturalness documentation |
| `tmp/run_competition_freeze_validation.py` | GC between cases + analyze error handling |
| `tmp/run_kitchen_naturalness_validation.py` | Kitchen naturalness/count/color validator |
| Tests | Color appliance cases + naturalness gender/plural tests |

---

## D. Files deleted

**None.** Unused-code audit was performed conservatively; nothing was provably unused with zero runtime/import risk, so no production deletions were made.

---

## E. Tests added

- `tests/unit/analysis/test_color_utils.py`
  - `test_refrigerator_rejects_cabinet_wood_beige`
  - `test_appliance_unknown_when_only_wood_evidence`
- `tests/unit/language/test_caption_naturalness_gender_plural.py`
  - gender → `two people`
  - humanize collapses `person and person`
  - coverage skips redundant census

---

## F. Full test result

```
617 passed, 12 warnings
```

(Focused suites: color + naturalness + object-count also green.)

---

## G. Real-image validation results

### Kitchen (`tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png`)

- **CAPTION:** `Two people are in a kitchen around a dining table. A white refrigerator, a brown couch, a sink, a tv, and beige cups are visible behind them. A beige cup and a brown vase sit on the table, while 4 brown chairs surround it.`
- **COUNTS:** refrigerator=1, chair=4, person=2  
- **FRIDGE_COLOR:** white  
- **RESULT:** PASS (`tmp/FINAL_KITCHEN_NATURALNESS_VALIDATION.md`)

### Critical freeze (horse / soccer / motorcycle / bicycle)

```
critical_fails=0
```

See `tmp/FINAL_COMPETITION_FREEZE_VALIDATION.md`.

Streamlit import: `streamlit_app.main` / `bootstrap` / `runtime` → **OK**.

---

## H. Color validation

| Check | Result |
|-------|--------|
| Kitchen fridge not beige/brown | **PASS** (white) |
| Synthetic fridge vs wood surround | **PASS** (unit) |
| Bicycle grass-green rejection | retained (prior unit) |
| Sports-ball white over ground | retained (prior unit) |
| Unknown preferred over wood-only appliance | **PASS** (unit) |

---

## I. Object-count validation

Kitchen verified counts and caption clamp: fridge **1**, chairs **4**, people **2**; no inflated refrigerator plural. Prior general count suite (A–M) remains in place; unit suite green.

---

## J. Relationship validation

Critical freeze relationship/activity checks: **PASS** (leading/holding, football/playing, riding motorcycle/bicycle).

---

## K. Environment validation

Kitchen caption + setting remain kitchen-grounded. Horse outdoor + fire retained.

---

## L. QA validation

Critical freeze QA: people counts, activities, colors, OCR `21` (soccer), no caption-append — **PASS**.

---

## M. Metric integrity validation

No metric calculation changes in this pass. Coverage remains instance-aware; Unavailable path unchanged. README still documents calculation-backed percentages.

---

## N. Export validation

`ExportManager` registered writers: **json, txt, md/markdown, pdf, html** (plus image). Required competition formats are present in `services/export/export_manager.py`. No export code changes in this pass.

---

## O. README status

Updated and truthful for:
- appliance entity-color behavior
- naturalness / gender-strip leftovers
- existing pipeline, counts, metrics, exports, hardware notes

---

## P. Remaining limitations

1. Ollama/VLM drafts can still be awkward before sanitize/arbitration; lock path repairs most cases but not every stylistic flaw (e.g. occasional redundant activity restatement).
2. Mid-pipeline drafts may briefly contain wrong quantities before final verified clamp.
3. Clothing/gender pronouns in alternate candidates may still appear until factuality filter runs.
4. SAM2 masks unavailable on this machine → bbox color sampling only (mitigated by appliance inset + bleed rejection).
5. Export writers verified present; this pass did not re-write all five artifacts to disk from a live kitchen run.
6. No production files deleted (conservative unused-code policy).

---

## Freeze decision

Systemic root causes for the reported kitchen caption/color regression were fixed and validated against kitchen + critical images + full unit suite. **Production code is FROZEN** — no further speculative patches after this validation.
