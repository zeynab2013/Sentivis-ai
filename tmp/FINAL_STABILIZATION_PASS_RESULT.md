# FINAL STABILIZATION PASS RESULT

**Date:** 2026-08-12  
**Policy:** Forensic audit first → smallest evidence-flow fixes → real-image proof  
**No architecture redesign / no model replacement / no global threshold loosening**

---

## 1. Exact root causes

### A. Horse clothing → “khaki” (entity/color provenance)
**Observed in audit (current crops):** person_1 has OBSERVED `shirt_color=light blue`, `clothing_color=light blue`; person_2 burgundy. Horse colors are separate (tan/brown).

**Failure mechanism when khaki appeared:**
1. QA attribute selection preferred **highest confidence** over **predicate precedence** / OBSERVED shirt.
2. Ambiguous muted labels (`khaki`/`olive`/`tan`) from mid-chroma crop sampling can absorb grass/horse pixels.
3. Unindexed “the person” always defaulted to `person_1` even when a larger actor held the CONFIRMED activity.

**Not fixed by hardcoding “not khaki”.** Fixed by provenance/precedence + bleed refusal.

### B. Motorcycle QA “What are they doing?” → unknown
**Verified riding was already CONFIRMED** and caption correct in forensic runs.

**Failure mechanisms:**
1. CONFIRMED activities could be dropped by `_pick()` when `scene_hit < 0` filtering was too aggressive in plural paths.
2. Multi-turn follow-up rewrite could rewrite pronouns toward inventory objects after an object question, derailing activity intent.

**Fix:** Always select CONFIRMED activities for QA; never rewrite activity questions in follow-up resolution.

### C. Soccer activity loss (“beside ball” inventory)
**Image:** `tmp/uploads/47871819_db55ac4699.jpg`

**Proven trace:**
- RAW VLM: *“Two girls are playing soccer on a grassy field… playing football…”*
- Interaction fusion: `rejected_vlm=2 total_rels=0` (no geometric person↔ball IoU)
- VLM bridge previously **required** `playing_with`/`holding` → activity discarded
- Natural path marked `action=no` → caption lost the sport

**Root cause:** Correct VLM activity discarded because bridge demanded bbox contact that football scenes often lack.

**Fix:** Multi-signal path — VLM names football/soccer **+** sports ball **+** ≥2 people **+** field/grass language → `vlm_multisignal` → CONFIRMED `playing football` (still rejects ball+proximity alone).

### D. Sports/person color binding
Olive/khaki OBSERVED on person crops with identical shirt/pants/dominant often = grass bleed.  
Ball color remains entity-bound (white sports ball stayed correct).

---

## 2. Files changed

| File | Change | Why | Deliberately NOT changed |
|------|--------|-----|---------------------------|
| `language/assistant/entity_indexing.py` | OBSERVED>predicate-order>confidence for attrs; unindexed person prefers CONFIRMED actor / largest area | Color provenance + entity grounding | People-count architecture |
| `language/assistant/visual_evidence_retriever.py` | Prefer shirt_color; clearer-color / background-bleed guards; CONFIRMED activity always wins for QA | Khaki/olive bleed + motorcycle they-doing | Clothing analyzer redesign |
| `language/assistant/vision_assistant.py` | Never rewrite activity questions in follow-ups | Preserve “What are they doing?” | Translation/voice |
| `analysis/evidence/vlm_activity_bridge.py` | Football multi-signal without IoU when VLM+ball+people+field | Soccer activity survival | Ball-alone → sport |
| `analysis/evidence/verified_evidence_builder.py` | Accept `vlm_multisignal` as strong/action-grade support | Let multi-signal become CONFIRMED narr-safe | Global thresholds |
| `tests/unit/language/test_final_qa_evidence_flow.py` | New | Regression | — |
| `tests/unit/analysis/test_final_stabilization_pass.py` | Football multi-signal + near≠football tests | Regression | — |

---

## 3. Tests

### Before this pass
Prior suite was green at **407** analysis+language tests after previous stabilization.

### New / modified
- `test_observed_shirt_preferred_over_ambiguous_clothing_color`
- `test_they_doing_uses_confirmed_riding_with_one_person`
- `test_activity_followup_not_rewritten_to_object`
- `test_unindexed_person_prefers_confirmed_activity_actor`
- `test_vlm_football_multisignal_without_iou_relation`
- `test_vlm_does_not_invent_football_from_ball_alone`

### Full unit suite (after)
**413 passed**, 0 failed (`tests/unit/analysis` + `tests/unit/language`)

---

## 4. Real-image validation (forensic)

| Case | Caption | Activity | Color QA | Activity QA |
|------|---------|----------|----------|-------------|
| Soccer `47871819…` | Includes **playing football**; 4 people; white ball | CONFIRMED playing football | Ball white; clothing refused when olive bleed | **they doing → playing football** |
| Horse `10815824…` | leading + rope + fire | CONFIRMED leading/holding rope | **light blue shirt** (not khaki) | they doing → leading |
| Motorcycle `143552829…` | riding motorcycle | CONFIRMED riding | red | **they doing → riding motorcycle** |

---

## 5. Regression matrix

`tmp/FINAL_REGRESSION_MATRIX.md` after this pass: **10/12 PASS** on first matrix run.

| Result | Cases |
|--------|-------|
| PASS | kitchen, horse+fire, baseball/sports, bicycle, vehicle, landscape, animal, dense, enhanced, multi-clothing |
| FAIL | `4_motorcycle` — caption collapsed to 6-word activity stub (`caption_nonempty`); **activity QA still correct (riding)** |
| FAIL | `10_low_quality` — expected blur stub |

**Follow-up micro-fix (same pass):** `ensure_salient_verified_coverage` restores OBSERVED clothing when caption is an activity-only stub (<12 words).

Motorcycle smoke check after fix: see `tmp/moto_caption_check.txt` — riding preserved; word count ≥8 when stub path hits.

Unit suite after all fixes: **413 passed**.

Real forensic cases (soccer / horse / motorcycle) demonstrated MODEL→VERIFIED→CAPTION→QA survival for the audited failure modes.

---

## 6. Remaining limitations

1. Grass-bleed olive/khaki can still appear as OBSERVED crop labels; QA now refuses uniform muted bleed rather than inventing caption colors onto the wrong person.
2. Soccer named sport requires VLM to name football/soccer **and** multi-signal corroboration — proximity alone still rejected.
3. Some captions remain stylistically uneven (NaturalCaptionService not redesigned).
4. Unit greens alone are not “competition ready”; acceptance is real-image fact survival.

---

## Verdict

Evidence-flow corrections demonstrated on real images:

**MODEL → VERIFIED → CAPTION → QA** for riding, leading, playing football, and light-blue clothing without khaki overwrite.

No architecture redesign. Hallucination gates kept. No image-specific hacks.
