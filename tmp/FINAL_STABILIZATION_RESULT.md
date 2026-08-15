# FINAL STABILIZATION RESULT

**Date:** 2026-08-12  
**Scope:** Final implementation pass after forensic root-cause audit  
**Policy:** Restore correct model information without reintroducing hallucinations; no architecture redesign

Evidence: `tmp/ROOT_CAUSE_DATA_FLOW_AUDIT.md`, `tmp/PRODUCTION_FACT_TRACE.md`, `tmp/FINAL_REGRESSION_MATRIX.md`, `tmp/FINAL_ROOT_CAUSE_REPORT.md`

---

## 1. Root causes (from audit — unchanged)

| ID | Cause |
|----|--------|
| RC-A | CONFIRMED holding/carrying/using had `narrative_safe=False` → dropped from caption projection |
| RC-B | Weaker INFERRED reasoner colors overwrote OBSERVED pixel-crop colors; clothing attrs on non-persons |
| RC-C | People count already aligned via verified people / `ordered_people` |
| RC-D | Motorcycle missing from cycling heuristic |

Additional loss points closed in this pass:

- VLM activities never entered the verified activity path without a controlled bridge
- Arbitration rejected entire rich captions for one unsupported noun
- Riding captions could still say “positioned to the left of” / “stationary beside” the ridden vehicle

---

## 2. Exact fixes (this pass)

| Fix | Behavior |
|-----|----------|
| Color precedence | Reasoner colors are never `OBSERVED`; cannot overwrite `pixel_crop` OBSERVED; aggregate clothing/dominant cannot replace OBSERVED shirt/pants |
| Non-person clothing | Still rejected (`clothing_attr_on_non_person`) |
| VLM activity bridge | `vlm_activity_bridge.py`: VLM phrase → ActivityEvidence only with corroborating entity + relation; no near→riding, no racket→tennis upgrade |
| Orchestrator wiring | Passes `vlm_caption` into `build_verified_scene_evidence` |
| Arbitration salvage | Score salvaged text (`filter_unsupported_claims_verified`); boost verified coverage vs thin stubs |
| Clause salvage | Strip unverified catcher/helmet clauses; repair “He is, a blue shirt” → “He is wearing a blue shirt” |
| Riding contradiction | Also clears “to the left of” and “remains stationary beside” when riding is CONFIRMED |
| People count API | `VerifiedSceneEvidence.people_count` = narrative-safe person entities |

---

## 3. Files changed

- `analysis/evidence/verified_evidence_builder.py`
- `analysis/evidence/vlm_activity_bridge.py` *(new)*
- `analysis/activity/heuristic_activity_analyzer.py` *(motorcycle; prior pass)*
- `services/pipeline/orchestrator.py`
- `core/contracts/verified_evidence.py`
- `language/refinement/caption_arbitration.py`
- `language/refinement/caption_coverage.py`
- `language/validation/caption_factuality.py`
- `tests/unit/analysis/test_final_stabilization_pass.py` *(new)*
- `tests/unit/analysis/test_final_stabilization_restore.py`
- `tests/unit/analysis/test_root_cause_forensics.py` *(prior)*
- `tmp/FINAL_STABILIZATION_RESULT.md`, `tmp/FINAL_ROOT_CAUSE_REPORT.md`, traces/matrix

---

## 4. Files intentionally untouched

- NaturalCaptionService architecture / redesign
- YOLO / Florence / Ollama / BLIP model selection
- Translation, TTS/voice, UI, enhancement, Competition Mode shell
- Hallucination invention patterns (roles/emotions/gender gates kept)
- Activity tier enum / VerifiedSceneEvidence overall architecture
- Global activity confidence thresholds (not lowered)

---

## 5–10. Before / after real-image examples

### Activity survival

| Scene | Before loss | After |
|-------|-------------|-------|
| Horse | VLM “holding rope”; leading survived; rope often missing | CONFIRMED `leading a horse` + `holding a rope` → caption + QA |
| Baseball (labeled football case) | VLM swinging correct; holding `narr=False` | CONFIRMED `holding a baseball bat` narr+qa; caption retains swinging/holding |
| Motorcycle | Riding verified but “next to motorcycle” contradiction | Riding; spatial contradiction stripped |
| Bicycle | Riding | Riding; no shopping/driving inventions in matrix |

### Color binding

| Scene | After |
|-------|-------|
| Baseball person_1 | OBSERVED cyan shirt retained; QA: cyan (not ball cream, not reasoner brown) |
| Horse | person_1 light blue / person_2 burgundy entity-bound |
| Motorcycle | red clothing entity-bound |

### People count

| Scene | verified / caption / QA |
|-------|-------------------------|
| Kitchen | 2 / two people / two people |
| Horse | 2 / Two people / two people |
| Baseball | 2 / Two people / two people |
| Dense / multi clothing | 9 / nine / nine |

### Caption examples (matrix)

- Horse: fire + leading + holding rope  
- Baseball: swinging bat + blue shirt/black pants + two people (unsupported catcher/helmet stripped)  
- Motorcycle: red jersey + riding (no “standing next to”)  

### QA examples

- Activity questions return CONFIRMED activities (leading / holding bat / riding)  
- Clothing questions use entity-bound OBSERVED colors  
- People questions match `people_count` / ordered people  

---

## 11. Unit tests

`tests/unit/analysis` + `tests/unit/language`: **407 passed**, 0 failed

Includes root-cause forensics, activity reliability (near≠riding), VLM bridge, color precedence, arbitration salvage.

---

## 12. Real-image regression

**11/12 PASS** (`tmp/FINAL_REGRESSION_MATRIX.md`)

| Result | Cases |
|--------|-------|
| PASS | kitchen, horse+fire, baseball/sports, motorcycle, bicycle, vehicle, landscape, animal, dense indoor, enhanced, multi-clothing |
| FAIL | `10_low_quality` only (blur stub — expected; not weakened) |

---

## 13. Remaining genuine limitations

1. The regression image named “football” is detector/VLM **baseball** (bat + sports ball). Named “playing football/soccer” only when VLM names that sport **and** multi-signal ball+interaction evidence exists — not invented from a generic ball.  
2. VLM activity bridge still requires relation corroboration; unsupported VLM prose does not become CONFIRMED.  
3. Some natural/Ollama phrasing remains awkward (“engaged in…”, meta confidence wording) — not redesigned in this pass.  
4. Blur / unusable inputs still produce stub captions.  
5. Unit-test green ≠ competition-ready by itself; acceptance is MODEL→EVIDENCE→CAPTION→QA survival on real images.

---

## Acceptance verdict

For the audited failure modes, real images show:

**CORRECT MODEL FACT → VERIFIED FACT → CAPTION → QA**

for activity (leading/holding/riding), entity-bound colors (cyan not overwritten), and consistent people counts — **without** removing hallucination gates or lowering global thresholds.

Final stabilization is **successful against the forensic acceptance criteria** for the regression corpus.  
Not claimed “competition ready” as a blank marketing statement; remaining limitations above still apply.
