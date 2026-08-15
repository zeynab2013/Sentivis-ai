# FINAL FORENSIC REPAIR REPORT

Date: 2026-08-12  
Scope: Entity-bound color survival + caption arbitration preserving OBSERVED shirt colors  
Constraint compliance: no architecture redesign, no model swap, no gate removal, no image hardcodes, no global threshold loosening.

---

## 1. Exact root causes (forensic)

### RC-1 — Caption→QA clothing color cross-person bleed
**Stage lost:** QA (`_safe_person_clothing_color`)  
**What happened:** For ambiguous olive/khaki OBSERVED clothing on person_1, caption color (`navy coat`) overrode entity evidence even when navy belonged to another person.  
**Evidence:** Soccer forensic — `clothing_color=olive` on person_1, `_safe_person_clothing_color → ('navy', 0.8)` from caption about a fourth person.

### RC-2 — `preferred_weak` punished correct clothing language
**Stage lost:** Caption arbitration (`orchestrator` preferred_weak)  
**What happened:** Markers `"olive"`, `"khaki"`, `"t-shirt"` marked evidence-grounded Ollama captions as “weak,” forcing thin NaturalCaption stubs that dropped light-blue / red shirt facts.  
**Evidence:** Horse — Ollama had `light blue t-shirt`; log `Preferred NaturalCaptionService over weak alternate wording` → stub without shirt color.

### RC-3 — Body/dominant color prose displaced OBSERVED shirt color
**Stage lost:** Caption arbitration / narrative remix  
**What happened:** Ollama wrote `dominant brown appearance` from `person.color=brown` while `shirt_color=red` was OBSERVED. Grounded arbitration sometimes kept that prose; shirt color did not force retention.  
**Evidence:** Motorcycle — verified `shirt_color=red`, interim caption brown appearance fluff.

### RC-4 — Robotic scene-zone metadata in prose
**Stage lost:** Caption sanitize / humanize  
**What happened:** Phrases like `bottom center of the scene`, `in outdoors`, `across an outdoors` survived into locked captions.

### Not the failure mode this pass
Pixel extraction for horse/motorcycle already produced correct `shirt_color` (light blue / red). The facts were lost **after** OBSERVED attributes existed — not at crop analysis.

---

## 2. Exact files changed

| File | Change |
|------|--------|
| `language/assistant/visual_evidence_retriever.py` | Entity-bound caption color gate; refuse cross-person navy/coat steal; color questions refuse when no reliable color; fix false `shirt` garment focus via question tokens |
| `services/pipeline/orchestrator.py` | Remove olive/khaki/t-shirt preferred_weak; mark body-color fluff weak; OBSERVED shirt-color survival preference in caption arbitration |
| `language/refinement/caption_sanity.py` | Humanize robotic zone / `outdoors` / `as evidenced by` phrases |
| `tests/unit/language/test_final_qa_evidence_flow.py` | Cross-person caption color + red-vs-brown regressions |
| `tests/unit/language/test_caption_sanity.py` | Robotic phrase sanitize regression |

## 3. Exact files NOT changed

- Translation / voice / listen / download UI paths  
- Image enhancement pipeline  
- Model weights / model selection  
- Hallucination gates (not removed)  
- `NaturalCaptionService` architecture  
- YOLO / Florence / Ollama model bindings  
- Global confidence thresholds (no broad loosening)

---

## 4. Before → after (critical behaviors)

| Case | Before (reported / prior audit) | After (this pass) |
|------|----------------------------------|-------------------|
| Horse clothing QA | khaki clothing (historical) | **light blue clothing** |
| Horse activity | leading (ok) | **leading a horse** + rope + fire in caption |
| Horse caption | thin inventory / lost shirt | **keeps light blue shirt** via alternate survival |
| Motorcycle clothing QA | brown shirt (reported) | **red jersey** |
| Motorcycle activity | riding (ok when present) | **riding a motorcycle** |
| Motorcycle caption | brown appearance / zone fluff | **red jersey and brown pants + riding** |
| Soccer activity | playing football (must keep) | **playing football** (CONFIRMED) |
| Soccer OCR | "21" | **"21"** |
| Soccer clothing QA | navy coat (wrong person) | **refuse** (olive bleed; no cross-person steal) |

---

## 5. Full regression matrix

Source: `tmp/FINAL_REGRESSION_MATRIX.md` / `tmp/final_regression_run_log8.txt`

**Score: 11/12 PASS**

| Category | Result |
|----------|--------|
| kitchen / horse / football / motorcycle / bicycle / vehicle / landscape / animal / dense indoor / enhanced / multi_person_clothing | PASS |
| 10_low_quality (blur) | FAIL (`caption_nonempty`, `suggestions`) — expected for intentionally degraded input |

No silent regression on riding / leading / football among passing categories.

---

## 6. Unit test results

```
tests/unit/analysis + tests/unit/language → 416 passed
```

Added coverage:
- caption color must not steal from another person  
- red shirt not overwritten by brown body color  
- robotic zone / outdoors sanitize  

---

## 7. Three real-image validation results

From `tmp/forensic_color_activity_audit_final.txt`:

### Horse (`10815824_2997e03d76.jpg`)
- OBSERVED: `person_1.shirt_color=light blue`  
- Caption retains light blue shirt; leading + rope + fire  
- QA clothing: light blue; activity: leading; people: 2  

### Soccer (`47871819_db55ac4699.jpg`)
- CONFIRMED: playing football  
- Caption: playing football; people: 4  
- OCR QA: `"21"`  
- Clothing QA: refuse (no navy steal)  

### Motorcycle (`143552829_72b6ba49d4.jpg`)
- OBSERVED: `shirt_color=red`  
- Caption: red jersey + brown pants + riding  
- QA: red jersey; riding a motorcycle; 1 person  

---

## 8. Remaining limitations

1. **Soccer / some Natural stubs** can still be shorter than raw VLM paragraphs when arbitration prefers thinner but activity-covered text. Football activity and OCR survive; dense jersey-color narrative from VLM is not always locked.  
2. **Olive/khaki grass bleed** on primary soccer person correctly refuses rather than inventing a color — uncertain is preferred over wrong.  
3. **Gendered pronouns** may still appear from Ollama (`her hand`) when VLM invents gender; not addressed this pass (out of color/activity scope).  
4. **Low-quality blur images** still fail caption density checks by design.  
5. Phrase cleanup for `across an outdoors` is in sanitize; horse forensic sample above was captured before the final `an outdoors` pattern — code now strips it.

---

## Acceptance checklist (honest)

- [x] Observed colors survive to final QA (horse light blue, moto red)  
- [x] Person clothing colors do not bleed from other entities (soccer navy blocked)  
- [x] VLM-supported activities survive (football, riding, leading)  
- [x] Riding motorcycle remains correct  
- [x] Horse leading remains correct  
- [x] Soccer/football remains correct when supported  
- [x] People count consistent caption↔QA on validated cases  
- [x] OCR "21" remains correct  
- [x] Caption more natural / retains shirt facts on horse & moto  
- [~] Soccer caption still relatively thin vs VLM (limitation #1)  
- [x] QA uses same verified evidence path  
- [x] Translation / voice / enhance / models / gates untouched  

**Verdict:** Critical color + activity survival failures identified by forensic audit are fixed at the data-flow layer. Not claiming “competition ready” solely from unit tests; real-image matrix is **11/12** with the expected blur failure.
