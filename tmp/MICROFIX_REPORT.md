# MICRO-FIX REPORT — Caption metadata / QA append / entity color bleed

Date: 2026-08-12

## 1. Exact root causes

### Bug 1 — Pipeline metadata in caption
**Root cause:** `language/prompts/context_caption.py` `build_context_caption()` emitted internal labels:
- `"Observed activity: {activity}."`
- `"The location is {indoor_outdoor}."`
- inventory fragments like `"Person, and bicycle."`

When richer caption candidates failed or were weak, this fallback became the locked user-facing caption.

### Bug 2 — QA appends a second caption
**Root cause:** Assistant LLM prompt (`_format_prompt`) included `OPTIONAL CAPTION SUMMARY` with the full canonical caption. When the model answered (or concatenated draft + caption), replies could become:

```
2 horses are visible in the scene.

A khaki-colored person wearing a red shirt…
```

Direct answers were also not scrubbed against an appended caption block.

### Bug 3 — Non-entity color bleed
**Root cause:** Object color QA accepted crop colors that commonly come from background/grass/ground:
- bicycle `dark green` / `green` (grass)
- sports ball `beige` / `tan` (ground)
Person clothing path in `_color_of_object_answer` also bypassed `_safe_person_clothing_color`, allowing raw khaki aggregates.

---

## 2. Exact files changed

| File | Change |
|------|--------|
| `language/prompts/context_caption.py` | Natural-language fallback; no `Observed activity:` / location inventory |
| `language/refinement/caption_arbitration.py` | Hard-reject metadata / inventory caption candidates |
| `language/assistant/visual_evidence_retriever.py` | Remove caption from QA prompt; strip appended caption; entity-bound color bleed refuses; person colors via safe clothing path |
| `language/assistant/vision_assistant.py` | Strip appended caption on direct + LLM answers; LLM rule not to restate caption |
| `tests/unit/language/test_microfix_caption_qa_color.py` | New regressions |
| `tests/unit/language/test_vision_assistant.py` | Expect caption absent from prompt |
| `tests/unit/language/test_vision_assistant_grounding.py` | Expect caption absent from prompt |
| `tests/unit/language/test_vision_assistant_quality.py` | Expect caption absent from prompt |

## 3. Exact files NOT changed

- Detection / YOLO / Florence / Ollama model bindings
- Activity verification / VLM activity bridge
- Global thresholds / hallucination gates
- Translation, voice, download, enhancement, OCR pipelines
- Verified evidence builder architecture
- NaturalCaptionService redesign

---

## 4. Before → after

| Bug | Before | After |
|-----|--------|-------|
| 1 | `Person, and bicycle. The location is outdoor. Observed activity: riding a bicycle.` | Natural: `A person is riding a bicycle outdoors.` (from context fallback) / richer candidates preferred; metadata rejected in arbitration |
| 2 | Animals answer + full caption paragraph | Animals answer only; caption stripped; caption not in QA prompt |
| 3 | bicycle dark green / ball beige asserted | Refuse unreliable scene-bleed colors; ask for uncertainty |

---

## 5. Unit tests

**422 passed** (`tests/unit/analysis` + `tests/unit/language`)

---

## 6. Real-image regression matrix

Source: `tmp/final_regression_run_log9.txt` / `tmp/FINAL_REGRESSION_MATRIX.md`

**Score: 11/12 PASS** (expected FAIL: `10_low_quality` blur — `caption_nonempty` / `suggestions`)

Bicycle case caption sample from this run:
- Includes riding a bicycle; **no** `Observed activity:`
- Activity QA: riding a bicycle

Horse / football / motorcycle / bicycle activity checks: **PASS** where applicable.

---

## 7. Previously working behaviors

No intentional changes to:
- horse leading / rope / fire
- motorcycle / bicycle riding verification
- football/soccer activity
- people count
- OCR
- observed light-blue / red shirt survival logic from prior pass

Matrix PASS rows for horse / football / motorcycle / bicycle confirm activity checks still PASS where applicable.
