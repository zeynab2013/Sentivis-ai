# FINAL ROOT CAUSE REPORT

**Date:** 2026-08-12  
**Method:** Freeze → data-flow audit → production fact traces → smallest proven fixes → final stabilization → re-validation  
**Evidence:** `tmp/ROOT_CAUSE_DATA_FLOW_AUDIT.md`, `tmp/PRODUCTION_FACT_TRACE.md`, `tmp/forensic_fact_tracer.py`, `tmp/FINAL_STABILIZATION_RESULT.md`, `tmp/FINAL_REGRESSION_MATRIX.md`

---

## 1. Actual root cause(s)

### RC-A — Activity information loss (not model failure)
Florence/BLIP often **correctly** described activities (e.g. “swinging a baseball bat”, “riding a dirt bike”, “holding a rope”).

Loss happened **after** perception:

1. **Activities never ingested VLM** — heuristics/relations only; VLM text was caption-candidate only.  
2. **Relation→activity synthesis set `narrative_safe=False` for `holding`/`carrying`/`using`** — CONFIRMED holding existed for QA but was **excluded from caption projection**.  
3. **Arbitration rejected rich VLM natural captions** (extra nouns → unsupported sentences) in favor of thinner “safer” text.

### RC-B — Color entity contamination
Crop colors were often correct (e.g. person cyan shirt, sports ball cream).

Corruption in **`build_verified_scene_evidence`**:

1. **SceneReasoner boost overwrote OBSERVED crop colors** with weaker INFERRED values (and could mark reasoner facts OBSERVED by confidence alone).  
2. **Clothing attributes bound to non-person entities** (`bowl_1.pants_color`, `bicycle_1.pants_color`).

### RC-C — People count
QA via `ordered_people` ≈ narrative-safe verified people. Authoritative API: `VerifiedSceneEvidence.people_count`.

### RC-D — Motorcycle heuristic gap
`_cycling_activities` was bicycle-only.

---

## 2. Exact functions responsible

| Loss | Function / area | File |
|------|-----------------|------|
| Holding excluded from captions | relation→activity `narr_ok` set | `verified_evidence_builder.py` |
| Color overwrite | reasoner attribute boost | `verified_evidence_builder.py` |
| Clothing on bowls/bikes | attribute ingest without person gate | `verified_evidence_builder.py` |
| VLM activity never verified | no bridge into ActivityHints | *(fixed via)* `vlm_activity_bridge.py` |
| Rich VLM caption rejected | `score_caption_candidate` + whole-sentence drop | `caption_arbitration.py`, `caption_factuality.py` |
| Motorcycle heuristic miss | `_cycling_activities` | `heuristic_activity_analyzer.py` |
| Riding vs “left of / stationary” | incomplete contradiction patterns | `caption_coverage.py` |

---

## 3. What information was being lost

| Fact | Survived at | Lost at |
|------|-------------|---------|
| Swinging/holding bat (VLM) | Florence raw | Heuristic empty; holding narr=False → caption projection |
| Riding motorcycle (VLM) | Florence + riding relation | Spatial contradiction / thin stubs |
| Holding rope (VLM) | Florence raw | No verified activity without bridge |
| Cyan shirt (crop OBSERVED) | AttributeSet | Overwritten by brown INFERRED |
| Pants on bowl/bike | Mis-indexed crop | Non-person attribute ingest |

---

## 4. Were BLIP/VLM outputs correct before the loss?

**Yes, frequently** (see `PRODUCTION_FACT_TRACE.md`): swinging bat, riding dirt bike, holding rope / fire / brown horse, girl riding bike.

Models were **not** the primary failure. Downstream gates and binding were.

---

## 5. Color pipeline diagnosis

```
pixel_crop OBSERVED
    >
detector attributes
    >
INFERRED reasoner / scene-context   (never OBSERVED; cannot overwrite stronger crop)
```

Entity-bound shirt/pants/shoes preserved separately. Global/dominant reasoner colors blocked when they conflict with OBSERVED clothing.

---

## 6. Activity pipeline diagnosis

```
VLM prose ──(bridge + relation corroboration)──→ ActivityEvidence
Relations (holding/riding/leading) ──→ VerifiedActivity (narrative_safe for literals)
Heuristic activities ──→ same gates
        ↓
language_understanding_from_verified (CONFIRMED ∧ narrative_safe)
        ↓
Arbitration scores salvaged captions (drop unsupported clauses, keep rest)
        ↓
Coverage injects missing CONFIRMED acts / hazards / people cues
```

Rejected without corroboration: near≠riding, holding racket≠tennis, holding cup≠drinking.

---

## 7. People-count diagnosis

| Source | Role |
|--------|------|
| YOLO person nodes | raw |
| `VerifiedEntity` narrative-safe persons | language SoT |
| `VerifiedSceneEvidence.people_count` | authoritative count API |
| `ordered_people(packet)` | QA index (aligned) |

---

## 8. Caption arbitration diagnosis

| Before | After |
|--------|-------|
| One unsupported noun → whole rich caption rejected / loses | Salvage unsupported sentences/clauses; score remaining text |
| Short inventory favored at equal risk | Higher weight on verified coverage + activity hits |

---

## 9. Files changed

See `tmp/FINAL_STABILIZATION_RESULT.md` §3.

---

## 10. Files NOT changed

NaturalCaptionService redesign; YOLO/Florence/Ollama/BLIP selection; translation; TTS/UI; enhancement; Competition Mode; global activity threshold loosening; hallucination invention pattern removal.

---

## 11. Regression results

### Unit tests
**407 passed** (`tests/unit/analysis` + `tests/unit/language`), 0 failed.

### Real-image matrix
**11/12 PASS** — only `10_low_quality` fails (expected blur stub).

Key survivals: horse leading+rope+fire; baseball holding/swinging + cyan clothing; motorcycle riding; bicycle riding; people counts consistent.

---

## 12. Before / after examples

### Holding bat
- **Before:** CONFIRMED holding, `narr=False` → omitted from caption projection.  
- **After:** `narrative_safe=True`; VLM swinging can survive salvage into final caption; QA holding.

### Cyan shirt
- **Before:** OBSERVED cyan → INFERRED brown.  
- **After:** cyan OBSERVED retained; QA cyan.

### Holding rope
- **Before:** VLM only.  
- **After:** bridge + leading corroboration → CONFIRMED in caption/QA path.

### Motorcycle spatial contradiction
- **Before:** “positioned next to / to the left of motorcycle” + riding.  
- **After:** spatial clause removed; riding kept.

---

## 13. Remaining limitations

1. Named football/soccer only with VLM sport name + multi-signal evidence; the “football” regression image is baseball equipment.  
2. Bridge never promotes unsupported semantic upgrades.  
3. Occasional awkward Ollama/natural phrasing remains (no NaturalCaptionService redesign).  
4. Blur inputs stay stub.  
5. Do not treat unit tests alone as competition readiness.

---

## Bottom line

The system was **over-filtering and mis-binding correct information**, not primarily failing to perceive it.

Proven survival after this final pass:

**crop/relation/VLM-corroborated evidence → VerifiedSceneEvidence (no destructive overwrite) → salvaged arbitration / coverage → caption + QA**

Objective met: **highly factual and information-rich**, without globally weakening hallucination protection.
