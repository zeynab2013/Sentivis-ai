# SENTIVIS AI — Final Competition Reliability Audit

Date: 2026-08-12

## Verdict

Competition-ready on evidence gating for activities, relations, and venue labels.
Caption / QA / reports / suggestions are aligned on **VerifiedSceneEvidence** for
activities and relationships. Remaining gaps are mostly pre-arbitration caption
writers that still *read* raw SceneContext, but final lock/filter paths are verified-aware.

## Remaining problems (known, non-blocking / follow-up)

1. **Parallel raw caption writers** — Ollama semantic synthesis, prompt builder, and
   narrative generator still receive raw `SceneContext`. Final arbitration + factuality
   filter against verified evidence, but inventory/fluent VLM prose can still compete.
2. **Geometry-only interactions** — holding/carrying/riding can still form from boxes
   without VLM/pose corroboration (contact required; exclusive binding applied).
3. **Export object counts / color palette** — some export sections still use raw graph
   attributes; activities/relations/summary use verified evidence.
4. **Caption naturalness** — occasional robotic leftovers (“engaged in the activity of…”)
   when Ollama/VLM candidates win; inventory lists are now penalized/rejected in arbitration.

## Root causes addressed this pass

| Problem | Root cause | Fix |
|---------|------------|-----|
| Holding racket → tennis | `_infer_rich_activity` semantic upgrades | Literal relation/activity only |
| Holding cup → drinking | Heuristic maps holding→drinking | Require `drinking` relation |
| Kitchen → cooking | Holding/sitting_on kitchen objects; cluster without link | Require `using` cookware; cluster needs linked relation |
| Laptop → office/work | Single laptop venue; near+laptop SUPPORTED work | ≥2 compute cues; reject performance without action-grade support |
| Near dog → walking | Weak pet heuristics | Only leading/guiding → walking |
| Horse alone → farm | Single livestock venue | Multi-livestock or structure cues |
| Bus → highway | Already calibrated; kept | Road-cue requirement retained |
| Racket → tennis court | “playing with tennis racket” counted as tennis | Venue needs `playing tennis`; caption place label same |
| Long unsupported captions | Length ≤8 exception | Drop all UNSUPPORTED/CONTRADICTED sentences |
| Caption-only QA actions | `_caption_explicit_action` | Require matching CONFIRMED/SUPPORTED packet activity |

## Files changed

- `analysis/activity/heuristic_activity_analyzer.py`
- `analysis/evidence/verified_evidence_builder.py`
- `analysis/context/context_builder.py`
- `analysis/relationships/relationship_analyzer.py`
- `language/semantic/natural_caption_service.py`
- `language/validation/caption_factuality.py`
- `language/assistant/visual_evidence_retriever.py`
- `language/refinement/caption_arbitration.py` (prior pass inventory penalty)
- `tests/unit/analysis/test_final_competition_reliability.py` (new)
- `tests/unit/analysis/test_person_role_reliability.py` (prior)
- `tmp/run_final_competition_validation.py` (validation harness)

## Tests passed

- Full `tests/unit/analysis` + `tests/unit/language`: **390 passed**
- Focused reliability suites: person-role, activity levels, evidence QA, final competition: **all green**

## Real image validation

Harness: `tmp/run_final_competition_validation.py`  
Report: `tmp/final_competition_validation.txt`

| Case | Result highlights |
|------|-------------------|
| Kitchen | No cooking without interaction; indoor kitchen setting |
| Bike + handbag | CONFIRMED riding; QA “riding a bicycle”; no shopping/driving |
| Bus street | Bus only; no highway; no false driving |
| Baseball | Sports scene without forced venue overclaim |
| Ski | Activity gated to verified evidence |
| Multi-people tennis | Playing-with racket ≠ tennis court (calibrate fix verified) |
| Bear | Natural outdoor; no farm |
| Landscape | Outdoor; weak activities absent |
| Low-quality blur | Honest unknown / cannot determine |
| Enhanced horse/fire | CONFIRMED leading horse; QA matches |

Initial validation run: **1 failure** (tennis court from “playing with a tennis racket”) — **fixed and re-verified**.

## Regression confirmation

- No architecture redesign
- Prior person-role / bicycle / shopping / driving gates retained
- Activity CONFIRMED/SUPPORTED/UNKNOWN tiers retained
- Caption↔QA: narrative CONFIRMED activities remain QA-answerable
- Unit suites show **no regressions** (390 passed)
