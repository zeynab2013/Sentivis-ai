# Final Stabilization Baseline (Phase 0–1)

**Date:** 2026-08-12  
**Git:** not a repository (no `.git`) — cannot revert by commit; must restore by targeted code edits.

## Current observed behavior (from `tmp/final_competition_validation.txt`)

| Area | Current behavior | Severity |
|------|------------------|----------|
| Caption richness | Often short / fragment (kitchen bowl+sink only; bike = pants only) | High |
| Confirmed activity in caption | Bike has CONFIRMED `riding a bicycle` (narr=True) but caption omits riding | Critical |
| Caption ↔ QA | Bike QA: riding; caption: clothing only | Critical |
| Activity conservatism | Sports/baseball: no activity; kitchen: no activity | High |
| People count | Kitchen QA “two people” OK when entities survive; captions often omit count | Medium |
| Colors | Object color lookup missing sports ball; clothing prefers shirt over pants; caption-color fallback can bleed | High |
| Suggested questions | Present; activity Q when riding CONFIRMED | OK-ish |

## Files responsible

| Capability | Primary files |
|------------|---------------|
| Caption generation | `language/semantic/natural_caption_service.py`, `language/refinement/caption_arbitration.py`, `language/refinement/caption_coverage.py`, `language/refinement/caption_sanity.py`, `services/pipeline/orchestrator.py` |
| Color / attributes | `analysis/attributes/attribute_extractor.py`, `analysis/clothing/clothing_analyzer.py`, `language/assistant/visual_evidence_retriever.py`, `language/assistant/evidence_packet.py` |
| People count | `language/assistant/entity_indexing.py` (`ordered_people`), `vision/detection/narrative_gate.py`, caption person densify in natural_caption_service |
| Activity | `analysis/activity/heuristic_activity_analyzer.py`, `analysis/evidence/verified_evidence_builder.py`, `analysis/relationships/relationship_analyzer.py` |
| QA | `language/assistant/visual_evidence_retriever.py`, `language/assistant/vision_assistant.py` |
| Suggested Q | `language/assistant/suggested_questions.py` |
| Arbitration / lock | `language/refinement/caption_arbitration.py`, `services/pipeline/orchestrator.py`, `language/validation/caption_factuality.py` |

## Likely regression causes (recent reliability passes)

1. **Activity over-gating** in `verified_evidence_builder.py`: `playing_with` treated as possession-only → rejects `playing with a ball` / sport heuristics; weak co-occurrence path rejects all performance claims.
2. **Caption factuality gender drop** in `caption_factuality.py`: any sentence with man/woman not in evidence blob is dropped → guts rich captions; survivors are thin clothing/spatial stubs.
3. **No forced coverage for CONFIRMED activities** in `caption_coverage.py` (hazards only) → arbitration can prefer pants-only stub even when riding is verified.
4. **Object color QA gaps** in `_color_of_object_answer`: sports ball / many objects absent from label set; `find_attribute` substring matching; clothing predicates used for persons only but objects can inherit crop contamination without entity hard-bind.
5. **Clothing collapse**: `_safe_person_clothing_color` only checks `clothing_color`/`shirt_color`, ignoring `pants_color` for pants questions; caption fallback can override entity colors for person 1.

## Do NOT revert (keep)

- Bicycle not a container / shopping needs cart / driving ≠ bike
- Exclusive person↔object binding; riding supersedes holding same bike
- Gender/shoulder invention guards (rewrite, don’t invent)
- Racket ≠ tennis court claim; kitchen ≠ cooking without cookware use

## Stabilization strategy

Smallest restores:
1. Confirm literal interaction activities (`playing with X`, `riding X`, …) when relation evidence exists.
2. Ensure CONFIRMED narrative activities appear in final caption.
3. Neutralize unverified gender words instead of deleting whole sentences.
4. Hard entity-bound object/person color answers (pants vs shirt; no cross-entity color).
5. Keep anti-hallucination for office/tennis/cooking proximity.
