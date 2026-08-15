# FINAL COMPETITION CAPTION QUALITY PASS — Forensic Report

## A. Root cause of short captions

`NaturalCaptionService._competition_quality_pass` exited early when a caption had ≥22 words and ≥40% cue coverage. Rich scenes with people + setting already present were treated as “done,” so verified fixtures, relations, OCR, and secondary details never entered the paragraph.

Secondary contributors:

- `_evidence_coverage_gaps` ignored missing objects when people were already mentioned
- Healthy spines skipped evidence-support weaving
- Expand caps and coverage cue caps under-counted rich object sets
- Accessory/prop sentences could lead the locked caption (arbitration / densify), which felt thin or wrong-ordered even when facts existed

## B. Files changed

- `language/semantic/natural_caption_service.py`
- `language/refinement/caption_sanity.py`
- `ui/formatters/result_formatters.py`
- `streamlit_app/components/results.py`
- `README.md`
- `tests/unit/language/test_dynamic_caption_evidence_coverage.py` (new)
- `tmp/run_final_caption_quality_probe.py` (validation harness)

## C. Functions changed (primary)

- `_competition_quality_pass` — richness-aware early-exit / expand stop conditions
- `_assemble_coherent_caption` — weave support on under-covered healthy spines
- `_finalize_caption` — extra expand + support when still under-covered
- `_expand_to_target_length` — richer caps, OCR clauses, action-first order
- `_evidence_coverage_gaps` — rich/medium missing-object gaps even with people
- `_coverage_ratio` — more object cues for rich/medium scenes
- `_evidence_support_paragraph` — richer fixture budgets; no “behind them” without people in text
- `_missing_evidence_clauses` / `_synthesize_evidence_dense_narrative` — richer relation/object budgets
- `_sentence_objects` — up to 5 objects on rich scenes
- `_self_review` — continue expanding while still under-covered
- `_order_sentences_by_narrative_priority` / `_densify_choppy_sentences` — action-first + merge clothing/riding & leading/rope
- `order_action_first_sentences` + `sanitize_caption` — enforce action-first on every locked path
- `_fmt_metric` / Streamlit confidence bars — show **Unavailable** when metric is `None`

## D. Dynamic caption detail

Scene richness score from verified people, objects, relations, background, weather, OCR, interaction, action, place, clothing.

| Richness | Soft word band | Behavior |
|----------|----------------|----------|
| simple | ~24–55 | Concise; no forced expansion |
| medium | ~70–130 soft | Expand if thin or coverage low |
| rich | ~90–160 soft | Multi-sentence evidence coverage required |

No fixed quota. Expansion only adds missing verified clauses.

## E. Evidence coverage calculation

Internal (`_coverage_ratio`): fraction of high-value cues present in the paragraph (people, interaction/action tokens, place, weather/time, clothing/color, capped meaningful objects, OCR tokens).

UI object/relationship/activity coverage (`CaptionQualityEvaluator`): fraction of scene-graph labels / semantic relations / non-weak activities mentioned in the caption. Returns `None` → UI **Unavailable** when no verified inputs exist.

## F. Relationships → final caption

Verified relations enter via story relations → `_sentence_spatial_from_story`, `_missing_evidence_clauses` (natural prose, not raw labels), and activity phrases (leading/holding/riding). Densify merges leading + holding-a-rope when both appear.

## G. Entity-bound colors

Unchanged entity-aware color path. Captions only use colors already attached to subjects/clothing; no background substitution added in this pass.

## H. Environment specificity

Unchanged trail/kitchen authority from prior pass. Captions continue to use verified settings (`kitchen`, `outdoor trail`, `recreational area`, etc.).

## I. Displayed percentages / calculations

| Metric | Source | Calculation |
|--------|--------|-------------|
| Detection / relation confidences | YOLO / relation scores | Model/heuristic confidence on that edge |
| Activity confidence | `ActivityHints.confidence` | Aggregated activity report confidence |
| Caption quality (`overall_quality`) | `CaptionQualityEvaluator` | Weighted grammar + fluency + evidence consistency + object term + factuality |
| Evidence consistency | evaluator | Average of available coverage terms + hallucination safety |
| Object coverage | evaluator | Mentioned graph node labels / all node labels (`None` if no nodes) |
| Relationship coverage | evaluator | Mentioned semantic relations / semantic relations (`None` if none) |
| Activity coverage | evaluator | Mentioned non-weak activities / those activities (`None` if none) |
| Hallucination risk | evaluator | `min(1, 0.25 * unsupported_tokens)`; `None` if no graph/activity evidence |
| Hallucination safety (UI bar) | `1 - hallucination_risk` or Unavailable |
| Image quality / blur / sharpness / etc. | enhancement metrics | Measured image stats + before/after quality when enhancement runs |
| Sentence confidence | `SentenceEvidenceAnalyzer` | Per-sentence evidence grounding score |
| Progress % | pipeline stage runner | Stage progress, not caption quality |

## J. Tests passed

- **598** unit tests passed

## K. Critical-image results

Freeze validation: **critical_fails=0/4** (HORSE, SOCCER, MOTORCYCLE, BICYCLE).

Post-sanitize bicycle lead:
`A person is riding a bicycle. Two people are visible in the scene. A handbag…`

Activities, people counts, colors, OCR (soccer), and no-caption-append QA preserved.

## L. Complex-image results

- Kitchen (`coco_kitchen.jpg`): people/sink/bowls — YOLO keeps only those high-confidence entities (no table/fridge on this image), so length stays evidence-matched
- Dense recreational (`random_385406.jpg`): multi-sentence description with people, attire, rackets, activity when Ollama/natural arbitration selects the rich candidate
- Horse: leading + rope + fire + people survive
- Soccer: richer multi-sentence alternate can win with clothing/ball/field detail while activity QA still passes

## M. Remaining limitations

- Caption richness cannot exceed verified detections (sparse YOLO → concise caption is correct)
- Ollama semantic alternate can still compete with NaturalCaptionService; arbitration prefers grounded/natural when competitive
- Some captions remain multi-sentence checklists rather than fully literary prose when densify cannot safely merge
- OCR may appear as `Visible text reads "…"` rather than fully woven prose
- CPU/memory guard can block real-image probes when free RAM < ~512 MB
