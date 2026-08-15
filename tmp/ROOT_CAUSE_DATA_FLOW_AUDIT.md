# ROOT CAUSE DATA FLOW AUDIT

**Status:** FREEZE — no production code modified during this audit.  
**Date:** 2026-08-12  
**Entry:** `PipelineOrchestrator.analyze` → `services/pipeline/orchestrator.py`

## Authority ladder (intended)

1. **`VerifiedSceneEvidence`** — canonical for Caption + QA language  
2. **Projected `SceneUnderstanding`** via `language_understanding_from_verified`  
3. **`SceneContext`** — geometry / UI / weaker candidates  
4. **VLM / Ollama / Gemma prose** — caption *candidates* only until grounded

## End-to-end chain

```
image_path
 → ValidatedImage
 → PreprocessedImage (enhanced display + original_display_pixels for color)
 → DetectionResult (YOLO ± SAM2)
 → AttributeSet (color/clothing from ORIGINAL pixels)
 → Relation[] → SceneGraph
 → ActivityHints (heuristics ONLY — no LLM)
 → SceneContext
 → VisualObservations? (BLIP/Florence understand)     [in-memory only]
 → InteractionFusion (geometry ≫ VLM; overwrites relations)
 → SceneUnderstanding (SceneReasoner)
 → VerifiedSceneEvidence                              ★ language SoT
 → projected SceneUnderstanding
 → candidates: Natural(+VLM narrate) | Ollama | Gemma/refined
 → arbitrate + factuality + coverage + lock
 → PipelineResult
 → AssistantEvidencePacket → QA / SuggestedQ / Report
```

---

## Per-stage map

| # | Stage | Function / file | Input | Output | Source of truth | Can drop? | Can overwrite? | Raw model preserved? |
|---|--------|-----------------|-------|--------|-----------------|-----------|----------------|----------------------|
| 1 | Validation | `ImageValidator.validate` | path | `ValidatedImage` | file pixels | fail closed | RGB convert | No |
| 2 | Preprocess | `EnhancedPreprocessor.preprocess` | validated | `PreprocessedImage` | originals for color; enhanced for display | enhance alters chroma | display path replaced | No |
| 3 | Detect | `RefiningObjectDetector` → YOLO | preprocess | `DetectionResult` | YOLO boxes/labels | NMS/conf/class | SAM2 masks | **No** (DTOs only) |
| 4 | Attributes | `AttributeExtractor` + clothing | dets + **original** pixels | `AttributeSet` | crop/mask color | low-conf crops | clothing overrides | No |
| 5 | Relations | `RelationshipAnalyzer.analyze` | dets | `Relation[]` | geometry/IoU | weak pairs | no | No |
| 6 | Graph | `SceneGraphBuilder.build` | dets+rels | `SceneGraph` | detections | no | no | No |
| 7 | Activity | `HeuristicActivityAnalyzer` | graph | `ActivityHints` | heuristics | empty preferred | **no LLM** | No |
| 8 | Context | `ContextBuilder.build` | graph+attrs+acts | `SceneContext` | assembled | unknown env | derived | No |
| 9 | VLM understand | `ManagedVisionModel.understand` | image+ctx | `VisualObservations` | prose candidate | fail→None | failover adapter | **In-run only** `raw_caption` |
| 10 | Pose/OCR | pose + OCR extractors | dets/pixels | poses, ocr | extractors | empty OK | no | OCR snippets later |
| 11 | Fusion | `InteractionEvidenceFuser` | ctx+obs+poses | fused relations | geometry ≫ VLM | speculative VLM | **YES replaces relations** | VLM phrases candidates |
| 12 | Reasoner | `SceneReasoner.reason` | fused ctx | `SceneUnderstanding` | multi-signal facts | discarded list | rebuild facts | VLM agreement only |
| 13 | Verified | `build_verified_scene_evidence` | ctx+understanding | `VerifiedSceneEvidence` | **canonical** | RejectedClaim | reasoner attrs can replace | rejected audit, not raw LLM |
| 14 | Project | `language_understanding_from_verified` | verified | understanding | CONFIRMED narr activities | SUPPORTED QA-only | replaces raw understanding | No |
| 15 | Natural | `NaturalCaptionService.generate` | verified understanding | paragraph | evidence spine + optional VLM narrate | anti-hallucination | blend/score | narrate logged only |
| 16 | Ollama semantic | `SemanticReasoningService.synthesize` | ctx+obs | RawCaption? | prompt from ctx | disabled/fail | cascade | parsed only |
| 17 | Gemma/fallback | `_run_reasoning` / refine | prompt | RefinedCaption | config prefer | fail→fallback | refine rewrite | intermediate |
| 18 | Arbitrate | `arbitrate_captions` + orchestrator heuristics | candidates+verified | preferred text | verified factuality | high risk stubs | **YES picks winner** | losers discarded |
| 19 | Factuality | `filter_unsupported_claims_verified` | preferred | filtered | verified blob | **YES drops sentences** | gender neutralize | No |
| 20 | Coverage | `ensure_salient_verified_coverage` | text+verified | patched | CONFIRMED acts/hazards/people | no | injects facts | No |
| 21 | Lock | final `RefinedCaption` | draft | locked | locked EN caption | soft QA recovery | immutable after | winner only |
| 22 | QA packet | `build_evidence_packet` | verified | packet | qa_safe items | UNCERTAIN/UNKNOWN | caption not KB | No |
| 23 | QA answer | `VisualEvidenceRetriever` + VisionAssistant | packet+Q | answer | packet | refusals | LLM paraphrase | chat only |
| 24 | Report | `result_formatters` | PipelineResult | UI strings | prefer verified | weak skip | localize | No |

---

## Critical loss / overwrite points (hypotheses to prove with traces)

1. **Activity never enters from VLM** — `ActivityReasoningService` is heuristic-only; BLIP/Florence activities must be re-derived from graph or survive only as caption prose.  
2. **Verified projection** — SUPPORTED activities are **QA-only**, not caption facts (`language_understanding_from_verified`).  
3. **Arbitration** — shorter inventory can beat richer VLM if factuality/coverage scores invert.  
4. **Factuality filter** — entire sentences dropped when token overlap with verified blob is low.  
5. **Color** — entity-bound in attributes, but QA/caption can still use wrong attr / caption fallback / crop bleed.  
6. **People count** — YOLO raw ≠ narrative_safe entities ≠ caption phrasing; QA uses `ordered_people(packet)`.  
7. **Raw VLM not on PipelineResult** — correct VLM activity can vanish if never promoted to VerifiedActivity.

---

## What this audit will prove next

For each regression image, `tmp/forensic_fact_tracer.py` logs:

- YOLO people/objects  
- Attribute colors per entity  
- Relations + heuristic activities  
- **BLIP/Florence `raw_caption`**  
- Verified entities / activities / rejected  
- Caption candidates + arbitration scores  
- Final caption + QA answers  

→ `tmp/PRODUCTION_FACT_TRACE.md`
