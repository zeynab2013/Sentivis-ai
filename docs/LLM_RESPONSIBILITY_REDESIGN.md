# LLM Responsibility Redesign

Measured benchmark on the same 20-image COCO val2017 real-photo dataset (`validation/real_world/`). Results file: `validation/real_world/llm_responsibility_benchmark.json`.

## Problem

The prior architecture let Ollama infer activities from sparse object lists. On this dataset, that approach scored **23.7%** activity reasoning vs **67.9%** for heuristics (`validation/real_world/activity_benchmark_comparison.json`). Ollama was kept, but its role was narrowed.

## Responsibility Split

| Layer | Owner | Responsibility |
|-------|-------|----------------|
| Low-level activity detection | `HeuristicActivityAnalyzer` via `ActivityReasoningService` | Detect and score activities from scene graph, spatial layout, and object co-occurrence. Always heuristic — Ollama does not guess activities. |
| High-level semantic synthesis | `SemanticReasoningService` (Ollama) | Explain the scene, improve fluency, combine evidence, reject contradictions, and write a natural caption. Must preserve every verified fact. |
| Caption validation | `CaptionEvidenceValidator` | Strip unsupported sentences before a caption is accepted. |

### Evidence passed to Ollama

Ollama receives structured, verified context only (no raw pixels):

- Verified objects (scene graph nodes)
- Attributes
- Validated spatial relationships
- Pre-verified activities (from heuristics)
- Environment classification
- Scene graph summary
- BLIP visual observations

Ollama is instructed **not** to invent, replace, or add activities. Prompt builder: `analysis/semantic/semantic_evidence_prompt.py`.

### Pipeline order

1. Scene graph construction  
2. Heuristic activity detection  
3. Scene context assembly (includes heuristic activities)  
4. BLIP observations  
5. Ollama semantic synthesis (caption only)  
6. Prompt building → Gemma fallback if no Ollama caption  
7. Refinement → QA  

Configuration: `[analysis.semantic_reasoning]` in `config/analysis.default.toml`. Environment override: `SENTIVIS_SEMANTIC_MODE=off|ollama`.

## Benchmark Setup

| Parameter | Value |
|-----------|-------|
| Dataset | 20 COCO val2017 photographs, 10 scene types |
| Manifest | `validation/real_world/manifest.json` |
| Runner | `scripts/run_llm_responsibility_benchmark.py` |
| Activity mode | `SENTIVIS_ACTIVITY_MODE=heuristic` (both arms) |
| Ollama status | Installed and responding |
| Primary Ollama model | `gemma2:2b` (HTTP 404 — not installed) |
| Fallback model used | `gemma3:1b` (succeeded on **2 / 20** images; remaining images used validated template/Gemma fallback captions) |
| Hardware | CPU (CUDA unavailable) |

### Compared architectures

1. **Heuristic-only** — `SENTIVIS_SEMANTIC_MODE=off`. Activities and captions from heuristics + Gemma/template path; no Ollama semantic layer.
2. **Split architecture** — `SENTIVIS_SEMANTIC_MODE=ollama`. Heuristic activities + Ollama semantic synthesis for caption.
3. **Legacy Ollama-activity** (reference only) — prior design where Ollama inferred activities; from `activity_benchmark_comparison.json`.

## Measured Results (20-image averages)

| Metric | Heuristic-only | Split architecture | Legacy Ollama-activity |
|--------|---------------:|-------------------:|-------------------------:|
| Caption quality | **0.700** | **0.709** | 0.711 |
| Hallucination rate | 0.534 | **0.484** | 0.534 |
| Evidence consistency | **0.653** | 0.648 | — |
| Semantic score | 0.745 | **0.754** | 0.643 |
| Activity reasoning | **0.963** | **0.963** | 0.237 |
| Runtime (total) | **247.0 s** | 431.9 s | 440.5 s |

Evidence consistency was not recorded in the legacy benchmark.

### Deltas: split vs heuristic-only

| Metric | Change |
|--------|--------|
| Caption quality | +0.009 (+1.3%) |
| Hallucination rate | −0.050 (−9.4%) |
| Evidence consistency | −0.005 (−0.8%) |
| Semantic score | +0.009 (+1.2%) |
| Activity reasoning | 0 (unchanged — both use heuristics) |
| Runtime | +184.9 s (+74.9%) |

### Deltas: split vs legacy Ollama-activity

| Metric | Change |
|--------|--------|
| Activity reasoning | +0.726 (+305%) |
| Semantic score | +0.111 (+17.3%) |
| Hallucination rate | −0.050 (−9.4%) |
| Caption quality | −0.002 (−0.3%) |
| Runtime | −8.6 s (−2.0%) |

## Per-Image Ollama Impact

Only 2 of 20 captions differed between heuristic-only and split architecture:

| Image | Heuristic-only caption (truncated) | Split caption | Hallucination (off → split) |
|-------|------------------------------------|---------------|----------------------------|
| `000000562581.jpg` | Template listing objects and relations | "A man stands near a tennis racket." | 0.500 → **0.000** |
| `000000089648.jpg` | Template listing objects and relations | "Chairs are present in the scene." | 0.536 → **0.036** |

On the 18 images where Ollama did not produce an accepted caption, outputs were identical between arms.

## Interpretation (from measured data only)

1. **Activity reasoning is preserved.** Split and heuristic-only both score **0.963** activity reasoning. Legacy Ollama-activity scored **0.237** on the same metric definition.
2. **Hallucination drops when Ollama captions are accepted.** The −5.0 pp average hallucination improvement comes entirely from the two images where `gemma3:1b` produced shorter, evidence-aligned captions that passed validation.
3. **Semantic score improves slightly.** +0.9 pp over heuristic-only, +11.1 pp over legacy Ollama-activity.
4. **Caption quality is essentially flat.** Split (+0.709) vs heuristic-only (+0.700) vs legacy (+0.711) — differences are within one percentage point.
5. **Evidence consistency is essentially flat.** Split (−0.005 pp) vs heuristic-only.
6. **Runtime cost remains high when Ollama is invoked.** Split adds ~185 s (+75%) vs heuristic-only, comparable to legacy Ollama-activity (432 s vs 441 s), because Ollama is still called per image even when it falls back.

## Architecture Files

| File | Role |
|------|------|
| `analysis/activity/activity_reasoning_service.py` | Heuristic-only activity detection |
| `analysis/semantic/semantic_reasoning_service.py` | Ollama caption/scene synthesis |
| `analysis/semantic/semantic_evidence_prompt.py` | Verified-evidence prompt (forbids activity invention) |
| `analysis/semantic/semantic_response_parser.py` | Parses JSON response |
| `services/pipeline/orchestrator.py` | Pipeline wiring |
| `scripts/run_llm_responsibility_benchmark.py` | Benchmark runner |

## Reproduce

```powershell
python scripts/run_llm_responsibility_benchmark.py
```

Output: `validation/real_world/llm_responsibility_benchmark.json`
