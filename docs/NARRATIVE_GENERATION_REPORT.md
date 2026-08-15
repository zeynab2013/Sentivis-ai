# Narrative Generation Report

Measured on the 20-image COCO val2017 real-photo dataset (`validation/real_world/`).  
Results file: `validation/real_world/results.json` (evaluated 2026-07-31 11:02:04 UTC).

## Architecture (unchanged)

| Layer | Responsibility |
|-------|----------------|
| Heuristics | Activity detection, environment inference, scene graph |
| Ollama | Evidence rewrite only (no object/activity invention) |
| `NarrativeGenerator` | Converts verified evidence into competition-facing prose |

Pipeline hook: `NarrativeGenerator.generate()` runs after QA in `PipelineOrchestrator`, before `PipelineResult` is returned.

## New Module

`language/semantic/narrative_generator.py`

**Inputs:** scene graph, attributes, relationships, activities, environment, BLIP observations, Ollama semantic summary, quality report.

**Outputs:**
- `narrative_full` — evidence-backed paragraph (target 120–250 words when evidence supports expansion)
- `narrative_short` — one sentence, max 25 words

Both outputs pass through `CaptionEvidenceValidator` before acceptance.

## Benchmark Averages (20 images)

| Metric | Measured | Target |
|--------|----------:|-------:|
| Caption quality (narrative full) | **89.2%** | > 90% |
| Environment accuracy | **88.5%** | > 92% |
| Activity accuracy | **95.0%** | > 90% |
| Hallucination rate | **3.4%** | < 3% |
| Evidence consistency | **76.1%** | > 95% |
| Narrative fluency | **73.5%** | > 95% |
| Overall semantic score | **84.2%** | — |

### Targets met

- Activity accuracy (95.0%)

### Targets near-miss

- Caption quality: 89.2% (target 90%)
- Hallucination: 3.4% (target 3%)

### Targets not met

- Environment accuracy: 88.5%
- Evidence consistency: 76.1%
- Narrative fluency: 73.5%

## Comparison vs Pre-Narrative Benchmark

From `validation/real_world/llm_responsibility_benchmark.json` (heuristic-only arm, same 20 images):

| Metric | Pre-narrative | Post-narrative | Delta |
|--------|--------------:|---------------:|------:|
| Caption quality | 70.0% | 89.2% | +19.2 pp |
| Hallucination rate | 53.4% | 3.4% | −50.0 pp |
| Evidence consistency | 65.3% | 76.1% | +10.8 pp |
| Activity reasoning | 96.3% | 95.0% | −1.3 pp |
| Overall semantic score | 74.5% | 84.2% | +9.7 pp |

## Sample Output

**Image:** `000000562581.jpg` (sports / tennis)

**Short caption:**  
`A person is playing tennis in tennis court.`

**Full caption (excerpt):**  
`The photograph captures an outdoor tennis court, with one person visible. A person appears to be playing tennis...`

## UI & Export

- Desktop UI: **Narrative Caption** panel added at top of results (FULL CAPTION + Short Caption).
- Exports (TXT, MD, PDF, JSON): narrative sections appear before technical caption sections.

## Reproduce

```powershell
python scripts/run_real_world_evaluation.py
```

Output: `validation/real_world/results.json`
