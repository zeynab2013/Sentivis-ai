# Semantic Enhancement Report

## Activity recognition

`analysis/activity/heuristic_activity_analyzer.py` expanded with evidence-backed activities:

- Ball sports: football, basketball, tennis, frisbee, etc.
- Cooking, kitchen preparation, restaurant dining, eating
- Typing, using laptop, office work
- Shopping, pet interaction, crossing street, children playing
- Cycling, driving, walking, waiting, conversation, classroom learning

All activities require co-occurring objects and/or validated scene-graph relations.

## Scene enrichment

`language/semantic/scene_enrichment.py` infers (when supported):

- Age groups, crowd behaviour, emotional atmosphere
- Scene purpose, environment type (indoor/outdoor/urban/commercial/etc.)

## Ollama responsibility

Ollama **does not** detect objects or invent activities. It only:

- Combines verified evidence into fluent captions
- Detects contradictions
- Produces readable reports

Factual content originates from YOLO, BLIP, scene graph, relationships, activities, and environment heuristics.

## Quality targets

| Metric | Baseline | Target |
|--------|----------|--------|
| Caption quality | 89.2% | >95% |
| Evidence consistency | 76.1% | >95% |
| Activity accuracy | 95.0% | >95% |
| Environment accuracy | ~85% | >95% |
| Hallucination rate | 3.4% | <2% |
| Overall semantic score | 84.2% | >92% |

Enhanced activities and enrichment improve consistency; full target achievement depends on Ollama availability during synthesis.
