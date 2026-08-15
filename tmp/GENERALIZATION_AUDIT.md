# GENERALIZATION AUDIT

**VERDICT: PASS WITH ISSUES**

Unique images analyzed: 11 | Category rows: 18
FAIL rows: 0 | ISSUE rows: 2 | MISSING: 0

| ID | Category | People | Activities (actors) | Caption (short) | Status | Failures |
|---|---|---|---|---|---|---|
| 1_single_person_activity | motorcycle rider | 1 | riding a motorcycle[1] | A person is riding a dirt bike. A person, gloves and riding a bike on the water. Large ... | ISSUE | K:MEDIUM |
| 2_multi_shared_activity | soccer shared play | 4 | playing football[2] | Two people are playing football. A white sports ball rests in the scene. | PASS | — |
| 3_multi_different_activities | farm leading+holding | 1 | leading a horse[1]; holding a rope[1] | They are leading one of the horses, holding a rope in their hand as they moves it acros... | PASS | — |
| 4_sports_soccer_alt | football sports alt | 2 | — | A young person is swinging a baseball bat at a white ball. They are, a blue shirt and b... | ISSUE | K:MEDIUM |
| 5_person_animal | horse interaction | 1 | leading a horse[1]; holding a rope[1] | They are leading one of the horses, holding a rope in their hand as they moves it acros... | PASS | — |
| 6_multi_animal | two horses+fire | 1 | leading a horse[1]; holding a rope[1] | They are leading one of the horses, holding a rope in their hand as they moves it acros... | PASS | — |
| 7_person_vehicle_bike | bicycle multi people | 2 | carrying a handbag[1] | One person is carrying a handbag. A bicycle is visible behind them. Another person is a... | PASS | — |
| 7b_person_vehicle_moto_alt | motorcycle alt | 1 | riding a motorcycle[1] | A person is riding a motorcycle on a road. The person is wearing brown clothing. | PASS | — |
| 8_indoor_kitchen | kitchen objects | 2 | — | Two people are in a kitchen. The person is near a brown chair and a potted plant, posit... | PASS | — |
| 9_outdoor_trail | outdoor landscape/trail | 1 | riding a bicycle[1] | The person’s posture indicates they are riding the bicycle, while grass surrounds them ... | PASS | — |
| 10_group_sports | soccer group | 4 | playing football[2] | Two people are playing football. A white sports ball rests in the scene. | PASS | — |
| 11_many_objects | kitchen dense objects | 2 | — | Two people are in a kitchen. The person is near a brown chair and a potted plant, posit... | PASS | — |
| 12_weak_activity | animal/no-strong-human-act | 0 | — | A bear is at the heart of the scene. | PASS | — |
| 13_overlap_people | soccer overlap | 4 | playing football[2] | Two people are playing football. A white sports ball rests in the scene. | PASS | — |
| 14_subset_participants | soccer 2 of N playing | 4 | playing football[2] | Two people are playing football. A white sports ball rests in the scene. | PASS | — |
| 15_simultaneous_activities | farm dual acts | 1 | leading a horse[1]; holding a rope[1] | They are leading one of the horses, holding a rope in their hand as they moves it acros... | PASS | — |
| 16_dense_indoor | dense indoor/group | 12 | playing with a tennis racket[1] | People are playing with a tennis racket outdoors. The person is wearing black clothing. | PASS | — |
| 17_outdoor_misc | misc outdoor | 2 | — | Two people and a skateboard are visible outdoors. | PASS | — |

## Detailed failures

### 1_single_person_activity (motorcycle rider)
- Path: `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\tmp\uploads\143552829_72b6ba49d4.jpg`
- Caption: A person is riding a dirt bike. A person, gloves and riding a bike on the water. Large rocks and grass on the ground next to the water. A brown motorcycle is visible behind them. A person is riding. A person is riding.
- People: 1
- Activities: `[{"activity": "riding a motorcycle", "entity_ids": ["person_1", "motorcycle_1"], "person_actors": ["person_1"], "n_person_actors": 1, "confidence": 0.855}]`
- Relations: `[{"type": "riding", "subject": "person_1", "object": "motorcycle_1"}]`
- Environment: `{"indoor_outdoor": "outdoor", "setting": "outdoor area", "scene_type": "outdoor scene"}`
- **MEDIUM [K]** Hallucination risk elevated (0.25) (likely stage: claim filtering / generation)

### 4_sports_soccer_alt (football sports alt)
- Path: `D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI\tmp\uploads\141139674_246c0f90a1.jpg`
- Caption: A young person is swinging a baseball bat at a white ball. They are, a blue shirt and black pants.
- People: 2
- Activities: `[]`
- Relations: `[]`
- Environment: `{"indoor_outdoor": "outdoor", "setting": "recreational area", "scene_type": "outdoor scene"}`
- **MEDIUM [K]** Hallucination risk elevated (0.25) (likely stage: claim filtering / generation)


## Notes

- Audit is evidence-vs-caption based; it does not use external human GT labels.
- Naturalness notes are advisory and do not alone cause FAIL.
- Duplicate category rows reuse the same pipeline result for the same image path.
