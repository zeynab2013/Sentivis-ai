# Sentivis AI — Real-World Evaluation

**Evaluated:** 2026-07-31 15:04:05 UTC
**Dataset:** COCO val2017 (real photographs)
**Images:** 20 real photographs

## Methodology

- Photos sourced from [COCO val2017](http://cocodataset.org/) — real-world photography, not synthetic.
- Ground truth from COCO instance annotations (labels, bounding boxes, areas).
- Full Sentivis pipeline executed per image (YOLO → attributes → relationships → scene graph → activities → context → BLIP → prompt → Gemma → refinement → QA).
- Metrics compare pipeline output to COCO ground truth and scene-type expectations.

## Average Scores

| Metric | Average |
|--------|---------|
| Object detection accuracy | 69.8% |
| Attribute accuracy | 93.3% |
| Relationship correctness | 55.1% |
| Activity reasoning | 93.3% |
| Environment reasoning | 88.5% |
| Caption quality | 87.1% |
| Hallucination rate | 3.2% |
| **Overall semantic score** | **83.4%** |

## Per-Image Comparison Table

| Scene | Image | Obj Det | Attr | Rel | Activity | Env | Caption | Halluc | Overall |
|-------|-------|---------|------|-----|----------|-----|---------|--------|---------|
| sports | `000000562581.jpg` | 0.58 | 1.00 | 0.25 | 1.00 | 1.00 | 0.96 | 0.00 | **0.83** |
| sports | `000000559547.jpg` | 0.80 | 1.00 | 0.00 | 1.00 | 1.00 | 0.84 | 0.10 | **0.79** |
| indoor | `000000319935.jpg` | 0.71 | 0.93 | 1.00 | 1.00 | 1.00 | 0.77 | 0.04 | **0.91** |
| indoor | `000000014038.jpg` | 0.15 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.00 | **0.86** |
| outdoor | `000000548780.jpg` | 0.91 | 0.97 | 0.50 | 1.00 | 1.00 | 0.80 | 0.00 | **0.88** |
| outdoor | `000000120420.jpg` | 0.57 | 1.00 | 0.50 | 1.00 | 0.00 | 0.90 | 0.00 | **0.71** |
| streets | `000000542625.jpg` | 0.83 | 0.91 | 0.14 | 1.00 | 1.00 | 0.91 | 0.00 | **0.82** |
| streets | `000000480944.jpg` | 0.67 | 1.00 | 0.35 | 1.00 | 1.00 | 0.96 | 0.04 | **0.85** |
| vehicles | `000000254814.jpg` | 0.65 | 0.90 | 0.32 | 1.00 | 1.00 | 0.81 | 0.00 | **0.81** |
| vehicles | `000000568290.jpg` | 0.56 | 1.00 | 0.22 | 1.00 | 1.00 | 0.83 | 0.19 | **0.78** |
| people | `000000581357.jpg` | 0.92 | 0.93 | 0.50 | 0.50 | 1.00 | 0.94 | 0.00 | **0.83** |
| people | `000000580197.jpg` | 0.80 | 0.78 | 0.00 | 1.00 | 0.70 | 0.85 | 0.00 | **0.73** |
| animals | `000000193162.jpg` | 0.90 | 0.83 | 1.00 | 1.00 | 1.00 | 0.94 | 0.00 | **0.95** |
| animals | `000000554291.jpg` | 0.46 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.17 | **0.88** |
| kitchens | `000000540502.jpg` | 0.75 | 0.87 | 1.00 | 1.00 | 1.00 | 0.81 | 0.00 | **0.92** |
| kitchens | `000000445658.jpg` | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.00 | **0.95** |
| offices | `000000507575.jpg` | 0.65 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | 0.06 | **0.92** |
| offices | `000000172595.jpg` | 0.83 | 0.96 | 1.00 | 1.00 | 1.00 | 0.85 | 0.00 | **0.95** |
| classrooms | `000000466125.jpg` | 0.67 | 0.78 | 0.00 | 0.50 | 0.00 | 0.95 | 0.00 | **0.56** |
| classrooms | `000000089648.jpg` | 0.81 | 0.82 | 0.25 | 0.67 | 1.00 | 0.80 | 0.04 | **0.76** |

## Confusion Summary

- Total failure notes: 115
- Images with missing important objects: 15/20
- Images with incorrect relations: 0/20

### Root Cause Frequency

- Detection precision gap (false positive): **17** images
- Detection recall gap (YOLO threshold or object scale): **15** images
- Attribute zone boundary edge case: **11** images
- Relationship proximity/threshold gap: **9** images
- Activity rule coverage gap: **3** images
- Environment label inference gap: **2** images

## Scores by Scene Type

| Scene Type | Images | Avg Overall | Avg Obj Det | Avg Env |
|------------|--------|-------------|-------------|---------|
| animals | 2 | 0.92 | 0.68 | 1.00 |
| classrooms | 2 | 0.66 | 0.74 | 0.50 |
| indoor | 2 | 0.89 | 0.43 | 1.00 |
| kitchens | 2 | 0.93 | 0.75 | 1.00 |
| offices | 2 | 0.93 | 0.74 | 1.00 |
| outdoor | 2 | 0.80 | 0.74 | 0.50 |
| people | 2 | 0.78 | 0.86 | 0.85 |
| sports | 2 | 0.81 | 0.69 | 1.00 |
| streets | 2 | 0.84 | 0.75 | 1.00 |
| vehicles | 2 | 0.79 | 0.60 | 1.00 |

## Most Common Reasoning Mistakes

1. **Detection precision gap (false positive)** — 17 occurrences
2. **Detection recall gap (YOLO threshold or object scale)** — 15 occurrences
3. **Attribute zone boundary edge case** — 11 occurrences
4. **Relationship proximity/threshold gap** — 9 occurrences
5. **Activity rule coverage gap** — 3 occurrences

## Per-Image Failures and Root Causes

### `000000466125.jpg` (classrooms) — score 0.56
- **Caption:** Two people are present.
- **Detected:** person, umbrella
- **Ground truth:** backpack, book, chair, dining table, laptop, person, umbrella
- **Missing:** backpack, book, chair, dining table, laptop, person
- Missed GT objects: backpack, book, chair, dining table, laptop, person
- Zone mismatch for person: expected middle-left, got bottom-left
- Zone mismatch for person: expected top-right, got middle-right
- Missing expected relation: person holding book
- Missing expected relation: person holding laptop
- Expected activities ['dining', 'people present', 'reading', 'working'], got ['having a conversation', 'people present']
- Environment outdoor contradicts GT/scene indoor
- *Root cause:* Activity rule coverage gap
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Environment label inference gap
- *Root cause:* Relationship proximity/threshold gap

### `000000120420.jpg` (outdoor) — score 0.71
- **Caption:** This appears to be an indoor setting (group gathering, crowded crowd level). Objects include person (middle-center), umbrella (top-center), person (top-right), person (top-right), ...
- **Detected:** bench, bird, cell phone, person, umbrella
- **Ground truth:** bench, bird, cell phone, person, umbrella
- **Missing:** bird, person
- Missed GT objects: bird, person
- Extra detections: bench, umbrella
- Environment indoor contradicts GT/scene outdoor
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Environment label inference gap

### `000000580197.jpg` (people) — score 0.73
- **Caption:** The scene suggests photographed scene with group gathering and high interaction complexity. Objects include person (middle-left), person (middle-right), person (bottom-center), tie...
- **Detected:** person, tie
- **Ground truth:** person, tie
- Extra detections: tie, tie
- Zone mismatch for person: expected bottom-center, got middle-left
- Zone mismatch for person: expected middle-left, got bottom-center
- Missing expected relation: person standing_beside person
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Relationship proximity/threshold gap

### `000000089648.jpg` (classrooms) — score 0.76
- **Caption:** This appears to be an indoor setting (group gathering, crowded crowd level). Objects include person (middle-left), person (bottom-right), laptop (middle-right), person (middle-left...
- **Detected:** chair, handbag, laptop, person, tie
- **Ground truth:** backpack, book, cell phone, chair, laptop, person
- **Missing:** book, person
- Missed GT objects: book, person
- Extra detections: chair, chair, chair, chair, handbag, laptop, laptop, tie
- Zone mismatch for person: expected top-left, got bottom-right
- Zone mismatch for person: expected middle-right, got bottom-left
- Zone mismatch for chair: expected bottom-center, got top-left
- Zone mismatch for chair: expected middle-center, got bottom-center
- Zone mismatch for chair: expected middle-right, got top-left
- Zone mismatch for chair: expected top-left, got bottom-left
- Zone mismatch for chair: expected middle-right, got top-center
- Zone mismatch for chair: expected top-center, got middle-center
- Zone mismatch for person: expected bottom-left, got top-center
- Zone mismatch for person: expected bottom-right, got middle-center
- Zone mismatch for chair: expected bottom-left, got top-center
- Missing expected relation: person holding cell phone
- Missing expected relation: person holding book
- Expected activities ['people present', 'reading', 'working'], got ['classroom learning', 'gym training', 'meeting', 'office work', 'people present', 'shopping', 'teaching']
- *Root cause:* Activity rule coverage gap
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Relationship proximity/threshold gap

### `000000568290.jpg` (vehicles) — score 0.78
- **Caption:** This appears to be an outdoor setting (small social interaction, pair crowd level). Objects include bus (middle-center), truck (middle-right), car (middle-right), person (middle-ri...
- **Detected:** bicycle, bus, car, person, truck
- **Ground truth:** bus, car, motorcycle, person, truck
- **Missing:** motorcycle, person
- Missed GT objects: motorcycle, person
- Extra detections: bicycle, bicycle, bicycle
- Missing expected relation: bus near_vehicle motorcycle
- Missing expected relation: motorcycle near_vehicle motorcycle
- Missing expected relation: motorcycle near_vehicle person
- Missing expected relation: person standing_beside person
- Missing expected relation: motorcycle near_vehicle bus
- Missing expected relation: car near_vehicle motorcycle
- Missing expected relation: motorcycle near_vehicle car
- Missing expected relation: person near_vehicle motorcycle
- Missing expected relation: motorcycle near_vehicle truck
- Missing expected relation: truck near_vehicle motorcycle
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Relationship proximity/threshold gap

### `000000559547.jpg` (sports) — score 0.79
- **Caption:** This appears to be an outdoor setting (group gathering, crowded crowd level). Objects include person (middle-right), person (middle-center), person (middle-left), baseball bat (bot...
- **Detected:** baseball bat, person, tennis racket
- **Ground truth:** baseball bat, baseball glove, person, sports ball
- **Missing:** sports ball
- Missed GT objects: sports ball
- Extra detections: tennis racket
- Missing expected relation: person playing_with baseball bat
- Missing expected relation: person playing_with sports ball
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Relationship proximity/threshold gap

### `000000254814.jpg` (vehicles) — score 0.81
- **Caption:** This appears to be an outdoor setting (group gathering, crowded crowd level). Objects include truck (top-left), car (bottom-right), car (bottom-center), car (middle-left), traffic ...
- **Detected:** bicycle, car, motorcycle, person, traffic light, truck
- **Ground truth:** bicycle, bus, car, motorcycle, person, traffic light, truck
- **Missing:** bus, motorcycle, person
- Missed GT objects: bus, motorcycle, person
- Extra detections: bicycle, car, car, traffic light, truck
- Zone mismatch for bicycle: expected top-right, got bottom-right
- Zone mismatch for car: expected middle-left, got bottom-right
- Zone mismatch for person: expected bottom-right, got middle-right
- Zone mismatch for person: expected middle-right, got bottom-right
- Zone mismatch for bicycle: expected middle-left, got top-left
- Zone mismatch for truck: expected top-left, got bottom-right
- Missing expected relation: car near_vehicle traffic light
- Missing expected relation: truck near_vehicle truck
- Missing expected relation: traffic light near_vehicle bus
- Missing expected relation: bus near_vehicle bicycle
- Missing expected relation: motorcycle near_vehicle bus
- Missing expected relation: bus near_vehicle car
- Missing expected relation: bicycle near_vehicle bus
- Missing expected relation: car near_vehicle bus
- Missing expected relation: truck near_vehicle motorcycle
- Missing expected relation: bus near_vehicle person
- Missing expected relation: person near_vehicle bus
- Missing expected relation: bus near_vehicle traffic light
- Missing expected relation: traffic light near_vehicle truck
- Missing expected relation: motorcycle near_vehicle truck
- Missing expected relation: truck near_vehicle traffic light
- Missing expected relation: bus near_vehicle motorcycle
- Missing expected relation: traffic light near_vehicle car
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Relationship proximity/threshold gap

### `000000542625.jpg` (streets) — score 0.82
- **Caption:** This appears to be an outdoor setting (no people detected, empty crowd level). Objects include car (bottom-right), car (middle-right), bird (middle-center), car (middle-right), par...
- **Detected:** bird, car, parking meter, traffic light, truck
- **Ground truth:** bird, car, fire hydrant, parking meter, traffic light, truck
- **Missing:** car
- Missed GT objects: car
- Extra detections: traffic light
- Zone mismatch for car: expected middle-right, got bottom-right
- Zone mismatch for car: expected top-center, got middle-right
- Missing expected relation: car near_vehicle traffic light
- Missing expected relation: car near_vehicle bird
- Missing expected relation: car near_vehicle fire hydrant
- Missing expected relation: traffic light near_vehicle car
- Missing expected relation: fire hydrant near_vehicle car
- Missing expected relation: bird near_vehicle car
- Missing expected relation: car near_vehicle truck
- Missing expected relation: truck near_vehicle car
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Relationship proximity/threshold gap

### `000000562581.jpg` (sports) — score 0.83
- **Caption:** there is a man that is playing tennis on the court
- **Detected:** person, tennis racket
- **Ground truth:** person, sports ball, tennis racket
- **Missing:** sports ball
- Missed GT objects: sports ball
- Extra detections: tennis racket
- Missing expected relation: person playing_with sports ball
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Relationship proximity/threshold gap

### `000000581357.jpg` (people) — score 0.83
- **Caption:** This appears to be an outdoor setting (group gathering, crowded crowd level). Objects include person (middle-center), skateboard (middle-center), person (bottom-right), person (bot...
- **Detected:** person, skateboard
- **Ground truth:** bench, person, skateboard
- **Missing:** person
- Missed GT objects: person
- Zone mismatch for person: expected middle-right, got bottom-right
- Zone mismatch for person: expected bottom-center, got middle-left
- Expected activities ['people present', 'playing sports'], got ['exercising', 'people present', 'skateboarding']
- *Root cause:* Activity rule coverage gap
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection recall gap (YOLO threshold or object scale)

### `000000480944.jpg` (streets) — score 0.85
- **Caption:** This appears to be an outdoor setting (no people detected, empty crowd level). Objects include car (middle-left), traffic light (top-right), car (middle-center), traffic light (top...
- **Detected:** bus, car, traffic light, truck
- **Ground truth:** bus, car, stop sign, traffic light
- **Missing:** bus, car, stop sign
- Missed GT objects: bus, car, stop sign
- Extra detections: traffic light, traffic light, traffic light, truck
- Missing expected relation: bus near_vehicle bus
- Missing expected relation: car near_vehicle stop sign
- Missing expected relation: stop sign near_vehicle car
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)
- *Root cause:* Relationship proximity/threshold gap

### `000000014038.jpg` (indoor) — score 0.86
- **Caption:** This appears to be an indoor setting (no people detected, empty crowd level). Objects include refrigerator (bottom-left), cell phone (bottom-right), bottle (top-left), bottle (top-...
- **Detected:** book, bottle, cell phone, refrigerator
- **Ground truth:** bed, book, bottle, cell phone, chair, couch, dining table, microwave, potted plant, refrigerator, tv
- **Missing:** bed, dining table, microwave, potted plant, tv
- Missed GT objects: bed, dining table, microwave, potted plant, tv
- Extra detections: book, book, book, bottle, bottle, cell phone
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)

### `000000554291.jpg` (animals) — score 0.88
- **Caption:** This appears to be an indoor setting (no people detected, empty crowd level). Objects include bird (middle-center), chair (top-right), dining table (bottom-center). 8 spatial relat...
- **Detected:** bird, chair, dining table
- **Ground truth:** bowl, cat, chair, couch, dining table, dog, mouse
- **Missing:** bowl, cat, couch, dog, mouse
- Missed GT objects: bowl, cat, couch, dog, mouse
- Extra detections: bird
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)

### `000000548780.jpg` (outdoor) — score 0.88
- **Caption:** This appears to be an outdoor setting (group gathering, crowded crowd level). Objects include bench (middle-center), bird (bottom-left), person (middle-center), handbag (middle-cen...
- **Detected:** bench, bird, handbag, person, potted plant
- **Ground truth:** backpack, bench, bird, handbag, person, potted plant
- **Missing:** person
- Missed GT objects: person
- Extra detections: handbag
- Zone mismatch for potted plant: expected middle-left, got bottom-left
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)

### `000000319935.jpg` (indoor) — score 0.91
- **Caption:** This appears to be an indoor setting (no people detected, empty crowd level). Objects include couch (bottom-right), chair (bottom-left), book (bottom-center), couch (middle-left), ...
- **Detected:** bed, book, chair, couch, dining table, potted plant, tv, vase
- **Ground truth:** bed, book, chair, couch, dining table, potted plant, tv
- Extra detections: book, book, chair, chair, dining table, tv, vase
- Zone mismatch for couch: expected bottom-left, got middle-left
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)

### `000000540502.jpg` (kitchens) — score 0.92
- **Caption:** This appears to be an indoor setting (no people detected, empty crowd level). Objects include refrigerator (middle-left), oven (middle-center), microwave (middle-center), potted pl...
- **Detected:** bowl, chair, dining table, microwave, oven, potted plant, refrigerator, sink
- **Ground truth:** apple, bowl, chair, dining table, microwave, orange, oven, potted plant, refrigerator, sink, spoon, toaster, vase
- Extra detections: bowl, chair, oven, potted plant, sink
- Zone mismatch for dining table: expected middle-center, got bottom-center
- Zone mismatch for chair: expected middle-left, got bottom-right
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)

### `000000507575.jpg` (offices) — score 0.92
- **Caption:** This appears to be an indoor setting (no people detected, empty crowd level). Objects include laptop (top-left), mouse (middle-right), laptop (top-center), cell phone (bottom-left)...
- **Detected:** book, cell phone, keyboard, laptop, mouse, scissors
- **Ground truth:** book, cell phone, keyboard, laptop, mouse, tv
- **Missing:** tv
- Missed GT objects: tv
- Extra detections: cell phone, laptop, mouse, scissors
- *Root cause:* Detection precision gap (false positive)
- *Root cause:* Detection recall gap (YOLO threshold or object scale)

### `000000172595.jpg` (offices) — score 0.95
- **Caption:** This appears to be an indoor setting (no people detected, empty crowd level). Objects include chair (middle-left), keyboard (middle-center), tv (top-center), mouse (middle-center),...
- **Detected:** backpack, chair, handbag, keyboard, laptop, mouse, tv
- **Ground truth:** backpack, book, bottle, cell phone, chair, handbag, keyboard, laptop, mouse, tv
- Extra detections: chair, keyboard, mouse, tv
- Zone mismatch for backpack: expected bottom-center, got middle-center
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection precision gap (false positive)

### `000000445658.jpg` (kitchens) — score 0.95
- **Caption:** This appears to be an indoor setting (no people detected, empty crowd level). Objects include oven (bottom-center), bowl (middle-right), refrigerator (middle-left), oven (bottom-ri...
- **Detected:** bottle, bowl, microwave, oven, refrigerator
- **Ground truth:** bottle, bowl, microwave, oven, refrigerator, sink, teddy bear
- Extra detections: bottle, bottle, bottle, oven
- *Root cause:* Detection precision gap (false positive)

### `000000193162.jpg` (animals) — score 0.95
- **Caption:** This appears to be an outdoor setting (individual presence, single person crowd level). Objects include cow (middle-center), dog (bottom-left), dog (middle-left), person (top-right...
- **Detected:** cow, dog, person
- **Ground truth:** cow, dog, person, sheep
- **Missing:** sheep
- Missed GT objects: sheep
- Zone mismatch for dog: expected middle-left, got bottom-left
- Zone mismatch for dog: expected bottom-left, got middle-left
- *Root cause:* Attribute zone boundary edge case
- *Root cause:* Detection recall gap (YOLO threshold or object scale)

## Recommendations Before Competition

- Lower YOLO confidence threshold or use larger input size for small/occluded COCO objects.
- Tune `near_distance_ratio` or add co-occurrence-based relation inference for distant but semantically linked objects.
- Add scene-type-specific activity templates (classroom, kitchen, street).
- Run competition mode with deterministic seed on target hardware and cache warmed models.
- Review lowest-scoring images in the table above before demo — they indicate domain gaps.

## Heuristic Improvements Applied During Evaluation

- Expanded kitchen/classroom indoor environment boosts in `context_builder.py`.
- Added classroom reading activity inference when people co-occur with study objects.
- Ensured `people present` is emitted whenever persons are detected alongside other activities.
- Increased relationship `near_distance_ratio` from 0.18 to 0.20 for better proximity relations.
- Expanded caption validator vocabulary for evidence-based context caption terms.
- Down-weight umbrella-only outdoor cues when indoor/classroom/kitchen objects are present.

Raw results: `validation/real_world/results.json`
