# FINAL REGRESSION MATRIX

Real-image validation after emergency stabilization pass.

| Category | Result | People (verified) | Confirmed activity | Notes |
|----------|--------|-------------------|--------------------|-------|
| 1_kitchen_multi_people | PASS | 2 | (none) | ok |
| 2_horse_person_fire | PASS | 2 | leading a horse; holding a rope | ok |
| 3_football_sports | PASS | 1 | (none) | ok |
| 4_motorcycle | PASS | 1 | riding a motorcycle | ok |
| 5_bicycle_multi_people | PASS | 2 | riding a bicycle | ok |
| 6_vehicle | PASS | 0 | (none) | ok |
| 7_landscape | PASS | 1 | riding a bicycle | ok |
| 8_animal | PASS | 0 | (none) | ok |
| 9_dense_indoor | PASS | 9 | playing with a tennis racket; playing with a tennis racket | ok |
| 10_low_quality | FAIL | 0 | (none) | caption_nonempty, suggestions |
| 11_enhanced | PASS | 2 | leading a horse; holding a rope | ok |
| 12_multi_person_clothing | PASS | 9 | playing with a tennis racket; playing with a tennis racket | ok |

**Score:** 11/12 categories PASS

## Per-case captions & QA

### 1_kitchen_multi_people
- Image: `tmp\coco_kitchen.jpg`
- Caption: In a kitchen, a person wearing black clothing and black pants is near a sink and 2 brown bowls, while another person remains farther back.
- People QA: There are two people.
- Activity QA: I can't determine the person's exact activity from the available visual evidence.
- Color QA: I can't determine which person's clothing you mean from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'N/A', 'activity_qa_uses_confirmed': 'N/A', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 2_horse_person_fire
- Image: `tmp\uploads\10815824_2997e03d76.jpg`
- Caption: Two people are with a brown horse outdoors. A fire is burning nearby. A person is leading a horse. A person is holding a rope.
- People QA: There are two people.
- Activity QA: The first person is leading a horse.
- Color QA: The first person is wearing black clothing.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'PASS', 'activity_qa_uses_confirmed': 'PASS', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 3_football_sports
- Image: `tmp\uploads\141139674_246c0f90a1.jpg`
- Caption: A person and a baseball bat are visible outdoors.
- People QA: There is one person.
- Activity QA: I can't determine the person's exact activity from the available visual evidence.
- Color QA: I can't reliably determine the clothing color from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'N/A', 'activity_qa_uses_confirmed': 'N/A', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 4_motorcycle
- Image: `tmp\uploads\143552829_72b6ba49d4.jpg`
- Caption: A man is riding a dirt bike. He is wearing a red jacket and orange pants. He has a helmet on his head and goggles on his eyes. The bike has a number on the front of it. Large rocks and grass on the ground next to the water. A person wearing a helmet, gloves and riding a bike on the water.
- People QA: There is one person.
- Activity QA: The person is riding a motorcycle.
- Color QA: The person is wearing a red jacket.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'PASS', 'activity_qa_uses_confirmed': 'PASS', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 5_bicycle_multi_people
- Image: `tmp\uploads\191003284_1025b0fb7d.jpg`
- Caption: A person is riding a bicycle. Two people are visible in the scene.
- People QA: There are two people.
- Activity QA: The first person is riding a bicycle.
- Color QA: I can't reliably determine the clothing color from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'PASS', 'activity_qa_uses_confirmed': 'PASS', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 6_vehicle
- Image: `tmp\competition_e2e_street.jpg`
- Caption: A large blue bus is parked in a parking lot.
- People QA: No people are clearly visible.
- Activity QA: I can't determine the person's exact activity from the available visual evidence.
- Color QA: I can't reliably determine the clothing color from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'N/A', 'activity_qa_uses_confirmed': 'N/A', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 7_landscape
- Image: `tmp\uploads\95728660_d47de66544.jpg`
- Caption: Him riding the bicycle on a dirt trail, as indicated by the 90% confidence level associated with this activity.
- People QA: There is one person.
- Activity QA: The person is riding a bicycle.
- Color QA: The person is wearing a black coat.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'PASS', 'activity_qa_uses_confirmed': 'PASS', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 8_animal
- Image: `tmp\uploads\random_850976.jpg`
- Caption: A beige bear is at the heart of the scene.
- People QA: No people are clearly visible.
- Activity QA: I can't determine the person's exact activity from the available visual evidence.
- Color QA: I can't reliably determine the clothing color from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'N/A', 'activity_qa_uses_confirmed': 'N/A', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 9_dense_indoor
- Image: `tmp\uploads\random_385406.jpg`
- Caption: A crowded outdoor outdoors hosts nine people engaged in what is a tennis training session. Several people are holding tennis rackets, with one person wearing a white jersey and black shoes beside a person dressed in a red coat and burgundy pants. A person is playing with a tennis racket.
- People QA: There are nine people.
- Activity QA: The first person is training.
- Color QA: I can't determine which person's clothing you mean from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'PASS', 'activity_qa_uses_confirmed': 'PASS', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 10_low_quality
- Image: `tmp\enhance_blur.jpg`
- Caption: An indoor moment unfolds around everyday objects.
- People QA: No people are clearly visible.
- Activity QA: I can't determine the person's exact activity from the available visual evidence.
- Color QA: I can't reliably determine the clothing color from the available visual evidence.
- Checks: `{'caption_nonempty': 'FAIL', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'N/A', 'activity_qa_uses_confirmed': 'N/A', 'no_weak_inventions': 'PASS', 'suggestions': 'FAIL'}`

### 11_enhanced
- Image: `tmp\pipeline_enhanced_500x333.png`
- Caption: One person is leading a brown horse outdoors. To the right of the horse, another individual wearing a beige coat is observing. A second brown horse is in the background, adding to the outdoor setting. A fire is burning nearby. Two people are visible in the scene. A person is holding a rope.
- People QA: There are two people.
- Activity QA: The first person is leading a horse.
- Color QA: I can't reliably determine the clothing color from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'PASS', 'activity_qa_uses_confirmed': 'PASS', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`

### 12_multi_person_clothing
- Image: `tmp\uploads\random_385406.jpg`
- Caption: A crowded outdoor outdoors hosts nine people engaged in what is a tennis training session. Several people are holding tennis rackets, with one person wearing a white jersey and black shoes beside a person dressed in a red coat and burgundy pants. A person is playing with a tennis racket.
- People QA: There are nine people.
- Activity QA: The first person is training.
- Color QA: I can't determine which person's clothing you mean from the available visual evidence.
- Checks: `{'caption_nonempty': 'PASS', 'people_qa_matches_verified': 'PASS', 'people_caption_qa_consistent': 'PASS', 'confirmed_activity_in_caption': 'PASS', 'activity_qa_uses_confirmed': 'PASS', 'no_weak_inventions': 'PASS', 'suggestions': 'PASS'}`
