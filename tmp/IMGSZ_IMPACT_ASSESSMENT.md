# IMGSZ IMPACT ASSESSMENT (read-only)

Images compared: **11** unique real validation images
(generalization audit set ∪ competition freeze critical set).

Production code was **not** modified. Legacy arm forces `imgsz=640` only inside this harness.

## Recommendation

**C) Multi-scale detection is justified as a future improvement, but keep 1280 for final freeze**

Across 11 unique validation images, imgsz=1280 wins or ties on person/sports/small-object recall for the competition distribution. The only clear person regression at 1280 is the known farm second-person miss. That single-image failure justifies future multi-scale work, but does not justify a global revert to 640 (which would regress soccer/dense/sports scenes).

## Preference counts

- 1280 better: 6 — 47871819_db55ac4699.jpg, 141139674_246c0f90a1.jpg, 5e4dc9b3-589c-43c6-84b0-5620936c9df4.png, random_850976.jpg, random_385406.jpg, 47870024_73a4481f7d.jpg
- 640 better: 0 — (none)
- Equivalent: 5 — 143552829_72b6ba49d4.jpg, 10815824_2997e03d76.jpg, 191003284_1025b0fb7d.jpg, 166321294_4a5e68535f.jpg, 95728660_d47de66544.jpg

## Metric winner tallies (per image)

- person recall: {'tie': 6, '1280': 4, '640': 1}
- sports-object recall: {'tie': 10, '1280': 1}
- animal recall: {'tie': 9, '1280': 2}
- small-object recall: {'tie': 8, '1280': 3}
- false-positive pressure (fewer weak clutter better): {'tie': 11}

## Downstream stability

- identical scene graphs: 4/11
- identical confirmed activities: 9/11
- identical captions: 0/11

## Important regressions @1280 vs @640

- person losses: ['10815824_2997e03d76.jpg']
- person gains: ['47871819_db55ac4699.jpg', '141139674_246c0f90a1.jpg', 'random_385406.jpg', '47870024_73a4481f7d.jpg']
- sports losses: none
- sports gains: ['47870024_73a4481f7d.jpg']
- animal losses: none
- animal gains: ['10815824_2997e03d76.jpg', 'random_850976.jpg']

## Per-image detail

### 143552829_72b6ba49d4.jpg (`1_single_person_activity` / motorcycle rider)
- preference: **equivalent** flags=['caption_diff']
- persons 1280/640: 1/1 (confs [0.928] vs [0.926])
- sports 0/0 {} vs {}
- animals 0/0 {} vs {}
- small 0/0; weak_clutter 0/0
- graph_n 2/2 jaccard=1.0 identical=True
- confirmed activities 1280: ['riding a motorcycle']
- confirmed activities 640: ['riding a motorcycle']
- quality/qa 1280: None/False | 640: None/False
- caption 1280: A person is riding a dirt bike. A person, gloves and riding a bike on the water. Large rocks and grass on the ground next to the water. A brown motorcycle is visible behind them. A person is riding. A person is riding.
- caption 640: A person is riding a dirt bike. A person, gloves and riding a bike on the water. Large rocks and grass on the ground next to the water. A motorcycle is visible behind them. A person is riding. A person is riding.

### 47871819_db55ac4699.jpg (`2_multi_shared_activity` / soccer shared play)
- preference: **1280** flags=['person_delta=4-3', 'graph_diff', 'caption_diff']
- persons 1280/640: 4/3 (confs [0.839, 0.714, 0.579, 0.539] vs [0.937, 0.875, 0.867])
- sports 1/1 {'sports ball': 1} vs {'sports ball': 1}
- animals 0/0 {} vs {}
- small 2/2; weak_clutter 0/0
- graph_n 5/4 jaccard=0.8 identical=False
- confirmed activities 1280: ['playing football']
- confirmed activities 640: ['playing football']
- quality/qa 1280: None/True | 640: None/True
- caption 1280: Two people are playing football. A white sports ball rests in the scene.
- caption 640: Two people are playing football. A white sports ball sits at the bottom of the scene.

### 10815824_2997e03d76.jpg (`3_multi_different_activities` / farm leading+holding)
- preference: **equivalent** flags=['person_delta=1-2', 'animal_delta=4-2', 'graph_diff', 'caption_diff']
- persons 1280/640: 1/2 (confs [0.86] vs [0.901, 0.866])
- sports 0/0 {} vs {}
- animals 4/2 {'horse': 4} vs {'horse': 2}
- small 0/0; weak_clutter 0/0
- graph_n 5/4 jaccard=0.5 identical=False
- confirmed activities 1280: ['leading a horse', 'holding a rope']
- confirmed activities 640: ['leading a horse', 'holding a rope']
- quality/qa 1280: None/True | 640: None/True
- caption 1280: A person in a cream-colored garment is on the grass, holding a rope with their hand as they leads one of the three brown horses. This single person is leading a horse, demonstrating a clear connection between them. A fir
- caption 640: Two people are situated outdoors in an open area, with A person leading a brown horse across the grassy ground. One person is holding a rope. A fire is burning nearby.

### 141139674_246c0f90a1.jpg (`4_sports_soccer_alt` / football sports alt)
- preference: **1280** flags=['person_delta=2-1', 'graph_diff', 'caption_diff']
- persons 1280/640: 2/1 (confs [0.759, 0.434] vs [0.916])
- sports 2/2 {'baseball bat': 1, 'sports ball': 1} vs {'baseball bat': 1, 'sports ball': 1}
- animals 0/0 {} vs {}
- small 1/1; weak_clutter 0/0
- graph_n 4/3 jaccard=0.75 identical=False
- confirmed activities 1280: []
- confirmed activities 640: []
- quality/qa 1280: None/True | 640: None/True
- caption 1280: A young person is swinging a baseball bat at a white ball. Another person is also visible.
- caption 640: A single person in black clothing is outdoors, holding a green baseball bat towards a white sports ball.

### 191003284_1025b0fb7d.jpg (`7_person_vehicle_bike` / bicycle multi people)
- preference: **equivalent** flags=['activity_diff', 'caption_diff']
- persons 1280/640: 2/2 (confs [0.571, 0.318] vs [0.837, 0.721])
- sports 0/0 {} vs {}
- animals 0/0 {} vs {}
- small 1/1; weak_clutter 0/0
- graph_n 4/4 jaccard=1.0 identical=True
- confirmed activities 1280: ['carrying a handbag']
- confirmed activities 640: ['riding a bicycle']
- quality/qa 1280: None/True | 640: None/True
- caption 1280: One person is carrying a handbag. A bicycle is visible behind them. Another person is also visible.
- caption 640: One person is riding a bicycle. A handbag is visible behind them. Another person is also visible.

### 166321294_4a5e68535f.jpg (`7b_person_vehicle_moto_alt` / motorcycle alt)
- preference: **equivalent** flags=['caption_diff']
- persons 1280/640: 1/1 (confs [0.896] vs [0.889])
- sports 0/0 {} vs {}
- animals 0/0 {} vs {}
- small 0/0; weak_clutter 0/0
- graph_n 2/2 jaccard=1.0 identical=True
- confirmed activities 1280: ['riding a motorcycle']
- confirmed activities 640: ['riding a motorcycle']
- quality/qa 1280: None/True | 640: None/True
- caption 1280: A person is riding a motorcycle on a road. The person is wearing brown clothing.
- caption 640: A person is riding a motorcycle on a road. The person is wearing burgundy clothing.

### 5e4dc9b3-589c-43c6-84b0-5620936c9df4.png (`8_indoor_kitchen` / kitchen objects)
- preference: **1280** flags=['graph_diff', 'caption_diff']
- persons 1280/640: 2/2 (confs [0.859, 0.641] vs [0.821, 0.703])
- sports 0/0 {} vs {}
- animals 0/0 {} vs {}
- small 13/11; weak_clutter 1/1
- graph_n 23/20 jaccard=0.87 identical=False
- confirmed activities 1280: []
- confirmed activities 640: []
- quality/qa 1280: None/True | 640: None/True
- caption 1280: Two people are together in a kitchen. The person is to the left of the scene, near a brown chair and a potted plant.
- caption 640: Two people are in a kitchen around a dining table. A white refrigerator, a brown couch, a sink, a tv, and beige cups are visible behind them. A beige cup and a brown vase sit on the table, while 4 brown chairs surround i

### 95728660_d47de66544.jpg (`9_outdoor_trail` / outdoor landscape/trail)
- preference: **equivalent** flags=['caption_diff']
- persons 1280/640: 1/1 (confs [0.868] vs [0.886])
- sports 0/0 {} vs {}
- animals 0/0 {} vs {}
- small 0/0; weak_clutter 0/0
- graph_n 2/2 jaccard=1.0 identical=True
- confirmed activities 1280: ['riding a bicycle']
- confirmed activities 640: ['riding a bicycle']
- quality/qa 1280: None/True | 640: None/False
- caption 1280: The person is riding the bicycle on a dirt trail. The person is wearing black clothing.
- caption 640: A person is riding a bike on a dirt trail. A person riding a bicycle on the ground. The person is wearing a blue jacket. There is grass on the ground behind the person. There are large white clouds in the sky.

### random_850976.jpg (`12_weak_activity` / animal/no-strong-human-act)
- preference: **1280** flags=['animal_delta=4-1', 'graph_diff', 'caption_diff']
- persons 1280/640: 0/0 (confs [] vs [])
- sports 0/0 {} vs {}
- animals 4/1 {'bear': 4} vs {'bear': 1}
- small 0/0; weak_clutter 0/0
- graph_n 4/1 jaccard=0.25 identical=False
- confirmed activities 1280: []
- confirmed activities 640: []
- quality/qa 1280: None/True | 640: None/True
- caption 1280: A bear is at the heart of the scene.
- caption 640: A beige bear is at the heart of the scene.

### random_385406.jpg (`16_dense_indoor` / dense indoor/group)
- preference: **1280** flags=['person_delta=12-10', 'graph_diff', 'activity_diff', 'caption_diff']
- persons 1280/640: 12/10 (confs [0.921, 0.885, 0.819, 0.769, 0.754, 0.703, 0.693, 0.662, 0.64, 0.626, 0.484, 0.366] vs [0.911, 0.889, 0.867, 0.861, 0.854, 0.822, 0.778, 0.743, 0.617, 0.394])
- sports 2/2 {'tennis racket': 2} vs {'tennis racket': 2}
- animals 0/0 {} vs {}
- small 4/2; weak_clutter 0/0
- graph_n 14/12 jaccard=0.857 identical=False
- confirmed activities 1280: ['playing with a tennis racket']
- confirmed activities 640: ['playing with a tennis racket', 'training', 'holding a tennis racket']
- quality/qa 1280: None/True | 640: None/False
- caption 1280: Outdoors, a person is playing with a tennis racket, while several other people remain farther back.
- caption 640: Another person is playing with a tennis racket. Outdoors, a person wearing red jacket and black pants is holding a tennis racket, while several other people remain farther back. Two people are training.

### 47870024_73a4481f7d.jpg (`17_outdoor_misc` / misc outdoor)
- preference: **1280** flags=['person_delta=2-1', 'sports_delta=1-0', 'graph_diff', 'caption_diff']
- persons 1280/640: 2/1 (confs [0.786, 0.697] vs [0.955])
- sports 1/0 {'skateboard': 1} vs {}
- animals 0/0 {} vs {}
- small 1/0; weak_clutter 0/0
- graph_n 3/1 jaccard=0.333 identical=False
- confirmed activities 1280: []
- confirmed activities 640: []
- quality/qa 1280: None/True | 640: None/True
- caption 1280: 2 people are visible. Alongside a skateboard.
- caption 640: A person wearing light blue clothing is visible in the scene.
