# Kitchen naturalness + count + color validation

- CAPTION: 2 people are in a kitchen around a dining table. A white refrigerator, a brown couch, a sink, a tv, and beige cups are visible behind them. A beige cup and a brown vase sit on the table, while 4 brown chairs surround it.
- COUNTS: {'chair': 4, 'refrigerator': 1, 'dining table': 1, 'cup': 1, 'vase': 1, 'clock': 1, 'couch': 1, 'potted plant': 6, 'person': 2, 'sink': 1, 'tv': 1}
- FRIDGE_COLOR: white
- CHECKS:
  - fridge_count_1: PASS
  - chair_count_4: PASS
  - person_count_2: PASS
  - kitchen_env: PASS
  - no_inflated_fridge: PASS
  - fridge_not_brown_beige: PASS
  - fridge_color_ok: PASS
  - no_person_and_person: PASS
  - no_gender_noun: PASS
  - natural_not_census_only: PASS
- RESULT: PASS