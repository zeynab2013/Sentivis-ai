# FINAL COMPETITION FREEZE VALIDATION

## IMAGE: HORSE
- Path: `tmp\uploads\10815824_2997e03d76.jpg`
- CAPTION: One person is leading a horse. Another person is holding a rope. 2 people are with a brown horse outdoors. A fire is burning nearby.
- CONFIRMED activities: ['leading a horse', 'holding a rope']
- QA TESTS:
  - How many people are visible? → There are two people.
  - What is the person doing? → The first person is leading a horse.
  - What color clothing is the person wearing? → The first person is wearing black clothing.
  - What color is the horse? → The horse appears brown.
  - What other animals are visible? → Two horses are visible in the scene.
  - Is there fire or smoke visible in the scene? → Yes. A fire is clearly visible. Smoke cannot be confirmed.
- SUGGESTED:
  - What other animals are visible?
  - What color clothing is the second person wearing?
  - What color clothing is the person wearing?
- FACT CHECK:
  - caption_naturalness: PASS
  - activity: PASS
  - people_count: PASS
  - qa_no_caption_append: PASS
  - colors: PASS
  - ocr: N/A
  - no_hallucination: PASS
  - relationships: PASS
- CASE RESULT: PASS

## IMAGE: SOCCER
- Path: `tmp\uploads\47871819_db55ac4699.jpg`
- CAPTION: One person is playing football. A white sports ball rests in the scene. 3 people are visible.
- CONFIRMED activities: ['playing football']
- QA TESTS:
  - How many people are visible? → There are three people.
  - What are they doing? → The people appear to be playing football.
  - What color is the sports ball? → The sports ball appears white.
  - What color clothing is the person wearing? → The first person is wearing a white jersey.
  - What readable text appears in the scene? → Readable text in the image includes "21".
- SUGGESTED:
  - What readable text appears in the scene?
  - What color clothing is the second person wearing?
- FACT CHECK:
  - caption_naturalness: PASS
  - activity: PASS
  - people_count: PASS
  - qa_no_caption_append: PASS
  - colors: PASS
  - ocr: PASS
  - no_hallucination: PASS
  - relationships: PASS
- CASE RESULT: PASS

## IMAGE: MOTORCYCLE
- Path: `tmp\uploads\143552829_72b6ba49d4.jpg`
- CAPTION: A man is riding a dirt bike. A person wearing a helmet, gloves and riding a bike on the water. Large rocks and grass on the ground next to the water. A motorcycle is visible behind them. He is wearing a red jacket and orange pants. He has a helmet on his head and goggles on his eyes. The bike has a number on the front of it.
- CONFIRMED activities: ['riding a motorcycle']
- QA TESTS:
  - How many people are visible? → There is one person.
  - What are they doing? → The person is riding a motorcycle.
  - What is the person doing? → The person is riding a motorcycle.
  - What color clothing is the person wearing? → The person is wearing a red jacket.
- SUGGESTED:
  - What color is the motorcycle?
  - What vehicle is visible in the scene?
- FACT CHECK:
  - caption_naturalness: PASS
  - activity: PASS
  - people_count: PASS
  - qa_no_caption_append: PASS
  - colors: PASS
  - ocr: N/A
  - no_hallucination: PASS
  - relationships: PASS
- CASE RESULT: PASS

## IMAGE: BICYCLE
- Path: `tmp\uploads\191003284_1025b0fb7d.jpg`
- CAPTION: One person is riding a bicycle. A handbag is visible behind them. Another person is also visible.
- CONFIRMED activities: ['riding a bicycle']
- QA TESTS:
  - How many people are visible? → There are two people.
  - What is the person doing? → The first person is riding a bicycle.
  - What color is the bicycle? → I can't reliably determine the color of the bicycle from the available visual evidence.
  - What color clothing is the person wearing? → I can't reliably determine the clothing color from the available visual evidence.
- SUGGESTED:
  - What color is the handbag?
  - What vehicle is visible in the scene?
- FACT CHECK:
  - caption_naturalness: PASS
  - activity: PASS
  - people_count: PASS
  - qa_no_caption_append: PASS
  - colors: PASS
  - ocr: N/A
  - no_hallucination: PASS
  - relationships: PASS
- CASE RESULT: PASS

**Critical cases failed:** 0/4