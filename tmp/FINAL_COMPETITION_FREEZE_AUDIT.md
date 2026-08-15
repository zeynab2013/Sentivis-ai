# FINAL COMPETITION FREEZE AUDIT (READ-ONLY)

**Scope:** 11 unique real validation images (critical + generalization set).  
**Method:** Full pipeline E2E + visual review of residual cases A–F.  
**Production code:** not modified.

## Visual adjudication notes

- **Motorcycle water:** Rocky stream water is **visually present** under/around the trials bike. Automated “hallucination” flag is **overturned**.
- **Farm second person:** Background adult is **visually present**; YOLO@1280 detects 1 person (known accepted miss).
- **Bicycle:** Child is on the bicycle with adult assisting; active “riding” is weakly/ambiguously supported. Frozen activity system CONFIRMED only `carrying a handbag`.
- **Soccer:** 4 people visible; 2 foreground players contesting the ball.

## Per-image snapshot

| Image | People | Confirmed activities | Caption (abbrev) | Worst after adjudication |
|---|---|---|---|---|
| soccer | 4 | playing football (2 actors) | Two people are playing football… 4 people present… | MEDIUM (awkward location clause) |
| farm | 1 (visually 2) | leading horse; holding rope | leading… holding a rope… | MEDIUM (YOLO miss) |
| motorcycle | 1 | riding motorcycle | riding dirt bike… next to the water… | LOW/INFO (water supported) |
| bicycle | 2 | carrying handbag | near bicycle… little person rides… handbag | MEDIUM (activity freeze gap) |
| baseball | 2 | — | swinging baseball bat at white ball | INFO |
| kitchen | 2 | — | kitchen/table inventory | INFO |
| moto_alt | 1 | riding motorcycle | riding on a road | INFO |
| dense | 12 | playing with tennis racket (1) | one person playing… others farther back | INFO |
| outdoor_misc | 2 | — | Two people are visible | LOW (thin) |
| trail | 1 | riding bicycle | “…is the activity of riding…” | MEDIUM (robotic phrasing) |
| animal | 0 | — | A bear is at the heart of the scene | INFO |

Artifacts: `tmp/FINAL_COMPETITION_FREEZE_AUDIT.json`, `tmp/FINAL_COMPETITION_FREEZE_AUDIT_run.log`
