# Sentivis AI

**SEE. UNDERSTAND. INSPIRE.**

Evidence-grounded visual understanding for the **InnoVerse USA** competition: detect objects, reason about relationships and activities, generate natural English captions, extract readable text, and answer follow-up questions from locked scene evidence — without re-running the VLM for every question.

---

## 1. Project purpose

Sentivis AI turns a single uploaded image into:

- a verified object/scene representation
- a natural English caption grounded in that evidence
- a Vision Assistant that answers questions from the same locked evidence
- optional enhancement, multilingual UI/caption translation, and TTS playback/download

The system prefers **honest uncertainty** over fabricated facts.

---

## 2. Main features

- Image validation and optional enhancement / super-resolution
- YOLO object detection (optional SAM2 masks when weights exist)
- Attribute extraction including **entity-aware color** and clothing bands
- Relationship and activity detection with evidence levels
- OCR / readable-text extraction (orientation-aware when needed)
- Natural English caption generation with grammar/factuality cleanup
- Vision Assistant QA + suggested questions (no caption-append spam)
- Multilingual UI + caption translation
- Voice / TTS playback and audio download
- Streamlit competition UI (optional legacy PySide6 desktop entry)

---

## 3. System architecture / pipeline overview

```
UI (Streamlit / optional PySide6)
  → Application services (startup + pipeline orchestrator)
    → Domain (vision / analysis / language)
      → Infrastructure (YOLO, Florence-2, OCR, Ollama, TTS)
```

High-level stages:

1. Validate image  
2. Optional enhance / SR  
3. YOLO detect (+ optional SAM2)  
4. Attributes, relationships, activities, OCR, scene context  
5. Verified evidence assembly  
6. Caption generation → sanitize / factuality / count clamp → lock `canonical_caption_en`  
7. Vision Assistant answers from the evidence packet (no VLM re-perception)

Domain packages (`language`, `core`, `analysis`, `vision`, …) must not import `streamlit_app`. UI language is injected via `core.config.ui_language`.

---

## 4. Image analysis workflow

1. Load & validate the image  
2. Optionally enhance (kept only when objectively better)  
3. Run YOLO detection with post-filters (confidence, accessories, same-label / nested-box dedupe)  
4. Extract per-entity attributes (colors, clothing bands when person)  
5. Build relationships and activity hints  
6. Run OCR on the (display) pixels  
7. Fuse into SceneContext + VerifiedSceneEvidence  

---

## 5. Caption generation workflow

1. VLM perceive (Florence-2) + structured scene understanding  
2. NaturalCaptionService composes evidence-grounded prose  
3. **Complexity-aware coverage:** rich scenes expand with verified people / objects / activities / relations / attributes / OCR; simple scenes stay concise (no fixed word quota, no filler)  
4. Coverage / arbitration prefer verified OBSERVED facts (activities, colors)  
5. Final path: strip detector phrasing → factuality filter → **sanitize_caption** (grammar/metadata + action-first sentence order) → **clamp_caption_object_counts** → lock English caption  
6. UI may translate the locked English caption; translation does **not** re-run perception  

Captions must be natural English: no `Observed activity:`, no `dominant color is…` inventory style, no `Two people are an outdoor activity…` templates, no `A person and person` gender-strip leftovers. Length tracks verified evidence density — not a fixed word target. Factuality repair rewrites unsupported claims without collapsing rich scenes into detector inventories.

### Caption quality evaluation (displayed metrics)

`CaptionQualityEvaluator` scores the locked caption against `SceneContext` evidence:

| Metric | Meaning | Calculation |
|--------|---------|-------------|
| Object coverage | Verified entity instances accounted for | For each class: `min(stated_qty, verified_n)` if an explicit quantity appears; otherwise all verified instances of a mentioned class. `None` → **Unavailable** |
| Relationship coverage | Semantic relations mentioned | `covered_semantic_relations / semantic_relations` (`None` → **Unavailable**) |
| Activity coverage | Distinct verified activities mentioned | `covered_activities / verified_activities` over non-weak activities (`None` → **Unavailable**). Multi-person scenes require each distinct CONFIRMED activity to be representable for full credit. |
| Hallucination risk | Unsupported object tokens | `min(1, 0.25 × unsupported_token_count)` (`None` when no graph/activity evidence) |
| Evidence consistency | Aggregate grounding | Average of available coverage terms + hallucination safety |
| Overall quality | Composite score | Weighted grammar + fluency + evidence consistency + object term + factuality |

**100% object coverage means every verified object instance is accounted for** under the quantity rules above — not merely that a class name appears once, and not that the caption is non-empty. **100% activity coverage means every non-weak verified activity is reflected in the caption.** Percentages are never fabricated. When a metric cannot be computed from evidence, the UI shows **Unavailable** (not a decorative number).

### What “Unavailable” means

Displayed metrics that return `None` from the evaluator (missing graph, no activities, no relations, etc.) must render as **Unavailable**. That is not a failure of the run — it means there was no defined denominator for that percentage. Never substitute 100%, 99%, or any other placeholder.

### Object counting (verified entities)

Final captions must not invent or inflate quantities. Distinct verified entity IDs are the single source of truth:

1. Detector boxes are IoU/IoA-deduplicated into semantic objects  
2. `VerifiedSceneEvidence` assigns stable entity IDs (`chair#1`, `person_2`, …)  
3. `count(class) = number of distinct narrative-safe entity IDs`  
4. Caption lock runs `clamp_caption_object_counts`, which forces explicit quantities to that count and removes cross-sentence recounts of already-covered classes  

Late caption stages continue expanding while high-importance verified labels remain uncovered on medium/rich scenes, and must not drop newly introduced fixtures solely because a dining table was already named.

---

## 6. Relationship / activity detection

- Geometric + heuristic relationships from detections  
- Activity evidence levels (`CONFIRMED` / `SUPPORTED` / …)  
- Optional Ollama semantic synthesis when available  
- Caption and QA consume verified activities (riding, leading, playing football, etc.)  
- **Multi-person activities:** each CONFIRMED narrative-safe activity stays bound to its actor entity IDs. Caption planning keeps all distinct person→activity pairs (`person_activities`) and weaves a secondary person’s verified activity into the lead (“…while another person …”) instead of replacing it with spatial filler. Coverage repair only skips an activity when its content tokens are already present — sharing a single verb (e.g. two different “holding …” actions) does **not** drop the second. Uncertain / non-narrative activities are omitted, never invented.

---

## 7. Entity-aware color detection

Colors are bound to entities — never whole-image dominant color.

- Object colors: inset crop / mask, vegetation & ground rejection  
- **Appliances / large fixtures** (refrigerator, oven, microwave, TV, …): deeper inset + rejection of warm cabinet/wood bleed; recoverable bright panels prefer **white** over beige/brown; otherwise **unknown** rather than inventing wood tones  
- Clothing: shirt / pants / shoes bands per person  
- Vehicles / animals: vegetation bleed rejection (e.g. bicycle must not become green from grass)  
- Sports balls: ground/earth rejection with high-luminance white recovery  
- OBSERVED pixel evidence outranks weaker INFERRED colors  
- Ambiguous multi-person clothing questions may honestly refuse  
- **UNKNOWN is preferred over an incorrect color** when evidence is insufficient

---

## 8. OCR / readable text

- Primary: EasyOCR; fallbacks: Tesseract → PaddleOCR  
- Orientation retry when the primary pass is empty, weak, or letter-fragmented  
- Token dedupe so repeated OCR hits are not treated as extra objects  
- OCR text is associated as readable text evidence — it does **not** create duplicate object instances  

---

## 9. Vision Assistant / QA

- Answers from the locked evidence packet  
- Concise direct answers (no full-caption append)  
- People count from verified people index  
- Object counts from verified object instances  
- OCR answers from verified readable text  
- Ambiguous “the person” clothing asks may request clarification / refuse  

---

## 10. Suggested questions

Generated from answerable verified evidence (appearance, activity, OCR, counts) while avoiding caption duplicates and weak shoe/color guesses.

---

## 11. Image enhancement

Optional Real-ESRGAN / enhancement path. Results are kept only when quality checks prefer them. Enhancement is **not** required for core detection/caption/QA.

---

## 12. Voice / TTS and download

Locked caption (or selected text) can be synthesized for playback and downloaded as an audio artifact via the Streamlit UI TTS helpers.

---

## 13. Report exports

From the Streamlit UI, analysis results can be exported as:

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| HTML | `.html` |
| Markdown | `.md` |
| TXT | `.txt` |
| JSON | `.json` |

Exports are built from the locked pipeline result (caption, detections, metrics, optional assistant transcript). They do **not** re-run perception.

---

## 14. Supported languages

Canonical analysis language is **English**. UI and locked-caption presentation can be translated using packaged resources (`core/resources/translations/` and project `translations/`). Supported UI/caption languages:

| Language | Code |
|----------|------|
| English | `en` |
| Persian (Farsi) | `fa` |
| German | `de` |
| Chinese | `zh` |
| Spanish | `es` |

Translation is a presentation-layer step over locked English evidence. It does **not** re-run perception or invent new visual facts.

---

## 15. Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
```

Requires Python **3.10.x** (`requires-python = ">=3.10,<3.11"`).

Equivalent lock-style file: `requirements.txt` (kept in sync with `pyproject.toml`).

---

## 16. Environment setup

```bash
set SENTIVIS_PROJECT_ROOT=C:\path\to\sentivis-tree
set SENTIVIS_UI_LANGUAGE=en
```

Defaults live in `config/*.default.toml`.

---

## 17. Required dependencies

Declared in `pyproject.toml` / `requirements.txt`, including:

- torch / torchvision stack  
- ultralytics (YOLO)  
- Streamlit UI stack  
- EasyOCR (+ optional Tesseract / PaddleOCR)  
- OpenCV, Pillow, numpy  

Install with `pip install -e ".[dev]"` for runtime + test extras.

---

## 18. Model requirements

Place or download weights under `models/` per `config/models.default.toml`.

| Model | Role | Missing behavior |
|-------|------|------------------|
| YOLO11x | Detection | Logged; pipeline cannot detect without it |
| Florence-2 | VLM perceive | Failover / structured fallback caption |
| EasyOCR | OCR | Falls back to Tesseract → PaddleOCR |
| SAM2 | Segmentation | Disabled when weights missing |
| Ollama gemma3:4b | Semantic / synthesis | Optional; QA can use direct evidence |
| Real-ESRGAN | Enhancement | Optional |

**Do not swap models for competition demos unless weights are intentionally replaced by the team.**

---

## 19. Hardware requirements

- Windows 11 recommended for the competition demo (Linux/macOS supported for core pipelines)  
- NVIDIA GPU optional (CPU fallback is supported and logged)  
- 8 GB RAM minimum (16 GB recommended)  

---

## 20. How to run the application

```bash
sentivis-ai
# or
python -m app.main
# or
sentivis-streamlit
```

Legacy desktop (PySide6):

```bash
sentivis-desktop
```

---

## 21. Project structure (high level)

| Path | Role |
|------|------|
| `app/` | Startup, container, entrypoints |
| `streamlit_app/` | Competition Streamlit UI |
| `vision/` | Detection, validation, crop analysis |
| `analysis/` | Attributes, clothing, OCR, evidence, activities |
| `language/` | Captioning, QA assistant, TTS, refinement |
| `services/` | Pipeline orchestration, models, plugins |
| `core/` | Contracts, config, logging |
| `ui/` | Shared UI helpers / formatters |
| `config/` | Default TOML configuration |
| `models/` | Local model weights |
| `tests/` | Unit / certification / acceptance tests |
| `scripts/` | Validation and utility CLIs |
| `tmp/` | Local runs / validation artifacts (not product code) |

---

## 22. Testing

```bash
python -m pytest tests/unit -q
python -m pytest tests -q
```

Markers include: `acceptance`, `e2e`, `ui`, `stress`, `performance`, `real_models`.

Focused regressions cover caption grammar, OCR/orientation dedupe, nested stop-sign dedupe, entity-bound colors, and activity survival.

---

## 23. Validation / regression process

Critical real-image gate (horse / soccer / motorcycle / bicycle):

```bash
python tmp/run_competition_freeze_validation.py
```

Broader real-image matrix and production audits live under `tmp/` / `scripts/` (e.g. `scripts/validate_clean_release.py`, certification via `python -m certification`).

Before demo: unit tests green + critical 4/4 + no metadata leakage in captions.

---

## 24. Competition / demo usage

1. Start `sentivis-ai`  
2. Confirm sidebar readiness (models / OCR / GPU notes)  
3. Upload an image  
4. Review caption + detections  
5. Use Vision Assistant suggested questions / free-form QA  
6. Optionally enhance, translate, or play/download TTS  
7. Prefer questions that map to verified entities (“first person”, “sports ball”, “readable text”)  

---

## 25. Known limitations

- Without SAM2 masks, some object colors may be **unknown** rather than wrong (by design)  
- Multi-person “the person” clothing questions may refuse when the referent is ambiguous  
- Pose from boxes alone is heuristic  
- OCR empty ≠ OCR unavailable (see status field)  
- Caption richness cannot exceed **verified** detections (sparse YOLO → concise caption is correct)  
- Optional Ollama alternate captions may compete with NaturalCaptionService; arbitration prefers grounded natural text when competitive  
- Low-quality / heavy blur images may fail density / coverage gates  
- Optional Ollama / SAM2 / enhancement paths degrade gracefully when missing  
- Do **not** claim 100% accuracy  

---

## 26. Troubleshooting

| Symptom | Check |
|---------|--------|
| No detections | YOLO weights under `models/`; device logs |
| No OCR text | EasyOCR install; status `unavailable` vs `empty` |
| Slow CPU runs | Expected without CUDA; close other heavy apps |
| Caption looks robotic | Ensure latest sanitize path; re-run analysis |
| Assistant invents facts | Answers must come from evidence; refresh session after new image |
| Translation missing | `translations/` / packaged resources present |

Preflight:

```bash
python -m certification
# or
sentivis-certify
```

---

## 27. Build / release

```bash
python -m release development --validate-only
python -m release development
python -m release production
```

Release trees include `streamlit_app`, `model_management`, `translations`, `config`, and core packages.

---

## License

Proprietary — Sentivis AI Team
