# Vision Intelligence Improvement Report

## Scope

Enhanced the computer vision intelligence pipeline while preserving Architecture v2.3 and the DI system.

No new pipeline stages were added. Improvements plug into existing stages:

1. `ATTRIBUTE_EXTRACTION` — crop + clothing analysis
2. `RELATIONSHIP_ANALYSIS` — existing semantic relations retained
3. `BLIP_UNDERSTANDING` — Florence-2 plugin with BLIP fallback
4. Caption refinement / narrative — fusion caption generator

## Components

| Module | Role |
|--------|------|
| `vision/crop_analysis/object_crop_analyzer.py` | Post-YOLO crop analysis (color, material, texture, description) |
| `analysis/clothing/clothing_analyzer.py` | Clothing type, garment colors, accessories, palette |
| `language/florence/florence_engine.py` | Florence-2 detailed captioning; auto BLIP fallback for 2GB VRAM / offline |
| `language/semantic/natural_caption_service.py` | Evidence-first caption plan from SceneReasoner + VLM reconciliation |
| Plugin `language.florence2` | Registered VLM plugin (default vision_language) |

## VRAM strategy

- YOLO remains exclusive in its stage (ModelManager single-slot policy).
- Florence-2-base (`microsoft/Florence-2-base-ft`) loads in float16 on CUDA when possible.
- If Florence cannot load (missing weights, OOM, offline), the engine falls back to BLIP automatically.
- Crop/clothing analysis is CPU-only and runs after YOLO release.

## Configuration

```toml
[florence]
model_id = "microsoft/Florence-2-base-ft"
preferred_device = "cuda"
max_new_tokens = 128
fallback_to_blip = true

[plugins]
vision_language = "language.florence2"
```

To force classic BLIP only:

```toml
[plugins]
vision_language = "language.blip_base"
```

## Final caption content

Fusion + narrative now emphasize:

- detected objects
- dominant / secondary colors
- clothing type and garment colors
- scene / environment
- activities
- relationships / interactions
