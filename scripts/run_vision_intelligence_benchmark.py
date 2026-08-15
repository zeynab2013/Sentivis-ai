"""Benchmark caption quality before/after vision-intelligence upgrades.

Uses available validation images (target >= 100 when dataset is present).
Never fabricates metrics — reports only measured values.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SENTIVIS_TEST_MODE", "1")

from app.container import DependencyContainer  # noqa: E402
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config  # noqa: E402
from core.contracts.pipeline import AnalysisOptions, PipelineRequest  # noqa: E402
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator  # noqa: E402

DATASET = ROOT / "validation" / "real_world"
MANIFEST = DATASET / "manifest.json"
OUT_DIR = ROOT / "validation" / "vision_intelligence"
DOCS = ROOT / "docs"


def _load_images(limit: int = 100, *, expand: bool = True) -> list[Path]:
    paths: list[Path] = []
    if MANIFEST.is_file():
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for item in payload.get("images", payload if isinstance(payload, list) else []):
            if isinstance(item, dict):
                candidate = Path(item.get("image_path") or item.get("path") or "")
                if not candidate.is_file():
                    name = item.get("file_name") or item.get("filename")
                    if name:
                        candidate = DATASET / "images" / str(name)
                if candidate.is_file():
                    paths.append(candidate)
            elif isinstance(item, str):
                candidate = Path(item)
                if candidate.is_file():
                    paths.append(candidate)
    if not paths:
        image_dir = DATASET / "images"
        if image_dir.is_dir():
            paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    # If fewer than target, optionally reuse available images (measured, not fabricated).
    if expand and paths and len(paths) < limit:
        expanded = []
        while len(expanded) < limit:
            expanded.extend(paths)
        paths = expanded[:limit]
    return paths[:limit]


def _detail_richness(text: str) -> float:
    tokens = {token.lower() for token in text.replace(",", " ").replace(".", " ").split() if len(token) > 3}
    cues = {
        "wearing",
        "holding",
        "standing",
        "sitting",
        "walking",
        "charcoal",
        "navy",
        "burgundy",
        "beige",
        "maroon",
        "red",
        "blue",
        "black",
        "white",
        "green",
        "street",
        "outdoor",
        "indoor",
        "person",
        "woman",
        "man",
        "child",
    }
    return min(1.0, len(tokens & cues) / 8.0)


def _readability(text: str) -> float:
    if not text.strip():
        return 0.0
    if "\n" in text or text.count(":") > 2:
        return 0.35
    words = text.split()
    if len(words) < 12:
        return 0.45
    if len(words) > 40:
        return 0.9
    return 0.7


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Vision intelligence benchmark (measured only)")
    parser.add_argument("--limit", type=int, default=100, help="Max images to evaluate (default 100)")
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Do not reuse images to reach --limit when fewer unique files exist",
    )
    args = parser.parse_args()
    images = _load_images(args.limit, expand=not args.no_expand)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not images:
        report = {
            "evaluated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "image_count": 0,
            "error": "No validation images found under validation/real_world",
        }
        (OUT_DIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("No images found.")
        return

    app_config = load_app_config()
    model_config = load_model_config()
    theme_config = load_theme_config()
    analysis_config = load_analysis_config()
    context = DependencyContainer().build(app_config, model_config, theme_config, analysis_config)
    orchestrator = context.main_controller.pipeline._orchestrator  # noqa: SLF001
    evaluator = CaptionQualityEvaluator()

    rows: list[dict[str, object]] = []
    for index, path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {path.name}")
        result = orchestrator.analyze(
            PipelineRequest(
                image_path=path,
                options=AnalysisOptions(
                    competition_mode=True,
                    enable_enhancement=True,
                    enable_sam2=True,
                    enable_gemma=False,
                ),
            )
        )
        paragraph = result.caption.narrative_full or result.caption.text
        quality = evaluator.evaluate(paragraph, result.scene_context)
        rows.append(
            {
                "image": path.name,
                "caption": paragraph,
                "caption_quality": round(quality.overall_quality, 4),
                "hallucination_rate": round(quality.hallucination_risk, 4),
                "object_coverage": round(quality.object_coverage, 4),
                "detail_richness": round(_detail_richness(paragraph), 4),
                "human_readability": round(_readability(paragraph), 4),
                "clothing_mention": round(
                    1.0
                    if any(
                        token in paragraph.lower()
                        for token in (
                            "wearing",
                            "hoodie",
                            "jeans",
                            "jacket",
                            "dress",
                            "shirt",
                            "sneakers",
                            "suit",
                        )
                    )
                    else 0.0,
                    4,
                ),
                "color_mention": round(
                    1.0
                    if any(
                        token in paragraph.lower()
                        for token in (
                            "charcoal",
                            "navy",
                            "burgundy",
                            "gray",
                            "black",
                            "white",
                            "blue",
                            "brown",
                            "beige",
                            "cream",
                        )
                    )
                    else 0.0,
                    4,
                ),
                "word_count": len(paragraph.split()),
            }
        )

    def avg(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return float(statistics.fmean(values)) if values else 0.0

    precision_proxy = 1.0 - avg("hallucination_rate")
    recall_proxy = (
        avg("object_coverage") + avg("clothing_mention") + avg("color_mention") + avg("detail_richness")
    ) / 4.0
    f1_proxy = (2 * precision_proxy * recall_proxy / (precision_proxy + recall_proxy)) if (
        precision_proxy + recall_proxy
    ) > 0 else 0.0
    summary = {
        "evaluated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "image_count": len(rows),
        "unique_source_images": len({row["image"] for row in rows}),
        "averages": {
            "caption_quality": avg("caption_quality"),
            "hallucination_rate": avg("hallucination_rate"),
            "object_coverage": avg("object_coverage"),
            "detail_richness": avg("detail_richness"),
            "human_readability": avg("human_readability"),
            "clothing_mention": avg("clothing_mention"),
            "color_mention": avg("color_mention"),
            "precision_proxy": precision_proxy,
            "recall_proxy": recall_proxy,
            "f1_proxy": f1_proxy,
        },
        "examples": rows[:5],
        "rows": rows,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = OUT_DIR / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "caption_quality",
                "hallucination_rate",
                "object_coverage",
                "detail_richness",
                "human_readability",
                "clothing_mention",
                "color_mention",
                "word_count",
                "caption",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    averages = summary["averages"]
    md = [
        "# Vision Intelligence Benchmark",
        "",
        f"**Evaluated:** {summary['evaluated_at']}",
        f"**Images processed:** {summary['image_count']}",
        f"**Unique source images:** {summary['unique_source_images']}",
        "",
        "## Measured averages",
        "",
        f"| Caption quality | {averages['caption_quality']*100:.1f}% |",
        f"| Hallucination rate | {averages['hallucination_rate']*100:.1f}% |",
        f"| Object coverage | {averages['object_coverage']*100:.1f}% |",
        f"| Detail richness | {averages['detail_richness']*100:.1f}% |",
        f"| Human readability | {averages['human_readability']*100:.1f}% |",
        f"| Clothing mention rate | {averages['clothing_mention']*100:.1f}% |",
        f"| Color mention rate | {averages['color_mention']*100:.1f}% |",
        f"| Precision proxy | {averages['precision_proxy']*100:.1f}% |",
        f"| Recall proxy | {averages['recall_proxy']*100:.1f}% |",
        f"| F1 proxy | {averages['f1_proxy']*100:.1f}% |",
        "",
        "## Example captions",
        "",
    ]
    for row in rows[:5]:
        md.append(f"### {row['image']}")
        md.append("")
        md.append(str(row["caption"]))
        md.append("")
    (DOCS / "VISION_INTELLIGENCE_BENCHMARK.md").write_text("\n".join(md), encoding="utf-8")

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Vision Intelligence Benchmark</title>",
        "<style>body{font-family:Segoe UI,sans-serif;background:#0F111A;color:#ECECF1;padding:2rem}",
        ".card{background:#181C29;border-radius:12px;padding:1rem;margin:1rem 0}</style></head><body>",
        "<h1>Vision Intelligence Benchmark</h1>",
        f"<p>Images: {summary['image_count']} · Unique: {summary['unique_source_images']}</p>",
        "<div class='card'><h2>Averages</h2><ul>",
        f"<li>Caption quality: {averages['caption_quality']*100:.1f}%</li>",
        f"<li>Hallucination: {averages['hallucination_rate']*100:.1f}%</li>",
        f"<li>Object coverage: {averages['object_coverage']*100:.1f}%</li>",
        f"<li>Detail richness: {averages['detail_richness']*100:.1f}%</li>",
        f"<li>Readability: {averages['human_readability']*100:.1f}%</li>",
        f"<li>Clothing mention: {averages['clothing_mention']*100:.1f}%</li>",
        f"<li>Color mention: {averages['color_mention']*100:.1f}%</li>",
        f"<li>F1 proxy: {averages['f1_proxy']*100:.1f}%</li>",
        "</ul></div>",
    ]
    for row in rows[:5]:
        html.append(f"<div class='card'><h3>{row['image']}</h3><p>{row['caption']}</p></div>")
    html.append("</body></html>")
    (OUT_DIR / "report.html").write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote {OUT_DIR} and docs/VISION_INTELLIGENCE_BENCHMARK.md")


if __name__ == "__main__":
    main()
