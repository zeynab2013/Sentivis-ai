"""Generate final semantic optimization benchmark reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "validation" / "real_world" / "results.json"
FINAL_JSON = ROOT / "validation" / "final_results.json"
DOCS = ROOT / "docs"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    if not RESULTS_PATH.is_file():
        from scripts.run_real_world_evaluation import run_evaluation

        payload = run_evaluation()
    else:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    FINAL_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    avg = payload["averages"]

    semantic_md = [
        "# Sentivis AI — Final Semantic Report",
        "",
        f"**Evaluated:** {payload['evaluated_at']}",
        f"**Images:** {payload['image_count']}",
        "",
        "## Measured Metrics",
        "",
        "| Metric | Measured |",
        "|--------|----------|",
        f"| Caption quality | {_pct(avg['caption_quality'])} |",
        f"| Hallucination rate | {_pct(avg['hallucination_rate'])} |",
        f"| Object detection accuracy | {_pct(avg['object_detection_accuracy'])} |",
        f"| Relationship correctness | {_pct(avg['relationship_correctness'])} |",
        f"| Environment reasoning | {_pct(avg['environment_reasoning'])} |",
        f"| Activity reasoning | {_pct(avg['activity_reasoning'])} |",
        f"| Narrative fluency | {_pct(avg['narrative_fluency'])} |",
        f"| Evidence consistency | {_pct(avg['evidence_consistency'])} |",
        f"| Overall semantic score | {_pct(avg['overall_semantic_score'])} |",
        "",
        "Raw results: `validation/final_results.json`",
        "",
    ]
    (DOCS / "FINAL_SEMANTIC_REPORT.md").write_text("\n".join(semantic_md), encoding="utf-8")

    caption_md = [
        "# Sentivis AI — Caption Report",
        "",
        f"**Evaluated:** {payload['evaluated_at']}",
        "",
        f"- Caption quality: **{_pct(avg['caption_quality'])}**",
        f"- Hallucination rate: **{_pct(avg['hallucination_rate'])}**",
        f"- Narrative fluency: **{_pct(avg['narrative_fluency'])}**",
        f"- Evidence consistency: **{_pct(avg['evidence_consistency'])}**",
        "",
    ]
    (DOCS / "CAPTION_REPORT.md").write_text("\n".join(caption_md), encoding="utf-8")

    competition_md = [
        "# Sentivis AI — Competition Report",
        "",
        f"**Evaluated:** {payload['evaluated_at']}",
        "",
        "## Summary",
        "",
        "Semantic optimization phase applied: enhanced preprocessing, rich object attributes,",
        "segmentation-aware relationships, expanded activities/environments, sentence-level",
        "evidence validation, and premium Streamlit presentation.",
        "",
        "## Measured Benchmark",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall semantic score | {_pct(avg['overall_semantic_score'])} |",
        f"| Caption quality | {_pct(avg['caption_quality'])} |",
        f"| Hallucination rate | {_pct(avg['hallucination_rate'])} |",
        f"| Activity accuracy | {_pct(avg['activity_reasoning'])} |",
        f"| Environment accuracy | {_pct(avg['environment_reasoning'])} |",
        f"| Relationship accuracy | {_pct(avg['relationship_correctness'])} |",
        "",
        "See also: FINAL_SEMANTIC_REPORT.md, CAPTION_REPORT.md, STREAMLIT_UI_REPORT.md",
        "",
    ]
    (DOCS / "COMPETITION_REPORT.md").write_text("\n".join(competition_md), encoding="utf-8")
    print("Reports written to docs/ and validation/final_results.json")


if __name__ == "__main__":
    main()
