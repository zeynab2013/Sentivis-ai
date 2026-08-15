"""Final UI/report regressions: short caption, executive summary, Streamlit chrome."""

from __future__ import annotations

from pathlib import Path

from language.semantic.narrative_generator import (
    NarrativeGenerator,
    executive_summary_from_paragraph,
    short_caption_from_paragraph,
)

FARM_CAPTION = (
    "On a grassy field, a person in a black sweatshirt holds a rope while leading a large "
    "brown horse, while another person and another horse stand farther back in the field. "
    "In the foreground, a fire burns."
)

_DANGLING = {
    "while",
    "and",
    "or",
    "but",
    "another",
    "because",
    "with",
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "than",
}


def test_short_caption_never_ends_mid_sentence() -> None:
    short = short_caption_from_paragraph(FARM_CAPTION)
    assert short
    assert not short.endswith("...")
    assert short.endswith((".", "!", "?"))
    last = short.rstrip(".!?").split()[-1].lower().strip(",;:")
    assert last not in _DANGLING
    # Must preserve core verified facts from the first complete sentence.
    lower = short.lower()
    assert "black sweatshirt" in lower
    assert "leading" in lower
    assert "horse" in lower
    assert "while another" in lower or "farther back" in lower


def test_executive_summary_not_identical_to_narrative() -> None:
    narrative = NarrativeGenerator().from_natural_paragraph(FARM_CAPTION)
    assert narrative.full_caption == FARM_CAPTION
    assert narrative.executive_summary.strip() != narrative.full_caption.strip()
    # Still factual — no khaki/smoke inventions.
    lower = narrative.executive_summary.lower()
    assert "khaki" not in lower
    assert "smoke" not in lower
    assert "black sweatshirt" in lower or "leading" in lower
    # Short must also be a complete sentence.
    assert narrative.short_caption.endswith((".", "!", "?"))
    assert not narrative.short_caption.endswith("...")


def test_executive_summary_helper_differs_on_multi_sentence() -> None:
    executive = executive_summary_from_paragraph(FARM_CAPTION)
    assert executive.strip() != FARM_CAPTION.strip()
    assert executive.endswith((".", "!", "?"))
    assert "fire" in FARM_CAPTION.lower()
    # Prefer keeping main action facts even if fire is left to the narrative.
    assert "horse" in executive.lower()


def test_short_caption_heals_ellipsis_truncation() -> None:
    broken = (
        "On a grassy field, a person in a black sweatshirt holds a rope while leading "
        "a large brown horse, while another..."
    )
    short = short_caption_from_paragraph(broken)
    assert not short.endswith("...")
    assert short.endswith((".", "!", "?"))
    last = short.rstrip(".!?").split()[-1].lower().strip(",;:")
    assert last not in _DANGLING
    assert "while another" not in short.lower()
    assert "black sweatshirt" in short.lower()
    assert "leading" in short.lower()


def test_farm_facts_preserved_in_full_caption() -> None:
    narrative = NarrativeGenerator().from_natural_paragraph(FARM_CAPTION)
    lower = narrative.full_caption.lower()
    for token in (
        "black sweatshirt",
        "rope",
        "leading",
        "brown horse",
        "another person",
        "another horse",
        "fire",
    ):
        assert token in lower
    assert "khaki" not in lower
    assert "smoke" not in lower
    assert "closer to the camera" not in lower


def test_empty_analysis_results_header_not_rendered() -> None:
    main = Path("streamlit_app/main.py").read_text(encoding="utf-8")
    # Standalone empty Analysis Results title card must not precede render_results.
    assert 't("streamlit.results.title")' not in main
    assert "render_results(st.session_state.result)" in main


def test_cancel_hidden_unless_analysis_running() -> None:
    main = Path("streamlit_app/main.py").read_text(encoding="utf-8")
    assert 'if analysis_running:' in main
    assert 'cancel = st.button(t("button.cancel")' in main
    assert "st.session_state.analysis_running = True" in main
    assert "st.session_state.analysis_running = False" in main
    # Cancel must not be rendered unconditionally beside Run Analysis.
    assert (
        'with btn_col2:\n        cancel = st.button(t("button.cancel")'
        not in main.replace("\r\n", "\n")
    )
