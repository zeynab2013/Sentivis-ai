"""Export controls for Streamlit."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from core.contracts.pipeline import PipelineResult
from services.export.export_manager import ExportManager
from streamlit_app.i18n import t


def render_exports(result: PipelineResult | None, export_manager: ExportManager, exports_dir: Path) -> None:
    if result is None:
        return

    st.markdown(
        f'<div class="section-heading">{t("streamlit.exports.title")}</div>'
        f'<p style="color:var(--text-muted);font-size:0.88rem;margin:0 0 0.65rem 0;">'
        f'TXT · HTML · Markdown · PDF · JSON</p>',
        unsafe_allow_html=True,
    )
    exports_dir.mkdir(parents=True, exist_ok=True)
    if "export_stamp" not in st.session_state:
        st.session_state.export_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp = st.session_state.export_stamp
    stem = result.request.image_path.stem

    formats = [
        ("pdf", "PDF"),
        ("html", "HTML"),
        ("md", "Markdown"),
        ("txt", "TXT"),
        ("json", "JSON"),
    ]

    st.markdown('<div class="export-panel">', unsafe_allow_html=True)
    # Wide: up to 5; narrow rows wrap via two responsive rows.
    row1 = st.columns(3, gap="medium")
    row2 = st.columns(2, gap="medium")
    column_map = {
        "pdf": row1[0],
        "html": row1[1],
        "md": row1[2],
        "txt": row2[0],
        "json": row2[1],
    }
    for fmt, label in formats:
        out_path = exports_dir / f"{stem}_{stamp}.{fmt}"
        with column_map[fmt]:
            if st.button(label, key=f"export_{fmt}", use_container_width=True):
                try:
                    transcript = ""
                    session = st.session_state.get("vision_assistant_session")
                    if session is not None and getattr(session, "turns", None):
                        lines = []
                        for turn in session.turns:
                            who = "You" if turn.role == "user" else "Sentivis AI"
                            lines.append(f"{who}: {turn.text}")
                        transcript = "\n".join(lines)
                    from services.export.export_manager import set_export_assistant_transcript

                    set_export_assistant_transcript(transcript)
                    export_manager.export(result, fmt, out_path)
                    set_export_assistant_transcript("")
                    st.session_state[f"export_path_{fmt}"] = str(out_path)
                    st.success(t("streamlit.exports.saved"))
                except Exception as exc:
                    st.error(str(exc))
            path = st.session_state.get(f"export_path_{fmt}")
            if path and Path(path).is_file():
                st.download_button(
                    f"Download {label}",
                    data=Path(path).read_bytes(),
                    file_name=Path(path).name,
                    key=f"dl_{fmt}",
                    use_container_width=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)
