"""Premium Sentivis AI Streamlit theme — commercial Vision AI look."""

from __future__ import annotations

THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');

:root {
  --primary: #FF4FAD;
  --primary-hover: #FF73C2;
  --primary-soft: rgba(255, 79, 173, 0.14);
  --primary-glow: rgba(255, 79, 173, 0.22);
  --primary-border: rgba(255, 79, 173, 0.32);
  --bg: #0E0818;
  --surface: #171028;
  --card: #21163A;
  --text: #F7F4FF;
  --text-secondary: #D2C9E6;
  --text-muted: #A89BBF;
  --border: rgba(255, 79, 173, 0.18);
  --success: #3DDC97;
  --warning: #FFC857;
  --danger: #FF6B6B;
  --radius: 16px;
  --shadow-soft: 0 10px 32px rgba(0, 0, 0, 0.36);
  --shadow-glow: 0 0 0 transparent;
}

html, body, [class*="css"] {
  font-family: 'Outfit', 'Segoe UI', sans-serif;
}

.stApp {
  background:
    radial-gradient(900px 480px at 8% -8%, rgba(255, 79, 173, 0.10), transparent 55%),
    radial-gradient(700px 420px at 92% 4%, rgba(88, 48, 160, 0.14), transparent 52%),
    linear-gradient(180deg, #0E0818 0%, #171028 46%, #0E0818 100%) !important;
  color: var(--text);
}

.block-container {
  padding-top: 1.15rem !important;
  padding-bottom: 2.75rem !important;
  max-width: 1240px;
}

h1, h2, h3, .brand-mark {
  font-family: 'Source Serif 4', Georgia, serif !important;
  letter-spacing: -0.02em;
  color: var(--text) !important;
}

.gradient-header {
  background: linear-gradient(120deg, #FFFFFF 0%, #FF63B8 55%, #FF3FA4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-header {
  text-align: center;
  margin: 0.1rem 0 1.1rem 0;
  padding: 1.35rem 1.15rem 1.2rem;
  border-radius: 20px;
  background: linear-gradient(145deg, rgba(33, 22, 58, 0.88), rgba(23, 16, 40, 0.72));
  border: 1px solid var(--border);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: var(--shadow-soft);
  animation: fade-rise 0.45s ease-out;
}

.hero-title {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: clamp(1.95rem, 3.6vw, 2.55rem);
  font-weight: 700;
  margin: 0;
  line-height: 1.18;
}

.hero-subtitle {
  margin: 0.45rem 0 0;
  color: var(--text-secondary);
  font-size: 0.98rem;
  font-weight: 500;
  letter-spacing: 0.01em;
  max-width: 42rem;
  margin-left: auto;
  margin-right: auto;
}

.toolbar-shell {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  align-items: center;
  justify-content: center;
  padding: 0.85rem 1rem;
  margin-bottom: 1.1rem;
  border-radius: 20px;
  background: rgba(36, 24, 67, 0.55);
  border: 1px solid var(--border);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
  animation: fade-rise 0.65s ease-out;
}

.glass-card {
  background: linear-gradient(160deg, rgba(33, 22, 58, 0.92), rgba(23, 16, 40, 0.84));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.1rem;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: var(--shadow-soft);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  animation: fade-rise 0.4s ease-out;
}

.glass-card:hover {
  border-color: rgba(255, 79, 173, 0.34);
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
}

.caption-panel {
  background: linear-gradient(155deg, rgba(36, 24, 62, 0.96), rgba(23, 16, 40, 0.9));
  border: 1px solid rgba(255, 79, 173, 0.30);
  border-radius: 18px;
  padding: 1.45rem 1.5rem 1.35rem;
  box-shadow: var(--shadow-soft);
  margin-bottom: 0.85rem;
  animation: caption-in 0.55s ease-out;
}

.caption-kicker {
  color: var(--primary);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin: 0 0 0.7rem 0;
}

.caption-body {
  color: var(--text);
  font-size: 1.18rem;
  line-height: 1.78;
  font-weight: 450;
  margin: 0;
  font-family: 'Source Serif 4', Georgia, serif;
  overflow-wrap: break-word;
  word-break: normal;
  white-space: normal;
  hyphens: none;
  max-width: 100%;
}

.caption-tts-bar {
  margin: 0.15rem 0 0.55rem 0;
  padding: 0.35rem 0.15rem;
}

.caption-tts-bar [data-testid="stHorizontalBlock"] {
  gap: 0.55rem !important;
  align-items: stretch !important;
}

.caption-tts-bar .stButton > button,
.caption-tts-bar .stDownloadButton > button {
  min-height: 2.55rem !important;
  border-radius: 12px !important;
}

.language-bar {
  margin: 0.15rem 0 0.35rem 0;
}
.language-bar-label {
  color: var(--primary);
  font-weight: 700;
  letter-spacing: 0.04em;
  font-size: 0.92rem;
}

.vision-assistant-panel {
  margin: 0.35rem 0 0.85rem 0;
  border: 1px solid rgba(255, 200, 87, 0.28);
  background: linear-gradient(155deg, rgba(36, 24, 67, 0.92), rgba(27, 18, 50, 0.78));
}

.metric-pill {
  display: inline-block;
  padding: 0.22rem 0.7rem;
  border-radius: 999px;
  background: var(--primary-soft);
  color: var(--primary);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
  word-break: normal;
  overflow-wrap: normal;
  hyphens: none;
}

.stat-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  width: 100%;
  align-items: stretch;
}

.stat-card {
  flex: 1 1 11rem;
  min-width: 11rem;
  max-width: 100%;
  box-sizing: border-box;
  text-align: center;
  padding: 1rem 0.9rem;
  border-radius: 16px;
  background: linear-gradient(165deg, rgba(36, 24, 67, 0.9), rgba(27, 18, 50, 0.7));
  border: 1px solid var(--border);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
  transition: transform 0.2s ease;
}

.stat-card:hover { transform: translateY(-3px); }

.stat-value {
  font-size: 1.55rem;
  font-weight: 700;
  margin: 0.35rem 0 0;
  background: linear-gradient(120deg, #FFFFFF, #FF63B8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  white-space: nowrap;
  word-break: normal;
  overflow-wrap: normal;
}

.stat-label {
  color: var(--text-secondary);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  white-space: normal;
  word-break: normal;
  overflow-wrap: normal;
  hyphens: none;
  line-height: 1.3;
}

.confidence-track {
  height: 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
  margin-top: 0.35rem;
}

.confidence-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #FF3FA4, #FF63B8, #FF9AD0);
  box-shadow: 0 0 12px rgba(255, 63, 164, 0.55);
  transition: width 0.6s ease;
}

.viewer-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.65rem 0 0.35rem;
}

.viewer-chip {
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  background: rgba(36, 24, 67, 0.85);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
  word-break: normal;
  overflow-wrap: normal;
  hyphens: none;
  max-width: 100%;
}

.viewer-chip strong { color: var(--primary); font-weight: 700; }

.loading-orbit {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  border: 3px solid rgba(255, 63, 164, 0.2);
  border-top-color: var(--primary);
  animation: spin 0.85s linear infinite;
  margin: 0.5rem auto;
}

.status-banner {
  text-align: center;
  padding: 0.85rem;
  border-radius: 14px;
  background: rgba(255, 63, 164, 0.1);
  border: 1px solid rgba(255, 63, 164, 0.35);
  color: var(--text-secondary);
  animation: pulse-soft 1.8s ease-in-out infinite;
}

div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1B1232 0%, #12091F 100%) !important;
  border-right: 1px solid var(--border);
}

div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stSidebar"] label {
  color: var(--text-muted) !important;
}

.stButton > button,
.stDownloadButton > button,
.stFileUploader button {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex-wrap: nowrap !important;
  white-space: nowrap !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
  hyphens: none !important;
  line-height: 1.25 !important;
  min-height: 2.75rem !important;
  height: auto !important;
  min-width: min-content !important;
  max-width: 100% !important;
  width: 100% !important;
  padding: 0.55rem 0.95rem !important;
  font-size: clamp(0.80rem, 1.55vw, 0.95rem) !important;
  overflow: visible !important;
  text-overflow: clip !important;
  box-sizing: border-box !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--primary-hover), var(--primary)) !important;
  color: #12091F !important;
  border: 1px solid var(--primary-border) !important;
  border-radius: 14px !important;
  font-weight: 650 !important;
  letter-spacing: 0.01em !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
  box-shadow: 0 4px 18px var(--primary-glow) !important;
}

/* Keep label glyphs intact across narrow columns / CJK / RTL scripts */
.stButton > button p,
.stButton > button div,
.stButton > button span,
.stDownloadButton > button p,
.stDownloadButton > button div,
.stDownloadButton > button span {
  white-space: nowrap !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
  hyphens: none !important;
  margin: 0 !important;
  line-height: 1.25 !important;
}

.toolbar-shell .stButton,
.action-bar .stButton {
  width: 100%;
}

div[data-testid="stSidebar"] .stButton > button,
div[data-testid="stSidebar"] .stSelectbox {
  min-width: 0 !important;
}

div[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
  white-space: nowrap !important;
  min-height: 2.6rem !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
  hyphens: none !important;
  line-height: 1.35 !important;
}

div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] p,
div[data-testid="stMarkdownContainer"] p,
.stCaption, [data-testid="stCaptionContainer"] {
  word-break: normal !important;
  overflow-wrap: normal !important;
  hyphens: none !important;
}

.glass-card, .caption-panel, .hero-header, .toolbar-shell {
  word-break: normal;
  overflow-wrap: normal;
  hyphens: none;
}

.export-panel .stButton > button,
.export-panel .stDownloadButton > button {
  min-width: 7.5rem !important;
  margin-bottom: 0.35rem !important;
}

@media (max-width: 720px) {
  .stat-card {
    flex: 1 1 calc(50% - 0.75rem);
    min-width: 9.5rem;
  }
}

@media (max-width: 420px) {
  .stat-card {
    flex: 1 1 100%;
    min-width: 100%;
  }
}

@media (max-width: 900px) {
  .stButton > button,
  .stDownloadButton > button {
    font-size: clamp(0.78rem, 2.4vw, 0.9rem) !important;
    padding: 0.5rem 0.7rem !important;
    min-height: 2.6rem !important;
  }
  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
}

.stButton > button:hover {
  transform: translateY(-2px);
  filter: brightness(1.05);
  box-shadow: 0 10px 30px rgba(255, 63, 164, 0.50) !important;
  background: linear-gradient(135deg, #FF7AC6, var(--primary-hover)) !important;
}

.stButton > button:active { transform: translateY(0); }

.stButton > button:disabled {
  opacity: 0.45 !important;
  filter: grayscale(0.35) !important;
  transform: none !important;
  box-shadow: none !important;
}

.stDownloadButton > button {
  background: var(--primary-soft) !important;
  color: var(--primary) !important;
  border: 1px solid rgba(255, 63, 164, 0.55) !important;
  border-radius: 14px !important;
  font-weight: 600 !important;
}

.stProgress > div > div {
  background: linear-gradient(90deg, #FF3FA4, var(--primary-hover), #FF7AC6) !important;
  border-radius: 999px !important;
}

.stProgress > div {
  background: rgba(255, 63, 164, 0.16) !important;
  border-radius: 999px !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; }

.stTabs [data-baseweb="tab"] {
  background: rgba(36, 24, 67, 0.65);
  border-radius: 12px;
  border: 1px solid transparent;
  color: var(--text-muted);
  padding: 0.45rem 0.9rem;
}

.stTabs [aria-selected="true"] {
  background: var(--primary-soft) !important;
  border-color: rgba(255, 63, 164, 0.55) !important;
  color: var(--primary) !important;
}

[data-testid="stFileUploader"] {
  background: rgba(36, 24, 67, 0.45);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 0.75rem;
}

[data-testid="stFileUploader"] section { border-radius: 14px !important; }

div[data-testid="stImage"] {
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 10px 36px rgba(0, 0, 0, 0.45), 0 0 24px rgba(255, 63, 164, 0.08);
  border: 1px solid rgba(255, 63, 164, 0.28);
}

.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
  background: rgba(36, 24, 67, 0.92) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}

[data-testid="stStatusWidget"],
.stAlert {
  border-left: 3px solid var(--primary) !important;
  border-radius: 12px !important;
  background: rgba(36, 24, 67, 0.55) !important;
}

hr { border-color: rgba(255, 63, 164, 0.28) !important; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255, 63, 164, 0.18); }
  50% { box-shadow: 0 0 28px 2px rgba(255, 63, 164, 0.35); }
}

@keyframes pulse-soft {
  0%, 100% { opacity: 0.85; }
  50% { opacity: 1; }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes fade-rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes caption-in {
  from { opacity: 0; transform: translateY(14px) scale(0.985); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

.analyze-ready .stButton > button {
  animation: none;
  box-shadow: 0 0 0 1px rgba(255, 79, 173, 0.35), 0 8px 24px rgba(255, 79, 173, 0.18) !important;
}

.vision-assistant-panel {
  border-color: rgba(255, 79, 173, 0.22) !important;
}

.va-turn {
  border-radius: 14px;
  padding: 0.75rem 0.9rem;
  margin: 0.4rem 0;
  border: 1px solid var(--border);
  background: rgba(23, 16, 40, 0.72);
}

.va-user {
  border-color: rgba(255, 200, 87, 0.22);
}

.va-assistant {
  border-color: rgba(255, 79, 173, 0.28);
  background: linear-gradient(155deg, rgba(36, 24, 62, 0.9), rgba(23, 16, 40, 0.82));
}

.va-role {
  opacity: 0.72;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-weight: 600;
}

.va-text {
  color: var(--text);
  font-size: 0.96rem;
  line-height: 1.55;
}

.va-suggestions .stButton > button {
  text-align: left !important;
  justify-content: flex-start !important;
  min-height: 2.55rem !important;
  border-radius: 12px !important;
  background: rgba(33, 22, 58, 0.72) !important;
  border: 1px solid rgba(255, 79, 173, 0.2) !important;
  font-weight: 500 !important;
  white-space: normal !important;
  height: auto !important;
  padding: 0.55rem 0.8rem !important;
}

.va-suggestions .stButton > button:hover {
  border-color: rgba(255, 79, 173, 0.45) !important;
  background: rgba(45, 30, 78, 0.85) !important;
}

.export-panel {
  margin-top: 0.35rem;
  padding: 0.85rem 0.15rem 0.25rem;
  border-top: 1px solid rgba(255, 79, 173, 0.16);
}

.export-panel .stButton > button,
.export-panel .stDownloadButton > button {
  min-height: 2.6rem !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
}

div[data-testid="stExpander"] {
  background: rgba(23, 16, 40, 0.55) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  margin-bottom: 0.55rem !important;
}

div[data-testid="stExpander"] details summary {
  font-weight: 650 !important;
  letter-spacing: 0.02em !important;
}

.section-heading {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.15rem;
  font-weight: 650;
  margin: 0.85rem 0 0.45rem;
  color: var(--text);
  letter-spacing: -0.01em;
}

.status-banner {
  animation: none;
}
"""


def inject_theme(*, high_contrast: bool = False) -> None:
    import streamlit as st

    from streamlit_app.i18n import sync_language_from_prefs
    from streamlit_app import preferences as prefs

    css = THEME_CSS
    if high_contrast:
        css += """
        .glass-card, .caption-panel {
          border-color: #FF3FA4;
          background: rgba(18, 9, 31, 0.96);
        }
        .stApp { color: #FFFFFF; }
        .caption-body { color: #FFFFFF; }
        """
    # Persian requires RTL for captions and primary text surfaces.
    try:
        import os

        import streamlit as st_runtime

        lang = (
            st_runtime.session_state.get("ui_language")
            or os.environ.get("SENTIVIS_UI_LANGUAGE")
            or prefs.load_language()
        )
        if str(lang).lower() == "fa":
            css += """
            .caption-panel, .caption-body, .glass-card p, .stMarkdown, .stCaption,
            [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
            .stButton > button {
              direction: rtl;
              text-align: right;
            }
            """
    except Exception:  # noqa: BLE001
        pass
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
