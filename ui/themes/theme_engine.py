"""Generate QSS stylesheets from design tokens."""

from ui.design.tokens import DesignTokens


def render_stylesheet(tokens: DesignTokens) -> str:
    """Render a complete application stylesheet from tokens."""
    t = tokens
    pad_sm = t.spacing("sm")
    pad_md = t.spacing("md")
    radius_md = t.radius("md")
    radius_lg = t.radius("lg")
    anim = t.animation_ms

    return f"""
/* Sentivis AI — token-generated theme */

* {{
    outline: none;
}}

SentivisButton:focus,
QLineEdit:focus,
QComboBox:focus,
QListWidget:focus,
QToolButton:focus,
QCheckBox:focus,
QTabBar::tab:focus {{
    border: 1px solid {t.focus_ring};
}}

QMainWindow, QWidget {{
    background-color: {t.background};
    color: {t.text_primary};
    font-family: "{t.font_family}";
    font-size: {t.font_size("md")}pt;
}}

QWidget[class="Sidebar"] {{
    background-color: {t.surface};
    border: 1px solid {t.border};
    border-radius: {radius_lg}px;
    padding: {pad_md}px;
}}

QWidget[class="Card"] {{
    background-color: {t.card};
    border: 1px solid {t.border};
    border-radius: {radius_lg}px;
    padding: {pad_md}px;
}}

QWidget[class="Card"]:hover {{
    border-color: {t.accent};
}}

QLabel[class="BrandTitle"] {{
    font-size: {t.font_size("xl")}pt;
    font-weight: 700;
    color: {t.accent};
}}

QLabel[class="BrandSlogan"] {{
    font-size: {t.font_size("sm")}pt;
    letter-spacing: 1px;
    color: {t.secondary};
    margin-bottom: {pad_sm}px;
}}

QLabel[class="SectionTitle"] {{
    font-size: {t.font_size("lg")}pt;
    font-weight: 600;
    color: {t.text_secondary};
    margin-top: {pad_sm}px;
}}

QLabel[class="StatusLabel"] {{
    color: {t.text_secondary};
    font-size: {t.font_size("sm")}pt;
}}

QLabel[class="Placeholder"] {{
    color: {t.text_secondary};
    padding: {t.spacing("lg")}px;
    border: 1px dashed {t.border};
    border-radius: {radius_lg}px;
}}

QLabel[class="stage-pending"] {{
    color: {t.text_secondary};
    padding: 2px 0;
}}

QLabel[class="stage-running"] {{
    color: {t.secondary};
    font-weight: 600;
}}

QLabel[class="stage-completed"] {{
    color: {t.success};
}}

QLabel[class="stage-failed"] {{
    color: {t.error};
}}

QLabel[class="stage-skipped"] {{
    color: {t.text_secondary};
    font-style: italic;
}}

QLabel[class="StageRow"] {{
    padding: 2px 0;
}}

QWidget[presentation="true"] QLabel[class="SectionTitle"] {{
    font-size: {t.font_size("lg")}pt;
}}

SentivisButton {{
    border: none;
    padding: {pad_sm}px {pad_md}px;
    border-radius: {radius_md}px;
    font-weight: 600;
    min-height: {t.icon_size("md")}px;
}}

SentivisButton[variant="primary"] {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.primary}, stop:1 {t.secondary});
    color: {t.text_primary};
}}

SentivisButton[variant="primary"]:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent}, stop:1 {t.secondary});
}}

SentivisButton[variant="secondary"] {{
    background-color: {t.secondary};
    color: #1A1523;
    border: 1px solid {t.secondary};
    font-weight: 600;
}}

SentivisButton[variant="secondary"]:hover {{
    border-color: {t.secondary};
    background-color: {t.secondary};
}}

SentivisButton[variant="danger"] {{
    background-color: {t.error};
    color: {t.text_primary};
}}

SentivisButton:disabled {{
    background-color: {t.border};
    color: {t.text_secondary};
}}

SentivisButton:pressed {{
    padding-top: {pad_sm + 1}px;
    padding-bottom: {pad_sm - 1}px;
}}

StatusBadge {{
    border-radius: {t.radius("sm")}px;
    padding: 2px {pad_sm}px;
    font-size: {t.font_size("sm")}pt;
    font-weight: 600;
}}

StatusBadge[status="ready"] {{
    background-color: {t.surface};
    color: {t.text_secondary};
    border: 1px solid {t.border};
}}

StatusBadge[status="loading"],
StatusBadge[status="running"],
StatusBadge[status="waiting"] {{
    background-color: {t.secondary};
    color: {t.text_primary};
}}

StatusBadge[status="recovering"] {{
    background-color: {t.warning};
    color: {t.background};
}}

StatusBadge[status="completed"] {{
    background-color: {t.success};
    color: {t.background};
}}

StatusBadge[status="warning"] {{
    background-color: {t.warning};
    color: {t.background};
}}

StatusBadge[status="failed"] {{
    background-color: {t.error};
    color: {t.text_primary};
}}

QTextEdit, QListWidget, QLineEdit, QSpinBox, QComboBox {{
    background-color: {t.surface};
    border: 1px solid {t.border};
    border-radius: {radius_md}px;
    padding: {pad_sm}px;
    selection-background-color: {t.primary};
    color: {t.text_primary};
}}

QProgressBar {{
    border: 1px solid {t.border};
    border-radius: {radius_md}px;
    text-align: center;
    background-color: {t.surface};
    min-height: {t.icon_size("sm")}px;
}}

QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.primary}, stop:1 {t.secondary});
    border-radius: {t.radius("sm")}px;
}}

QScrollArea {{
    background: transparent;
    border: none;
}}

QSplitter::handle {{
    background-color: {t.border};
    width: 2px;
}}

QToolButton {{
    background: transparent;
    border: none;
    color: {t.text_primary};
    font-weight: 600;
    text-align: left;
    padding: {t.spacing("xs")}px 0;
}}

QToolButton:hover {{
    color: {t.accent};
}}

QCheckBox {{
    spacing: {pad_sm}px;
    color: {t.text_primary};
}}

QCheckBox::indicator {{
    width: {t.icon_size("sm")}px;
    height: {t.icon_size("sm")}px;
    border-radius: {t.radius("sm")}px;
    border: 1px solid {t.secondary};
    background: {t.surface};
}}

QCheckBox::indicator:checked {{
    background: {t.primary};
}}

QTabWidget::pane {{
    border: 1px solid {t.border};
    border-radius: {radius_md}px;
    background: {t.card};
}}

QTabBar::tab {{
    background: {t.surface};
    color: {t.text_secondary};
    padding: {pad_sm}px {pad_md}px;
    border-top-left-radius: {radius_md}px;
    border-top-right-radius: {radius_md}px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background: {t.card};
    color: {t.text_primary};
    border-bottom: 2px solid {t.accent};
}}

QGraphicsView {{
    background-color: {t.background};
    border: 1px solid {t.border};
    border-radius: {radius_lg}px;
}}

QDialog {{
    background-color: {t.card};
}}

QMenu {{
    background-color: {t.card};
    border: 1px solid {t.border};
    padding: {pad_sm}px;
}}

QMenu::item:selected {{
    background-color: {t.primary};
}}

QLabel[class="EmptyStateTitle"] {{
    font-size: {t.font_size("lg")}pt;
    font-weight: 600;
    color: {t.text_primary};
}}

QLabel[class="EmptyStateMessage"] {{
    color: {t.text_secondary};
    font-size: {t.font_size("md")}pt;
    padding: {pad_sm}px;
}}

QLabel[class="EmptyStateHint"] {{
    color: {t.accent};
    font-size: {t.font_size("sm")}pt;
    font-style: italic;
}}

QLabel[class="SkeletonLine"] {{
    background-color: {t.border};
    border-radius: {t.radius("sm")}px;
    min-height: 12px;
}}

QWidget[class="NotificationToast"] {{
    background-color: {t.card};
    border: 1px solid {t.border};
    border-radius: {radius_md}px;
    min-width: 280px;
    max-width: 420px;
}}

QWidget[class="NotificationToast"][level="success"] {{
    border-left: 4px solid {t.success};
}}

QWidget[class="NotificationToast"][level="warning"] {{
    border-left: 4px solid {t.warning};
}}

QWidget[class="NotificationToast"][level="error"] {{
    border-left: 4px solid {t.error};
}}

QWidget[class="NotificationToast"][level="info"] {{
    border-left: 4px solid {t.secondary};
}}

QLabel[class="NotificationMessage"] {{
    color: {t.text_primary};
    font-size: {t.font_size("sm")}pt;
}}

/* Micro-interaction hint: keep transitions conceptual via instant hover states (<{anim}ms perceived) */
"""
