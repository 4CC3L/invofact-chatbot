"""
ui/styles.py — Tema visual InvoFact (azul · verde sobre fondo oscuro)
======================================================================
Paleta basada en la identidad de marca de invofact.com,
aplicada sobre un fondo oscuro para máxima legibilidad.
"""

# ── Paleta de colores InvoFact ──────────────────────────────
PRIMARY_BLUE  = "#3b82f6"    # azul brillante visible en oscuro
ACCENT_GREEN  = "#25c07e"    # verde InvoFact
BG_MAIN       = "#0f1117"    # fondo principal oscuro
BG_SIDEBAR    = "#1a1d27"    # sidebar ligeramente más claro
BG_CARD       = "#1e2130"    # tarjetas / bloques
BORDER_COLOR  = "#2d3148"    # bordes sutiles
TEXT_PRIMARY  = "#e8eaf0"    # texto principal (casi blanco)
TEXT_MUTED    = "#8b92a8"    # texto secundario

# ── Colores por categoría (fondo oscuro, texto claro) ────────
CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    "ventas":        ("#1e3a6e", "#60a5fa"),
    "productos":     ("#14442a", "#4ade80"),
    "usuarios":      ("#3d2e0e", "#fbbf24"),
    "caja":          ("#2d1b5e", "#a78bfa"),
    "compras":       ("#4a1010", "#f87171"),
    "reportes":      ("#0d3d35", "#34d399"),
    "configuracion": ("#3d360a", "#facc15"),
    "general":       ("#252a3a", "#94a3b8"),
    "fallback":      ("#4a1010", "#f87171"),
    "saludo":        ("#0c2e4a", "#38bdf8"),
    "despedida":     ("#0d3d35", "#34d399"),
    "confirmacion":  ("#0d3d35", "#34d399"),
}

MAIN_CSS = f"""
<style>
/* ── Layout general ── */
[data-testid="stAppViewContainer"] {{
    background-color: {BG_MAIN};
}}
[data-testid="stSidebar"] {{
    background-color: {BG_SIDEBAR};
    border-right: 1px solid {BORDER_COLOR};
}}
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}

/* ── Texto global ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
.stMarkdown, .stText {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── Sidebar texto ── */
[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY};
}}

/* ── Sidebar componentes custom ── */
.sidebar-logo {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.5rem;
}}
.sidebar-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {PRIMARY_BLUE};
    margin: 0;
}}
.sidebar-subtitle {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    margin: 0;
}}
.sidebar-divider {{
    border: none;
    border-top: 1px solid {BORDER_COLOR};
    margin: 0.8rem 0;
}}
.sidebar-section-title {{
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUTED};
    margin-bottom: 0.5rem;
}}
.topic-item {{
    font-size: 0.88rem;
    color: {TEXT_PRIMARY};
    padding: 3px 0;
}}

/* ── Mensajes de chat ── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.15rem 0 !important;
}}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── Input de chat ── */
[data-testid="stChatInputTextArea"] {{
    background-color: {BG_CARD} !important;
    border: 1.5px solid {PRIMARY_BLUE} !important;
    border-radius: 12px !important;
    color: {TEXT_PRIMARY} !important;
}}
[data-testid="stBottomBlockContainer"] {{
    background-color: {BG_MAIN} !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-weight: 600;
    color: {TEXT_MUTED} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {PRIMARY_BLUE} !important;
    border-bottom-color: {PRIMARY_BLUE} !important;
}}
[data-testid="stTabsContent"] {{
    background: transparent !important;
}}

/* ── Métricas ── */
[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
    padding: 0.6rem 1rem;
}}
[data-testid="stMetricValue"] {{
    color: {PRIMARY_BLUE} !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
}}

/* ── Botón sidebar ── */
[data-testid="stSidebar"] .stButton button {{
    border: 1.5px solid #ef4444;
    color: #ef4444 !important;
    background: transparent;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    transition: all 0.2s;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: #ef4444 !important;
    color: white !important;
}}

/* ── Botones generales ── */
.stDownloadButton button {{
    background-color: {PRIMARY_BLUE} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}}
.stButton button {{
    border-radius: 8px !important;
}}

/* ── Tabla / DataFrame ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    overflow: hidden;
}}
[data-testid="stDataFrame"] th {{
    background-color: {BG_CARD} !important;
    color: {PRIMARY_BLUE} !important;
}}
[data-testid="stDataFrame"] td {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] {{
    background-color: {BG_CARD} !important;
    border-color: {BORDER_COLOR} !important;
    color: {TEXT_PRIMARY} !important;
}}

/* ── Caption / small text ── */
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption {{
    color: {TEXT_MUTED} !important;
}}

/* ── Ocultar controles nativos del sidebar (usamos botones propios) ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}
</style>
"""


