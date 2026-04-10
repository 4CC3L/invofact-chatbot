"""
hub.py - ChatHub Invofact-Sunat
Interfaz unificada: InvoFact Soporte + TAX-BOT SUNAT.
Ejecutar con: .venv\\Scripts\\streamlit.exe run hub.py
"""

# ── Imports ────────────────────────────────────────────────────────────────
import json
import uuid
import pandas as pd
import streamlit as st
from groq import Groq
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from chatbot_engine import ChatbotEngine
from core.db import (
    init_db,
    nuevo_thread_id,
    guardar_mensaje,
    obtener_historial,
    limpiar_historial,
    stats_historial,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChatHub Invofact-Sunat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. PALETA
# ══════════════════════════════════════════════════════════════════════════════
_BLUE   = "#3b82f6"
_AMBER  = "#f59e0b"
_GREEN  = "#25c07e"
_MUTED  = "#8b92a8"
_CARD   = "#1a1e2e"
_CARD2  = "#1e2130"
_BORDER = "#2d3148"
_BG     = "#0f1117"
_TEXT   = "#e8eaf0"
_SIDEBAR_BG = "#12151f"

# ══════════════════════════════════════════════════════════════════════════════
# 3. CSS
# ══════════════════════════════════════════════════════════════════════════════
HUB_CSS = f"""
<style>
/* ── Base ────────────────────────────────── */
[data-testid="stAppViewContainer"] {{ background: {_BG}; }}
[data-testid="stHeader"]  {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
footer {{ display: none !important; }}

/* ── Sidebar — forzar siempre visible ────── */
[data-testid="stSidebar"] {{
    background-color: {_SIDEBAR_BG} !important;
    border-right: 1px solid {_BORDER};
    min-width: 240px !important;
    max-width: 240px !important;
    /* Anular el transform: translateX(-300px) que Streamlit inyecta por JS */
    transform: none !important;
    transition: none !important;
    visibility: visible !important;
    display: block !important;
}}
/* Ocultar el botón de colapso para que no se pueda cerrar */
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"] {{
    display: none !important;
}}
[data-testid="stSidebar"] * {{ color: {_TEXT} !important; }}

/* ── Texto global ────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
.stMarkdown, .stText {{ color: {_TEXT} !important; }}

/* ── Chat input ──────────────────────────── */
[data-testid="stChatInputTextArea"] {{
    background: {_CARD2} !important;
    border: 1.5px solid {_BLUE} !important;
    border-radius: 12px !important;
    color: {_TEXT} !important;
}}
[data-testid="stBottomBlockContainer"] {{
    background: {_BG} !important;
    border-top: 1px solid {_BORDER};
}}

/* ── Chat messages ───────────────────────── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
}}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {{ color: {_TEXT} !important; }}

/* ── Tabs ────────────────────────────────── */
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-weight: 600;
    color: {_MUTED} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {_BLUE} !important;
    border-bottom-color: {_BLUE} !important;
}}
[data-testid="stTabsContent"] {{ background: transparent !important; }}

/* ── Métricas ────────────────────────────── */
[data-testid="stMetric"] {{
    background: {_CARD2};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    padding: 0.6rem 1rem;
}}
[data-testid="stMetricValue"] {{ color: {_BLUE} !important; font-weight: 700 !important; }}
[data-testid="stMetricLabel"] {{ color: {_MUTED} !important; }}

/* ── DataFrame ───────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {_BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
[data-testid="stDataFrame"] th {{ background: {_CARD2} !important; color: {_BLUE} !important; }}
[data-testid="stDataFrame"] td {{ color: {_TEXT} !important; }}

/* ── Alertas ─────────────────────────────── */
[data-testid="stAlert"] {{
    background: {_CARD2} !important;
    border-color: {_BORDER} !important;
    color: {_TEXT} !important;
}}

/* ── Botones ─────────────────────────────── */
.stButton button {{
    border-radius: 8px !important;
    transition: all 0.18s !important;
}}
.stDownloadButton button {{
    background: {_BLUE} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}}

/* ── Sidebar: botones de nav ─────────────── */
[data-testid="stSidebar"] .stButton button {{
    background: transparent !important;
    border: 1px solid {_BORDER} !important;
    color: {_TEXT} !important;
    text-align: left !important;
    font-size: 0.88rem !important;
    width: 100% !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: {_CARD2} !important;
    border-color: {_BLUE} !important;
}}

/* ── Nav bar top ─────────────────────────── */
.hub-topnav {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.7rem 1rem;
    background: {_CARD};
    border: 1px solid {_BORDER};
    border-radius: 14px;
    margin-bottom: 1.6rem;
}}
.hub-logo {{
    font-size: 1.25rem;
    font-weight: 800;
    color: {_TEXT};
    white-space: nowrap;
    margin-right: 0.5rem;
}}
.hub-logo span.b {{ color: {_BLUE}; }}
.hub-logo span.a {{ color: {_AMBER}; }}
.hub-divider-v {{
    width: 1px;
    height: 20px;
    background: {_BORDER};
    margin: 0 0.3rem;
}}

/* ── Home cards ──────────────────────────── */
.bot-card {{
    background: {_CARD};
    border: 1px solid {_BORDER};
    border-radius: 18px;
    padding: 2rem 1.8rem;
    min-height: 310px;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}}
.bot-card:hover {{ border-color: {_BLUE}; box-shadow: 0 8px 28px rgba(59,130,246,.18); transform: translateY(-3px); }}
.bot-card.tax:hover {{ border-color: {_AMBER}; box-shadow: 0 8px 28px rgba(245,158,11,.18); }}

/* ── Chips ───────────────────────────────── */
.tag {{ display:inline-block; font-size:0.7rem; padding:3px 10px;
        border-radius:99px; font-weight:600; margin:3px 2px; }}

/* ── HR ──────────────────────────────────── */
.hr {{ border:none; border-top:1px solid {_BORDER}; margin:0.8rem 0; }}

/* ── Caption / small ─────────────────────── */
small, .stCaption, caption {{ color: {_MUTED} !important; }}

/* ── password input ──────────────────────── */
input[type="password"] {{
    background: {_CARD2} !important;
    border-color: {_BORDER} !important;
    color: {_TEXT} !important;
}}
</style>
"""
st.markdown(HUB_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. INIT DB + ESTADO
# ══════════════════════════════════════════════════════════════════════════════
init_db()


def _init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "invofact_messages" not in st.session_state:
        st.session_state.invofact_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Bienvenido a **InvoFact Soporte**! 👋\n\n"
                    "Puedo ayudarte con:\n"
                    "- 📦 Ventas y comprobantes electrónicos\n"
                    "- 🏷️ Productos e inventario\n"
                    "- 👤 Usuarios y permisos\n"
                    "- 💰 Caja: apertura, cierre y egresos\n"
                    "- 🛒 Compras y proveedores\n"
                    "- 📊 Reportes y exportación a Excel\n\n"
                    "¿En qué te puedo ayudar hoy?"
                ),
            }
        ]
    if "invofact_thread_id" not in st.session_state:
        st.session_state.invofact_thread_id = nuevo_thread_id()
    if "taxbot_messages" not in st.session_state:
        st.session_state.taxbot_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Hola! Soy **TAX-BOT SUNAT** 🏛️\n\n"
                    "Estoy entrenado con la normativa tributaria peruana vigente al 2024. "
                    "Puedo orientarte sobre:\n"
                    "- 📋 Regímenes tributarios (NRUS, RER, RMT, Régimen General)\n"
                    "- 🧾 Comprobantes de pago y facturación electrónica\n"
                    "- 💰 IGV, crédito fiscal y exportaciones\n"
                    "- 📚 Libros contables obligatorios\n"
                    "- ⚠️ Infracciones, sanciones y TIM\n\n"
                    "¿En qué te puedo ayudar hoy?"
                ),
            }
        ]
    if "taxbot_api_key" not in st.session_state:
        st.session_state.taxbot_api_key = ""


_init_state()


def _nav(page: str) -> None:
    st.session_state.page = page


# ══════════════════════════════════════════════════════════════════════════════
# 5. RECURSOS CACHEADOS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Iniciando motor InvoFact…")
def _cargar_motor() -> ChatbotEngine:
    return ChatbotEngine()


@st.cache_resource(show_spinner="Cargando base de conocimiento SUNAT…")
def _cargar_sunat():
    try:
        with open("sunat_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None

    docs: list[Document] = []

    for reg in data.get("regimenes", []):
        nombre   = reg["nombre_completo"]
        acronimo = reg["acronimo"]

        req_obj  = reg.get("quien_puede_acogerse", {})
        docs.append(Document(
            page_content=(
                f"Régimen: {nombre} ({acronimo}). {reg['descripcion']} "
                f"Requisitos: {req_obj.get('descripcion', '')} "
                f"{' | '.join(req_obj.get('requisitos', []))}"
            ),
            metadata={"fuente": f"Normas {acronimo}", "regimen": acronimo},
        ))

        excluidos = reg.get("quien_NO_puede_acogerse", [])
        if excluidos:
            docs.append(Document(
                page_content="Excluidos del {}: {}".format(
                    acronimo,
                    " | ".join(e if isinstance(e, str) else e.get("actividad", str(e))
                               for e in excluidos),
                ),
                metadata={"fuente": f"Exclusiones {acronimo}", "regimen": acronimo},
            ))

        tributos = reg.get("tributos_que_paga", {})
        if "impuesto_a_la_renta" in tributos:
            ir = tributos["impuesto_a_la_renta"]
            docs.append(Document(
                page_content=f"IR en {acronimo}: {ir.get('tasa','')}. {ir.get('descripcion','')}",
                metadata={"fuente": f"IR {acronimo}", "regimen": acronimo},
            ))
        if "igv" in tributos:
            docs.append(Document(
                page_content=f"IGV en {acronimo}: {tributos['igv'].get('descripcion','')}",
                metadata={"fuente": f"IGV {acronimo}", "regimen": acronimo},
            ))
        if "cuota_nrus" in tributos:
            docs.append(Document(
                page_content=f"Cuota NRUS: {tributos['cuota_nrus'].get('descripcion','')}",
                metadata={"fuente": "Cuota NRUS", "regimen": "NRUS"},
            ))

        for cat in reg.get("categorias", []):
            docs.append(Document(
                page_content=(
                    f"NRUS {cat.get('descripcion','Categoría')} ({cat.get('categoria','')}): "
                    f"cuota S/ {cat.get('cuota_mensual_soles',0)}. "
                    f"Límite ingresos: S/ {cat.get('limite_ingresos_brutos_mensuales_soles','')}. "
                    f"Límite adquisiciones: S/ {cat.get('limite_adquisiciones_mensuales_soles','')}."
                ),
                metadata={"fuente": "Categorías NRUS", "regimen": "NRUS"},
            ))

        libros = reg.get("libros_contables", {})
        tramos = libros.get("tramos", [])
        if tramos:
            for t in tramos:
                docs.append(Document(
                    page_content=(
                        f"Libros {acronimo} — {t.get('nivel','')}: "
                        f"{' | '.join(t.get('libros',[]))}. "
                        f"Ref. S/ {t.get('soles_2024', t.get('soles_2024_hasta',''))}."
                    ),
                    metadata={"fuente": f"Libros {acronimo}", "regimen": acronimo},
                ))
        else:
            req_lb = " | ".join(
                lb.get("libro","") if isinstance(lb, dict) else str(lb)
                for lb in libros.get("libros_requeridos",[])
            )
            txt = f"{libros.get('descripcion','')} {req_lb}".strip()
            if txt:
                docs.append(Document(
                    page_content=f"Libros {acronimo}: {txt}",
                    metadata={"fuente": f"Libros {acronimo}", "regimen": acronimo},
                ))

        for comp_key, label in [
            ("comprobantes_que_puede_emitir", "SÍ puede emitir"),
            ("comprobantes_que_NO_puede_emitir", "NO puede emitir"),
        ]:
            comp = reg.get(comp_key, [])
            if comp:
                docs.append(Document(
                    page_content=f"Comprobantes {label} en {acronimo}: {' | '.join(comp)}.",
                    metadata={"fuente": f"Comprobantes {acronimo}", "regimen": acronimo},
                ))

        decl   = reg.get("declaraciones", {})
        dm     = decl.get("mensual", {})
        da     = decl.get("anual", {})
        dtxt   = (f"Mensual: {dm.get('descripcion','')} " if dm else "") + \
                 (f"Anual: {da.get('descripcion','')}"   if da else "")
        if dtxt.strip():
            docs.append(Document(
                page_content=f"Declaraciones {acronimo}: {dtxt.strip()}",
                metadata={"fuente": f"Declaraciones {acronimo}", "regimen": acronimo},
            ))

        cambio = reg.get("cambio_de_regimen", {})
        if cambio:
            puede = " | ".join(cambio.get("puede_cambiar_a", []))
            excl  = cambio.get("exclusion_automatica", "")
            ctxt  = (f"Puede cambiar a: {puede}. " if puede else "") + \
                    (f"Exclusión: {excl}" if excl else "")
            if ctxt.strip():
                docs.append(Document(
                    page_content=f"Cambio de régimen desde {acronimo}: {ctxt.strip()}",
                    metadata={"fuente": f"Cambio {acronimo}", "regimen": acronimo},
                ))

        for caso in reg.get("casos_practicos", []):
            detalles = []
            for campo, lbl in [
                ("ingresos_mensuales","Ingresos mensuales"),
                ("ingresos_netos_anuales","Ingresos anuales"),
                ("cuota_a_pagar","Cuota"),
                ("total_impuestos_mes","Impuestos/mes"),
                ("renta_neta_imponible","Renta neta imponible"),
                ("alerta","ALERTA"),
            ]:
                if campo in caso:
                    detalles.append(f"{lbl}: {caso[campo]}")
            docs.append(Document(
                page_content=f"Caso práctico {acronimo}: {caso.get('descripcion','')}. {' | '.join(detalles)}",
                metadata={"fuente": f"Caso {acronimo}", "regimen": acronimo},
            ))

        for faq in reg.get("preguntas_frecuentes", []):
            docs.append(Document(
                page_content=f"FAQ {acronimo} — {faq.get('pregunta','')}: {faq.get('respuesta','')}",
                metadata={"fuente": f"FAQ {acronimo}", "regimen": acronimo},
            ))

        if reg.get("ventajas"):
            docs.append(Document(
                page_content=f"Ventajas {acronimo}: {' | '.join(reg['ventajas'])}",
                metadata={"fuente": f"Ventajas {acronimo}", "regimen": acronimo},
            ))
        if reg.get("desventajas"):
            docs.append(Document(
                page_content=f"Desventajas {acronimo}: {' | '.join(reg['desventajas'])}",
                metadata={"fuente": f"Desventajas {acronimo}", "regimen": acronimo},
            ))

        ded = reg.get("deducciones_y_gastos", reg.get("gastos_deducibles", {}))
        if ded:
            gastos = " | ".join(
                g.get("gasto","") if isinstance(g, dict) else str(g)
                for g in ded.get("gastos_deducibles_principales", [])
            )
            principio = ded.get("principio_causalidad", ded.get("principio",""))
            if gastos or principio:
                docs.append(Document(
                    page_content=f"Gastos deducibles {acronimo}: {principio} {gastos}".strip(),
                    metadata={"fuente": f"Gastos {acronimo}", "regimen": acronimo},
                ))

    for fila in data.get("tabla_comparativa", {}).get("filas", []):
        docs.append(Document(
            page_content=(
                f"Comparativa — {fila.get('criterio','')}: "
                f"NRUS={fila.get('nuevo_rus','')} | RER={fila.get('rer','')} | "
                f"RMT={fila.get('rmt','')} | RG={fila.get('regimen_general','')}"
            ),
            metadata={"fuente": "Tabla Comparativa", "regimen": "todos"},
        ))

    igv_g = data.get("igv_general", {})
    if igv_g:
        docs.append(Document(
            page_content=(
                f"IGV general — tasa: {igv_g.get('tasa','')}. "
                f"Gravadas: {' | '.join(igv_g.get('operaciones_gravadas',[]))}. "
                f"Exoneradas: {' | '.join(igv_g.get('operaciones_exoneradas_principales',[]))}."
            ),
            metadata={"fuente": "IGV General", "regimen": "todos"},
        ))
        cred = igv_g.get("credito_fiscal", {})
        docs.append(Document(
            page_content=(
                f"Crédito fiscal IGV — sustanciales: {' | '.join(cred.get('requisitos_sustanciales',[]))}. "
                f"Formales: {' | '.join(cred.get('requisitos_formales',[]))}."
            ),
            metadata={"fuente": "Crédito Fiscal", "regimen": "todos"},
        ))
        docs.append(Document(
            page_content=f"IGV exportaciones: {igv_g.get('exportaciones',{}).get('descripcion','')}",
            metadata={"fuente": "IGV Exportaciones", "regimen": "todos"},
        ))

    inf = data.get("infracciones_y_sanciones", {})
    for infrac in inf.get("infracciones_principales", []):
        docs.append(Document(
            page_content=(
                f"Infracción — {infrac.get('articulo','')}: {infrac.get('infraccion','')}. "
                f"Sanción I: {infrac.get('sancion_tabla_I','')} | "
                f"II: {infrac.get('sancion_tabla_II','')} | "
                f"III: {infrac.get('sancion_tabla_III','')}. "
                f"Gradualidad: {infrac.get('gradualidad_maxima','')}"
            ),
            metadata={"fuente": "Infracciones", "regimen": "todos"},
        ))
    if inf.get("tim"):
        tim = inf["tim"]
        docs.append(Document(
            page_content=f"TIM: {tim.get('tasa_mensual','')} mensual. {tim.get('descripcion','')}",
            metadata={"fuente": "TIM", "regimen": "todos"},
        ))
    if inf.get("regimen_de_gradualidad"):
        docs.append(Document(
            page_content=f"Gradualidad: {inf['regimen_de_gradualidad'].get('descripcion','')}",
            metadata={"fuente": "Gradualidad", "regimen": "todos"},
        ))

    for ob in data.get("obligaciones_tributarias_comunes", {}).get("obligaciones", []):
        docs.append(Document(
            page_content=(
                f"Obligación — {ob.get('obligacion','')}: Aplica a {ob.get('aplica_a','')}. "
                f"Tasa: {ob.get('tasa', ob.get('tasas',''))}. "
                f"{ob.get('descripcion', ob.get('nota',''))}"
            ).strip().rstrip(".") + ".",
            metadata={"fuente": "Obligaciones Comunes", "regimen": "todos"},
        ))

    uit = data.get("uit_historico", {})
    docs.append(Document(
        page_content="Valor histórico UIT: " + " | ".join(
            f"{v['año']}: S/ {v['uit_soles']}" for v in uit.get("valores", [])
        ) + ".",
        metadata={"fuente": "UIT Histórico", "regimen": "todos"},
    ))

    for renta in data.get("rentas_de_trabajo", []):
        docs.append(Document(
            page_content=(
                f"Renta {renta.get('categoria','')}: {renta.get('descripcion','')} "
                f"Tasas: {renta.get('tasas_y_retenciones','')} "
                f"Declaración: {renta.get('declaracion','')}"
            ),
            metadata={"fuente": "Rentas de Trabajo", "regimen": "todos"},
        ))

    docs.append(Document(
        page_content=(
            "Boletas de Venta — Monto para DNI: Es obligatorio consignar datos de "
            "identificación (Nombres, Apellidos y DNI) cuando el importe total supere S/ 700.00."
        ),
        metadata={"fuente": "RS 123-2022/SUNAT", "regimen": "todos"},
    ))
    docs.append(Document(
        page_content=(
            "Anulación de Facturas: Para anular una Factura Electrónica con Nota de Crédito, "
            "el plazo máximo es hasta el séptimo (7mo) día calendario desde la emisión."
        ),
        metadata={"fuente": "RS 193-2020/SUNAT", "regimen": "todos"},
    ))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return Chroma.from_documents(
        docs, embeddings, collection_name=f"taxbot_{uuid.uuid4().hex}"
    )


motor   = _cargar_motor()
vdb     = _cargar_sunat()

# ══════════════════════════════════════════════════════════════════════════════
# 6. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown(
        f"""
        <div style="padding:1.2rem 0.5rem 0.6rem;text-align:center;">
            <p style="font-size:1.65rem;font-weight:800;margin:0;line-height:1.1;">
                🤖&nbsp;<span style="color:{_BLUE};">Chat</span><span style="color:{_AMBER};">Hub</span>
            </p>
            <p style="font-size:0.72rem;color:{_MUTED};margin:0.25rem 0 0;">
                PUCP · Asistentes Inteligentes
            </p>
        </div>
        <hr style="border:none;border-top:1px solid {_BORDER};margin:0.5rem 0 1rem;">
        """,
        unsafe_allow_html=True,
    )

    # Navegación
    st.markdown(
        f'<p style="font-size:0.68rem;color:{_MUTED};text-transform:uppercase;'
        f'letter-spacing:.08em;margin:0 0 .4rem .25rem;">Navegación</p>',
        unsafe_allow_html=True,
    )

    _p = st.session_state.page

    pages = [
        ("home",     "🏠", "Inicio"),
        ("invofact", "🤖", "InvoFact Soporte"),
        ("taxbot",   "🏛️", "TAX-BOT SUNAT"),
    ]
    for pid, icon, label in pages:
        active = "▶  " if _p == pid else "      "
        color  = _BLUE if _p == pid else _TEXT
        st.markdown(
            f'<div style="background:{"rgba(59,130,246,.12)" if _p==pid else "transparent"}; '
            f'border-radius:8px;margin:2px 0;">',
            unsafe_allow_html=True,
        )
        if st.button(f"{icon}  {label}", key=f"sb_{pid}", use_container_width=True):
            _nav(pid)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'<hr style="border:none;border-top:1px solid {_BORDER};margin:1rem 0;">', unsafe_allow_html=True)

    # Controles contextuales
    if _p == "invofact":
        st.markdown(
            f'<p style="font-size:0.68rem;color:{_MUTED};text-transform:uppercase;'
            f'letter-spacing:.08em;margin:0 0 .5rem .25rem;">Temas</p>',
            unsafe_allow_html=True,
        )
        for ico, tema in [
            ("📦", "Ventas y comprobantes"),
            ("🏷️", "Productos e inventario"),
            ("👤", "Usuarios y permisos"),
            ("💰", "Caja y egresos"),
            ("🛒", "Compras y proveedores"),
            ("📊", "Reportes"),
            ("⚙️", "Configuración"),
        ]:
            st.markdown(
                f'<p style="font-size:0.8rem;padding:2px 4px;color:#c4c9d8;">{ico} {tema}</p>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<hr style="border:none;border-top:1px solid {_BORDER};margin:.8rem 0;">', unsafe_allow_html=True)
        n_usr = sum(1 for m in st.session_state.invofact_messages if m["role"] == "user")
        st.caption(f"Hilo activo: **{st.session_state.invofact_thread_id[:8]}…**")
        st.caption(f"Consultas en sesión: **{n_usr}**")
        if st.button("🗑️ Nueva conversación", use_container_width=True, key="sb_inv_new"):
            st.session_state.invofact_messages  = []
            st.session_state.invofact_thread_id = nuevo_thread_id()
            st.rerun()

    elif _p == "taxbot":
        st.markdown(
            f'<p style="font-size:0.68rem;color:{_MUTED};text-transform:uppercase;'
            f'letter-spacing:.08em;margin:0 0 .5rem .25rem;">Configuración</p>',
            unsafe_allow_html=True,
        )
        new_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=st.session_state.taxbot_api_key,
            help="Clave de GroqCloud para habilitar la IA",
            key="sb_api_key",
        )
        if new_key != st.session_state.taxbot_api_key:
            st.session_state.taxbot_api_key = new_key
        if st.button("🧹 Limpiar chat", use_container_width=True, key="sb_tax_clear"):
            st.session_state.taxbot_messages = []
            st.rerun()
        st.markdown(f'<hr style="border:none;border-top:1px solid {_BORDER};margin:.8rem 0;">', unsafe_allow_html=True)
        st.warning("⚖️ No solicitamos Clave SOL ni datos privados. Uso académico.")

    # Footer del sidebar
    
    


# ══════════════════════════════════════════════════════════════════════════════
# 7. BARRA DE NAVEGACIÓN SUPERIOR (siempre visible en el área principal)
# ══════════════════════════════════════════════════════════════════════════════
def _topnav() -> None:
    """Barra de navegación siempre presente en el área de contenido."""
    n1, n2, n3, n4 = st.columns([2, 1.3, 1.5, 1.5])
    with n1:
        st.markdown(
            f'<p style="font-size:1.25rem;font-weight:800;margin:0;padding-top:2px;">'
            f'🤖 <span style="color:{_BLUE};">Chat</span>'
            f'<span style="color:{_AMBER};">Hub</span></p>',
            unsafe_allow_html=True,
        )
    _p = st.session_state.page
    with n2:
        style_h = f"background:rgba(59,130,246,.15);border:1px solid {_BLUE};" if _p == "home"     else ""
        style_i = f"background:rgba(59,130,246,.15);border:1px solid {_BLUE};" if _p == "invofact" else ""
        style_t = f"background:rgba(245,158,11,.15);border:1px solid {_AMBER};"if _p == "taxbot"   else ""

        if st.button("🏠 Inicio",           key="tn_home",     use_container_width=True):
            _nav("home");     st.rerun()
    with n3:
        if st.button("🤖 InvoFact Soporte", key="tn_invofact", use_container_width=True):
            _nav("invofact"); st.rerun()
    with n4:
        if st.button("🏛️ TAX-BOT SUNAT",   key="tn_taxbot",   use_container_width=True):
            _nav("taxbot");   st.rerun()

    st.markdown(f'<hr style="border:none;border-top:1px solid {_BORDER};margin:0.5rem 0 1.2rem;">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 8. PÁGINA: HOME
# ══════════════════════════════════════════════════════════════════════════════
def _page_home() -> None:
    _topnav()

    st.markdown(
        f"""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <p style="font-size:2.4rem;font-weight:800;color:{_TEXT};margin:0;line-height:1.1;">
                Elige tu asistente
            </p>
            <p style="font-size:1rem;color:{_MUTED};margin:0.5rem 0 0;">
                Dos especialistas, una sola plataforma
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f"""
            <div class="bot-card">
                <div style="font-size:2.4rem;margin-bottom:.7rem;">🤖</div>
                <p style="font-size:1.15rem;font-weight:700;color:{_BLUE};margin:0 0 .3rem;">InvoFact Soporte</p>
                <p style="font-size:0.8rem;color:{_MUTED};margin:0 0 1rem;">
                    Asistente virtual de ayuda al cliente para el sistema de
                    facturación electrónica InvoFact. Motor NLP/ML local.
                </p>
                <div style="margin-bottom:1rem;">
                    <span class="tag" style="background:#1e3a6e;color:#60a5fa;">📦 Ventas</span>
                    <span class="tag" style="background:#14442a;color:#4ade80;">🏷️ Inventario</span>
                    <span class="tag" style="background:#3d2e0e;color:#fbbf24;">💰 Caja</span>
                    <span class="tag" style="background:#0d3d35;color:#34d399;">📊 Reportes</span>
                    <span class="tag" style="background:#252a3a;color:#94a3b8;">⚙️ Config</span>
                </div>
                <p style="font-size:0.73rem;color:#6b7280;margin:0;">
                    ⚡ Sin conexión a internet requerida
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if st.button("▶  Abrir InvoFact Soporte", key="home_btn_inv", use_container_width=True):
            _nav("invofact")
            st.rerun()

    with col2:
        st.markdown(
            f"""
            <div class="bot-card tax">
                <div style="font-size:2.4rem;margin-bottom:.7rem;">🏛️</div>
                <p style="font-size:1.15rem;font-weight:700;color:{_AMBER};margin:0 0 .3rem;">TAX-BOT SUNAT</p>
                <p style="font-size:0.8rem;color:{_MUTED};margin:0 0 1rem;">
                    Experto en tributación peruana con IA generativa (Groq / Llama 3.1).
                    Normativa SUNAT actualizada al 2024.
                </p>
                <div style="margin-bottom:1rem;">
                    <span class="tag" style="background:#3d2e0e;color:#fbbf24;">📋 NRUS</span>
                    <span class="tag" style="background:#3d2e0e;color:#fbbf24;">📋 RER</span>
                    <span class="tag" style="background:#3d2e0e;color:#fbbf24;">📋 RMT</span>
                    <span class="tag" style="background:#3d2e0e;color:#fbbf24;">📋 RG</span>
                    <span class="tag" style="background:#1e3a6e;color:#93c5fd;">🧾 IGV</span>
                    <span class="tag" style="background:#4a1010;color:#f87171;">⚠️ Sanciones</span>
                </div>
                <p style="font-size:0.73rem;color:#6b7280;margin:0;">
                    🔑 Requiere Groq API Key
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if st.button("▶  Abrir TAX-BOT SUNAT", key="home_btn_tax", use_container_width=True):
            _nav("taxbot")
            st.rerun()

    # Estadísticas
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_BORDER};margin:2rem 0 1rem;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:0.68rem;color:{_MUTED};text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:.8rem;">Estadísticas — InvoFact Soporte</p>',
        unsafe_allow_html=True,
    )
    try:
        s = stats_historial()
        c1, c2, c3 = st.columns(3)
        c1.metric("Consultas totales",  s["total_mensajes"])
        c2.metric("Sesiones únicas",    s["total_hilos"])
        c3.metric("Confianza promedio", f"{s['confianza_prom']}%")
    except Exception:
        st.caption("Estadísticas no disponibles aún.")


# ══════════════════════════════════════════════════════════════════════════════
# 9. PÁGINA: INVOFACT SOPORTE
# ══════════════════════════════════════════════════════════════════════════════
def _page_invofact() -> None:
    _topnav()

    # Sub-tabs Chat | Historial
    tab_chat, tab_hist = st.tabs(["💬  Chat", "📋  Historial de conversaciones"])

    # ── Chat ──────────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;
                 padding:0 0 .6rem;border-bottom:2px solid {_BLUE};margin-bottom:1rem;">
                <span style="font-size:1.6rem;">🤖</span>
                <div>
                    <p style="font-size:1.15rem;font-weight:700;color:{_BLUE};margin:0;">InvoFact Soporte</p>
                    <p style="font-size:0.75rem;color:{_MUTED};margin:0;">Asistente virtual de ayuda al cliente</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for msg in st.session_state.invofact_messages:
            avatar = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if pregunta := st.chat_input("Escribe tu consulta sobre InvoFact…", key="ci_inv"):
            st.session_state.invofact_messages.append({"role": "user", "content": pregunta})
            with st.chat_message("user", avatar="👤"):
                st.markdown(pregunta)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner(""):
                    resultado = motor.responder(pregunta)
                st.markdown(resultado["texto"])
            st.session_state.invofact_messages.append(
                {"role": "assistant", "content": resultado["texto"]}
            )
            if resultado["categoria"] not in ("Saludo", "Despedida", "Confirmacion"):
                guardar_mensaje(
                    thread_id        = st.session_state.invofact_thread_id,
                    pregunta_usuario = pregunta,
                    pregunta_dataset = resultado.get("pregunta_dataset"),
                    categoria        = resultado["categoria"],
                    confianza        = resultado["score"],
                    respuesta        = resultado["texto"],
                )
            st.rerun()

    # ── Historial ─────────────────────────────────────────────────────────────
    with tab_hist:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;
                 padding:0 0 .6rem;border-bottom:2px solid {_BLUE};margin-bottom:1rem;">
                <span style="font-size:1.6rem;">📋</span>
                <div>
                    <p style="font-size:1.15rem;font-weight:700;color:{_BLUE};margin:0;">Historial</p>
                    <p style="font-size:0.75rem;color:{_MUTED};margin:0;">Registro de consultas y métricas</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s = stats_historial()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total consultas",    s["total_mensajes"])
        c2.metric("Hilos únicos",       s["total_hilos"])
        c3.metric("Confianza promedio", f"{s['confianza_prom']}%")

        registros = obtener_historial(limit=200)
        if not registros:
            st.info("No hay conversaciones registradas aún.")
        else:
            df = pd.DataFrame(registros).rename(columns={
                "id": "ID", "thread_id": "Hilo", "timestamp": "Fecha/Hora",
                "pregunta_usuario": "Pregunta del usuario",
                "pregunta_dataset": "Pregunta en dataset",
                "categoria": "Categoría", "confianza": "Confianza",
            })
            df["Confianza"] = df["Confianza"].apply(lambda x: f"{x:.0%}" if x else "—")
            st.dataframe(df, use_container_width=True, hide_index=True,
                column_config={
                    "ID":                   st.column_config.NumberColumn(width="small"),
                    "Hilo":                 st.column_config.TextColumn(width="small"),
                    "Fecha/Hora":           st.column_config.TextColumn(width="medium"),
                    "Pregunta del usuario": st.column_config.TextColumn(width="large"),
                    "Pregunta en dataset":  st.column_config.TextColumn(width="large"),
                    "Categoría":            st.column_config.TextColumn(width="small"),
                    "Confianza":            st.column_config.TextColumn(width="small"),
                })
            col_exp, col_del, _ = st.columns([1, 1, 3])
            with col_exp:
                st.download_button("⬇️ Exportar CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="historial_invofact.csv", mime="text/csv",
                    use_container_width=True)
            with col_del:
                if st.button("🗑️ Limpiar historial", use_container_width=True, key="hist_del"):
                    limpiar_historial(); st.success("Historial eliminado."); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 10. PÁGINA: TAX-BOT SUNAT
# ══════════════════════════════════════════════════════════════════════════════
_SALUDOS  = {"hola","buenas","buenos dias","buenas tardes","buenas noches",
             "ayuda","tengo dudas","ayudame","hola bot"}
_PRIV_KW  = ["clave sol","mi ruc","mi deuda"]
_TAX_AV   = {"user": "👤", "assistant": "🏛️"}


def _page_taxbot() -> None:
    _topnav()

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;
             padding:0 0 .6rem;border-bottom:2px solid {_AMBER};margin-bottom:1rem;">
            <span style="font-size:1.6rem;">🏛️</span>
            <div>
                <p style="font-size:1.15rem;font-weight:700;color:{_AMBER};margin:0;">TAX-BOT SUNAT</p>
                <p style="font-size:0.75rem;color:{_MUTED};margin:0;">
                    Asistente tributario inteligente para MYPES y emprendedores
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Botones rápidos — sólo en el primer turno
    pregunta_rapida: str | None = None
    if len(st.session_state.taxbot_messages) == 1:
        st.markdown(
            f'<p style="font-size:0.8rem;color:{_MUTED};margin-bottom:.4rem;">💡 Consultas frecuentes:</p>',
            unsafe_allow_html=True,
        )
        q1, q2 = st.columns(2)
        with q1:
            if st.button("📊 Límites del Régimen MYPE",  use_container_width=True, key="tq1"):
                pregunta_rapida = "¿Cuáles son los límites de ingresos para el Régimen MYPE Tributario (RMT)?"
            if st.button("🏪 Cuánto se paga en el NRUS", use_container_width=True, key="tq2"):
                pregunta_rapida = "¿Cuánto se paga mensualmente en el Nuevo RUS y cuáles son sus categorías?"
        with q2:
            if st.button("🧾 Monto para DNI en boleta",  use_container_width=True, key="tq3"):
                pregunta_rapida = "¿A partir de qué monto es obligatorio pedir el DNI en una boleta de venta?"
            if st.button("📚 Libros en Régimen General", use_container_width=True, key="tq4"):
                pregunta_rapida = "¿Qué libros contables debo llevar si estoy en el Régimen General?"

    # Historial
    for msg in st.session_state.taxbot_messages:
        with st.chat_message(msg["role"], avatar=_TAX_AV[msg["role"]]):
            st.markdown(msg["content"])

    # Input
    prompt = st.chat_input("Escribe tu consulta tributaria aquí…", key="ci_tax")
    if pregunta_rapida:
        prompt = pregunta_rapida
    if not prompt:
        return

    # Procesar
    st.session_state.taxbot_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=_TAX_AV["user"]):
        st.markdown(prompt)

    groq_key   = st.session_state.taxbot_api_key
    p_lower    = prompt.lower().strip()
    respuesta: str

    if not groq_key:
        respuesta = (
            "⚠️ **API Key requerida:** Ingresa tu Groq API Key en el "
            "panel lateral izquierdo para activar la IA generativa."
        )
    elif p_lower in _SALUDOS:
        respuesta = (
            "¡Hola! Soy **TAX-BOT SUNAT**, tu asistente tributario. "
            "¿Tienes consultas sobre regímenes SUNAT, facturación electrónica, "
            "IGV o libros contables?"
        )
    elif any(k in p_lower for k in _PRIV_KW):
        respuesta = (
            "⚠️ **Privacidad:** No accedo a tu Clave SOL, RUC ni datos privados. "
            "Para revisar tu situación tributaria personal visita **sunat.gob.pe**."
        )
    else:
        # RAG
        contexto = ""
        fuente   = ""
        resultados = []
        if vdb:
            _REGIMENES_KW = ["NRUS", "Nuevo RUS", "RER", "RMT",
                             "Régimen General", "Régimen Especial", "Régimen MYPE"]
            _historial_txt = " ".join(
                m["content"] for m in st.session_state.taxbot_messages[-6:]
                if m["role"] in ("user", "assistant")
            )
            _regimen_det = next(
                (r for r in _REGIMENES_KW if r.lower() in _historial_txt.lower()), ""
            )
            _query_enr = f"{prompt} {_regimen_det}".strip()
            resultados = vdb.similarity_search(_query_enr, k=8)
            if resultados:
                contexto = "\n".join(d.page_content for d in resultados)
                fuente   = resultados[0].metadata.get("fuente", "SUNAT")

        hist = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.taxbot_messages[-(6 * 2):-1]
            if m["role"] in ("user", "assistant")
        ]

        sys_prompt = f"""Eres TAX-BOT SUNAT, asistente experto en tributación peruana.
Año de referencia: 2026. UIT vigente: S/ 5,500 (D.S. N° 301-2025-EF).
Cuando el contexto exprese límites en UIT, recalcúlalos multiplicando por 5,500.

CONTEXTO DE LA BASE DE CONOCIMIENTO:
{contexto}

REGLAS:
1. Responde SOLO usando el CONTEXTO anterior.
2. Si el contexto no cubre la pregunta: "No tengo esa información en mi base certificada. Consulta en sunat.gob.pe"
3. Usa viñetas (•) para listas y negritas para cifras clave.
4. Considera el historial para dar continuidad.
5. No inventes tasas, montos ni normas."""

        try:
            client = Groq(api_key=groq_key)
            comp   = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    *hist,
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-8b-instant",
                temperature=0.2,
                stream=False,
            )
            rx = comp.choices[0].message.content
            _FRASE_NO_SABE = "no tengo ese dato en mi base certificada"
            _bot_no_sabe = _FRASE_NO_SABE in rx.lower() or "no tengo esa información" in rx.lower()
            if contexto and not _bot_no_sabe:
                _fuentes_unicas = list(dict.fromkeys(
                    d.metadata.get("fuente", "SUNAT") for d in resultados
                ))
                respuesta = f"{rx}\n\n---\n*📜 Fuentes legales consultadas: **{' | '.join(_fuentes_unicas)}***"
            else:
                respuesta = rx
        except Exception as e:
            respuesta = f"❌ Error de conexión con Groq: {e}"

    with st.chat_message("assistant", avatar=_TAX_AV["assistant"]):
        st.markdown(respuesta)

    st.session_state.taxbot_messages.append({"role": "assistant", "content": respuesta})
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 11. ROUTER
# ══════════════════════════════════════════════════════════════════════════════
_page_map = {
    "home":     _page_home,
    "invofact": _page_invofact,
    "taxbot":   _page_taxbot,
}

_page_map.get(st.session_state.page, _page_home)()
