r"""
app.py — Interfaz Web del Chatbot InvoFact
==========================================
Punto de entrada de Streamlit. Ejecutar con:
    .\.venv\Scripts\streamlit.exe run app.py

Estructura del proyecto:
    core/engine.py   → Motor NLP/NLU/NLG  (lógica pura, sin UI)
    core/db.py       → Historial SQLite
    ui/styles.py     → Tema visual InvoFact
    ui/components.py → Componentes HTML reutilizables
    chatbot_engine.py → ChatbotEngine (importado desde core)
"""

import pandas as pd
import streamlit as st

from chatbot_engine import ChatbotEngine
from core.db import init_db, nuevo_thread_id, guardar_mensaje, obtener_historial, limpiar_historial, stats_historial
from ui.styles import MAIN_CSS, PRIMARY_BLUE, ACCENT_GREEN, TEXT_MUTED, BORDER_COLOR, BG_CARD
from ui.components import badge_html, header_html

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="InvoFact Soporte",
    page_icon="https://invofact.com/favicon.ico",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(MAIN_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ══════════════════════════════════════════════════════════════

init_db()

@st.cache_resource(show_spinner="Iniciando asistente...")
def cargar_motor() -> ChatbotEngine:
    return ChatbotEngine()

motor = cargar_motor()

# ID de hilo único por sesión
if "thread_id" not in st.session_state:
    st.session_state.thread_id = nuevo_thread_id()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":    "assistant",
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

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-logo">
            <img src="https://invofact.com/favicon.ico" width="32" style="border-radius:6px;">
            <div>
                <p class="sidebar-title">InvoFact</p>
                <p class="sidebar-subtitle">Centro de soporte</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-title">Temas disponibles</p>', unsafe_allow_html=True)
    temas = [
        ("📦", "Ventas y comprobantes"),
        ("🏷️", "Productos e inventario"),
        ("👤", "Usuarios y permisos"),
        ("💰", "Caja y egresos"),
        ("🛒", "Compras y proveedores"),
        ("📊", "Reportes"),
        ("⚙️", "Configuración"),
    ]
    for icon, tema in temas:
        st.markdown(f'<p class="topic-item">{icon} {tema}</p>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    n_msgs   = sum(1 for m in st.session_state.messages if m["role"] == "user")
    thread   = st.session_state.thread_id
    st.caption(f"Hilo: **{thread}**")
    st.caption(f"Preguntas en esta sesión: **{n_msgs}**")

    st.markdown("")
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = nuevo_thread_id()
        st.rerun()

    st.markdown("")

# ══════════════════════════════════════════════════════════════
# TABS: CHAT | HISTORIAL
# ══════════════════════════════════════════════════════════════

# Botón para mostrar el sidebar cuando está oculto

tab_chat, tab_historial = st.tabs(["💬  Chat", "📋  Historial de conversaciones"])

# ──────────────────────────────────────────────────────────────
# TAB 1: CHAT
# ──────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown(
        header_html("InvoFact Soporte", "Asistente virtual de ayuda al cliente", "🤖"),
        unsafe_allow_html=True,
    )

    # Renderizar historial
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input
    if pregunta := st.chat_input("Escribí tu consulta sobre InvoFact..."):
        # Mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta)

        # Respuesta del asistente
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(""):
                resultado = motor.responder(pregunta)

            st.markdown(resultado["texto"])

        # Guardar en sesión
        st.session_state.messages.append({
            "role":    "assistant",
            "content": resultado["texto"],
        })

        # Guardar en SQLite (solo mensajes con intención real)
        if resultado["categoria"] not in ("Saludo", "Despedida", "Confirmacion"):
            guardar_mensaje(
                thread_id        = st.session_state.thread_id,
                pregunta_usuario = pregunta,
                pregunta_dataset = resultado.get("pregunta_dataset"),
                categoria        = resultado["categoria"],
                confianza        = resultado["score"],
                respuesta        = resultado["texto"],
            )

        st.rerun()

# ──────────────────────────────────────────────────────────────
# TAB 2: HISTORIAL
# ──────────────────────────────────────────────────────────────
with tab_historial:
    st.markdown(
        header_html("Historial de conversaciones", "Registro de consultas y métricas de confianza", "📋"),
        unsafe_allow_html=True,
    )

    stats = stats_historial()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total consultas",  stats["total_mensajes"])
    c2.metric("Hilos únicos",     stats["total_hilos"])
    c3.metric("Confianza promedio", f"{stats['confianza_prom']}%")

    st.markdown("")

    registros = obtener_historial(limit=200)

    if not registros:
        st.info("No hay conversaciones registradas todavía. ¡Empieza a chatear!")
    else:
        df = pd.DataFrame(registros)
        df = df.rename(columns={
            "id":               "ID",
            "thread_id":        "Hilo",
            "timestamp":        "Fecha/Hora",
            "pregunta_usuario":  "Pregunta del usuario",
            "pregunta_dataset":  "Pregunta en dataset",
            "categoria":        "Categoría",
            "confianza":        "Confianza",
        })
        df["Confianza"] = df["Confianza"].apply(
            lambda x: f"{x:.0%}" if x is not None else "—"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID":                   st.column_config.NumberColumn(width="small"),
                "Hilo":                 st.column_config.TextColumn(width="small"),
                "Fecha/Hora":           st.column_config.TextColumn(width="medium"),
                "Pregunta del usuario": st.column_config.TextColumn(width="large"),
                "Pregunta en dataset":  st.column_config.TextColumn(width="large"),
                "Categoría":            st.column_config.TextColumn(width="small"),
                "Confianza":            st.column_config.TextColumn(width="small"),
            },
        )

        st.markdown("")
        col_export, col_clear, _ = st.columns([1, 1, 3])

        with col_export:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Exportar CSV",
                data=csv,
                file_name="historial_invofact.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_clear:
            if st.button("🗑️ Limpiar historial", use_container_width=True):
                limpiar_historial()
                st.success("Historial eliminado.")
                st.rerun()

