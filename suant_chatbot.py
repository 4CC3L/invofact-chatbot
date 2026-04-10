import streamlit as st
import json
import uuid
import os
from groq import Groq
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="TAX-BOT PUCP", page_icon="🤖", layout="centered")

# ==========================================
# 2. CACHÉ DE LA BASE DE DATOS (RAG)
#    Chunking completo del dataset +
#    modelo de embeddings multilingüe
# ==========================================
@st.cache_resource
def cargar_base_conocimiento():
    try:
        with open('sunat_regimenes_tributarios.json', 'r', encoding='utf-8') as f:
            data_maestra = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo sunat_regimenes_tributarios.json")
        return None

    documentos = []

    # ── POR RÉGIMEN ───────────────────────────────────────────────────────────
    for reg in data_maestra.get("regimenes", []):
        nombre  = reg["nombre_completo"]
        acronimo = reg["acronimo"]

        # Descripción + requisitos de acogimiento
        req_obj = reg.get("quien_puede_acogerse", {})
        req_desc = req_obj.get("descripcion", "Aplica a todos sin límite.")
        req_lista = " | ".join(req_obj.get("requisitos", []))
        documentos.append(Document(
            page_content=(
                f"Régimen: {nombre} ({acronimo}). {reg['descripcion']} "
                f"Requisitos para acogerse: {req_desc} {req_lista}"
            ),
            metadata={"fuente": f"Normas {acronimo}", "regimen": acronimo}
        ))

        # Quién NO puede acogerse
        excluidos = reg.get("quien_NO_puede_acogerse", [])
        if excluidos:
            excluidos_txt = " | ".join(
                e if isinstance(e, str) else e.get("actividad", str(e))
                for e in excluidos
            )
            documentos.append(Document(
                page_content=f"Actividades o sujetos EXCLUIDOS del {acronimo}: {excluidos_txt}",
                metadata={"fuente": f"Exclusiones {acronimo}", "regimen": acronimo}
            ))

        # Tributos
        tributos = reg.get("tributos_que_paga", {})
        if "impuesto_a_la_renta" in tributos:
            ir = tributos["impuesto_a_la_renta"]
            ir_txt = ir.get("tasa", "") + ". " + ir.get("descripcion", "")
            documentos.append(Document(
                page_content=f"Impuesto a la Renta (IR) en el {acronimo}: {ir_txt.strip()}",
                metadata={"fuente": f"IR {acronimo}", "regimen": acronimo}
            ))
        if "igv" in tributos:
            igv = tributos["igv"]
            igv_txt = igv.get("descripcion", "Paga IGV 18%.")
            documentos.append(Document(
                page_content=f"IGV en el {acronimo}: {igv_txt}",
                metadata={"fuente": f"IGV {acronimo}", "regimen": acronimo}
            ))
        if "cuota_nrus" in tributos:
            cuota = tributos["cuota_nrus"]
            documentos.append(Document(
                page_content=f"Cuota mensual del NRUS: {cuota.get('descripcion', '')}",
                metadata={"fuente": "Cuota NRUS", "regimen": "NRUS"}
            ))

        # Categorías NRUS
        for cat in reg.get("categorias", []):
            cat_id = cat.get("categoria", "")
            cuota = cat.get("cuota_mensual_soles", 0)
            lim_ing = cat.get("limite_ingresos_brutos_mensuales_soles", "")
            lim_adq = cat.get("limite_adquisiciones_mensuales_soles", "")
            desc = cat.get("descripcion", f"Categoría {cat_id}")
            documentos.append(Document(
                page_content=(
                    f"NRUS {desc}: cuota mensual S/ {cuota}. "
                    f"Límite ingresos brutos mensuales: S/ {lim_ing}. "
                    f"Límite adquisiciones mensuales: S/ {lim_adq}."
                ),
                metadata={"fuente": f"Categorías NRUS", "regimen": "NRUS"}
            ))

        # Libros contables
        libros_obj = reg.get("libros_contables", {})
        tramos = libros_obj.get("tramos", [])
        if tramos:
            for tramo in tramos:
                libros_txt = " | ".join(tramo.get("libros", []))
                soles = tramo.get("soles_2024", tramo.get("soles_2024_hasta", ""))
                documentos.append(Document(
                    page_content=(
                        f"Libros contables en el {acronimo} — {tramo.get('nivel', '')}: "
                        f"{libros_txt}. Referencia en soles 2024: S/ {soles}."
                    ),
                    metadata={"fuente": f"Libros {acronimo}", "regimen": acronimo}
                ))
        else:
            libros_desc = libros_obj.get("descripcion", "")
            
            # Extraemos inteligentemente el nombre del libro si es un diccionario
            lista_libros = libros_obj.get("libros_requeridos", [])
            libros_req = " | ".join([
                libro.get("libro", "") if isinstance(libro, dict) else str(libro) 
                for libro in lista_libros
            ])
            
            if libros_desc or libros_req:
                documentos.append(Document(
                    page_content=f"Libros contables en el {acronimo}: {libros_desc} {libros_req}".strip(),
                    metadata={"fuente": f"Libros {acronimo}", "regimen": acronimo}
                ))

        # Comprobantes
        comp_puede = reg.get("comprobantes_que_puede_emitir", [])
        comp_no    = reg.get("comprobantes_que_NO_puede_emitir", [])
        if comp_puede:
            documentos.append(Document(
                page_content=(
                    f"Comprobantes que SÍ puede emitir en {acronimo}: "
                    f"{' | '.join(comp_puede)}."
                ),
                metadata={"fuente": f"Comprobantes {acronimo}", "regimen": acronimo}
            ))
        if comp_no:
            documentos.append(Document(
                page_content=(
                    f"Comprobantes que NO puede emitir en {acronimo}: "
                    f"{' | '.join(comp_no)}."
                ),
                metadata={"fuente": f"Comprobantes {acronimo}", "regimen": acronimo}
            ))

        # Declaraciones
        decl = reg.get("declaraciones", {})
        decl_mensual = decl.get("mensual", {})
        decl_anual   = decl.get("anual", {})
        decl_txt = ""
        if decl_mensual:
            decl_txt += f"Declaración mensual: {decl_mensual.get('descripcion', '')} "
        if decl_anual:
            decl_txt += f"Declaración anual: {decl_anual.get('descripcion', '')}"
        if decl_txt.strip():
            documentos.append(Document(
                page_content=f"Declaraciones en el {acronimo}: {decl_txt.strip()}",
                metadata={"fuente": f"Declaraciones {acronimo}", "regimen": acronimo}
            ))

        # Cambio de régimen
        cambio = reg.get("cambio_de_regimen", {})
        if cambio:
            puede_cambiar_a = " | ".join(cambio.get("puede_cambiar_a", []))
            exclusion = cambio.get("exclusion_automatica", "")
            cambio_txt = f"Puede cambiar a: {puede_cambiar_a}. " if puede_cambiar_a else ""
            if exclusion:
                cambio_txt += f"Exclusión automática: {exclusion}"
            if cambio_txt.strip():
                documentos.append(Document(
                    page_content=f"Cambio de régimen desde el {acronimo}: {cambio_txt.strip()}",
                    metadata={"fuente": f"Cambio de Régimen {acronimo}", "regimen": acronimo}
                ))

        # Casos prácticos
        for caso in reg.get("casos_practicos", []):
            desc = caso.get("descripcion", "")
            detalles = []
            if "ingresos_mensuales" in caso:
                detalles.append(f"Ingresos mensuales: S/ {caso['ingresos_mensuales']}")
            if "ingresos_netos_anuales" in caso:
                detalles.append(f"Ingresos anuales: S/ {caso['ingresos_netos_anuales']}")
            if "cuota_a_pagar" in caso:
                detalles.append(f"Cuota a pagar: S/ {caso['cuota_a_pagar']}")
            if "total_impuestos_mes" in caso:
                detalles.append(f"Total impuestos mes: S/ {caso['total_impuestos_mes']}")
            if "renta_neta_imponible" in caso:
                detalles.append(f"Renta neta imponible: S/ {caso['renta_neta_imponible']}")
            if "alerta" in caso:
                detalles.append(f"ALERTA: {caso['alerta']}")
            detalles_txt = " | ".join(detalles)
            documentos.append(Document(
                page_content=f"Caso práctico {acronimo}: {desc}. {detalles_txt}",
                metadata={"fuente": f"Caso {acronimo}", "regimen": acronimo}
            ))

        # Preguntas frecuentes
        for faq in reg.get("preguntas_frecuentes", []):
            documentos.append(Document(
                page_content=f"FAQ {acronimo} — {faq.get('pregunta', '')}: {faq.get('respuesta', '')}",
                metadata={"fuente": f"FAQ {acronimo}", "regimen": acronimo}
            ))

        # Ventajas y desventajas
        ventajas    = reg.get("ventajas", [])
        desventajas = reg.get("desventajas", [])
        if ventajas:
            documentos.append(Document(
                page_content=f"Ventajas del {acronimo}: {' | '.join(ventajas)}",
                metadata={"fuente": f"Ventajas {acronimo}", "regimen": acronimo}
            ))
        if desventajas:
            documentos.append(Document(
                page_content=f"Desventajas del {acronimo}: {' | '.join(desventajas)}",
                metadata={"fuente": f"Desventajas {acronimo}", "regimen": acronimo}
            ))

        # Deducciones / Gastos deducibles (RMT / RG)
        ded = reg.get("deducciones_y_gastos", reg.get("gastos_deducibles", {}))
        if ded:
            lista_gastos_cruda = ded.get("gastos_deducibles_principales", [])
            
            # Extraemos inteligentemente si es un texto simple (RMT) o un diccionario (RG)
            gastos_lista = " | ".join([
                g.get("gasto", "") if isinstance(g, dict) else str(g)
                for g in lista_gastos_cruda
            ])
            
            principio = ded.get("principio_causalidad", ded.get("principio", ""))
            if gastos_lista or principio:
                documentos.append(Document(
                    page_content=(
                        f"Gastos deducibles en el {acronimo}: {principio} "
                        f"Principales gastos: {gastos_lista}"
                    ).strip(),
                    metadata={"fuente": f"Gastos {acronimo}", "regimen": acronimo}
                ))

    # ── SECCIONES TRANSVERSALES ───────────────────────────────────────────────

    # Tabla comparativa
    tc = data_maestra.get("tabla_comparativa", {})
    for fila in tc.get("filas", []):
        criterio = fila.get("criterio", "")
        linea = (
            f"Comparativa tributaria — {criterio}: "
            f"NRUS={fila.get('nuevo_rus', '')} | "
            f"RER={fila.get('rer', '')} | "
            f"RMT={fila.get('rmt', '')} | "
            f"RG={fila.get('regimen_general', '')}"
        )
        documentos.append(Document(
            page_content=linea,
            metadata={"fuente": "Tabla Comparativa SUNAT", "regimen": "todos"}
        ))

    # IGV General
    igv_g = data_maestra.get("igv_general", {})
    if igv_g:
        ops_grav = " | ".join(igv_g.get("operaciones_gravadas", []))
        ops_exon = " | ".join(igv_g.get("operaciones_exoneradas_principales", []))
        documentos.append(Document(
            page_content=(
                f"IGV general — tasa: {igv_g.get('tasa', '')}. "
                f"Operaciones gravadas: {ops_grav}. "
                f"Operaciones exoneradas: {ops_exon}."
            ),
            metadata={"fuente": "IGV General SUNAT", "regimen": "todos"}
        ))
        cred = igv_g.get("credito_fiscal", {})
        req_sust = " | ".join(cred.get("requisitos_sustanciales", []))
        req_form = " | ".join(cred.get("requisitos_formales", []))
        documentos.append(Document(
            page_content=(
                f"Crédito fiscal IGV — requisitos sustanciales: {req_sust}. "
                f"Requisitos formales: {req_form}."
            ),
            metadata={"fuente": "Crédito Fiscal IGV", "regimen": "todos"}
        ))
        exp = igv_g.get("exportaciones", {})
        documentos.append(Document(
            page_content=f"IGV en exportaciones: {exp.get('descripcion', '')}",
            metadata={"fuente": "IGV Exportaciones", "regimen": "todos"}
        ))

    # Infracciones y sanciones
    inf = data_maestra.get("infracciones_y_sanciones", {})
    for infrac in inf.get("infracciones_principales", []):
        documentos.append(Document(
            page_content=(
                f"Infracción tributaria — {infrac.get('articulo', '')}: "
                f"{infrac.get('infraccion', '')}. "
                f"Sanción Tabla I: {infrac.get('sancion_tabla_I', '')} | "
                f"Tabla II: {infrac.get('sancion_tabla_II', '')} | "
                f"Tabla III: {infrac.get('sancion_tabla_III', '')}. "
                f"Gradualidad: {infrac.get('gradualidad_maxima', 'Ver régimen de gradualidad.')}"
            ),
            metadata={"fuente": "Infracciones SUNAT", "regimen": "todos"}
        ))
    tim = inf.get("tim", {})
    if tim:
        documentos.append(Document(
            page_content=(
                f"Tasa de Interés Moratorio (TIM): {tim.get('tasa_mensual', '')} mensual. "
                f"{tim.get('descripcion', '')}"
            ),
            metadata={"fuente": "TIM SUNAT", "regimen": "todos"}
        ))
    grad = inf.get("regimen_de_gradualidad", {})
    if grad:
        documentos.append(Document(
            page_content=f"Régimen de gradualidad de infracciones: {grad.get('descripcion', '')}",
            metadata={"fuente": "Gradualidad SUNAT", "regimen": "todos"}
        ))

    # Obligaciones tributarias comunes
    oc = data_maestra.get("obligaciones_tributarias_comunes", {})
    for ob in oc.get("obligaciones", []):
        tasas = ob.get("tasa", ob.get("tasas", ""))
        documentos.append(Document(
            page_content=(
                f"Obligación tributaria común — {ob.get('obligacion', '')}: "
                f"Aplica a: {ob.get('aplica_a', '')}. "
                f"Tasa: {tasas}. "
                f"{ob.get('descripcion', ob.get('nota', ''))}"
            ).strip().rstrip(".") + ".",
            metadata={"fuente": "Obligaciones Comunes SUNAT", "regimen": "todos"}
        ))

    # UIT histórico
    uit = data_maestra.get("uit_historico", {})
    valores_txt = " | ".join(
        f"{v['año']}: S/ {v['uit_soles']}" for v in uit.get("valores", [])
    )
    documentos.append(Document(
        page_content=f"Valor histórico de la UIT: {valores_txt}.",
        metadata={"fuente": "UIT Histórico SUNAT", "regimen": "todos"}
    ))

# ── REGLAS ESPECÍFICAS ──────────────────────────────────────────────────────────
    documentos.append(Document(
        page_content="Boletas de Venta - Monto para DNI: Según la normativa vigente de SUNAT, es obligatorio consignar los datos de identificación (Nombres, Apellidos y DNI) del cliente en una Boleta de Venta Electrónica cuando el importe total de la operación supere los S/ 700.00 (Setecientos y 00/100 Soles).", 
        metadata={"fuente": "Resolución de Superintendencia N° 123-2022/SUNAT", "regimen": "todos"}
    ))
    documentos.append(Document(
        page_content="Anulación de Facturas: Para anular una Factura Electrónica mediante una Nota de Crédito, el plazo máximo excepcional es hasta el séptimo (7mo) día calendario contado desde el día siguiente de la fecha de emisión del comprobante.", 
        metadata={"fuente": "Resolución de Superintendencia N° 193-2020/SUNAT", "regimen": "todos"}
    ))
    # ── VECTOR DB ─────────────────────────────────────────────────────────────
    # Modelo multilingüe optimizado para español
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma.from_documents(
        documentos,
        embeddings,
        collection_name=f"taxbot_{uuid.uuid4().hex}"
    )
    return vector_db

vector_db = cargar_base_conocimiento()

# ==========================================
# 3. BARRA LATERAL
# ==========================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/PUCP_logo_con_nombre.svg/1024px-PUCP_logo_con_nombre.svg.png",
        width=200
    )
    st.title("⚙️ Panel de Control")
    st.write("Autenticación requerida para consultar la base legal.")

    groq_api_key = st.text_input(
        "🔑 Groq API Key", type="password",
        help="Ingresa tu clave de GroqCloud para activar la IA."
    )

    st.divider()
    st.warning(
        "⚖️ **Disclaimer:** Este bot no solicita tu Clave SOL ni accede "
        "a tu información tributaria privada. Uso netamente académico."
    )

    if st.button("🧹 Limpiar Historial de Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("🚀 Proyecto Final - Diseño de Chatbots")

# ==========================================
# 4. ENCABEZADO Y BOTONES DE INICIO RÁPIDO
# ==========================================
st.title("🏛️ TAX-BOT PUCP")
st.markdown("### Asistente Tributario Inteligente para MYPES y Emprendedores")

avatares = {"user": "👤", "assistant": "🤖"}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy TAX-BOT. Estoy entrenado con el marco normativo "
                "de SUNAT. ¿En qué te puedo ayudar hoy?"
            )
        }
    ]

# Botones de inicio rápido (solo en el primer mensaje)
pregunta_rapida = None  # CORREGIDO: siempre definida antes del bloque de lógica
if len(st.session_state.messages) == 1:
    st.write("💡 **Consultas frecuentes:**")
    col1, col2 = st.columns(2)
    with col1:
        btn_mype = st.button("📊 Límites del Régimen MYPE", use_container_width=True)
        btn_rus  = st.button("🏪 Cuánto se paga en el NRUS", use_container_width=True)
    with col2:
        btn_dni    = st.button("🧾 Monto para pedir DNI en boleta", use_container_width=True)
        btn_libros = st.button("📚 Libros para Régimen General", use_container_width=True)

    if btn_mype:
        pregunta_rapida = "¿Cuáles son los límites de ingresos para estar en el Régimen MYPE Tributario (RMT)?"
    if btn_rus:
        pregunta_rapida = "¿Cuánto se paga mensualmente en el Nuevo RUS y cuáles son sus categorías?"
    if btn_dni:
        pregunta_rapida = "Sobre las boletas de venta: ¿A partir de qué monto es obligatorio pedir el DNI?"
    if btn_libros:
        pregunta_rapida = "¿Qué libros contables estoy obligado a llevar si estoy en el Régimen General?"

# ==========================================
# 5. HISTORIAL + LÓGICA DEL CHAT
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatares[message["role"]]):
        st.markdown(message["content"])

# Entrada: teclado o botón rápido
prompt = st.chat_input("Escribe tu consulta tributaria aquí...")
if pregunta_rapida:       
    prompt = pregunta_rapida

if prompt:
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatares["user"]):
        st.markdown(prompt)

    if not groq_api_key:
        st.error("⚠️ Falta la API Key: Por favor, ingresa tu clave en el panel lateral.")
        st.stop()

    client = Groq(api_key=groq_api_key)

    with st.chat_message("assistant", avatar=avatares["assistant"]):
        mensaje_placeholder = st.empty()
        prompt_lower = prompt.lower().strip()

        # ── NLU: Saludos (bypass) ─────────────────────────────────────────────
        saludos = [
            "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
            "ayuda", "tengo dudas", "ayudame", "hola bot"
        ]
        if prompt_lower in saludos:
            respuesta_final = (
                "¡Hola! Qué gusto saludarte. Soy TAX-BOT PUCP, tu asistente "
                "tributario. ¿Tienes alguna consulta sobre regímenes de SUNAT, "
                "facturación electrónica, IGV o libros contables?"
            )
            mensaje_placeholder.markdown(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
            st.stop()

        # ── Filtro de privacidad ──────────────────────────────────────────────
        if any(p in prompt_lower for p in ["clave sol", "mi ruc", "mi deuda"]):
            respuesta_final = (
                "⚠️ **Protección de Privacidad:** Como asistente virtual, "
                "no accedo a tu Clave SOL ni a datos privados. Por favor, "
                "revisa tus datos en el portal oficial de SUNAT Operaciones en Línea."
            )
            mensaje_placeholder.info(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
            st.stop()

        # ── Búsqueda RAG ──────────────────────────────────────────────────────
        # CORREGIDO: k=5 para más contexto
        contexto = ""
        fuente = ""
        if vector_db:
            resultados = vector_db.similarity_search(prompt, k=5)
            if resultados:
                contexto = "\n".join([doc.page_content for doc in resultados])
                fuente = resultados[0].metadata.get("fuente", "SUNAT")

        # ── Historial de conversación (últimos 6 turnos) ──────────────────────
        # CORREGIDO: memoria conversacional real enviada al LLM
        MAX_TURNOS = 6
        historial_reciente = st.session_state.messages[-(MAX_TURNOS * 2):-1]
        mensajes_historial = [
            {"role": m["role"], "content": m["content"]}
            for m in historial_reciente
            if m["role"] in ("user", "assistant")
        ]

        # ── NLG con Groq ──────────────────────────────────────────────────────
        prompt_sistema = f"""
Eres TAX-BOT PUCP, un asistente virtual empático y experto en tributación peruana (SUNAT).
Año de referencia: 2024. UIT vigente: S/ 5,150.

CONTEXTO RECUPERADO DE LA BASE DE CONOCIMIENTO:
{contexto}

REGLAS DE RESPUESTA:
1. Responde SOLO basándote en el CONTEXTO RECUPERADO anterior.
2. Si el contexto no cubre la pregunta del usuario, responde amablemente:
   "La normativa tributaria es amplia y no tengo el dato exacto de esa consulta en mi base certificada. ¿Podrías detallar tu pregunta o consultar directamente en sunat.gob.pe?"
3. Usa viñetas (•) para listas y negritas para cifras clave.
4. Si el usuario hace referencia a una pregunta anterior, considera el historial del chat para dar continuidad.
5. No inventes tasas, montos ni normas que no estén en el contexto.
"""

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    *mensajes_historial,   # CORREGIDO: historial incluido
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.2,
                stream=False
            )

            respuesta_groq = chat_completion.choices[0].message.content

            if contexto and "no tengo el dato exacto" not in respuesta_groq.lower():
                respuesta_final = (
                    f"{respuesta_groq}\n\n"
                    f"---\n*📜 Fuente legal consultada: **{fuente}***"
                )
            else:
                respuesta_final = respuesta_groq

            mensaje_placeholder.markdown(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

        except Exception as e:
            mensaje_placeholder.error(f"❌ Error de conexión: {e}")
