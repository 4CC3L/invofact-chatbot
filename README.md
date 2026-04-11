# ChatHub — InvoFact + TAX-BOT SUNAT 🤖🏛️

Plataforma unificada de asistentes conversacionales con dos bots especializados:

- **InvoFact Soporte** — Asistente de soporte técnico para el sistema de facturación [InvoFact](https://invofact.com), basado en NLP/NLU/NLG local con `SentenceTransformers`.
- **TAX-BOT SUNAT** — Asistente tributario peruano con RAG (Retrieval-Augmented Generation) + Groq LLM, especializado en regímenes tributarios SUNAT.

Interfaz construida con **Streamlit** (`hub.py`). Ambos bots también pueden ejecutarse de forma independiente.

---

## Estructura del proyecto

```
proyectoChatbot/
├── .venv/                      # Entorno virtual Python
│
├── core/
│   ├── __init__.py
│   └── db.py                   # Persistencia SQLite: historial de conversaciones InvoFact
│
├── ui/
│   ├── __init__.py
│   ├── styles.py               # Tema visual InvoFact (azul · verde · oscuro)
│   └── components.py           # Componentes HTML reutilizables (badges, headers)
│
├── hub.py                      # ★ Punto de entrada principal — ChatHub unificado
├── app.py                      # Punto de entrada alternativo — solo InvoFact
├── suant_chatbot.py            # Punto de entrada alternativo — solo TAX-BOT
│
├── chatbot_engine.py           # ChatbotEngine: NLP · NLU · NLG (motor InvoFact)
├── dataset.json                # Dataset InvoFact (78 entradas, 8 categorías)
├── sunat_dataset.json          # Dataset TAX-BOT (4 regímenes, normas, FAQs, UIT histórico)
│
├── chatbot.ipynb               # Notebook de desarrollo y demostración
├── invofact_chat.db            # Base de datos SQLite (generada automáticamente)
└── README.md
```

---

## Requisitos previos

- **Python 3.10+**
- **pip** actualizado
- **Groq API Key** (gratis en [console.groq.com](https://console.groq.com)) — requerida solo para TAX-BOT

---

## 1. Crear el entorno virtual

```powershell
python -m venv .venv
```

### Activar el entorno

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

---

## 2. Instalar dependencias

```powershell
pip install streamlit pandas numpy nltk scikit-learn sentence-transformers groq langchain-core langchain-community langchain-huggingface chromadb
```

> La primera ejecucion descarga automaticamente el modelo `paraphrase-multilingual-MiniLM-L12-v2`
> (~470 MB) desde HuggingFace y lo guarda en cache local. Las ejecuciones posteriores son instantaneas.

---

## 3. Levantar el ChatHub (recomendado)

```powershell
.\.venv\Scripts\streamlit.exe run hub.py
```

Abre el navegador en **http://localhost:8501**.

Para ejecutar los bots de forma independiente:

```powershell
# Solo InvoFact
.\.venv\Scripts\streamlit.exe run app.py

# Solo TAX-BOT
.\.venv\Scripts\streamlit.exe run suant_chatbot.py
```

---

## 4. Ejecutar el notebook (opcional)

```powershell
.\.venv\Scripts\jupyter.exe notebook chatbot.ipynb
```

O abrirlo desde **VS Code** con la extension Jupyter, seleccionando el kernel `.venv`.

---

## Chatbot 1 — InvoFact Soporte

Asistente de soporte tecnico del sistema de facturacion electronica InvoFact.

### Arquitectura

```
dataset.json
     |
     v
chatbot_engine.py
 |-- NLP  — NLTK: tokenizacion, stopwords, stemming (espanol)
 |-- NLU  — SentenceTransformer: embeddings semanticos + cosine similarity
 |            modelo : paraphrase-multilingual-MiniLM-L12-v2
 |            umbral : CONFIDENCE_THRESHOLD = 0.50
 +-- NLG  — Deteccion de intencion + respuesta dinamica con sugerencias
     |
     v
app.py / hub.py  (Streamlit)
 |-- Historial SQLite persistente (core/db.py)
 |-- Vista "Chat" y vista "Historial" con tabla de conversaciones
 |-- Badges de categoria por color
 +-- Estadisticas: total mensajes, hilos, confianza promedio
```

### Dataset InvoFact

| Categoria     | Entradas |
|---------------|:--------:|
| Ventas        | 18       |
| General       | 14       |
| Productos     | 10       |
| Usuarios      | 9        |
| Caja          | 8        |
| Reportes      | 8        |
| Compras       | 6        |
| Configuracion | 5        |
| **Total**     | **78**   |

### Configuracion

| Constante              | Valor                                   | Descripcion                                |
|------------------------|-----------------------------------------|--------------------------------------------|
| `MODEL_NAME`           | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo SentenceTransformer (HuggingFace)   |
| `CONFIDENCE_THRESHOLD` | `0.50`                                  | Umbral minimo de similitud coseno          |
| `DATASET_PATH`         | `dataset.json`                          | Ruta al dataset de preguntas y respuestas  |
| `DB_PATH`              | `invofact_chat.db`                      | Base de datos SQLite del historial         |

### Persistencia SQLite (core/db.py)

Tabla `historial` con los campos: `thread_id`, `timestamp`, `pregunta_usuario`, `pregunta_dataset`, `categoria`, `confianza`, `respuesta`.

Funciones disponibles: `init_db`, `nuevo_thread_id`, `guardar_mensaje`, `obtener_historial`, `limpiar_historial`, `stats_historial`.

---

## Chatbot 2 — TAX-BOT SUNAT

Asistente tributario peruano con IA generativa. Responde consultas sobre los cuatro regimenes de SUNAT usando RAG sobre un dataset normativo curado.

### Arquitectura

```
sunat_dataset.json
     |  (chunking: ~30 documentos por regimen + secciones transversales)
     v
Chroma (vector store en memoria)
 +-- HuggingFaceEmbeddings: paraphrase-multilingual-MiniLM-L12-v2
     |
     v  similarity_search(query_enriquecida, k=8)
Groq API — llama-3.1-8b-instant
 |-- temperature : 0.2
 |-- Sistema: SOLO responder con el contexto recuperado
 |-- Anno referencia: 2026 · UIT vigente: S/ 5,500
 +-- Memoria conversacional: ultimos 6 turnos
     |
     v
suant_chatbot.py / hub.py  (Streamlit)
 |-- Botones de inicio rapido (4 consultas frecuentes)
 |-- Filtro de privacidad (Clave SOL, RUC, datos privados)
 |-- Bypass para saludos (sin consumo de API)
 |-- Query enriquecida: detecta el regimen en el historial reciente
 +-- Fuentes legales mostradas en cada respuesta
```

### Dataset SUNAT (sunat_dataset.json)

| Seccion                       | Contenido                                                                    |
|-------------------------------|------------------------------------------------------------------------------|
| **Regimenes (4)**             | Nuevo RUS, RER, Regimen MYPE Tributario (RMT), Regimen General               |
| Por regimen                   | Descripcion, requisitos, tributos, libros, comprobantes, declaraciones, cambio de regimen, casos practicos, FAQs, ventajas/desventajas |
| **Tabla comparativa**         | Comparacion de los 4 regimenes en 13 criterios                               |
| **IGV general**               | Tasa 18%, operaciones gravadas/exoneradas, credito fiscal, exportaciones     |
| **Infracciones y sanciones**  | Tablas I/II/III, TIM (1% mensual / 0.03% diario), regimen de gradualidad    |
| **Obligaciones comunes**      | RUC, comprobantes, declaraciones, PLAME                                      |
| **UIT historico**             | Valores 2017-2026 (UIT 2026: S/ 5,500)                                       |
| **Comprobantes de pago**      | Boleta, factura, nota de credito — umbral S/ 700 para identificacion DNI     |

### Configuracion TAX-BOT

| Parametro              | Valor                                   | Descripcion                                  |
|------------------------|-----------------------------------------|----------------------------------------------|
| Modelo LLM             | `llama-3.1-8b-instant`                  | Groq Cloud                                   |
| Temperature            | `0.2`                                   | Respuestas mas deterministicas               |
| Embeddings             | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace, multilingue                     |
| RAG k                  | `8`                                     | Documentos recuperados por consulta          |
| Memoria conversacional | 6 turnos                                | Ventana de historial enviada al LLM          |
| Anno de referencia     | 2026                                    | UIT vigente: S/ 5,500                        |
| API Key                | Ingresada en el panel lateral           | Requerida — obtener en console.groq.com      |

---

## Solucion de problemas

| Problema | Solucion |
|----------|----------|
| `ModuleNotFoundError` al importar | Verificar que el entorno `.venv` esta activado |
| Puerto 8501 ocupado | `streamlit run hub.py --server.port 8502` |
| Error de permisos en PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| El modelo no descarga | Verificar conexion a internet (primera ejecucion unicamente) |
| TAX-BOT no responde | Ingresar la Groq API Key en el panel lateral izquierdo |
| Advertencia `Pydantic V1` o `UNEXPECTED key` | Inofensiva — no afecta el funcionamiento |
