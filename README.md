# InvoFact Chatbot 🤖

Asistente conversacional para [InvoFact](https://invofact.com) con NLP · NLU · NLG, basado en
`SentenceTransformers` y una interfaz web construida con **Streamlit**.

---

## Estructura del proyecto

```
proyectoChatbot/
├── .venv/                      # Entorno virtual Python
├── .streamlit/
│   └── config.toml             # Configuración Streamlit (sin telemetría, sin file watcher)
│
├── core/                       # Motor del chatbot (lógica pura, sin interfaz)
│   ├── __init__.py
│   └── db.py                   # Persistencia SQLite: historial de conversaciones
│
├── ui/                         # Módulos de interfaz web
│   ├── __init__.py
│   ├── styles.py               # Tema visual InvoFact (azul · verde · blanco)
│   └── components.py           # Componentes HTML reutilizables (badges, headers)
│
├── chatbot_engine.py           # ChatbotEngine: NLP · NLU · NLG (usado por app y notebook)
├── dataset.json                # Dataset de entrenamiento (78 entradas, 8 categorías)
├── app.py                      # Punto de entrada Streamlit (Chat + Historial)
├── chatbot.ipynb               # Notebook de desarrollo y demostración (Fases 1–7)
├── invofact_chat.db            # Base de datos SQLite (generada al ejecutar)
├── Sílabo.pdf                  # Documento de referencia académica
├── Tarea Académica.pdf         # Documento de referencia académica
└── README.md
```

---

## Requisitos previos

- **Python 3.10+**
- **pip** actualizado

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
pip install pandas numpy nltk scikit-learn ipywidgets sentence-transformers streamlit
```

> La primera vez que se ejecute el chatbot, descargará automáticamente el modelo
> `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) desde HuggingFace y lo guardará en caché local.
> Las ejecuciones posteriores son instantáneas.

---

## 3. Levantar la interfaz web

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Abre el navegador en **http://localhost:8501**.

---

## 4. Ejecutar el notebook (opcional)

Si preferís usar el chatbot desde el notebook de Jupyter:

```powershell
.\.venv\Scripts\jupyter.exe notebook chatbot.ipynb
```

O abrirlo directamente desde **VS Code** con la extensión Jupyter seleccionando el kernel `.venv`.

Ejecutá todas las celdas en orden (Kernel → Restart & Run All).

---

## Arquitectura

```
dataset.json
     │
     ▼
chatbot_engine.py
 ├── NLP  — NLTK: tokenización, stopwords, stemming (español)
 ├── NLU  — SentenceTransformer: embeddings + cosine similarity
 │            modelo: paraphrase-multilingual-MiniLM-L12-v2
 │            umbral de confianza: 0.50
 └── NLG  — Detección de intención + respuesta dinámica
              sugerencias extraídas del propio dataset
     │
     ▼
app.py  (Streamlit)
 ├── Tema oscuro personalizado
 ├── Sidebar con temas disponibles y contador de mensajes
 ├── Historial de conversación persistente (session_state)
 └── Badges de categoría por color
```

---

## Dataset

| Categoría     | Entradas |
|---------------|----------|
| Ventas        | 18       |
| General       | 14       |
| Productos     | 10       |
| Usuarios      | 9        |
| Caja          | 8        |
| Reportes      | 8        |
| Compras       | 6        |
| Configuracion | 5        |
| **Total**     | **78**   |

---

## Variables de entorno / configuración

No se requieren variables de entorno. Toda la configuración está en `chatbot_engine.py`:

| Constante             | Valor por defecto                           | Descripción                          |
|-----------------------|---------------------------------------------|--------------------------------------|
| `MODEL_NAME`          | `paraphrase-multilingual-MiniLM-L12-v2`     | Modelo SentenceTransformer           |
| `CONFIDENCE_THRESHOLD`| `0.50`                                      | Umbral mínimo de similitud para responder |
| `DATASET_PATH`        | `dataset.json` (mismo directorio)           | Ruta al dataset                      |

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError` al importar | Verificar que el entorno `.venv` está activado |
| Puerto 8501 ocupado | `streamlit run app.py --server.port 8502` |
| Error de permisos en PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| El modelo no descarga | Verificar conexión a internet (primera ejecución únicamente) |
| Advertencia de symlinks en Windows | Inofensiva — el caché de HuggingFace funciona correctamente |
