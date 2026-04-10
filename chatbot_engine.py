"""
chatbot_engine.py — Motor del Chatbot InvoFact
================================================
Contiene toda la lógica NLP · NLU · NLG desacoplada del notebook,
lista para ser importada por la aplicación web (app.py).
"""

import json
import re
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ── Descarga silenciosa de recursos NLTK ──────────────────
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("stopwords",  quiet=True)

# ── Constantes ─────────────────────────────────────────────
DATASET_PATH         = Path(__file__).parent / "dataset_invofact.json"
MODEL_NAME           = "paraphrase-multilingual-MiniLM-L12-v2"
CONFIDENCE_THRESHOLD = 0.50

# ══════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════

def _cargar_dataset() -> list:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ══════════════════════════════════════════════════════════════
# NLP — PIPELINE DE PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════

_stemmer    = SnowballStemmer("spanish")
_stop_words = set(stopwords.words("spanish"))


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sin tildes, sin puntuación."""
    texto = texto.lower()
    for orig, reempl in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u","ñ":"n"}.items():
        texto = texto.replace(orig, reempl)
    texto = re.sub(r'[¿?¡!.,;:()\-"\'`\u201c\u201d\u2018\u2019\/]', " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _preprocess(texto: str) -> str:
    tokens = word_tokenize(normalizar_texto(texto), language="spanish")
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]
    return " ".join(_stemmer.stem(t) for t in tokens)


# ══════════════════════════════════════════════════════════════
# NLU — SENTENCE TRANSFORMER
# ══════════════════════════════════════════════════════════════

class ChatbotEngine:
    """
    Encapsula el modelo y el estado del chatbot.
    Usar @st.cache_resource para cargar una sola vez en Streamlit.
    """

    def __init__(self):
        dataset = _cargar_dataset()
        self.preguntas   = [d["pregunta"]  for d in dataset]
        self.respuestas  = [d["respuesta"] for d in dataset]
        self.categorias  = [d["categoria"] for d in dataset]

        self._model      = SentenceTransformer(MODEL_NAME)
        self._embeddings = self._model.encode(self.preguntas, convert_to_numpy=True)

    # ── NLU ────────────────────────────────────────────────
    def detectar_intencion(self, pregunta: str):
        emb       = self._model.encode([pregunta], convert_to_numpy=True)
        sims      = cosine_similarity(emb, self._embeddings).flatten()
        idx       = int(np.argmax(sims))
        score     = float(sims[idx])
        if score < CONFIDENCE_THRESHOLD:
            return None, score, None
        return idx, score, self.categorias[idx]

    # ── NLG — sugerencias dinámicas ────────────────────────
    def _sugerir(self, categoria: str, idx_actual: int, n: int = 2) -> str:
        candidatos = [
            self.preguntas[i]
            for i, cat in enumerate(self.categorias)
            if cat == categoria and i != idx_actual
        ]
        if not candidatos:
            return ""
        elegidas = random.sample(candidatos, min(n, len(candidatos)))
        return "💡 **También puedes preguntarme:**\n" + "\n".join(f"- {p}" for p in elegidas)

    # ── NLG — respuesta principal ──────────────────────────
    def responder(self, mensaje: str) -> dict:
        """
        Procesa el mensaje del usuario y devuelve un dict con:
            texto     : str  — respuesta en Markdown
            categoria : str  — categoría detectada
            score     : float
        """
        texto_norm = normalizar_texto(mensaje)

        # Saludo
        saludos = ["hola","buenos dias","buenas tardes","buenas noches",
                   "buenas","hey","saludos","que tal","buen dia"]
        if any(s in texto_norm for s in saludos):
            return {
                "texto": (
                    "¡Hola! Soy el asistente virtual de **InvoFact**. 🤖\n\n"
                    "Puedo ayudarte con:\n"
                    "- Ventas y comprobantes electrónicos\n"
                    "- Productos e inventario\n"
                    "- Usuarios y permisos\n"
                    "- Caja: apertura, cierre y egresos\n"
                    "- Compras y proveedores\n"
                    "- Reportes y exportación a Excel\n\n"
                    "¿En qué te puedo ayudar hoy?"
                ),
                "categoria":        "Saludo",
                "score":            1.0,
                "pregunta_dataset": None,
            }

        # Despedida
        despedidas = ["adios","chau","bye","hasta luego","hasta pronto",
                      "gracias","muchas gracias","ok gracias","listo gracias"]
        if any(d in texto_norm for d in despedidas):
            return {
                "texto": (
                    "¡Hasta luego! Fue un gusto ayudarte. 😊\n\n"
                    "Si en el futuro tienes más preguntas sobre InvoFact, "
                    "no dudes en consultarme. ¡Éxito con tu negocio!"
                ),
                "categoria":        "Despedida",
                "score":            1.0,
                "pregunta_dataset": None,
            }

        # Confirmación
        confirmaciones = ["genial","perfecto","excelente","ok","okey","de acuerdo",
                          "entendido","entiendo","claro","listo","muy bien","super",
                          "que bueno","chevere","bien","dale","ya"]
        if len(texto_norm.split()) <= 4 and any(c in texto_norm for c in confirmaciones):
            return {
                "texto": (
                    "¡Genial! 😊 ¿Hay algo más en lo que te pueda ayudar?\n\n"
                    "Puedo responder preguntas sobre ventas, productos, comprobantes, "
                    "inventario, caja, reportes o configuración del sistema."
                ),
                "categoria":        "Confirmacion",
                "score":            1.0,
                "pregunta_dataset": None,
            }

        # NLU
        idx, score, categoria = self.detectar_intencion(mensaje)

        if idx is None:
            return {
                "texto": (
                    "Lo siento, no encontré una respuesta para esa consulta. 🤔\n\n"
                    "Puedo ayudarte con: ventas, comprobantes, productos, stock, "
                    "usuarios, caja, reportes o configuración.\n\n"
                    "¿Podrías reformular tu pregunta o mencionar el módulo del sistema?"
                ),
                "categoria":        "Fallback",
                "score":            score,
                "pregunta_dataset": None,
            }

        # Construir respuesta
        respuesta_base = self.respuestas[idx]
        pregunta_match = self.preguntas[idx]
        sugerencia     = self._sugerir(categoria, idx)

        texto = respuesta_base
        if sugerencia:
            texto += f"\n\n{sugerencia}"
        if score < 0.60:
            texto += "\n\n⚠️ *Si no es lo que buscabas, intenta reformular tu pregunta.*"

        return {
            "texto":            texto,
            "categoria":        categoria,
            "score":            score,
            "pregunta_dataset": pregunta_match,
        }
