"""
core/db.py — Persistencia de historial de conversaciones (SQLite)
=================================================================
Almacena cada intercambio del chatbot para análisis posterior.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "invofact_chat.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crea la tabla si no existe."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historial (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id        TEXT    NOT NULL,
            timestamp        TEXT    NOT NULL,
            pregunta_usuario TEXT    NOT NULL,
            pregunta_dataset TEXT,
            categoria        TEXT,
            confianza        REAL,
            respuesta        TEXT
        )
    """)
    conn.commit()
    conn.close()


def nuevo_thread_id() -> str:
    """Genera un identificador único de hilo de conversación."""
    return str(uuid.uuid4())[:8].upper()


def guardar_mensaje(
    thread_id: str,
    pregunta_usuario: str,
    pregunta_dataset: str | None,
    categoria: str,
    confianza: float,
    respuesta: str,
) -> None:
    """Inserta un registro en el historial."""
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO historial
            (thread_id, timestamp, pregunta_usuario, pregunta_dataset,
             categoria, confianza, respuesta)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            datetime.now().isoformat(timespec="seconds"),
            pregunta_usuario,
            pregunta_dataset,
            categoria,
            confianza,
            respuesta,
        ),
    )
    conn.commit()
    conn.close()


def obtener_historial(limit: int = 200) -> list[dict]:
    """Devuelve los últimos `limit` registros ordenados del más reciente al más antiguo."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, thread_id, timestamp, pregunta_usuario, pregunta_dataset, "
        "categoria, confianza FROM historial ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def limpiar_historial() -> None:
    """Elimina todos los registros del historial."""
    conn = _get_conn()
    conn.execute("DELETE FROM historial")
    conn.commit()
    conn.close()


def stats_historial() -> dict:
    """Retorna estadísticas básicas del historial."""
    conn = _get_conn()
    total   = conn.execute("SELECT COUNT(*) FROM historial").fetchone()[0]
    threads = conn.execute("SELECT COUNT(DISTINCT thread_id) FROM historial").fetchone()[0]
    avg_conf = conn.execute(
        "SELECT AVG(confianza) FROM historial WHERE categoria NOT IN "
        "('Saludo','Despedida','Confirmacion','Fallback')"
    ).fetchone()[0]
    conn.close()
    return {
        "total_mensajes": total,
        "total_hilos":    threads,
        "confianza_prom": round(avg_conf * 100, 1) if avg_conf else 0.0,
    }
