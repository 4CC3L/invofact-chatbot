"""
ui/components.py — Componentes reutilizables de la interfaz InvoFact
====================================================================
"""

from ui.styles import CATEGORY_COLORS, PRIMARY_BLUE, ACCENT_GREEN


def badge_html(categoria: str) -> str:
    """Genera un badge HTML con el color de la categoría."""
    key = (
        categoria.lower()
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("é", "e")
        .replace("á", "a")
        .replace("í", "i")
    )
    bg, color = CATEGORY_COLORS.get(key, ("#f1f5f9", "#475569"))
    return (
        f'<span style="display:inline-block;font-size:0.72rem;padding:2px 10px;'
        f'border-radius:99px;font-weight:600;background:{bg};color:{color};">'
        f"{categoria}</span>"
    )


def header_html(title: str, subtitle: str, icon: str = "🤖") -> str:
    """Genera el encabezado de página."""
    from ui.styles import PRIMARY_BLUE, TEXT_MUTED, BORDER_COLOR
    return f"""
    <div style="display:flex;align-items:center;gap:12px;padding:1rem 0 0.5rem;
         border-bottom:2px solid {PRIMARY_BLUE};margin-bottom:1.2rem;">
        <span style="font-size:1.8rem;">{icon}</span>
        <div>
            <p style="font-size:1.3rem;font-weight:700;color:{PRIMARY_BLUE};margin:0;">{title}</p>
            <p style="font-size:0.8rem;color:{TEXT_MUTED};margin:0;">{subtitle}</p>
        </div>
    </div>
    """


def stat_card_html(label: str, value: str, color: str = None) -> str:
    """Mini tarjeta estadística en HTML."""
    from ui.styles import PRIMARY_BLUE, BG_CARD, BORDER_COLOR, TEXT_MUTED
    c = color or PRIMARY_BLUE
    return f"""
    <div style="background:{BG_CARD};border:1px solid {BORDER_COLOR};border-radius:10px;
         padding:0.7rem 1.1rem;text-align:center;">
        <p style="font-size:1.5rem;font-weight:700;color:{c};margin:0;">{value}</p>
        <p style="font-size:0.75rem;color:{TEXT_MUTED};margin:0;">{label}</p>
    </div>
    """
