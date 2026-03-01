"""
Shared CSS & Plotly theme for the Positional Embeddings Explorer.
Import inject_styles() once in app.py.
"""
import streamlit as st


DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.85)",
    font_family="IBM Plex Sans",
    font_color="#e2e8f0",
    font_size=14,
)

# ── hex colours as plain rgba strings so Plotly accepts them ──────────────────
COLORS = {
    "sinusoidal": "#00d4ff",
    "learned":    "#a78bfa",
    "relative":   "#34d399",
    "rope":       "#fbbf24",
    "alibi":      "#f87171",
    "muted":      "#64748b",
    "surface":    "#111827",
    "surface2":   "#1a2235",
    "bg":         "#0a0e1a",
    "border":     "rgba(255,255,255,0.07)",
    "text":       "#e2e8f0",
}

# Convert hex → rgba with alpha for Plotly fillcolor
def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --surface2: #1a2235;
    --c1: #00d4ff;
    --c2: #a78bfa;
    --c3: #34d399;
    --c4: #fbbf24;
    --c5: #f87171;
    --text:  #e2e8f0;
    --muted: #94a3b8;
    --border: rgba(255,255,255,0.07);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 17px;
    background-color: var(--bg);
    color: var(--text);
}
p, li, td, th, label { font-size: 17px !important; line-height: 1.8 !important; }
.stApp { background-color: var(--bg); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px !important;
    color: var(--muted);
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,212,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    background: linear-gradient(90deg, var(--c1), var(--c2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem 0;
}
.hero p {
    color: var(--muted);
    font-size: 18px !important;
    max-width: 700px;
    margin: 0;
    line-height: 1.8 !important;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.4rem;
}
.card-accent { border-left: 3px solid var(--c1); }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 14px !important;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 6px;
    margin-bottom: 6px;
}
.badge-pro  { background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.3); }
.badge-con  { background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); }
.badge-use  { background:rgba(0,212,255,0.12);  color:#00d4ff; border:1px solid rgba(0,212,255,0.3); }
.badge-year { background:rgba(251,191,36,0.12); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }

/* ── Math block ── */
.math-block {
    background: var(--surface2);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 1.2rem 1.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px !important;
    color: var(--c1);
    margin: 1.2rem 0;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.9 !important;
}

/* ── Example block ── */
.example-block {
    background: rgba(167,139,250,0.06);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 8px;
    padding: 1.2rem 1.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px !important;
    color: #c4b5fd;
    margin: 1.2rem 0;
    white-space: pre-wrap;
    line-height: 1.9 !important;
}

/* ── Callout ── */
.callout {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 1.2rem 1.6rem;
    margin: 1.2rem 0;
    font-size: 17px !important;
    line-height: 1.8 !important;
}
.callout-warn {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 8px;
    padding: 1.2rem 1.6rem;
    margin: 1.2rem 0;
    font-size: 17px !important;
    line-height: 1.8 !important;
}

/* ── Section header ── */
.sec-header {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 1.25rem !important;
    color: var(--c1);
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Viz caption ── */
.viz-caption {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid var(--c4);
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.3rem 0 1.2rem 0;
    font-size: 15px !important;
    color: var(--muted);
    line-height: 1.7 !important;
}

/* ── Paper link ── */
.paper-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 15px !important;
    color: var(--c1);
    text-decoration: none;
    font-family: 'IBM Plex Mono', monospace;
}
.paper-link:hover { background: rgba(0,212,255,0.16); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border-radius: 8px 8px 0 0;
    border: 1px solid var(--border);
    border-bottom: none;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px !important;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--c1) !important;
    border-color: rgba(0,212,255,0.3) !important;
}

h2, h3, h4 { font-family: 'IBM Plex Sans', sans-serif; }
h2 { font-size: 1.7rem !important; }
h3 { font-size: 1.35rem !important; }
</style>
"""


def inject_styles():
    st.markdown(CSS, unsafe_allow_html=True)
