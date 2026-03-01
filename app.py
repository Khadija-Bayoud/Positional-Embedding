"""
Positional Embeddings Explorer — main entry point.

Run with:
    streamlit run app.py

Project structure:
    app.py                   ← you are here (routing only)
    utils/
        styles.py            ← CSS, colour constants, Plotly theme
        math_helpers.py      ← all NumPy math (PE formulas)
    components/
        ui.py                ← reusable HTML/Streamlit widgets
    pages/
        overview.py          ← home / taxonomy
        sinusoidal.py        ← Absolute (Sinusoidal) PE
        learned.py           ← Learned PE
        relative.py          ← Relative PE
        rope.py              ← RoPE
        alibi.py             ← ALiBi
        comparison.py        ← side-by-side comparison

To add a new PE method:
    1. Add math functions to utils/math_helpers.py
    2. Create pages/your_method.py  (copy any existing page as template)
    3. Import and register it in the PAGES dict below
    4. Done — navigation updates automatically.
"""

import streamlit as st

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Positional Embeddings Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global styles ──────────────────────────────────────────────────────
from utils.styles import inject_styles
inject_styles()

# ── Page registry — add new methods here ─────────────────────────────────────
from sections import overview, sinusoidal, learned, relative, rope, alibi, comparison

PAGES = {
    "🏠  Overview":             overview,
    "📐  Absolute (Sinusoidal)": sinusoidal,
    "🎓  Learned":              learned,
    "🔗  Relative":             relative,
    "🌀  RoPE":                 rope,
    "📏  ALiBi":                alibi,
    "⚖️  Comparison":           comparison,
}

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:13px;color:#64748b;
                letter-spacing:0.1em;text-transform:uppercase;
                padding:1rem 0 0.6rem 0;'>Navigation</div>
    """, unsafe_allow_html=True)

    page_key = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:13px;color:#64748b;line-height:1.8;'>
      Each method page has 4 tabs:<br>
      <span style='color:#00d4ff;'>💡 Intuition</span> — plain-English explanation<br>
      <span style='color:#00d4ff;'>📐 Math</span> — formula + worked example<br>
      <span style='color:#00d4ff;'>📊 Viz</span> — interactive charts<br>
      <span style='color:#00d4ff;'>⚖️ Pros/Cons</span> — honest trade-offs<br><br>
      Use <b style='color:#fbbf24;'>⚖️ Comparison</b> to see<br>
      all methods side-by-side.
    </div>
    """, unsafe_allow_html=True)

# ── Render selected page ──────────────────────────────────────────────────────
PAGES[page_key].render()
