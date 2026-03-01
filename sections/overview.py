"""
Overview / home page.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.styles import DARK_TEMPLATE, COLORS
from utils.math_helpers import (
    sinusoidal_pe, learned_pe_sim, relative_bias_matrix,
    rope_freqs, alibi_bias_matrix,
)
from components.ui import callout


def render():
    st.markdown("""
    <div class="hero">
      <h1>🧭 Positional Embeddings</h1>
      <p>
         Transformers are <em>permutation-invariant</em> by design (the output of a model
         remains the same regardless of the order or sequence of the input elements) — without positional information,
         the model has no sense of word order. Positional embeddings inject that structure.
         This explorer breaks down the 5 most important approaches: their intuition, math, 
         visualizations, and trade-offs.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Why order matters ──────────────────────────────────────────────────────
    st.markdown("### Why Does Word Order Matter?")
    callout("""
    Consider <b>"The dog bit the man"</b> versus <b>"The man bit the dog"</b>.
    Same words. Completely different meaning.<br><br>
    Self-attention computes every token's relationship with every other token —
    but it does this <em>symmetrically</em>: if you shuffled the input tokens,
    you'd get the same outputs (just shuffled). The model literally cannot tell position 1 from position 5
    unless we tell it.<br><br>
    Positional embeddings solve this by giving each token a unique positional <b>"fingerprint"</b>
    that gets mixed into its representation before any attention happens.
    """)

    # ── 5 method cards ────────────────────────────────────────────────────────
    st.markdown("### The Five Methods")
    methods = [
        ("📐", "Absolute\n(Sinusoidal)", COLORS["sinusoidal"], "2017", "A mathematical formula using sine & cosine waves"),
        ("🎓", "Learned",               COLORS["learned"],    "2018", "A trainable lookup table, one vector per position"),
        ("🔗", "Relative",              COLORS["relative"],   "2018", "Encodes the distance between pairs of tokens"),
        ("🌀", "RoPE",                  COLORS["rope"],       "2021", "Rotates Q & K vectors by a position-dependent angle"),
        ("📏", "ALiBi",                 COLORS["alibi"],      "2021", "Subtracts a distance penalty from attention scores"),
    ]
    cols = st.columns(5)
    for col, (icon, name, color, year, desc) in zip(cols, methods):
        with col:
            st.markdown(f"""
            <div style='background:var(--surface);border:1px solid {color}30;
                        border-top:3px solid {color};border-radius:10px;
                        padding:1.3rem;text-align:center;min-height:185px;'>
              <div style='font-size:1.9rem;margin-bottom:0.5rem;'>{icon}</div>
              <div style='font-family:IBM Plex Mono;font-size:15px;font-weight:600;
                          color:{color};margin-bottom:0.5rem;white-space:pre-line;'>{name}</div>
              <div style='font-size:14px;color:#94a3b8;line-height:1.6;margin-bottom:0.6rem;'>{desc}</div>
              <span class='badge badge-year'>{year}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Taxonomy ──────────────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("### Two Key Distinctions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card card-accent">
          <div class="sec-header">📦 Absolute vs Relative</div>
          <p style='color:#94a3b8;'>
            <b style='color:#e2e8f0;'>Absolute</b> methods stamp each token with its own position
            index (1, 2, 3 …). Simple and fast, but the model never directly knows
            <em>how far apart</em> two tokens are.<br><br>
            <b style='color:#e2e8f0;'>Relative</b> methods encode the <em>gap between</em> two
            tokens directly — making attention position-aware in a way that generalises
            better to longer sequences.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card card-accent">
          <div class="sec-header">🔧 Fixed vs Learned</div>
          <p style='color:#94a3b8;'>
            <b style='color:#e2e8f0;'>Fixed</b> methods (sinusoidal, ALiBi, RoPE) use a
            deterministic formula — zero extra parameters, no risk of overfitting to
            training lengths.<br><br>
            <b style='color:#e2e8f0;'>Learned</b> methods train position vectors end-to-end.
            More flexible and task-adaptive, but bounded by the maximum sequence length
            seen during training.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── At-a-glance heatmaps ─────────────────────────────────────────────────
    st.markdown("### Quick Visual Snapshot")
    st.markdown("""
    <div class='viz-caption'>
      Each panel below is a thumbnail of the full positional information each method produces.
      They look different because each method stores positional info in a completely different shape:
      some are <em>token × dimension</em> matrices (what you add to each token),
      others are <em>token × token</em> matrices (what you add to each attention score).
      Use the sidebar to explore any method in detail.
    </div>""", unsafe_allow_html=True)

    seq_len, d_model = 32, 64
    PE_sin   = sinusoidal_pe(seq_len, d_model)
    PE_learn = learned_pe_sim(seq_len, d_model)
    rel_bias = relative_bias_matrix(seq_len)
    cos_r, sin_r, angles = rope_freqs(seq_len, d_model)
    alibi_b, _ = alibi_bias_matrix(seq_len, 4)

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=["Sinusoidal", "Learned", "Relative", "RoPE angles", "ALiBi"],
        horizontal_spacing=0.04,
    )
    for i, (data, cs) in enumerate([
        (PE_sin,    "RdBu"),
        (PE_learn,  "RdBu"),
        (rel_bias,  "RdBu"),
        (angles,    "Plasma"),
        (alibi_b[0],"Blues_r"),
    ], 1):
        fig.add_trace(go.Heatmap(z=data, colorscale=cs, showscale=False), row=1, col=i)

    fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10), **DARK_TEMPLATE)
    for ax in fig.layout:
        if ax.startswith(("xaxis", "yaxis")):
            fig.layout[ax].update(showticklabels=False)
    st.plotly_chart(fig, width='stretch')
