"""
Side-by-side comparison of all PE methods.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.styles import DARK_TEMPLATE, COLORS, hex_to_rgba
from utils.math_helpers import (
    sinusoidal_pe, learned_pe_sim, relative_bias_matrix,
    rope_freqs, apply_rope, alibi_bias_matrix,
)
from components.ui import callout, viz_caption


def render():
    st.markdown("""
    <div class="hero">
      <h1>⚖️ Side-by-Side Comparison</h1>
      <p>
        All five methods in one place. Compare their structure, similarity decay,
        feature table, decision guide, and radar chart — everything you need to
        pick the right one for your project.
      </p>
    </div>
    """, unsafe_allow_html=True)

    seq_len = st.slider("Sequence length for all charts", 16, 96, 48, key="cmp_seq")
    d_model = 64

    # ── Pre-compute all matrices ──────────────────────────────────────────────
    PE_sin   = sinusoidal_pe(seq_len, d_model)
    PE_learn = learned_pe_sim(seq_len, d_model)
    rel_bias = relative_bias_matrix(seq_len)
    cos_r, sin_r, angles = rope_freqs(seq_len, d_model)
    alibi_b, slopes = alibi_bias_matrix(seq_len, 8)

    rng = np.random.default_rng(1)
    Q   = rng.normal(0, 1, (seq_len, d_model))
    K   = rng.normal(0, 1, (seq_len, d_model))
    Q_r = apply_rope(Q, cos_r, sin_r)
    K_r = apply_rope(K, cos_r, sin_r)
    rope_attn = Q_r @ K_r.T / np.sqrt(d_model)

    # ── 1. Heatmap snapshot ───────────────────────────────────────────────────
    st.markdown("### Structural Heatmaps — What Each Method Stores")
    st.markdown("""
    <div class='callout'>
      These heatmaps show <em>fundamentally different shapes</em> because each method stores
      positional information in a different place:<br>
      • <b>Sinusoidal</b> and <b>Learned PE</b> produce a <em>(positions × dimensions)</em> matrix —
        a vector added to each token's embedding.<br>
      • <b>Relative</b>, <b>RoPE</b>, and <b>ALiBi</b> produce a <em>(positions × positions)</em> matrix —
        a value added to each attention score between two tokens.
    </div>
    """, unsafe_allow_html=True)

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=[
            "Sinusoidal PE\n(pos × dim)",
            "Learned PE\n(pos × dim)",
            "Relative bias\n(pos × pos)",
            "RoPE attn scores\n(pos × pos)",
            "ALiBi bias h1\n(pos × pos)",
        ],
        horizontal_spacing=0.04,
    )
    data_list = [
        (PE_sin,       "RdBu",    True),
        (PE_learn,     "RdBu",    True),
        (rel_bias,     "RdBu",    True),
        (rope_attn,    "RdBu",    True),
        (alibi_b[0],   "Blues_r", False),
    ]
    for ci, (data, cs, mid) in enumerate(data_list, 1):
        kw = dict(zmid=0) if mid else {}
        fig.add_trace(go.Heatmap(z=data, colorscale=cs, showscale=False, **kw), row=1, col=ci)
    fig.update_layout(height=310, margin=dict(t=55, b=10, l=10, r=10), **DARK_TEMPLATE)
    for ax in fig.layout:
        if ax.startswith(("xaxis", "yaxis")):
            fig.layout[ax].update(showticklabels=False)
    st.plotly_chart(fig, width='stretch')

    # ── 2. Similarity decay ───────────────────────────────────────────────────
    st.markdown("### How Fast Does Positional Similarity Decay?")
    viz_caption(
        "Starting from position 0, how similar is each other position to it? "
        "A sharp drop = the method strongly differentiates nearby vs distant positions. "
        "A flat line = all positions look about the same from position 0's perspective. "
        "Sinusoidal and Learned use cosine similarity of PE vectors. "
        "RoPE uses the actual dot-product attention score. "
        "Relative and ALiBi use their bias values (normalised to [-1, 1] for comparison)."
    )
    dists = np.arange(seq_len)
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    sin_sims  = np.array([cosine_sim(PE_sin[0], PE_sin[k]) for k in dists])
    lrn_sims  = np.array([cosine_sim(PE_learn[0], PE_learn[k]) for k in dists])
    rope_row  = rope_attn[0]
    rope_norm = (rope_row - rope_row.min()) / (rope_row.max() - rope_row.min() + 1e-8) * 2 - 1
    rel_row   = rel_bias[0]
    rel_norm  = (rel_row - rel_row.min()) / (rel_row.max() - rel_row.min() + 1e-8) * 2 - 1
    alibi_row = -slopes[0] * dists
    alibi_norm = (alibi_row - alibi_row.min()) / (alibi_row.max() - alibi_row.min() + 1e-8) * 2 - 1

    fig2 = go.Figure()
    traces = [
        (sin_sims,   COLORS["sinusoidal"], "Sinusoidal"),
        (lrn_sims,   COLORS["learned"],    "Learned"),
        (rope_norm,  COLORS["rope"],       "RoPE"),
        (rel_norm,   COLORS["relative"],   "Relative"),
        (alibi_norm, COLORS["alibi"],      "ALiBi"),
    ]
    for y, color, name in traces:
        fig2.add_trace(go.Scatter(x=dists, y=y, mode="lines", name=name,
                                   line=dict(color=color, width=2.5)))
    fig2.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
    fig2.update_layout(
        title="Normalised positional similarity from position 0 as distance increases",
        height=360, xaxis_title="Distance from position 0", yaxis_title="Similarity (normalised)",
        **DARK_TEMPLATE, margin=dict(t=55, b=60),
        legend=dict(orientation="h", y=-0.22, font_size=13),
    )
    st.plotly_chart(fig2, width='stretch')

    # ── 3. Feature table ──────────────────────────────────────────────────────
    st.markdown("### Feature Comparison Table")
    data = {
        "Method":               ["Sinusoidal", "Learned", "Relative", "RoPE", "ALiBi"],
        "Extra parameters":     ["0",          "L × d",   "2(2k+1)d_k", "0",  "0"],
        "Encodes absolute pos": ["✅",         "✅",       "❌",       "✅",   "❌"],
        "Encodes relative pos": ["⚠️ implicit","❌",       "✅",       "✅",   "✅"],
        "Length extrapolation": ["Medium",     "❌ capped","Good",    "Good", "Excellent"],
        "KV-cache friendly":    ["✅",         "✅",       "⚠️",       "✅",   "✅"],
        "Implementation":       ["Easy",       "Trivial",  "Complex", "Medium","Trivial"],
        "Year":                 ["2017",       "2018",     "2018",    "2021", "2021"],
        "Common in LLMs":       ["Legacy",     "Legacy",   "Encoders","Dominant","Some"],
    }
    df = pd.DataFrame(data).set_index("Method")
    st.dataframe(df, width='stretch', height=260)

    # ── 4. Decision guide ─────────────────────────────────────────────────────
    st.markdown("### When Should You Use Each?")
    guides = [
        ("📐 Sinusoidal", COLORS["sinusoidal"],
         ["Research baseline", "Encoder-only models", "No parameter budget constraint", "Classic seq2seq"]),
        ("🎓 Learned", COLORS["learned"],
         ["Short fixed-length inputs", "BERT-style fine-tuning", "Task needs position specialisation", "Max length is known & fixed"]),
        ("🔗 Relative", COLORS["relative"],
         ["NLU tasks (GLUE, SQuAD)", "DeBERTa-style models", "Code or music generation", "Explicit distance matters"]),
        ("🌀 RoPE", COLORS["rope"],
         ["Modern decoder LLMs", "Long context (4K–128K)", "Open-weight models", "General purpose LLM pre-training"]),
        ("📏 ALiBi", COLORS["alibi"],
         ["Extreme length extrapolation", "Zero embedding overhead", "BLOOM / MPT style models", "Decoder LM with strong locality"]),
    ]
    cols = st.columns(5)
    for col, (name, color, use_cases) in zip(cols, guides):
        with col:
            items = "".join(f"<li style='margin-bottom:4px;'>{u}</li>" for u in use_cases)
            st.markdown(f"""
            <div style='background:var(--surface);border:1px solid {color}30;border-top:3px solid {color};
                        border-radius:10px;padding:1.3rem;min-height:200px;'>
              <div style='font-family:IBM Plex Mono;font-size:14px;font-weight:700;
                          color:{color};margin-bottom:0.8rem;'>{name}</div>
              <ul style='color:#94a3b8;font-size:15px;line-height:1.8;
                          padding-left:1.1rem;margin:0;'>{items}</ul>
            </div>
            """, unsafe_allow_html=True)

    # ── 5. Radar chart ────────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("### Capability Radar", unsafe_allow_html=True)
    callout("""
    Scores (0–5) are qualitative judgements across six axes:<br>
    <b>Length Extrapolation</b> — works on sequences longer than training length<br>
    <b>Simplicity</b> — ease of implementation and zero extra parameters<br>
    <b>Relative Info</b> — directly encodes pairwise token distances<br>
    <b>Absolute Info</b> — encodes where each token is absolutely<br>
    <b>KV-Cache</b> — compatible with standard autoregressive KV caching<br>
    <b>Param-Free</b> — no additional learned parameters required
    """)

    categories = ["Length Extrapolation", "Simplicity", "Relative Info",
                  "Absolute Info", "KV-Cache", "Param-Free"]
    scores = {
        "Sinusoidal": [3, 5, 2, 5, 5, 5],
        "Learned":    [1, 5, 1, 5, 5, 2],
        "Relative":   [4, 2, 5, 2, 2, 3],
        "RoPE":       [4, 3, 5, 5, 5, 5],
        "ALiBi":      [5, 5, 4, 1, 5, 5],
    }
    color_list = [
        COLORS["sinusoidal"], COLORS["learned"], COLORS["relative"],
        COLORS["rope"], COLORS["alibi"],
    ]

    fig_r = go.Figure()
    for (method, vals), color in zip(scores.items(), color_list):
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=method,
            line=dict(color=color, width=2.5),
            fillcolor=hex_to_rgba(color, 0.10),   # ← fixed: proper rgba string
        ))
    fig_r.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 5],
                tickfont=dict(size=12),
                gridcolor="#1f2937",
            ),
            angularaxis=dict(tickfont=dict(size=13), gridcolor="#1f2937"),
            bgcolor="rgba(17,24,39,0.5)",
        ),
        showlegend=True, height=500,
        legend=dict(font_size=14),
        **DARK_TEMPLATE,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_r, width='stretch')
