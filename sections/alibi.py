"""
ALiBi Positional Embedding page.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.styles import DARK_TEMPLATE, COLORS
from utils.math_helpers import alibi_bias_matrix, alibi_slopes
from components.ui import hero, sec_header, callout, math_block, example_block, viz_caption, pros_cons, notable_uses


PAPER_URL   = "https://arxiv.org/abs/2108.12409"
PAPER_TITLE = "Train Short, Test Long: ALiBi — Press et al. 2021"


def render():
    hero(
        "📏", "ALiBi — Attention with Linear Biases",
        "ALiBi doesn't add any embeddings at all. Instead it subtracts a linear penalty "
        "from attention scores based on the distance between tokens. "
        "Radical simplicity, zero parameters, and excellent length extrapolation.",
        PAPER_TITLE, PAPER_URL,
    )

    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        seq_len      = st.slider("Sequence length", 8, 128, 48, key="ali_seq")
        n_heads      = st.select_slider("Num attention heads", [4, 8, 16, 32], value=8, key="ali_heads")
        head_to_show = st.slider("Head to display in detail", 0, n_heads - 1, 0, key="ali_head")
        causal       = st.checkbox("Causal (decoder-style) mask", value=True, key="ali_causal")

    with col_main:
        t1, t2, t3, t4 = st.tabs(["💡 Intuition & Need", "📐 Math", "📊 Visualization", "⚖️ Pros & Cons"])

        # ── INTUITION ─────────────────────────────────────────────────────────
        with t1:
            st.markdown("""
            <div class="card">
            <div class="sec-header">💡 Core Idea</div>
            <p style='color:#94a3b8;'>
              We know from experiments that transformers <em>naturally tend to attend more to nearby tokens</em>
              than to distant ones. ALiBi simply says: <b style='color:#e2e8f0;'>let's make that explicit
              and hard-code it into the attention score.</b><br><br>
              For every pair (query i, key j), ALiBi subtracts a penalty proportional to how far apart
              they are. It's like a parking ticket: the further you park from the centre, the bigger the fine.
              Nearby tokens pay a small fine; distant tokens pay a large one.<br><br>
              Different attention heads get different slope sizes — some heads are strict (steep fines, very local),
              others are lenient (shallow fines, more global). This gives the model a diversity of attentional ranges.
            </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
            <div class="sec-header">🎯 Why Do We Need It?</div>
            <p style='color:#94a3b8;'>
              The killer feature of ALiBi is <b style='color:#e2e8f0;'>length extrapolation</b>.
              Sinusoidal and learned PE both hit a wall: train on 512 tokens, and the model
              struggles at 1024 because it encounters position embeddings it's never seen.<br><br>
              ALiBi has no positional embeddings at all. When the sequence gets longer, the penalty
              just keeps growing linearly — the same rule the model trained with.
              There's nothing new to see. A model trained on 1024 tokens can extrapolate to
              4096 without any degradation in quality.
            </p>
            </div>
            """, unsafe_allow_html=True)

        # ── MATH ──────────────────────────────────────────────────────────────
        with t2:
            st.markdown("""
            <div class="card">
            <div class="sec-header">📐 The Formula</div>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"""
            e_{ij}^{\text{ALiBi}} \;=\;
            \underbrace{\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}}_{\text{standard attention}}
            \;-\; \underbrace{m_h \cdot |i - j|}_{\text{linear distance penalty}}
            """)
            st.latex(r"""
            m_h \;=\; \frac{1}{2^{\,8h/n_{\text{heads}}}}, \quad h = 1, 2, \ldots, n_{\text{heads}}
            """)

            callout("""
            <b>Reading the formula:</b><br>
            • <code>e_ij</code> = the attention score from query position i to key position j.<br>
            • <code>m_h</code> = the slope for head h. Larger = steeper penalty = more local head.<br>
            • <code>|i − j|</code> = the absolute distance between the two tokens.<br>
            • The slopes form a geometric sequence: head 1 is the most local (steepest), 
              head n_heads is the most global (shallowest).
            """)

            st.markdown("""
            <div class="card">
            <div class="sec-header">📐 Slopes for 8 heads</div>
            </div>
            """, unsafe_allow_html=True)
            m = alibi_slopes(8)
            slopes_display = "  ".join([f"h{i+1}: {v:.4f}" for i, v in enumerate(m)])
            st.code(slopes_display, language="text")

            callout("""
            <b>Why a geometric sequence of slopes?</b> It creates a natural spectrum:
            head 1 attends within ~2 tokens, head 8 attends across ~256 tokens.
            Together, the heads cover all distance scales from very local to global.
            """, warning=True)

            st.markdown("---")
            sec_header("🔢 Worked Example")
            st.markdown("<p style='color:#94a3b8;'>4 heads, positions i=0 and j=3. Standard dot product = 0.8.</p>", unsafe_allow_html=True)

            m4 = alibi_slopes(4)
            example_block(
                f"Slopes for 4 heads:\n"
                f"  m₁ = {m4[0]:.4f}   m₂ = {m4[1]:.4f}   m₃ = {m4[2]:.4f}   m₄ = {m4[3]:.4f}\n"
                f"\n"
                f"Standard attention score (same for all heads):  q·k/√d = 0.800\n"
                f"Distance |i−j| = |0 − 3| = 3\n"
                f"\n"
                f"ALiBi score per head:\n"
                f"  Head 1:  0.800 − {m4[0]:.4f} × 3 = {0.8 - m4[0]*3:.4f}   ← steep penalty (local head)\n"
                f"  Head 2:  0.800 − {m4[1]:.4f} × 3 = {0.8 - m4[1]*3:.4f}\n"
                f"  Head 3:  0.800 − {m4[2]:.4f} × 3 = {0.8 - m4[2]*3:.4f}\n"
                f"  Head 4:  0.800 − {m4[3]:.4f} × 3 = {0.8 - m4[3]*3:.4f}   ← gentle penalty (global head)\n"
                f"\n"
                f"Head 1 is now much less likely to attend to position 3 than head 4."
            )

        # ── VISUALIZATION ─────────────────────────────────────────────────────
        with t3:
            biases, slopes = alibi_bias_matrix(seq_len, n_heads)
            vmin = np.min(biases)
            vmax = np.max(biases)

            st.markdown(f"#### Visualization 1 — Bias Matrix for Head {head_to_show + 1}")
            viz_caption(
                f"Each cell (i, j) shows the penalty subtracted from the attention score between query i and key j. "
                f"Darker blue = larger penalty (token is far away → strongly penalised). "
                f"The diagonal (i = j) has zero penalty (a token attending to itself). "
                f"The causal mask (lower triangle only) makes future tokens invisible — decoder style. "
                f"<b>Try different heads</b>: head 1 has the steepest slope (mostly dark), "
                f"the last head has a gentle slope (mostly light). "
                f"<b>Toggle the causal mask</b> to switch between encoder and decoder style."
            )
            bias_head = biases[head_to_show].copy()
            if causal:
                mask = np.tril(np.ones((seq_len, seq_len)))
                bias_head[mask == 0] = np.nan

            fig1 = go.Figure(go.Heatmap(
                z=bias_head, colorscale="Blues_r",
                colorbar=dict(title="Penalty (negative)"),
                zmin=biases[head_to_show].min(),
            ))
            fig1.update_layout(
                title=f"ALiBi Bias Matrix — Head {head_to_show + 1}  ·  slope = {slopes[head_to_show]:.5f}",
                height=380, xaxis_title="Key position (j)", yaxis_title="Query position (i)",
                **DARK_TEMPLATE, margin=dict(t=55, b=30)
            )
            st.plotly_chart(fig1, width='stretch')

            st.markdown("#### Visualization 2 — All Heads Side-by-Side")
            viz_caption(
                f"Each small panel = one attention head. Head 1 (top-left) has the steepest slope: "
                f"you can barely see anything beyond the diagonal. "
                f"The last head (bottom-right) has a gentle slope: most of the matrix is visible. "
                f"Together these heads let the model attend at every possible scale simultaneously. "
                f"<b>Change n_heads</b> to see more or fewer heads; changing it also rescales all slopes."
            )
            n_cols = min(n_heads, 4)
            n_rows = (n_heads + n_cols - 1) // n_cols
            fig2 = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f"h{h+1} m={slopes[h]:.3f}" for h in range(n_heads)],
                horizontal_spacing=0.06, vertical_spacing=0.12,
            )
            for h in range(n_heads):
                r, c = h // n_cols + 1, h % n_cols + 1
                b = biases[h].copy()
                if causal:
                    b[np.triu(np.ones_like(b), 1) == 1] = np.nan
                fig2.add_trace(go.Heatmap(z=b, zmin=vmin, zmax=vmax, colorscale="Blues_r", showscale=False), row=r, col=c)
            fig2.update_layout(height=max(300, 200 * n_rows), **DARK_TEMPLATE, margin=dict(t=40, b=10))
            for ax in fig2.layout:
                if ax.startswith(("xaxis", "yaxis")):
                    fig2.layout[ax].update(showticklabels=False)
            st.plotly_chart(fig2, width='stretch')

            st.markdown("#### Visualization 3 — Penalty vs Distance per Head")
            viz_caption(
                f"Each line shows how the penalty grows as two tokens move further apart. "
                f"Head 1 (steep) reaches a large penalty very quickly — tokens more than ~4 apart get suppressed. "
                f"The last head barely penalises even tokens 50 positions away. "
                f"This is the key to ALiBi's extrapolation: the lines just keep going linearly past the training length. "
                f"<b>Increase seq_len</b> to see the lines extend — the model always knows what to do."
            )
            fig3 = go.Figure()
            import plotly.express as px
            palette = px.colors.sequential.Plasma
            dists = np.arange(seq_len)
            for h in range(n_heads):
                c = palette[int(h / n_heads * (len(palette) - 1))]
                fig3.add_trace(go.Scatter(
                    x=dists, y=-slopes[h] * dists, mode="lines",
                    name=f"head {h+1}  m={slopes[h]:.4f}",
                    line=dict(color=c, width=2),
                ))
            fig3.update_layout(
                title="ALiBi penalty vs distance  ·  steeper = more local head",
                height=320, xaxis_title="Distance |i−j|", yaxis_title="Penalty subtracted from score",
                **DARK_TEMPLATE, margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig3, width='stretch')

        # ── PROS / CONS ───────────────────────────────────────────────────────
        with t4:
            pros_cons(
                pros=[
                    ("Best length extrapolation",
                     "A model trained on 1K tokens extrapolates to 4K or more with no special tricks — the linear penalty is always well-defined, no unseen position indices."),
                    ("Truly zero parameters",
                     "Not even the slopes are trained — they're fixed formulas. Simpler than even sinusoidal PE."),
                    ("Induces useful inductive bias",
                     "Most NLP tasks benefit from local context. ALiBi bakes this in for free, often helping models focus on relevant nearby tokens without being taught to."),
                    ("Trivial to implement",
                     "Just subtract a precomputed distance matrix from your attention scores. Works in any framework in ~5 lines."),
                ],
                cons=[
                    ("Forced locality may hurt",
                     "Some tasks genuinely need global attention (e.g., summarisation of long documents where the conclusion depends on the opening). ALiBi's penalty can suppress relevant distant tokens."),
                    ("No absolute position",
                     "ALiBi tells the model how far two tokens are from each other, but doesn't tell each token where it is absolutely. Tasks where absolute position matters (e.g., structured text at known line numbers) can suffer."),
                    ("Outperformed by RoPE at standard lengths",
                     "On most benchmarks at normal training lengths (4K-8K), RoPE models score higher. ALiBi's advantage is mainly at extreme extrapolation distances."),
                    ("Fixed slopes by formula",
                     "There's no learned adaptation — if your task has an unusual attention-distance profile, ALiBi can't adjust its slopes to fit."),
                ],
            )
            notable_uses(["BLOOM (176B)", "MPT", "OpenLLaMA variants", "XGLM"])
