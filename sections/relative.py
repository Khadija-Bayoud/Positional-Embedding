"""
Relative Positional Embedding page.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.styles import DARK_TEMPLATE, COLORS
from utils.math_helpers import relative_offset_matrix, relative_bias_matrix, t5_bucket_matrix
from components.ui import hero, sec_header, callout, math_block, example_block, viz_caption, pros_cons, notable_uses


PAPER_URL   = "https://arxiv.org/abs/1803.02155"
PAPER_TITLE = "Self-Attention with Relative Position Representations — Shaw et al. 2018"


def render():
    hero(
        "🔗", "Relative Positional Embedding",
        "Instead of encoding where a token is absolutely, Relative PE encodes "
        "the distance between every pair of tokens. The model learns to interpret "
        "'2 positions apart' rather than 'at position 5'.",
        PAPER_TITLE, PAPER_URL,
    )

    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        seq_len  = st.slider("Sequence length", 8, 64, 32, key="rel_seq")
        max_dist = st.slider("Max clipped distance (k)", 4, 32, 16, key="rel_k")
        variant  = st.selectbox("Variant", ["Shaw et al. (signed offset)", "T5 Buckets (log scale)", "Symmetric (absolute distance)"], key="rel_var")

    with col_main:
        t1, t2, t3, t4 = st.tabs(["💡 Intuition & Need", "📐 Math", "📊 Visualization", "⚖️ Pros & Cons"])

        # ── INTUITION ─────────────────────────────────────────────────────────
        with t1:
            st.markdown("""
            <div class="card">
            <div class="sec-header">💡 Core Idea</div>
            <p style='color:#94a3b8;'>
              When you read the sentence <em>"The dog bit the man"</em>,
              what matters for understanding "bit" isn't that it's at position index 2 globally.
              What matters is that <b style='color:#e2e8f0;'>"bit" is 1 position after "dog"</b>
              and 2 positions before "man".<br><br>
              Relative PE directly encodes these gaps. For every pair of tokens (query, key),
              instead of asking "where am I?", the model asks
              <b style='color:#e2e8f0;'>"how far apart are we?"</b>.
              The model learns a set of bias vectors — one for each possible distance offset —
              and these get added into the attention score computation.
            </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
            <div class="sec-header">🎯 Why Do We Need It?</div>
            <p style='color:#94a3b8;'>
              Absolute PE tells each token its own position, but it never tells the attention head
              how far two tokens are from each other — the model has to figure that out indirectly.<br><br>
              Relative PE <em>directly injects pairwise distance</em> into the attention calculation.
              This has two key benefits:<br>
              • <b style='color:#e2e8f0;'>Length generalisation:</b> A model trained on sequences of 512
                can more naturally handle 1024, because "5 positions apart" means the same thing at any length.<br>
              • <b style='color:#e2e8f0;'>Explicit structure:</b> The model doesn't need to reverse-engineer
                distances from absolute positions — the information is handed to it directly.
            </p>
            </div>
            """, unsafe_allow_html=True)

        # ── MATH ──────────────────────────────────────────────────────────────
        with t2:
            st.markdown("""
            <div class="card">
            <div class="sec-header">📐 Standard Attention (baseline)</div>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""
            e_{ij} = \frac{(\mathbf{x}_i\,\mathbf{W}_Q)(\mathbf{x}_j\,\mathbf{W}_K)^\top}{\sqrt{d_k}}
            """)

            st.markdown("""
            <div class="card">
            <div class="sec-header">📐 Shaw et al. Relative Attention</div>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""
            e_{ij} = \frac{(\mathbf{x}_i\,\mathbf{W}_Q)\bigl(\mathbf{x}_j\,\mathbf{W}_K + \mathbf{a}^K_{ij}\bigr)^\top}{\sqrt{d_k}}
            """)
            st.latex(r"""
            \mathbf{a}^K_{ij} = \mathbf{w}^K_{\,\text{clip}(j-i,\,-k,\,k)}
            """)
            st.latex(r"""
            \mathbf{w}^K \in \mathbb{R}^{(2k+1)\,\times\, d_k} \quad \leftarrow \text{learned relative bias table}
            """)

            callout("""
            <b>Reading the formula:</b><br>
            • <code>e_ij</code> = the raw attention score from query token i to key token j.<br>
            • <code>a^K_ij</code> = a learned vector that depends only on the signed distance (j − i).<br>
            • The <b>clip</b> function caps the distance at ±k — beyond k positions, all offsets share
              the same "far away" vector. This keeps the number of parameters manageable.<br>
            • The value side has an analogous term <code>a^V_ij</code> that modifies how values are summed.
            """)

            st.markdown("""
            <div class="card">
            <div class="sec-header">📐 T5 Variant — Logarithmic Buckets</div>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""
            \text{bucket}(j-i) = \begin{cases}
              j-i & \text{if } 0 \le j-i < k/2 \quad (\text{exact fine-grained}) \\
              k/2 + \left\lfloor \log\!\frac{j-i}{k/2} \Big/ \log\!\frac{L}{k/2} \cdot \frac{k}{2} \right\rfloor & \text{otherwise (log-spaced)}
            \end{cases}
            """)

            callout("""
            <b>T5 insight:</b> Nearby positions need fine resolution (did "bit" come just before or just after "dog"?),
            but for distant positions the exact number doesn't matter much — you just need to know "far away".
            T5's log-bucketing gives you many buckets for small distances and few buckets for large distances,
            which is a much more efficient use of parameters.
            """)

            st.markdown("---")
            sec_header("🔢 Worked Example")
            st.markdown("<p style='color:#94a3b8;'>Sequence: 4 tokens. d_k = 2, k = 2. Let's compute e₀₂ (attention from position 0 to position 2).</p>", unsafe_allow_html=True)

            example_block(
                "Tokens at positions: 0='The', 1='dog', 2='bit', 3='the'\n"
                "\n"
                "Step 1 — Standard attention part:\n"
                "  x₀·W_Q = [0.5, 0.3]   (query for 'The')\n"
                "  x₂·W_K = [0.8, 0.1]   (key for 'sat')\n"
                "  Standard dot product = 0.5×0.8 + 0.3×0.1 = 0.43\n"
                "\n"
                "Step 2 — Relative bias part:\n"
                "  offset = j − i = 2 − 0 = +2  (clipped to k=2 → bucket +2)\n"
                "  a^K[+2] = [0.2, -0.1]  (learned vector for 'two ahead')\n"
                "  x₀·W_Q · a^K[+2] = 0.5×0.2 + 0.3×(−0.1) = 0.07\n"
                "\n"
                "Step 3 — Final attention score:\n"
                "  e₀₂ = (0.43 + 0.07) / √2 ≈ 0.354\n"
                "\n"
                "The model effectively learned: 'things 2 positions ahead get this extra +0.07 boost'"
            )

        # ── VISUALIZATION ─────────────────────────────────────────────────────
        with t3:
            pos = np.arange(seq_len)
            rel = pos[None, :] - pos[:, None]  # j - i

            if "Shaw" in variant:
                display_mat = np.clip(rel, -max_dist, max_dist).astype(float)
                title_suffix = f"Signed offset (j−i), clipped to ±{max_dist}"
            elif "T5" in variant:
                display_mat = t5_bucket_matrix(seq_len)
                title_suffix = "T5 log-bucket index"
            else:
                display_mat = np.abs(rel).astype(float)
                title_suffix = "Absolute distance |i−j|"

            st.markdown("#### Visualization 1 — Relative Offset Matrix")
            viz_caption(
                f"Each cell (i, j) shows the positional offset from query position i to key position j. "
                f"This is what gets looked up in the learned bias table. "
                f"The bright red diagonal = offset 0 (token attending to itself). "
                f"Blue = key is to the left (j < i), red = key is to the right (j > i). "
                f"<b>Variant: {title_suffix}.</b> "
                f"<b>Try changing variant</b> — T5 Buckets compresses far-apart positions into fewer bins. "
                f"<b>Change k</b> (max clip distance) to see how far the fine-grained resolution extends."
            )
            fig1 = go.Figure(go.Heatmap(z=display_mat, colorscale="RdBu", zmid=0,
                                         colorbar=dict(title="Offset / Bucket")))
            fig1.update_layout(
                title=f"Relative offset matrix — {title_suffix}",
                height=380, xaxis_title="Key position (j)", yaxis_title="Query position (i)",
                **DARK_TEMPLATE, margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig1, width='stretch')

            st.markdown("#### Visualization 2 — Simulated Learned Bias Weights")
            viz_caption(
                "This bar chart shows what a learned relative bias table might look like after training. "
                "Each bar = one possible relative offset value, from −k on the left to +k on the right. "
                "Green bars mean 'attending to tokens at this distance gets a positive boost'. "
                "Red bars mean 'attending to tokens at this distance is penalised'. "
                "In practice, models often learn to prefer attending to nearby tokens (bars near 0 are larger). "
                "<b>The exact shape depends entirely on the task</b> — this is a simulation."
            )
            offsets = np.arange(-max_dist, max_dist + 1)
            rng = np.random.default_rng(42)
            learned_bias = rng.normal(0, 0.3, len(offsets)) * np.exp(-0.05 * np.abs(offsets))
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=offsets, y=learned_bias,
                marker_color=[COLORS["alibi"] if v < 0 else COLORS["relative"] for v in learned_bias],
            ))
            fig2.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
            fig2.update_layout(
                title="Simulated learned relative bias weights (one head)",
                height=280, xaxis_title="Relative offset (j − i)", yaxis_title="Bias weight added to attention score",
                **DARK_TEMPLATE, margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig2, width='stretch')

        # ── PROS / CONS ───────────────────────────────────────────────────────
        with t4:
            pros_cons(
                pros=[
                    ("Better length generalisation",
                     "Trained on 512 tokens? The same 'offset=5' vector works at position 500 and at position 5000 — the meaning of 'nearby' doesn't change."),
                    ("Explicit relative info",
                     "The distance between two tokens is directly injected into the attention score — no indirect reconstruction needed."),
                    ("Used in strong encoders",
                     "DeBERTa with relative PE holds top scores on many NLU benchmarks even years after its release."),
                ],
                cons=[
                    ("Compute overhead",
                     "You need to compute and store an (L × L) bias matrix for every head and every layer — O(L²) extra work per layer."),
                    ("Complex to implement",
                     "You have to modify the attention mechanism internals, not just add a vector to the input. More error-prone."),
                    ("KV-cache friction",
                     "Relative biases depend on the query position, so you can't simply cache keys and values for autoregressive generation without extra bookkeeping."),
                    ("Mostly in encoders",
                     "The compute overhead makes it less popular in large decoder-only LLMs, where RoPE or ALiBi dominate."),
                ],
            )
            notable_uses(["T5", "DeBERTa / DeBERTa-v3", "Transformer-XL", "Music Transformer", "Shaw et al. NMT"])
