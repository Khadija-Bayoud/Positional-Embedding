"""
Rotary Positional Embedding (RoPE) page.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.styles import DARK_TEMPLATE, COLORS
from utils.math_helpers import rope_freqs, apply_rope
from components.ui import hero, sec_header, callout, math_block, example_block, viz_caption, pros_cons, notable_uses


PAPER_URL   = "https://arxiv.org/abs/2104.09864"
PAPER_TITLE = "RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al. 2021"


def render():
    hero(
        "🌀", "Rotary Positional Embedding (RoPE)",
        "RoPE encodes position by rotating query and key vectors in the complex plane. "
        "The rotation angle is proportional to position — and, beautifully, the dot product "
        "of a rotated Q and K depends only on their relative angle (i.e. their distance). "
        "This is the dominant approach in modern LLMs.",
        PAPER_TITLE, PAPER_URL,
    )

    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        seq_len  = st.slider("Sequence length", 8, 128, 48, key="rope_seq")
        d_model  = st.select_slider("d_model", [16, 32, 64, 128], value=64, key="rope_d")
        dim_pair = st.slider("Dim pair to show in rotation plot", 0, d_model // 2 - 1, 0, key="rope_dp")
        base_val = st.select_slider("RoPE base θ", [1000, 5000, 10000, 50000, 500000], value=10000, key="rope_base")

    with col_main:
        t1, t2, t3, t4 = st.tabs(["💡 Intuition & Need", "📐 Math", "📊 Visualization", "⚖️ Pros & Cons"])

        # ── INTUITION ─────────────────────────────────────────────────────────
        with t1:
            st.markdown("""
            <div class="card">
            <div class="sec-header">💡 Core Idea</div>
            <p style='color:#94a3b8;'>
              Imagine a clock face. Every position in the sequence rotates the clock's hands
              by a fixed amount. Token at position 3 has been rotated 3 ticks. Token at position 7
              has been rotated 7 ticks.<br><br>
              Now, when two tokens compute their attention score (dot product), what matters is
              <b style='color:#e2e8f0;'>the angle between their hands</b> — and that angle is
              exactly the difference in their positions (7 - 3 = 4 ticks). You get
              <em>relative distance for free</em>, just from rotating by absolute position!<br><br>
              RoPE does this across many independent 2D planes (one per pair of embedding dimensions),
              each spinning at a different rate — just like the frequency layers in sinusoidal PE.
            </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
            <div class="sec-header">🎯 Why Do We Need It?</div>
            <p style='color:#94a3b8;'>
              RoPE elegantly solves a tension between absolute and relative PE:<br>
              • It <b style='color:#e2e8f0;'>encodes absolute position</b>
                (each token's Q and K vectors are uniquely rotated).<br>
              • It produces <b style='color:#e2e8f0;'>relative position in attention scores</b>
                (the dot product only depends on the distance, not the absolute positions).<br>
              • It adds <b style='color:#e2e8f0;'>zero parameters</b> — pure rotation.<br>
              • It is <b style='color:#e2e8f0;'>KV-cache friendly</b> — you only apply the rotation once
                per token at the position it was generated.<br><br>
              This combination makes RoPE the default choice for virtually all modern open-weight LLMs.
            </p>
            </div>
            """, unsafe_allow_html=True)

        # ── MATH ──────────────────────────────────────────────────────────────
        with t2:
            st.markdown("""
            <div class="card">
            <div class="sec-header">📐 The Rotation Formula</div>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"""
            \theta_i = \frac{1}{\,\theta_{\text{base}}^{\,2i/d}\,}, \quad i = 0, 1, \ldots, \tfrac{d}{2}-1
            """)
            st.latex(r"""
            \begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix}
            =
            \underbrace{\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}}_{\text{2D rotation matrix at position }m}
            \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
            """)

            callout("""
            <b>Reading the formula:</b><br>
            • We split the embedding vector into pairs of dimensions: (dim 0, dim 1), (dim 2, dim 3), etc.<br>
            • Each pair is a 2D plane. We <b>rotate</b> the vector in that plane by angle <code>m × θᵢ</code>,
              where m is the token's position.<br>
            • θᵢ decreases with i (same idea as sinusoidal frequencies) — early pairs rotate fast, later pairs rotate slowly.<br>
            • The same rotation is applied to Key vectors as well.
            """)

            st.markdown("""
            <div class="card">
            <div class="sec-header">✨ The Key Property — Relative Position in Dot Products</div>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""
            \langle\mathbf{q}_m,\, \mathbf{k}_n\rangle
            \;=\; \sum_{i=0}^{d/2-1}
            \bigl\langle R_i(m)\,\mathbf{q}_{2i:2i+1},\; R_i(n)\,\mathbf{k}_{2i:2i+1} \bigr\rangle
            \;=\; f\!\left(\mathbf{q}, \mathbf{k},\; m - n\right)
            """)
            callout("""
            The dot product depends only on <b>m - n</b> (the relative distance), not on m or n individually.
            This is the magic: by rotating with absolute position, the inner product becomes relative.
            """)

            st.markdown("---")
            sec_header("🔢 Worked Example")
            st.markdown("<p style='color:#94a3b8;'>d = 4 (two pairs), base θ = 10000. Positions m=0 and m=2. Focus on first pair (i=0).</p>", unsafe_allow_html=True)

            st.latex(r"\theta_0 = \frac{1}{10000^{0/4}} = 1.0")

            example_block(
                "Query vector at position 0: q = [1.0, 0.0, ...]  (just the first pair)\n"
                "  Angle = 0 × θ₀ = 0 × 1.0 = 0 rad\n"
                "  q'[0] = 1.0×cos(0) - 0.0×sin(0) = 1.0\n"
                "  q'[1] = 0.0×cos(0) + 1.0×sin(0) = 0.0\n"
                "  → rotated q at pos 0: [1.0, 0.0]  (no change, angle=0)\n"
                "\n"
                "Key vector at position 2:   k = [1.0, 0.0, ...]  (same original vector)\n"
                "  Angle = 2 × θ₀ = 2 × 1.0 = 2 rad\n"
                "  k'[0] = 1.0×cos(2) - 0.0×sin(2) ≈  1.0×(-0.416) = -0.416\n"
                "  k'[1] = 0.0×cos(2) + 1.0×sin(2) ≈  1.0×(0.909)  =  0.909\n"
                "  → rotated k at pos 2: [-0.416, 0.909]\n"
                "\n"
                "Attention score (first pair contribution):\n"
                "  q' · k' = 1.0×(-0.416) + 0.0×0.909 = -0.416 = cos(2) = cos(m-n)\n"
                "\n"
                "  ✓ Result depends only on (m-n)=2, not on m=0 or n=2 individually!"
            )

        # ── VISUALIZATION ─────────────────────────────────────────────────────
        with t3:
            cos_r, sin_r, angles = rope_freqs(seq_len, d_model, float(base_val))

            st.markdown("#### Visualization 1 — Rotation Angles per Position × Dimension Pair")
            viz_caption(
                f"Each cell (pos, dim_pair) shows the rotation angle in radians for that token at that dimension pair. "
                f"Row = a token position (0 to {seq_len-1}). Column = an embedding dimension pair (0 to {d_model//2-1}). "
                f"Bright = large angle (fast rotation). Dark = small angle (slow rotation). "
                f"Left columns complete many full rotations; right columns barely rotate — "
                f"same multi-frequency structure as sinusoidal PE. "
                f"<b>Increase base θ</b> (left panel) to spread the frequencies further and reduce max angle."
            )
            fig1 = go.Figure(go.Heatmap(
                z=angles, colorscale="Plasma",
                colorbar=dict(title="Angle (radians)"),
            ))
            fig1.update_layout(
                title="RoPE rotation angles  ·  rows=positions, cols=dimension pairs",
                height=320, xaxis_title="Dimension pair i", yaxis_title="Token position m",
                **DARK_TEMPLATE, margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig1, width='stretch')

            st.markdown("#### Visualization 2 — Rotation in 2D (Unit Circle)")
            viz_caption(
                f"For dimension pair {dim_pair}, every token position is represented as a point on the unit circle, "
                f"with its angle = pos × θ_{dim_pair}. "
                f"The arrow from the origin shows the direction of the rotated 2D vector. "
                f"Transparent arrows = early positions; bright arrows = later positions. "
                f"Two tokens whose arrows are close together will have high attention affinity from this dimension pair. "
                f"<b>Try different dim pairs</b>: low-index pairs spin fast (arrows spread out widely), "
                f"high-index pairs spin slowly (arrows cluster together)."
            )
            a = angles[:, dim_pair]
            fig2 = go.Figure()
            for idx in range(seq_len):
                alpha = max(0.15, idx / seq_len)
                rgba_c = f"rgba(251,191,36,{alpha:.2f})"
                fig2.add_trace(go.Scatter(
                    x=[0, np.cos(a[idx])], y=[0, np.sin(a[idx])],
                    mode="lines+markers",
                    line=dict(color=rgba_c, width=1.5),
                    marker=dict(size=[0, 6], color=rgba_c),
                    showlegend=False,
                ))
            fig2.add_shape(type="circle", x0=-1.05, y0=-1.05, x1=1.05, y1=1.05,
                           line=dict(color=COLORS["muted"], dash="dot"))
            fig2.update_layout(
                title=f"2D rotation — dimension pair {dim_pair}  ·  faint=early, bright=later positions",
                height=400, xaxis=dict(range=[-1.2, 1.2], scaleanchor="y"),
                yaxis=dict(range=[-1.2, 1.2]),
                **DARK_TEMPLATE, margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig2, width='stretch')

            st.markdown("#### Visualization 3 — Attention Scores with RoPE Applied")
            viz_caption(
                f"We generate random Q and K vectors (same as what a transformer would produce) "
                f"and then apply RoPE rotations. Cell (i, j) = the attention score from position i to position j. "
                f"Because RoPE makes scores depend on relative distance, "
                f"you'd expect a 'diagonal band' structure — positions close to each other (near the diagonal) "
                f"tend to have higher scores than distant ones. "
                f"The randomness in Q/K adds noise, but the band structure is visible. "
                f"<b>Change seq_len or d_model</b> to see how scale affects the pattern."
            )
            rng_v = np.random.default_rng(1)
            Q = rng_v.normal(0, 1, (seq_len, d_model))
            K = rng_v.normal(0, 1, (seq_len, d_model))
            Q_r = apply_rope(Q, cos_r, sin_r)
            K_r = apply_rope(K, cos_r, sin_r)
            attn = Q_r @ K_r.T / np.sqrt(d_model)
            fig3 = go.Figure(go.Heatmap(z=attn, colorscale="RdBu", zmid=0,
                                         colorbar=dict(title="Attention score")))
            fig3.update_layout(
                title="Attention scores (random Q,K with RoPE)  ·  diagonal band = nearby positions preferred",
                height=340, xaxis_title="Key position", yaxis_title="Query position",
                **DARK_TEMPLATE, margin=dict(t=55, b=30)
            )
            st.plotly_chart(fig3, width='stretch')

        # ── PROS / CONS ───────────────────────────────────────────────────────
        with t4:
            pros_cons(
                pros=[
                    ("Best of both worlds",
                     "Encodes absolute position (each Q/K is uniquely rotated) while making attention scores depend only on relative distance. No other method does both simultaneously."),
                    ("Zero parameters",
                     "Pure mathematical rotation — nothing to train, nothing to overfit."),
                    ("KV-cache compatible",
                     "You apply the rotation once when a token is generated, then cache it. Perfect for efficient autoregressive generation."),
                    ("Scales to long contexts",
                     "Techniques like YaRN, LongRoPE, and NTK interpolation extend RoPE to 128K+ tokens with minimal fine-tuning."),
                    ("Dominant in practice",
                     "LLaMA, Mistral, Gemma, Falcon, Qwen and most modern open-weight LLMs use RoPE — it's battle-tested at scale."),
                ],
                cons=[
                    ("Non-trivial implementation",
                     "Splitting into dimension pairs, applying 2×2 rotation matrices, getting strides right — more fiddly than just adding a vector."),
                    ("Default base limits context",
                     "With θ_base=10000, models start to degrade beyond ~4K tokens. You need larger bases or interpolation tricks for longer sequences."),
                    ("Needs fine-tuning for extreme lengths",
                     "Jumping from 4K to 128K context requires careful continued pre-training — you can't just swap the base at inference time."),
                ],
            )
            notable_uses(["LLaMA 1/2/3", "Mistral", "Gemma", "Falcon", "GPT-NeoX", "Qwen", "DeepSeek"])
