"""
Absolute (Sinusoidal) Positional Embedding page.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.styles import DARK_TEMPLATE, COLORS
from utils.math_helpers import sinusoidal_pe
from components.ui import hero, sec_header, callout, math_block, example_block, viz_caption, pros_cons, notable_uses


PAPER_URL   = "https://arxiv.org/abs/1706.03762"
PAPER_TITLE = "Attention Is All You Need — Vaswani et al. 2017"


def render():
    hero(
        "📐", "Absolute Positional Embedding (Sinusoidal)",
        "The original Transformer encoding. Each position gets a unique, fixed vector built "
        "from interleaved sine and cosine waves at carefully chosen frequencies. "
        "No parameters to train — pure maths.",
        PAPER_TITLE, PAPER_URL,
    )

    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        seq_len   = st.slider("Sequence length", 16, 128, 64,  key="abs_seq")
        d_model   = st.select_slider("d_model", [16, 32, 64, 128, 256], value=64, key="abs_d")
        show_dims = st.slider("Dims to show in waveform", 2, min(d_model, 16), 8, key="abs_dims")

    with col_main:
        t1, t2, t3, t4 = st.tabs(["💡 Intuition & Need", "📐 Math", "📊 Visualization", "⚖️ Pros & Cons"])

        # ── INTUITION ─────────────────────────────────────────────────────────
        with t1:
            st.markdown("""
            <div class="card">
            <div class="sec-header">💡 Core Idea</div>
            <p style='color:#94a3b8;'>
              Imagine you're tracking time during a full day using several clocks at once
              a second hand, a minute hand, an hour hand.
              If you only looked at the second hand, many different moments would look identical.
              If you only looked at the hour hand, you couldn’t distinguish nearby minutes.
              But when you look at all of them together — seconds, minutes, hours,
              their combined positions uniquely identify a specific moment in time.
              Each hand moves at a different speed, and together they create a precise timestamp.<br><br>
              Sinusoidal positional encoding works the same way for tokens in a sequence.
              Instead of labeling a token with just its position number,
              it represents that position as a vector (a list of numbers).
              Each number in that vector — each dimension — comes from a sine or cosine wave oscillating at a different frequency.
              Some waves change quickly (like the second hand) to distinguish nearby tokens,
              while others change slowly (like the hour hand) to separate tokens that are far apart.

            </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
            <div class="sec-header">🎯 Why Do We Need It?</div>
            <p style='color:#94a3b8;'>
              Self-attention computes relationships between <em>all pairs</em> of tokens simultaneously.
              If you shuffled the input sentence "The dog bit the man" into "The man bit the dog",
              you'd get <b style='color:#e2e8f0;'>exactly the same output</b> from a vanilla transformer
              (just reordered) — it is <em>permutation equivariant</em>.<br><br>
              Adding a positional encoding vector to each token's embedding breaks this symmetry.
              Now "dog" at position 1 has a different representation than "dog" at position 4,
              and the model can learn to use that difference.
            </p>
            </div>
            """, unsafe_allow_html=True)

        # ── MATH ──────────────────────────────────────────────────────────────
        with t2:
            st.markdown("""
            <div class="card">
            <div class="sec-header">📐 The Formula</div>
            <p style='color:#94a3b8;'>
              For position <code>pos</code> (which token it is) and dimension index <code>i</code>
              (which slot in the embedding vector):
            </p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"""
            \text{PE}(\text{pos},\, 2i)   = \sin\!\left(\frac{\text{pos}}{10000^{\,2i/d}}\right)
            """)
            st.latex(r"""
            \text{PE}(\text{pos},\, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{\,2i/d}}\right)
            """)

            callout("""
            <b>Reading the formula:</b><br>
            • <code>pos</code> = the token's position in the sentence (0, 1, 2, …)<br>
            • <code>i</code> = which pair of dimensions we're filling (0, 1, 2, …, d/2−1)<br>
            • Even dimensions (0, 2, 4…) get <b>sine</b> values; odd dimensions (1, 3, 5…) get <b>cosine</b>.<br>
            • The denominator <code>10000^(2i/d)</code> grows with i, making high-index dimensions
              oscillate <em>slowly</em> and low-index dimensions oscillate <em>fast</em>.
            """)

            st.markdown("""
            <div class="card">
            <div class="sec-header">✨ Hidden Property: Relative Offsets for Free</div>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"""
            \text{PE}(\text{pos}+k) = \mathbf{M}_k \cdot \text{PE}(\text{pos})
            """)

            callout("""
            <b>What this means:</b> Any fixed offset <em>k</em> can be expressed as a
            rotation matrix applied to the current position's encoding.
            This means the model can <em>learn to compute relative distances</em>
            through attention, even though the encoding itself is absolute.
            """)

            # ── EXAMPLE ───────────────────────────────────────────────────────
            st.markdown("---")
            sec_header("🔢 Worked Example")
            st.markdown("<p style='color:#94a3b8;'>Let's compute PE by hand for a tiny case: d_model = 4, positions 0 and 1.</p>", unsafe_allow_html=True)

            st.latex(r"""
            d = 4,\quad \text{so } i \in \{0, 1\} \text{ (two dimension pairs)}
            """)
            st.latex(r"""
            \omega_0 = \frac{1}{10000^{0/4}} = 1.0 \qquad
            \omega_1 = \frac{1}{10000^{2/4}} = \frac{1}{100} = 0.01
            """)

            example_block(
                "Position 0:\n"
                "  PE(0, dim 0) = sin(0 × 1.0)  = sin(0)  = 0.000\n"
                "  PE(0, dim 1) = cos(0 × 1.0)  = cos(0)  = 1.000\n"
                "  PE(0, dim 2) = sin(0 × 0.01) = sin(0)  = 0.000\n"
                "  PE(0, dim 3) = cos(0 × 0.01) = cos(0)  = 1.000\n"
                "\n"
                "  → PE(0) = [ 0.000,  1.000,  0.000,  1.000 ]\n"
                "\n"
                "Position 1:\n"
                "  PE(1, dim 0) = sin(1 × 1.0)  = sin(1)  ≈ 0.841\n"
                "  PE(1, dim 1) = cos(1 × 1.0)  = cos(1)  ≈ 0.540\n"
                "  PE(1, dim 2) = sin(1 × 0.01) = sin(0.01) ≈ 0.010\n"
                "  PE(1, dim 3) = cos(1 × 0.01) = cos(0.01) ≈ 1.000\n"
                "\n"
                "  → PE(1) = [ 0.841,  0.540,  0.010,  1.000 ]"
            )

            callout("""
            <b>Notice:</b> dim 0/1 (fast frequency) already differ a lot between positions 0 and 1.
            But dim 2/3 (slow frequency) are nearly identical — they only start to differ for positions
            very far apart.
            """)

        # ── VISUALIZATION ────────────────────────────────────────────────────
        with t3:
            PE = sinusoidal_pe(seq_len, d_model)

            # Heatmap 1: full matrix
            st.markdown("#### Visualization 1 — The Full PE Matrix")
            viz_caption(
                f"Each row is one token's positional vector (there are {seq_len} rows = {seq_len} positions). "
                f"Each column is one dimension (there are {d_model} columns). "
                "The colour shows the sine/cosine value: blue = +1, red = -1. "
                "Notice the fast-oscillating stripes on the left (small i = high frequency) "
                "and the slow gradient on the right (large i = low frequency). "
                "<br><b>Try changing sequence length and d_model</b> in the left panel — "
                "more positions add rows; larger d_model adds columns and more frequency bands."
            )
            fig1 = go.Figure(go.Heatmap(
                z=PE, colorscale="RdBu", zmid=0,
                colorbar=dict(title="Value", tickfont=dict(size=12)),
            ))
            fig1.update_layout(
                title="PE Matrix  ·  rows = positions, columns = embedding dimensions",
                height=340, xaxis_title="Embedding dimension", yaxis_title="Token position",
                **DARK_TEMPLATE, margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig1, width='stretch')

            # Heatmap 2: waveforms
            st.markdown("#### Visualization 2 — Dimension Waveforms")
            viz_caption(
                f"Each line traces a single embedding dimension across all {seq_len} positions. "
                "Low-index dimensions (left, warm colours) complete many cycles → fine-grained nearby discrimination. "
                "High-index dimensions (right, cool colours) barely move → coarse long-range structure. "
                "<b>Drag 'Dims to show'</b> to see more or fewer waveforms and observe how frequency drops."
            )
            fig2 = go.Figure()
            palette = px.colors.sequential.Plasma
            for idx in range(show_dims):
                c = palette[int(idx / show_dims * (len(palette) - 1))]
                fig2.add_trace(go.Scatter(
                    x=np.arange(seq_len), y=PE[:, idx], mode="lines",
                    name=f"dim {idx}", line=dict(color=c, width=2),
                ))
            fig2.update_layout(
                title=f"First {show_dims} dimensions across all positions",
                height=300, xaxis_title="Token position", yaxis_title="PE value",
                **DARK_TEMPLATE, margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig2, width='stretch')

            # Heatmap 3: dot-product similarity
            st.markdown("#### Visualization 3 — Positional Similarity")
            viz_caption(
                "Each cell (i, j) shows the dot-product between PE[i] and PE[j]. "
                "Bright = two positions whose PE vectors are similar (close together). "
                "Dark = very different PE vectors (far apart). "
                "The bright diagonal is each position compared with itself (maximum similarity). "
                "Off-diagonal bands show that nearby positions are more similar than distant ones — "
                "this is the implicit relative-position signal the model can use during attention. "
                "<b>Increase sequence length</b> to see the pattern extend."
            )
            sim = PE @ PE.T
            fig3 = go.Figure(go.Heatmap(z=sim, colorscale="Viridis",
                                        colorbar=dict(title="Dot product")))
            fig3.update_layout(
                title="PE similarity matrix  ·  bright diagonal = 'same position'",
                height=380, xaxis_title="Position j", yaxis_title="Position i",
                **DARK_TEMPLATE, margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig3, width='stretch')

        # ── PROS / CONS ───────────────────────────────────────────────────────
        with t4:
            pros_cons(
                pros=[
                    ("Zero parameters",
                     "The formula is fixed — no training needed, no risk of overfitting positional patterns."),
                    ("Works at any length",
                     "You can compute PE(pos) for any pos at inference time, even beyond training length."),
                    ("Unique per position",
                     "Every position gets a mathematically guaranteed distinct vector."),
                    ("Implicit relative info",
                     "The linear offset property means the model can extract relative distances via attention."),
                ],
                cons=[
                    ("Not task-aware",
                     "The fixed formula can't adapt — if your task has unusual positional patterns (i.e, instead of Paragraphs → sentences → words you have Code → functions → blocks → lines), the model can't learn a better encoding."),
                    ("Poor extrapolation in practice",
                     "Although the formula is defined for any length, models trained on short sequences often degrade badly on much longer ones, because attention statistics change."),
                    ("Only additive",
                     "PE is just added to token embeddings — the relative-position signal is implicit and must be reconstructed by the model, not explicit."),
                    ("Superseded",
                     "Modern LLMs use RoPE or ALiBi which provide genuine relative-position encoding with better length generalisation."),
                ],
            )
            notable_uses(["Original Transformer (2017)", "Early BERT experiments", "T5 (base version)"])
