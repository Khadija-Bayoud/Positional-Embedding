"""
Learned Positional Embedding page.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.styles import DARK_TEMPLATE, COLORS
from utils.math_helpers import sinusoidal_pe, learned_pe_sim
from components.ui import hero, sec_header, callout, math_block, example_block, viz_caption, pros_cons, notable_uses


PAPER_URL   = "https://arxiv.org/abs/1810.04805"
PAPER_TITLE = "BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al. 2018"


def render():
    hero(
        "🎓", "Learned Positional Embedding",
        "Instead of a fixed formula, each position gets a trainable vector optimised "
        "end-to-end alongside the rest of the model. Used in BERT, GPT-2, and many foundational models.",
        PAPER_TITLE, PAPER_URL,
    )

    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        seq_len = st.slider("Sequence length", 16, 512, 128, key="lrn_seq")
        d_model = st.select_slider("d_model", [16, 32, 64, 128, 256], value=64, key="lrn_d")
        seed    = st.slider("Simulated weight seed", 0, 50, 7, key="lrn_seed")

    with col_main:
        t1, t2, t3, t4 = st.tabs(["💡 Intuition & Need", "📐 Math", "📊 Visualization", "⚖️ Pros & Cons"])

        # ── INTUITION ─────────────────────────────────────────────────────────
        with t1:
            st.markdown("""
            <div class="card">
            <div class="sec-header">💡 Core Idea</div>
            <p style='color:#94a3b8;'>
              Sinusoidal encoding is hand-crafted by a human. But what if we just said:
              <b style='color:#e2e8f0;'>"Model, figure out for yourself what positional information
              is most useful for this task"</b>?<br><br>
              Learned PE does exactly that. We create a lookup table — a big matrix with
              one row per position and one column per embedding dimension. At the start
              of training these rows are random (or initialised like sinusoidal). 
              As training proceeds, gradient descent nudges each row until it encodes
              whatever positional pattern helps the model's loss go down.
            </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
            <div class="sec-header">🎯 Why Do We Need It?</div>
            <p style='color:#94a3b8;'>
              In many tasks, <em>specific absolute positions carry predictable semantic roles</em>.
              For example, in question answering the first token is often [CLS] (a classification signal),
              and the question tokens tend to come before the passage tokens. A learned PE can
              specialise position 0 to mean "classification anchor" in a way no fixed formula can.<br><br>
              Learned PE is also dead-simple to implement:
              just one <code>nn.Embedding(max_len, d_model)</code> layer.
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
            \mathbf{E} \in \mathbb{R}^{L_{\max} \times d} \quad \leftarrow \text{trainable parameter matrix}
            """)
            st.latex(r"""
            \mathbf{z}_i = \text{TokenEmb}(x_i) + \mathbf{E}[i], \quad i = 0,\ldots,L-1
            """)

            callout("""
            <b>Reading the formula:</b><br>
            • <b>E</b> is the positional embedding table — it has one row per position,
              trained from scratch (or sinusoidal init).<br>
            • <b>E[i]</b> just means "look up row i in the table".<br>
            • We add the position vector to the token's word embedding.
              The model then sees a combined "what word + where it is" representation.
            """)

            st.latex(r"""
            \mathcal{L} \xrightarrow{\text{backprop}} \nabla_{\mathbf{E}} \mathcal{L}
            \;\Rightarrow\; \mathbf{E} \leftarrow \mathbf{E} - \eta\,\nabla_{\mathbf{E}}\mathcal{L}
            """)

            callout("""
            <b>Training:</b> The position table <b>E</b> is updated by gradient descent exactly
            like any other weight matrix. After training, position 0's vector reflects
            everything the model learned about "being in first position" across all training examples.
            """)

            st.markdown("---")
            sec_header("🔢 Worked Example")
            st.markdown("<p style='color:#94a3b8;'>d_model = 4, max positions = 3. The table starts randomly:</p>", unsafe_allow_html=True)

            example_block(
                "E (before training) — random init:\n"
                "  E[0] = [-0.12,  0.87, -0.34,  0.45]   ← position 0\n"
                "  E[1] = [ 0.56, -0.23,  0.91, -0.11]   ← position 1\n"
                "  E[2] = [-0.78,  0.04,  0.22,  0.63]   ← position 2\n"
                "\n"
                "Token embedding for word 'dog' (learned separately):\n"
                "  TokenEmb('dog') = [0.3, 0.7, -0.5, 0.1]\n"
                "\n"
                "If 'dog' appears at position 1:\n"
                "  z = TokenEmb('dog') + E[1]\n"
                "    = [0.3+0.56, 0.7+(−0.23), −0.5+0.91, 0.1+(−0.11)]\n"
                "    = [0.86,     0.47,          0.41,      −0.01]"
            )

            callout("""
            After training, E[0] and E[1] will look very different
            because the model learned different roles for the two positions —
            not because a formula forced them to be different.
            """)

        # ── VISUALIZATION ─────────────────────────────────────────────────────
        with t3:
            PE_l = learned_pe_sim(seq_len, d_model, seed)
            PE_s = sinusoidal_pe(seq_len, d_model)

            st.markdown("#### Visualization 1 — Learned vs Sinusoidal Structure")
            viz_caption(
                "Left: the simulated learned PE table (sinusoidal init + small training noise). "
                "Right: the pure sinusoidal PE for reference. "
                "Row = a token position, column = an embedding dimension, colour = value. "
                "<b>Try different seeds</b> — each seed simulates a different 'trained' state. "
                "Notice that the overall structure is similar to sinusoidal, but with irregular patches "
                "where the model drifted toward task-specific patterns."
            )
            fig1 = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Learned PE (simulated)", "Sinusoidal PE (reference)"],
                                  horizontal_spacing=0.08)
            for ci, data in enumerate([PE_l, PE_s], 1):
                fig1.add_trace(go.Heatmap(z=data, colorscale="RdBu", zmid=0, showscale=(ci == 2)),
                               row=1, col=ci)
            fig1.update_layout(height=320, **DARK_TEMPLATE, margin=dict(t=50, b=10))
            for ax in fig1.layout:
                if ax.startswith(("xaxis", "yaxis")):
                    fig1.layout[ax].update(showticklabels=False)
            st.plotly_chart(fig1, width='stretch')

            st.markdown("#### Visualization 2 — Similarity Structure")
            viz_caption(
                "Cell (i, j) = cosine similarity between position i's vector and position j's vector. "
                "Bright diagonal = each position is most similar to itself. "
                "Off-diagonal brightness shows how similar nearby vs distant positions look. "
                "For learned PE the pattern is noisier than sinusoidal — positions whose "
                "training role was similar end up with similar vectors even if far apart. "
                "<b>Change the seed</b> to see how different training runs produce different similarity structures."
            )
            sim_l = PE_l @ PE_l.T
            sim_s = PE_s @ PE_s.T
            fig2 = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Learned: similarity", "Sinusoidal: similarity"],
                                  horizontal_spacing=0.08)
            for ci, sim in enumerate([sim_l, sim_s], 1):
                fig2.add_trace(go.Heatmap(z=sim, colorscale="Viridis", showscale=(ci == 2)),
                               row=1, col=ci)
            fig2.update_layout(height=320, **DARK_TEMPLATE, margin=dict(t=50, b=10))
            st.plotly_chart(fig2, width='stretch')

            st.markdown("#### Visualization 3 — Per-Position Norm")
            viz_caption(
                "The L2 norm (length) of each position's embedding vector. "
                "Sinusoidal PE has a near-constant norm (all vectors the same length). "
                "Learned PE can develop unequal norms — positions with more training signal "
                "or more discriminative roles may end up with larger vectors. "
                "A spike at a particular position means the model is 'shouting' that position. "
                "<b>Change d_model</b> to see how dimensionality affects the norm scale."
            )
            norms_l = np.linalg.norm(PE_l, axis=1)
            norms_s = np.linalg.norm(PE_s, axis=1)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=np.arange(seq_len), y=norms_l, name="Learned (sim)",
                                      line=dict(color=COLORS["learned"], width=2.5)))
            fig3.add_trace(go.Scatter(x=np.arange(seq_len), y=norms_s, name="Sinusoidal",
                                      line=dict(color=COLORS["sinusoidal"], width=2.5, dash="dot")))
            fig3.update_layout(title="L2 norm per position", height=260,
                               xaxis_title="Token position", yaxis_title="Norm (vector length)",
                               **DARK_TEMPLATE, margin=dict(t=50, b=30))
            st.plotly_chart(fig3, width='stretch')

        # ── PROS / CONS ───────────────────────────────────────────────────────
        with t4:
            pros_cons(
                pros=[
                    ("Task-adaptive",
                     "The model learns whatever positional pattern lowers its loss — it's not constrained by a human's design choice."),
                    ("Simple to implement",
                     "One line of code: nn.Embedding(max_len, d_model). Every framework supports it."),
                    ("Works well empirically",
                     "On short-sequence NLU tasks (GLUE, SQuAD), learned PE matches or beats sinusoidal."),
                ],
                cons=[
                    ("Hard length limit",
                     "You must fix max_len before training. At inference, the model crashes if given a longer input. There is no position 513 if you only trained up to 512."),
                    ("Later positions underfit",
                     "If long sequences are rare in training data, position vectors near max_len see very few gradient updates and may be poorly learned."),
                    ("More parameters",
                     "BERT-base adds 512 × 768 ≈ 393 K extra parameters just for positional embeddings."),
                    ("No generalisation",
                     "Unlike sinusoidal, there's no mathematical structure ensuring nearby positions are similar — the model must learn that from data alone."),
                ],
            )
            notable_uses(["BERT", "GPT / GPT-2", "RoBERTa", "XLNet", "DistilBERT"])
