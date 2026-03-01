"""
Pure-math / NumPy helper functions for every PE method.
No Streamlit imports here — only numpy.
"""
import numpy as np


# ── Sinusoidal ─────────────────────────────────────────────────────────────────
def sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    """Returns (seq_len, d_model) sinusoidal positional encoding matrix."""
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]          # (L, 1)
    i   = np.arange(d_model)[None, :]          # (1, D)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])
    return PE


# ── Learned (simulated) ────────────────────────────────────────────────────────
def learned_pe_sim(seq_len: int, d_model: int, seed: int = 42) -> np.ndarray:
    """
    Simulates a learned PE table by starting from sinusoidal and adding
    small random noise (as if slightly trained away from the init).
    """
    rng  = np.random.default_rng(seed)
    base = sinusoidal_pe(seq_len, d_model)
    noise = rng.normal(0, 0.18, base.shape)
    return base + noise


# ── Relative PE ───────────────────────────────────────────────────────────────
def relative_offset_matrix(seq_len: int) -> np.ndarray:
    """Returns (seq_len, seq_len) signed relative offset matrix (j - i)."""
    pos = np.arange(seq_len)
    return pos[None, :] - pos[:, None]   # row = query, col = key


def relative_bias_matrix(seq_len: int, max_rel: int = 16) -> np.ndarray:
    """
    Shaw et al. style: clips offset then applies a smooth learned-weight proxy.
    """
    rel  = relative_offset_matrix(seq_len)
    clip = np.clip(rel, -max_rel, max_rel)
    # proxy for learned weights: exponential decay from 0
    bias = -np.abs(clip) * 0.1
    return bias


def t5_bucket_matrix(seq_len: int, num_buckets: int = 32, max_dist: int = 128) -> np.ndarray:
    """T5-style log-bucket relative bias matrix."""
    rel = relative_offset_matrix(seq_len)   # signed (query→key)
    n   = -rel                              # distance from query to key
    max_exact = num_buckets // 2
    is_small  = (n >= 0) & (n < max_exact)
    val_if_large = max_exact + (
        np.log(np.maximum(n, 1).astype(float) / max_exact) /
        np.log(max_dist / max_exact) * (num_buckets - max_exact)
    ).astype(int)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    bucket = np.where(is_small, n, val_if_large)
    return bucket.astype(float)


# ── RoPE ──────────────────────────────────────────────────────────────────────
def rope_freqs(seq_len: int, d_model: int, base: float = 10000.0):
    """
    Returns:
        cos_mat : (seq_len, d_model//2)
        sin_mat : (seq_len, d_model//2)
        angle_mat: (seq_len, d_model//2)  — raw angles in radians
    """
    theta = 1.0 / (base ** (2 * np.arange(d_model // 2) / d_model))
    pos   = np.arange(seq_len)
    angles = np.outer(pos, theta)          # (seq, d/2)
    return np.cos(angles), np.sin(angles), angles


def apply_rope(X: np.ndarray, cos_mat: np.ndarray, sin_mat: np.ndarray) -> np.ndarray:
    """Apply RoPE rotation to a (seq_len, d_model) matrix."""
    X_out = X.copy()
    d = X.shape[1]
    for k in range(0, d, 2):
        j = k // 2
        X_out[:, k]   = X[:, k]   * cos_mat[:, j] - X[:, k+1] * sin_mat[:, j]
        X_out[:, k+1] = X[:, k+1] * cos_mat[:, j] + X[:, k]   * sin_mat[:, j]
    return X_out


# ── ALiBi ─────────────────────────────────────────────────────────────────────
def alibi_slopes(n_heads: int) -> np.ndarray:
    """Returns per-head slopes: mₕ = 1 / 2^(8h/n_heads)."""
    h = np.arange(1, n_heads + 1)
    return 1.0 / (2 ** (8 * h / n_heads))


def alibi_bias_matrix(seq_len: int, n_heads: int = 8):
    """
    Returns:
        biases : (n_heads, seq_len, seq_len) — bias[h,i,j] = -slope_h * |i-j|
        slopes : (n_heads,)
    """
    m    = alibi_slopes(n_heads)
    pos  = np.arange(seq_len)
    dist = np.abs(pos[:, None] - pos[None, :])      # (seq, seq) unsigned distance
    biases = -m[:, None, None] * dist[None, :, :]   # (heads, seq, seq)
    return biases, m
