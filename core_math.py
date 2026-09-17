"""
core_math.py -- every mathematical primitive RLT needs, implemented from scratch.

CONSTRAINT (project-wide): pure Python only.
    no numpy, no torch, no tensorflow, no jax, no autograd, no framework.
    allowed: math (sqrt/exp/erf/tanh/pi/log), and nothing else.

A "vector" is  list[float]                of length d.
A "matrix" is  list[list[float]]          with len(M) == d_out, len(M[0]) == d_in.
               The paper uses column vectors, so  matvec(M, x)  is  M @ x.
"heads"   are  list[list[float]]          of shape [H][d_head].

Nothing here knows what a Transformer is. These are the bricks.
Every non-obvious function carries the equation it implements.
"""

import math

# only these names from math are used anywhere in the project
__all__ = [
    "vadd", "vsub", "vscale", "vmul", "vneg", "dot", "vzeros", "vsum",
    "matvec", "matmul", "transpose", "mzeros",
    "mean", "var", "rms", "l2", "vmin", "vmax",
    "softmax", "log_softmax", "sigmoid", "tanh_", "gelu_tanh", "gelu_exact",
    "rmsnorm", "rmsnorm_raw",
    "concat", "slice_", "split_heads", "merge_heads",
    "causal_mask", "sliding_window_mask", "apply_mask",
    "attention", "multihead_attention",
    "LCG", "randn_vector", "randn_matrix",
    "shape", "summarize", "allclose",
]

NEG_INF = float("-inf")


# =============================================================================
# 1. VECTOR ARITHMETIC
# =============================================================================

def vadd(a, b):
    """(a + b)_i = a_i + b_i"""
    assert len(a) == len(b), f"vadd length mismatch: {len(a)} vs {len(b)}"
    return [a[i] + b[i] for i in range(len(a))]


def vsub(a, b):
    """(a - b)_i = a_i - b_i"""
    assert len(a) == len(b), f"vsub length mismatch: {len(a)} vs {len(b)}"
    return [a[i] - b[i] for i in range(len(a))]


def vscale(c, a):
    """(c * a)_i = c * a_i        scalar times vector"""
    return [c * a[i] for i in range(len(a))]


def vmul(a, b):
    """(a (*) b)_i = a_i * b_i    elementwise (Hadamard) product.
    This is the '(*)' in eq (2.11):  u_t = e_t + alpha * g_t (*) W_s r_{t-1}
    """
    assert len(a) == len(b), f"vmul length mismatch: {len(a)} vs {len(b)}"
    return [a[i] * b[i] for i in range(len(a))]


def vneg(a):
    return [-x for x in a]


def vzeros(n):
    return [0.0] * n


def vsum(a):
    """sum_i a_i -- written out rather than calling sum(), so the loop is visible."""
    total = 0.0
    for x in a:
        total += x
    return total


def dot(a, b):
    """<a, b> = sum_i a_i b_i"""
    assert len(a) == len(b), f"dot length mismatch: {len(a)} vs {len(b)}"
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


# =============================================================================
# 2. MATRIX ARITHMETIC
# =============================================================================

def matvec(M, x):
    """y = M x,  y_i = sum_j M_ij x_j.      M is [d_out][d_in], x is [d_in]."""
    assert len(M) > 0, "matvec on empty matrix"
    assert len(M[0]) == len(x), f"matvec shape mismatch: M is [{len(M)},{len(M[0])}], x is [{len(x)}]"
    return [dot(M[i], x) for i in range(len(M))]


def matmul(A, B):
    """C = A B,  C_ij = sum_k A_ik B_kj.    A is [n][k], B is [k][m], C is [n][m]."""
    assert len(A[0]) == len(B), f"matmul shape mismatch: A cols {len(A[0])} vs B rows {len(B)}"
    n, k, m = len(A), len(B), len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for p in range(k):
                acc += A[i][p] * B[p][j]
            C[i][j] = acc
    return C


def transpose(M):
    """(M^T)_ij = M_ji"""
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]


def mzeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


# =============================================================================
# 3. REDUCTIONS  (the statistics the shape tracker of Phase 5 will report)
# =============================================================================

def mean(a):
    """mean(a) = (1/n) sum_i a_i"""
    return vsum(a) / len(a)


def var(a):
    """var(a) = (1/n) sum_i (a_i - mean(a))^2      -- population variance, not sample."""
    m = mean(a)
    return vsum([(x - m) ** 2 for x in a]) / len(a)


def rms(a):
    """rms(a) = sqrt( (1/n) sum_i a_i^2 )

    NOTE this is the *uncentered* second moment -- it does NOT subtract the mean.
    That is the whole point of RMSNorm vs LayerNorm.
    """
    return math.sqrt(vsum([x * x for x in a]) / len(a))


def l2(a):
    """||a||_2 = sqrt( sum_i a_i^2 ).     Relation: ||a||_2 = sqrt(n) * rms(a)."""
    return math.sqrt(vsum([x * x for x in a]))


def vmin(a):
    m = a[0]
    for x in a:
        if x < m:
            m = x
    return m


def vmax(a):
    m = a[0]
    for x in a:
        if x > m:
            m = x
    return m


# =============================================================================
# 4. NONLINEARITIES
# =============================================================================

def softmax(z):
    """p_i = exp(z_i) / sum_j exp(z_j)

    Computed as  exp(z_i - max z) / sum_j exp(z_j - max z),  which is algebraically
    identical (the exp(max) cancels) but does not overflow.

    Entries equal to -inf (masked positions) map to exactly 0.0.
    """
    m = vmax(z)
    if m == NEG_INF:
        raise ValueError("softmax over an all-masked row: every logit is -inf")
    exps = []
    for x in z:
        if x == NEG_INF:
            exps.append(0.0)
        else:
            exps.append(math.exp(x - m))
    denom = vsum(exps)
    return [e / denom for e in exps]


def log_softmax(z):
    """log p_i = z_i - log sum_j exp(z_j)
                = (z_i - m) - log sum_j exp(z_j - m)
    Kept separate from log(softmax(z)) because taking the log of a tiny probability
    loses precision; this form does not. Used by the training loss in Phase 10.
    """
    m = vmax(z)
    if m == NEG_INF:
        raise ValueError("log_softmax over an all-masked row")
    shifted = [(x - m) if x != NEG_INF else NEG_INF for x in z]
    lse = math.log(vsum([math.exp(x) if x != NEG_INF else 0.0 for x in shifted]))
    return [(x - lse) if x != NEG_INF else NEG_INF for x in shifted]


def sigmoid(x):
    """sigma(x) = 1 / (1 + exp(-x))       -- the 'sigma' of eq (2.10).

    Branch on the sign so neither exp() overflows:
        x >= 0:  1 / (1 + exp(-x))
        x <  0:  exp(x) / (1 + exp(x))     (same value, multiply through by exp(x))
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def tanh_(x):
    """tanh(x) = (e^x - e^-x) / (e^x + e^-x). Delegates to math.tanh, which is stable."""
    return math.tanh(x)


def gelu_exact(x):
    """GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    Phi is the standard normal CDF. This is the definition from Hendrycks & Gimpel.
    """
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_tanh(x):
    """GELU(x) ~= 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))

    The tanh approximation. Agrees with gelu_exact to ~1e-3 absolute.

    [UNSPECIFIED BY PAPER] the paper writes only 'FFN_l' and never names an
    activation. Both forms are provided so the choice is visible and swappable;
    see research_notes.md section 8, item A6.
    """
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3)))


# =============================================================================
# 5. RMSNORM  -- eq (2.9) and the seven other norm sites
# =============================================================================

def rmsnorm_raw(x, eps=1e-5):
    """The normalization alone, with no learned gain:

        r_i = x_i / sqrt( mean(x^2) + eps )
            = x_i / sqrt( (1/n) sum_j x_j^2 + eps )

    Note eps sits INSIDE the sqrt (the usual convention). The mean is over the
    SQUARES -- the vector's own mean is never subtracted.
    """
    ms = vsum([v * v for v in x]) / len(x)
    denom = math.sqrt(ms + eps)
    return [v / denom for v in x]


def rmsnorm(x, gain=None, eps=1e-5):
    """Full RMSNorm:   out_i = gain_i * x_i / sqrt(mean(x^2) + eps)

    [UNSPECIFIED BY PAPER] whether the paper's RMSNorm carries a learned gain.
    We include one, initialized to all-ones, so the no-gain variant is exactly
    the initialization. See research_notes.md section 8, item B5.

    Property worth remembering (and tested in test_core_math.py):
        rmsnorm_raw is scale-invariant only in the limit eps -> 0.
        For c > 0:   rmsnorm_raw(c*x) == rmsnorm_raw(x)   when eps == 0.
        With eps > 0 the invariance is approximate, and the error grows as
        ||x|| -> 0. This matters: a state s_{t-1} that collapses toward zero
        does NOT get renormalized back to unit RMS.
    """
    r = rmsnorm_raw(x, eps)
    if gain is None:
        return r
    assert len(gain) == len(x), f"rmsnorm gain length {len(gain)} != input {len(x)}"
    return vmul(gain, r)


# =============================================================================
# 6. SHAPE PLUMBING
# =============================================================================

def concat(a, b):
    """[a ; b] -- the concatenation in eq (2.10), giving a vector of length 2d."""
    return list(a) + list(b)


def slice_(a, start, stop):
    """a[start:stop], written explicitly because Phase 8 slices caches constantly."""
    return a[start:stop]


def split_heads(x, n_heads):
    """[d] -> [H][d_head], contiguous split.

    Head h owns dimensions [h*d_head, (h+1)*d_head). This is a VIEW decision,
    not mathematics -- it is the standard contiguous convention, chosen so that
    merge_heads(split_heads(x, H)) == x exactly.
    """
    d = len(x)
    assert d % n_heads == 0, f"d={d} not divisible by n_heads={n_heads}"
    dh = d // n_heads
    return [x[h * dh:(h + 1) * dh] for h in range(n_heads)]


def merge_heads(heads):
    """[H][d_head] -> [d], the exact inverse of split_heads."""
    out = []
    for h in heads:
        out.extend(h)
    return out


# =============================================================================
# 7. MASKS
# =============================================================================

def causal_mask(T):
    """mask[i][j] = True iff query position i may attend to key position j.

    Causal: j <= i. Positions are 0-indexed here; the paper is 1-indexed.

        T=4:    j: 0 1 2 3
             i=0:  T . . .
             i=1:  T T . .
             i=2:  T T T .
             i=3:  T T T T
    """
    return [[j <= i for j in range(T)] for i in range(T)]


def sliding_window_mask(T, W):
    """Causal AND within a window of W positions INCLUDING the current one.

        mask[i][j] = True iff  (i - W + 1) <= j <= i

    The paper (sec 2.1) is explicit that W includes the current token, so W=1
    means 'attend to yourself only'.

        T=5, W=3:   j: 0 1 2 3 4
                 i=0:  T . . . .
                 i=1:  T T . . .
                 i=2:  T T T . .
                 i=3:  . T T T .      <- position 0 has left the window
                 i=4:  . . T T T
    """
    assert W >= 1, f"window W must be >= 1, got {W}"
    return [[(j <= i) and (j >= i - W + 1) for j in range(T)] for i in range(T)]


def apply_mask(scores_row, mask_row):
    """Set disallowed logits to -inf so softmax sends them to exactly 0."""
    assert len(scores_row) == len(mask_row)
    return [scores_row[j] if mask_row[j] else NEG_INF for j in range(len(scores_row))]


# =============================================================================
# 8. ATTENTION
# =============================================================================

def attention(q, keys, values, scale=None):
    """Single-head scaled dot-product attention over an explicit list of keys/values.

        score_j = <q, k_j> / sqrt(d_head)            (eq: the 'QK^T / sqrt(d_k)' step)
        p_j     = softmax(score)_j
        out     = sum_j p_j * v_j

    Returns (out, scores, probs) -- ALL THREE. The intermediates are returned
    rather than hidden because Phase 9 inspects every one of them. A function
    that returned only `out` would be exactly the opaque helper this project
    is built to avoid.

    keys/values are lists of vectors, already restricted to the allowed set.
    For SWA the caller passes only the window; no mask is then needed, because
    the window IS the mask. `scale` defaults to 1/sqrt(d_head).
    """
    assert len(keys) == len(values), f"{len(keys)} keys vs {len(values)} values"
    assert len(keys) > 0, "attention over an empty key set"
    dh = len(q)
    if scale is None:
        scale = 1.0 / math.sqrt(dh)

    scores = [dot(q, k) * scale for k in keys]
    probs = softmax(scores)

    out = vzeros(len(values[0]))
    for j in range(len(values)):
        out = vadd(out, vscale(probs[j], values[j]))
    return out, scores, probs


def multihead_attention(q_heads, k_heads_list, v_heads_list, scale=None):
    """Run `attention` independently per head and concatenate.

        q_heads       [H][d_head]              the query, already split
        k_heads_list  [n_keys][H][d_head]      one entry per key position
        v_heads_list  [n_keys][H][d_head]

    Returns (out [d], scores [H][n_keys], probs [H][n_keys]).

    The per-head scores and probs are returned as [H][n_keys] so that Phase 9
    can print the full [heads, query_positions, key_positions] tensor by
    stacking over query positions.
    """
    H = len(q_heads)
    n = len(k_heads_list)
    assert n == len(v_heads_list), f"{n} key entries vs {len(v_heads_list)} value entries"
    assert n > 0, "multihead_attention over an empty key set"

    out_heads, all_scores, all_probs = [], [], []
    for h in range(H):
        keys_h = [k_heads_list[j][h] for j in range(n)]
        vals_h = [v_heads_list[j][h] for j in range(n)]
        o, sc, pr = attention(q_heads[h], keys_h, vals_h, scale)
        out_heads.append(o)
        all_scores.append(sc)
        all_probs.append(pr)

    return merge_heads(out_heads), all_scores, all_probs


# =============================================================================
# 9. RANDOM INITIALIZATION
# =============================================================================

class LCG:
    """A linear congruential generator, so initialization has no hidden machinery.

        state <- (a * state + c) mod m       with the Numerical Recipes constants
        a = 1664525, c = 1013904223, m = 2^32

    Reason for not using `random`: reproducibility here should be readable in
    ten lines rather than inherited from a Mersenne Twister. Same seed gives
    the same weights on any platform, any Python version.

    LCG low bits are famously poor; we only ever use the high 24 bits via
    `uniform`, which is more than enough for weight init.
    """

    A = 1664525
    C = 1013904223
    M = 2 ** 32

    def __init__(self, seed=0):
        self.state = seed % self.M

    def next_u32(self):
        self.state = (self.A * self.state + self.C) % self.M
        return self.state

    def uniform(self):
        """Uniform on (0, 1). Uses the top 24 bits; never returns exactly 0."""
        u = self.next_u32() >> 8          # top 24 bits
        return (u + 0.5) / (2 ** 24)      # in (0,1), open at both ends

    def normal(self):
        """Standard normal via the Box-Muller transform:

            u1, u2 ~ Uniform(0,1) independent
            z = sqrt(-2 ln u1) * cos(2 pi u2)    is N(0,1)

        We discard the second output sin(...) for simplicity; it costs one extra
        uniform per sample and keeps the stream easy to follow.
        """
        u1 = self.uniform()
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def randn_vector(n, rng, std=1.0):
    """n iid samples from N(0, std^2)."""
    return [rng.normal() * std for _ in range(n)]


def randn_matrix(rows, cols, rng, std=1.0):
    """[rows][cols] iid N(0, std^2).

    [UNSPECIFIED BY PAPER] the paper gives no initialization scheme at all.
    Callers pass `std` explicitly (e.g. 1/sqrt(fan_in)) so the choice is always
    at the call site and never buried here. See research_notes.md section 8,
    Group C.
    """
    return [[rng.normal() * std for _ in range(cols)] for _ in range(rows)]


# =============================================================================
# 10. INSPECTION HELPERS  (seed of the Phase 5 shape tracker)
# =============================================================================

def shape(x):
    """Recursive shape of nested lists, as a tuple. shape(3.0) == ()."""
    if isinstance(x, (int, float)):
        return ()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return (0,)
        return (len(x),) + shape(x[0])
    raise TypeError(f"shape() does not handle {type(x)}")


def _flatten(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for item in x:
        out.extend(_flatten(item))
    return out


def summarize(name, x):
    """The Phase 5 record for one value: name, shape, dtype, min, max, mean, rms, l2.

    Returns a dict rather than a formatted string, so trajectory.json can hold
    the raw numbers. Formatting belongs in inspect.py, not here.
    """
    flat = _flatten(x)
    finite = [v for v in flat if v != NEG_INF and v == v]   # drop -inf and NaN
    rec = {
        "name": name,
        "shape": list(shape(x)),
        "dtype": "float",
        "n": len(flat),
        "n_masked": len(flat) - len(finite),
    }
    if finite:
        rec.update({
            "min": vmin(finite),
            "max": vmax(finite),
            "mean": mean(finite),
            "rms": rms(finite),
            "l2": l2(finite),
        })
    return rec


def allclose(a, b, atol=1e-9, rtol=1e-7):
    """Elementwise |a-b| <= atol + rtol*|b| over arbitrarily nested lists.

    Used by every invariant test. Returns (ok, max_abs_diff, index_of_worst).
    """
    fa, fb = _flatten(a), _flatten(b)
    assert len(fa) == len(fb), f"allclose shape mismatch: {len(fa)} vs {len(fb)}"
    worst, worst_i = 0.0, -1
    for i in range(len(fa)):
        d = abs(fa[i] - fb[i])
        if d > worst:
            worst, worst_i = d, i
        if d > atol + rtol * abs(fb[i]):
            return False, worst, worst_i
    return True, worst, worst_i
