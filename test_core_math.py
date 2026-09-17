"""
test_core_math.py -- numerical sanity checks for the Phase 2 primitives.

No pytest, no unittest: a 20-line harness, so this runs anywhere (laptop, Colab,
a bare interpreter) with zero dependencies. Run:  python test_core_math.py

Each check states the INVARIANT it is testing. Several of them print real numbers
rather than just passing, because a few are findings in their own right -- notably
CHECK 17 (RMSNorm's eps is not scale-invariant near zero) and CHECK 22 (the SWA
window/retention identity, which is claim C8 at the mask level).
"""

import math
import os
import re

import core_math as cm

# ---------------------------------------------------------------- harness ----
_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        _PASS.append(name)
        print(f"  ok   {name}" + (f"   {detail}" if detail else ""))
    else:
        _FAIL.append(name)
        print(f"  FAIL {name}" + (f"   {detail}" if detail else ""))


def close(a, b, atol=1e-9, rtol=1e-7):
    ok, worst, _ = cm.allclose(a, b, atol, rtol)
    return ok, f"max|diff|={worst:.3e}"


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


# =============================================================================
print("=" * 72)
print("core_math.py -- numerical sanity checks")
print("=" * 72)

section("CHECK 0: the purity constraint")

# A guard on sys.modules would be measuring the HOST (Colab pre-imports numpy),
# not this code. So check the source: what does core_math.py actually import?
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_math.py")).read()
imports = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", src, re.M))
check("core_math imports nothing but `math`", imports == {"math"}, f"found {sorted(imports)}")

BANNED = {"numpy", "torch", "tensorflow", "jax", "flax", "keras", "transformers", "scipy", "sklearn"}
check("no banned framework named in source", not (BANNED & imports), f"banned hits: {sorted(BANNED & imports)}")

rng = cm.LCG(seed=1234)

# =============================================================================
section("CHECK 1-7: vector and matrix algebra")

a = cm.randn_vector(6, rng)
b = cm.randn_vector(6, rng)

check("1  (a+b)-b == a", *close(cm.vsub(cm.vadd(a, b), b), a))
check("2  a(*)1 == a", *close(cm.vmul(a, [1.0] * 6), a))
check("3  <a,b> == <b,a>", *close([cm.dot(a, b)], [cm.dot(b, a)]))
check("4  <a,a> == ||a||^2", *close([cm.dot(a, a)], [cm.l2(a) ** 2]))
check("5  ||a|| == sqrt(n)*rms(a)", *close([cm.l2(a)], [math.sqrt(6) * cm.rms(a)]))

A = cm.randn_matrix(4, 5, rng)
B = cm.randn_matrix(5, 3, rng)
C = cm.randn_matrix(3, 2, rng)
x = cm.randn_vector(5, rng)

check("6  matvec(M,x)[i] == dot(M[i],x)",
      *close(cm.matvec(A, x), [cm.dot(A[i], x) for i in range(4)]))
check("7a transpose(transpose(A)) == A", *close(cm.transpose(cm.transpose(A)), A))
check("7b (AB)^T == B^T A^T",
      *close(cm.transpose(cm.matmul(A, B)), cm.matmul(cm.transpose(B), cm.transpose(A)), atol=1e-12, rtol=1e-9))
check("7c A(BC) == (AB)C  [associativity]",
      *close(cm.matmul(A, cm.matmul(B, C)), cm.matmul(cm.matmul(A, B), C), atol=1e-12, rtol=1e-9))
check("7d (AB)x == A(Bx)",
      *close(cm.matvec(cm.matmul(A, B), cm.randn_vector(3, cm.LCG(7))),
             cm.matvec(A, cm.matvec(B, cm.randn_vector(3, cm.LCG(7)))), atol=1e-12, rtol=1e-9))

# =============================================================================
section("CHECK 8-13: softmax and friends")

z = [1.0, -2.0, 0.5, 3.0, -0.25]
p = cm.softmax(z)

check("8  softmax sums to 1", *close([cm.vsum(p)], [1.0]))
check("9  softmax is strictly positive", all(v > 0 for v in p))
check("10 softmax(z+c) == softmax(z)  [shift invariance]",
      *close(cm.softmax([v + 17.3 for v in z]), p, atol=1e-12, rtol=1e-9))

# The stability test: naive exp(z_i) would overflow to inf here.
big = [1000.0, 999.0, 1001.0]
pb = cm.softmax(big)
check("11 softmax survives logits of 1e3 (no overflow)",
      abs(cm.vsum(pb) - 1.0) < 1e-12 and all(v == v for v in pb),
      f"p={[round(v,6) for v in pb]}")

masked = cm.apply_mask([1.0, 2.0, 3.0], [True, False, True])
pm = cm.softmax(masked)
check("12 masked entries get exactly 0.0 probability",
      pm[1] == 0.0 and abs(cm.vsum(pm) - 1.0) < 1e-12, f"p={[round(v,6) for v in pm]}")

ls = cm.log_softmax(z)
check("13a exp(log_softmax) == softmax", *close([math.exp(v) for v in ls], p))
check("13b log_softmax == log(softmax) for non-tiny p",
      *close(ls, [math.log(v) for v in p]))

# =============================================================================
section("CHECK 14-16: nonlinearities")

check("14a sigmoid(0) == 0.5", *close([cm.sigmoid(0.0)], [0.5]))
check("14b sigmoid(-x) == 1 - sigmoid(x)",
      *close([cm.sigmoid(-2.7)], [1.0 - cm.sigmoid(2.7)]))
check("14c sigmoid does not overflow at +/-1000",
      cm.sigmoid(1000.0) == 1.0 and cm.sigmoid(-1000.0) < 1e-300,
      f"sigmoid(-1000)={cm.sigmoid(-1000.0):.3e}")
check("14d sigmoid is in (0,1) on a wide sweep",
      all(0.0 <= cm.sigmoid(v) <= 1.0 for v in [-50, -5, -1, 0, 1, 5, 50]))

check("15a gelu(0) == 0", cm.gelu_exact(0.0) == 0.0 and cm.gelu_tanh(0.0) == 0.0)
worst_g = max(abs(cm.gelu_exact(v) - cm.gelu_tanh(v)) for v in [-4 + 0.05 * i for i in range(161)])
check("15b gelu_tanh approximates gelu_exact to <1e-2 on [-4,4]",
      worst_g < 1e-2, f"max|diff|={worst_g:.3e}")
check("15c gelu is not monotone (it dips below 0 for x<0)",
      cm.gelu_exact(-0.5) < 0.0, f"gelu(-0.5)={cm.gelu_exact(-0.5):.6f}")

# =============================================================================
section("CHECK 17: RMSNorm  -- eq (2.9)")

v = cm.randn_vector(8, rng)

r0 = cm.rmsnorm_raw(v, eps=0.0)
check("17a rms(rmsnorm_raw(x)) == 1   when eps=0", *close([cm.rms(r0)], [1.0]))
check("17b rmsnorm_raw does NOT centre: mean is not forced to 0",
      abs(cm.mean(r0)) > 1e-6, f"mean={cm.mean(r0):.6f}")
check("17c scale invariance at eps=0: rmsnorm(c*x) == rmsnorm(x)",
      *close(cm.rmsnorm_raw(cm.vscale(37.0, v), eps=0.0), r0, atol=1e-12, rtol=1e-9))
check("17d learned gain of ones is the identity",
      *close(cm.rmsnorm(v, gain=[1.0] * 8, eps=0.0), r0))

# A finding, not just a check. eps breaks scale invariance, and the break is
# severe exactly when the state collapses toward zero -- which is the regime a
# decaying recurrent state s_t lives in. Printed, not asserted away.
print("\n     eps=1e-5 breaks scale invariance as ||x|| -> 0:")
print(f"     {'scale c':>10} {'rms(x)':>12} {'rms(rmsnorm(c*x))':>20}")
for c in [1e2, 1e0, 1e-1, 1e-2, 1e-3]:
    xc = cm.vscale(c, v)
    print(f"     {c:>10.0e} {cm.rms(xc):>12.3e} {cm.rms(cm.rmsnorm(xc, eps=1e-5)):>20.6f}")
rms_big = cm.rms(cm.rmsnorm(cm.vscale(1e2, v), eps=1e-5))
rms_small = cm.rms(cm.rmsnorm(cm.vscale(1e-3, v), eps=1e-5))
check("17e  (documented) output rms falls well below 1 for a near-zero input",
      rms_big > 0.999 and rms_small < 0.9,
      f"rms@c=1e2 -> {rms_big:.6f}, rms@c=1e-3 -> {rms_small:.6f}")

# =============================================================================
section("CHECK 18-19: shape plumbing")

y = cm.randn_vector(12, rng)
check("18a merge_heads(split_heads(x,H)) == x  for H=1,2,3,4,6,12",
      all(cm.merge_heads(cm.split_heads(y, H)) == y for H in [1, 2, 3, 4, 6, 12]))
check("18b split_heads gives H blocks of d/H",
      cm.shape(cm.split_heads(y, 3)) == (3, 4), f"shape={cm.shape(cm.split_heads(y,3))}")
check("19a concat lengths add", len(cm.concat(a, b)) == len(a) + len(b))
check("19b concat([e;r]) puts e first",
      cm.concat([1.0, 2.0], [9.0, 9.0])[:2] == [1.0, 2.0])

# =============================================================================
section("CHECK 20-22: masks, and the SWA window identity (claim C8)")

cmask = cm.causal_mask(4)
check("20a causal row i has i+1 allowed", [sum(r) for r in cmask] == [1, 2, 3, 4])
check("20b causal is lower-triangular", all(not cmask[i][j] for i in range(4) for j in range(4) if j > i))

for W in [1, 2, 3, 8]:
    m = cm.sliding_window_mask(6, W)
    counts = [sum(r) for r in m]
    expect = [min(W, i + 1) for i in range(6)]
    check(f"21 W={W}: row i allows min(W,i+1) keys", counts == expect, f"{counts}")

check("21b W=1 is the diagonal only",
      all(cm.sliding_window_mask(5, 1)[i][j] == (i == j) for i in range(5) for j in range(5)))

# CLAIM C8, at the mask level.
#   read window at t    = [max(0, t-W+1), t]      (paper 1-indexed: max(1,t-W+1)..t)
#   retained after t    = [max(0, t-W+2), t]      (paper: max(1,t-W+2)..t)
#   assert: retained(t) UNION {t+1} == read_window(t+1), exactly.
print("\n     claim C8: retained(t) + current  ==  read window(t+1)")
c8_ok = True
for W in [1, 2, 3, 8]:
    T = 10
    m = cm.sliding_window_mask(T, W)
    for t in range(T - 1):
        read_t = {j for j in range(T) if m[t][j]}
        retained = {j for j in read_t if j >= t - W + 2}
        read_next = {j for j in range(T) if m[t + 1][j]}
        if retained | {t + 1} != read_next:
            c8_ok = False
            print(f"     MISMATCH W={W} t={t}: retained={sorted(retained)} "
                  f"+{t+1} != {sorted(read_next)}")
        if W == 3 and t < 5:
            print(f"     W=3 t={t}: read={sorted(read_t)}  retained={sorted(retained)}"
                  f"  -> next read={sorted(read_next)}")
check("22 C8 holds for W in {1,2,3,8}, all t", c8_ok,
      "retention rule and read window are exactly consistent")

# =============================================================================
section("CHECK 23-27: attention")

q = cm.randn_vector(4, rng)
keys = [cm.randn_vector(4, rng) for _ in range(5)]
vals = [cm.randn_vector(4, rng) for _ in range(5)]
out, scores, probs = cm.attention(q, keys, vals)

check("23a attention probs sum to 1", *close([cm.vsum(probs)], [1.0]))
check("23b scores match q.k/sqrt(d_head) by hand",
      *close(scores, [cm.dot(q, k) / math.sqrt(4) for k in keys]))
check("23c output == sum_j p_j v_j by hand",
      *close(out, [sum(probs[j] * vals[j][i] for j in range(5)) for i in range(4)]))

# A convex combination cannot leave the hull of the values. This catches sign
# and indexing errors that a sums-to-1 check would not.
hull_ok = all(min(vl[i] for vl in vals) - 1e-12 <= out[i] <= max(vl[i] for vl in vals) + 1e-12
              for i in range(4))
check("24 output lies inside the convex hull of the values", hull_ok)

_, _, p_eq = cm.attention(q, [keys[0]] * 4, vals[:4])
check("25 identical keys give a uniform distribution", *close(p_eq, [0.25] * 4))

out1, _, p1 = cm.attention(q, [keys[0]], [vals[0]])
check("26a a single key gives p=1 and out==that value",
      p1 == [1.0] and cm.allclose(out1, vals[0])[0])

# Hard-attention limit: scale one key up so its score dominates.
sharp_keys = [cm.vscale(80.0, q), keys[1], keys[2]]
out_s, sc_s, p_s = cm.attention(q, sharp_keys, vals[:3])
check("26b a dominant score drives p -> one-hot and out -> that value",
      p_s[0] > 1 - 1e-9 and cm.allclose(out_s, vals[0], atol=1e-8)[0],
      f"p={[round(u,9) for u in p_s]}")

# Multi-head with H=1 must reduce exactly to the single-head path.
qh = cm.split_heads(q, 1)
kh = [cm.split_heads(k, 1) for k in keys]
vh = [cm.split_heads(vv, 1) for vv in vals]
mo, ms, mp = cm.multihead_attention(qh, kh, vh)
check("27a multihead with H=1 == single-head attention", *close(mo, out))
check("27b multihead scores have shape [H][n_keys]",
      cm.shape(ms) == (1, 5), f"shape={cm.shape(ms)}")

q2 = cm.randn_vector(8, rng)
k2 = [cm.randn_vector(8, rng) for _ in range(3)]
v2 = [cm.randn_vector(8, rng) for _ in range(3)]
mo2, ms2, mp2 = cm.multihead_attention(cm.split_heads(q2, 2),
                                       [cm.split_heads(k, 2) for k in k2],
                                       [cm.split_heads(vv, 2) for vv in v2])
check("27c H=2: each head's probs sum to 1 independently",
      all(abs(cm.vsum(row) - 1.0) < 1e-12 for row in mp2))
check("27d H=2: heads do not mix -- head 0 output depends only on head-0 values",
      *close(mo2[:4],
             cm.attention(cm.split_heads(q2, 2)[0],
                          [cm.split_heads(k, 2)[0] for k in k2],
                          [cm.split_heads(vv, 2)[0] for vv in v2])[0]))
check("27e H=2 scale is 1/sqrt(d_head)=1/2, not 1/sqrt(d)",
      *close([ms2[0][0]], [cm.dot(cm.split_heads(q2, 2)[0], cm.split_heads(k2[0], 2)[0]) / 2.0]))

# =============================================================================
section("CHECK 28-30: the random generator")

# NOTE: draw from ONE generator. Writing [cm.LCG(99).normal() for _ in ...]
# rebuilds the generator every iteration and yields the same number 5 times --
# a vacuous test that passes. This bit the first draft of this file.
g99a, g99b, g100 = cm.LCG(99), cm.LCG(99), cm.LCG(100)
s1 = [g99a.normal() for _ in range(5)]
s2 = [g99b.normal() for _ in range(5)]
s3 = [g100.normal() for _ in range(5)]
check("28a same seed gives the same stream", s1 == s2)
check("28b different seeds differ", s1 != s3)
check("28c a stream does not repeat itself", len(set(s1)) == 5,
      f"{len(set(s1))} distinct of 5")

N = 20000
g = cm.LCG(2024)
samples = [g.normal() for _ in range(N)]
mu, sd = cm.mean(samples), math.sqrt(cm.var(samples))
check("29 Box-Muller gives mean~0, std~1", abs(mu) < 0.05 and abs(sd - 1.0) < 0.05,
      f"mean={mu:+.4f} std={sd:.4f} over N={N}")

g5 = cm.LCG(5)
u = [g5.uniform() for _ in range(1000)]
check("30a uniform stays strictly inside (0,1)", all(0.0 < t < 1.0 for t in u),
      f"min={min(u):.6f} max={max(u):.6f}")
check("30b uniform actually varies (catches the rebuilt-generator bug)",
      len(set(u)) > 990, f"{len(set(u))} distinct of 1000")
# crude uniformity: 10 equal bins should each hold ~100 of 1000 draws
bins = [0] * 10
for t in u:
    bins[min(9, int(t * 10))] += 1
check("30c uniform fills 10 bins within +/-40 of 100", all(60 <= c <= 140 for c in bins),
      f"bins={bins}")

# =============================================================================
section("CHECK 31: inspection helpers")

check("31a shape of a scalar is ()", cm.shape(3.0) == ())
check("31b shape of [H][n] nests", cm.shape([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) == (3, 2))
rec = cm.summarize("s_t", [3.0, -1.0, 0.0, 4.0])
check("31c summarize reports shape/min/max/mean/rms/l2",
      rec["shape"] == [4] and rec["min"] == -1.0 and rec["max"] == 4.0
      and abs(rec["mean"] - 1.5) < 1e-12 and abs(rec["l2"] - math.sqrt(26)) < 1e-12,
      f"rms={rec['rms']:.6f} l2={rec['l2']:.6f}")
check("31d summarize counts masked (-inf) entries",
      cm.summarize("m", [1.0, float('-inf'), 2.0])["n_masked"] == 1)
ok_d, worst_d, idx_d = cm.allclose([1.0, 2.0], [1.0, 2.5])
check("31e allclose reports the worst index", (not ok_d) and idx_d == 1, f"worst={worst_d}")

# =============================================================================
print("\n" + "=" * 72)
print(f"PASSED {len(_PASS)}   FAILED {len(_FAIL)}")
if _FAIL:
    for n in _FAIL:
        print(f"   FAILED: {n}")
    raise SystemExit(1)
print("all core_math invariants hold.")
print("=" * 72)
