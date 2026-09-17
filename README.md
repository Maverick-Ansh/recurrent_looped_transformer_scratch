# recurrent_looped_transformer_scratch

A computational dissection of **Recurrent Looped Transformer** (Yifan Zhang, September 12 2026), built in **pure Python** — no PyTorch, no TensorFlow, no JAX, no NumPy, no autograd, no framework of any kind.

**The primary artifact is a Colab notebook** — everything (paper notes, computation graph, all primitives, all tests) is written inline there, so there is no module to jump to while reading. This repo is the durable mirror.

The goal is not an implementation. The goal is to open the model up and be able to answer, with actual numbers: what is inside the recurrent state at token `t`, what crosses from `t` to `t+1`, what the gate does numerically, where information from early tokens survives, and where it disappears.

---

## Status — all 18 phases complete

Everything lives in **`RLT_dissection.ipynb`** (Colab). 44 cells, every one executed.

| Phase | What | State |
|---|---|---|
| 0–2 | paper extracted, computation graph, primitives | done — 67/67 invariants |
| 3–5 | tiny RLT (d=8, L=1+1), full forward trace, shape tracker | done |
| 6, 17 | trajectories across time, ASCII plots + raw data files | done |
| 7 | recurrence dissection, experiments A–I, 2x2 over (alpha, W) | done |
| 8–9 | SWA cache as a first-class object, attention as arithmetic | done |
| 10–12 | hand-written reverse-mode tape, training, parameter + gradient dissection | done — all VJPs finite-difference checked |
| 13–16 | information flow, all 11 claims, ablation lab (17 trained models) | done |
| 18 | findings | done |

**Headline results**

- **All 11 extracted claims CONFIRMED**, several at 0 ulp.
- **There are two recurrent channels, and `alpha = 0` only severs one.** With the state channel cut, the SWA cache still carries a decoder-internal perturbation forward for exactly `W-1` steps, then bit-exact zero. The only true non-recurrence control is `alpha = 0` **and** `W = 1`.
- **Claim C9 quantified:** the true `ds_t/ds_j` exceeds the product of single-step `ds/ds` factors by up to **8904x** at init (22–35x after training), and by exactly 1.00x at one step — as the structure requires.
- **Claim C10:** gradient reaches all 9 positions while the `W=3` cache reaches 3.
- **Truncated BPTT can be gradient-inflating** — "detach s only" has 100.6% of the full gradient norm. The norm is the wrong statistic; the angle is right: `cos(g, g_full) = 0.925`.
- **The merge is scale-invariant in `s_{t-1}`** (eq 2.9 normalizes first), so `||s_t||` is not an information channel, and `s := 0` is bit-identical to `alpha = 0`.
- **Ablation:** full 84.09% vs fully-non-recurrent 62.85% on running parity, complete separation across 3 seeds, exact one-sided p = 0.050 — the smallest this design can produce. Monotone ordering in the predicted direction. Underpowered; reported as such.
- **The depth-split sweep detected nothing and could not have** — confounded by a fixed step budget, n=1. Reported as a null result about the experiment.

---

## The one thing to know before reading further

**This paper reports no experiments.** It says so three times (§1, §3.2, §8). There is no task, no dataset, no baseline, no table of results, no hyperparameter appendix. Grep confirms it: zero occurrences of `512`, `1365`, `4+4`, `accuracy`, `we train`, `we evaluate`.

That is not a criticism — it is a design report, and it is explicit about being one. But it changes what "reproduction" means here:

- There is nothing to *re-run*. There are **propositions to verify** (§3.1, App. B.1) and a set of stated structural non-equivalences (§2.4, §2.6, App. B, App. C). These are exactly the kind of claims a pure-Python implementation can check **exactly**, because we control every float.
- Any experimental configuration or task in this repo is **ours**, chosen by us, and labelled as such. It is never presented as "the paper's setup".

`research_notes.md` §0 records this in full, including the provenance of a config that is easy to mistake for the paper's.

---

## What the paper actually specifies

Three equations carry the whole architecture.

**The merge** — how the previous state re-enters the model (eqs 2.9–2.11):

```
r_{t-1} = RMSNorm_s(s_{t-1})
g_t     = sigma( W_g [e_t ; r_{t-1}] + b_g )
u_t     = e_t + alpha * g_t (*) W_s r_{t-1}
```

Note `W_s` multiplies `r_{t-1}` — the *normalized* state — not `s_{t-1}`.

**The decoder block** — SWA, then cross-attention to encoder memory, then FFN (eqs 2.12–2.16):

```
b_t^l = z_t^{l-1} + Attn^D( q_t^l, {(k_j,v_j)}_{j=max(1,t-W+1)}^{t} )     causal SWA
a_t^l = b_t^l     + Attn^M( q^M_t, M_{<=t}^{g(l)} )                       cross-attn
z_t^l = a_t^l     + FFN( RMSNorm(a_t^l) )              s_t = z_t^{L_D}
```

**The recurrence** — and this is the part most implementations get wrong:

```
H_t = (s_t, C_t^D)
```

The state that crosses from token to token is **not** just the vector `s_t`. It is `s_t` **and every decoder layer's sliding-window KV cache**. Appendix B makes this explicit with a 2×2 Jacobian:

```
          [ ds_t/ds_{t-1}    ds_t/dC^D_{t-1}   ]
    J_t = [                                    ]
          [ dC^D_t/ds_{t-1}  dC^D_t/dC^D_{t-1} ]
```

> *"A product involving only `ds_t/ds_{t-1}` generally misses paths through decoder KV."* — App. B

**Consequence, already recorded as a design correction:** setting `alpha = 0` does *not* give you a non-recurrent model. It severs channel 1 and leaves channel 2 running. A genuine non-recurrence control needs `alpha = 0` **and** `W = 1` (at `W = 1` the paper's retained set is explicitly empty). Phase 7's ablation is therefore a 2×2 over `(alpha, W)`, not a single knob.

---

## Phase 2 — what is built

`core_math.py`, ~450 lines, imports `math` and nothing else. Vector and matrix algebra, softmax / log-softmax / sigmoid / GELU, RMSNorm, head splitting, causal and sliding-window masks, attention, an LCG + Box-Muller initializer, and the inspection helpers.

Two deliberate design rules:

1. **`attention()` returns `(out, scores, probs)` — all three.** A function returning only `out` would be exactly the opaque helper this project exists to avoid.
2. **Initialization has no hidden machinery.** A 10-line linear congruential generator plus Box-Muller, rather than inheriting a Mersenne Twister, so the same seed gives the same weights on any platform and the whole path from bits to weights is readable.

```
python test_core_math.py     ->   PASSED 67   FAILED 0
```

### Two results from the invariant suite worth keeping

**RMSNorm's `eps` is not scale-invariant near zero.** Textbook RMSNorm is described as scale-invariant, and at `eps = 0` it is. At `eps = 1e-5` it is not, and the failure is severe exactly in the regime a decaying recurrent state lives in:

```
   scale c       rms(x)    rms(rmsnorm(c*x))
     1e+02    9.778e+01             1.000000
     1e+00    9.778e-01             0.999995
     1e-01    9.778e-02             0.999477
     1e-02    9.778e-03             0.951480
     1e-03    9.778e-04             0.295411
```

So a state `s_t` that contracts toward zero does **not** get renormalized back to unit RMS — it keeps shrinking through the norm. Directly relevant to §3.3's caveat that *"contraction ... may suppress the practical contribution of long paths."*

**Claim C8 holds exactly.** The paper's read window is `[max(1, t-W+1), t]` and its retention rule is `[max(1, t-W+2), t]`. Those are different intervals, which is easy to misread as an off-by-one. They are consistent:

```
W=3 t=0: read=[0]        retained=[0]     -> next read=[0,1]
W=3 t=1: read=[0,1]      retained=[0,1]   -> next read=[0,1,2]
W=3 t=2: read=[0,1,2]    retained=[1,2]   -> next read=[1,2,3]
W=3 t=3: read=[1,2,3]    retained=[2,3]   -> next read=[2,3,4]
```

`retained(t) ∪ {t+1} == read_window(t+1)`, exactly, for every `W ∈ {1,2,3,8}` and every `t`. No slack, no gap. Verified at the mask level now; re-verified against the live cache object in Phase 8.

---

## Files

```
RLT_dissection.ipynb   PRIMARY. The whole dissection, inline: paper notes, computation
                       graph, every primitive, every test. Read this one.
research_notes.md      the paper, extracted: notation, every equation, every state and
                       cache, both propositions, the BPTT appendix, and a complete list
                       of what is UNSPECIFIED with our labelled assumptions
architecture.md        the computation graph: INPUT -> OPERATION -> OUTPUT -> SHAPE ->
                       MEANING for every step, with the recurrent path drawn unrolled
core_math.py           the primitives as an importable module. imports `math`, nothing else
test_core_math.py      the same 67 invariants as a standalone script
```

The notebook is authoritative; `core_math.py` / `test_core_math.py` are a runnable mirror of its §2 so the suite can be run headless.

Every claim in the notes is tagged **[PAPER]**, **[UNSPECIFIED BY PAPER]**, or **[IMPLEMENTATION ASSUMPTION]**. Paper facts and our choices are never mixed.

---

## Running it

The notebook needs nothing but Colab (or Jupyter). Headless:

```bash
git clone https://github.com/Maverick-Ansh/recurrent_looped_transformer_scratch
cd recurrent_looped_transformer_scratch
python test_core_math.py        # -> PASSED 67   FAILED 0
```

No install step. No requirements file. Python 3 and the standard library.

The notebook's last cell syncs itself back here (source-only, outputs stripped). It reads a `GITHUB_TOKEN` from Colab Secrets, which Colab only exposes when the cell is run by hand from the UI.
