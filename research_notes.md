# research_notes.md — Recurrent Looped Transformer, read as a scientist

Source: `Recurrent Looped Transformer`, Yifan Zhang, September 12 2026.
Local PDF: `C:\Users\ANSH\Downloads\Recurrent_Looped_Transformer.pdf` (19 pages, 8 sections + appendices A/B/C).
Project page cited by the paper: `https://github.com/yifanzhang-pro/recurrent-looped-tranformer` (sic, typo in the paper).

Every line below is tagged:

- **[PAPER]** — stated in the paper. Section/equation given.
- **[UNSPECIFIED BY PAPER]** — the paper does not say.
- **[IMPLEMENTATION ASSUMPTION]** — our choice, made because we need a number to run code.

Nothing is filled in silently.

---

## 0. The single most important fact about this paper

**[PAPER]** **This paper reports no experiments.** It says so three times:

- §1, last line: *"The report develops these mechanisms; it does not report measured efficiency or scaling results."*
- §3.2: *"no reduced-prefill speedup is claimed."*
- §8: *"The computational definitions are explicit; realized reasoning quality, hardware efficiency, and scaling behavior require future validation."*

Grep of the full text confirms it: zero occurrences of `512`, `1365`, `4+4`, `accuracy`, `we train`, `we evaluate`. There are no tables of results, no task, no dataset, no baseline run, no hyperparameter appendix. The one occurrence of the word "parity" is *"does not prove numerical kernel **parity**"* (§5.3) — a statement about kernel numerics, not the parity task.

### Consequence for this project

The brief for this dissection (Phase 15) states:

> The paper's experimental models use: width 512, FFN width 1365, 4 attention heads, 8 logical layers, SWA window 8, one shared encoder-memory group, alpha = 0.1, and compares 4+4 / 5+3 / 6+2 / 7+1 / 8+0 against an 8-layer decoder-only Transformer.

**That configuration is not in this paper.** It matches the config of an earlier reproduction of this same paper in `C:\Users\ANSH\rlt-reproduce` (`rlt/config.py`: `n_heads: int = 4`, `W: int = 8`, an `alpha` knob, an `alpha = 0` internal control), i.e. it is a *previously chosen* experimental design, not a paper fact. Phase 14's *"RLT can learn parity earlier for certain splits"* is likewise a hypothesis to be tested, not a paper claim to be reproduced.

This does not block anything. It changes what Phases 14–15 mean:

- Phase 14 verifies the paper's **actual** claims — two propositions and a set of stated non-equivalences (§3.1, §2.4, §2.6, App. B, App. C). These are *structural* claims about the computation, and they are exactly the kind of thing a pure-Python implementation can check exactly.
- Phase 15 builds an experiment the paper **defers**, not one it reports. The depth-split sweep is ours to design, and it must be labelled as ours.

Recorded here so it never gets laundered into "the paper says".

---

## 1. Notation

| Symbol | Type / shape | Meaning | Source |
|---|---|---|---|
| `x_{1:S}` | token ids, `x_1 = BOS` | one independent sequence | §2.1 |
| `S` | int | full sequence length | §2.1 |
| `T` | int | serving boundary (end of prompt). **"marks a serving boundary, not a change in the conditional model"** | §2.1 |
| `d` | int | residual width | §2.1 |
| `L_E`, `L_D` | int | encoder / decoder depth | §2.1 |
| `e_t` | `R^d`, column vector | encoder representation of token `t` | §2.1 |
| `s_t` | `R^d`, column vector | recurrent decoder output | §2.1 |
| `C_t^D` | per-layer list of (k,v) pairs | retained decoder SWA KV at **every** decoder layer | §2.1 |
| `H_t = (s_t, C_t^D)` | pair | **complete decoder state** | §2.1 |
| `W` | int `>= 1` | SWA window, **includes the current token** | §2.1 |
| `G` | int in `1..L_D` | number of encoder-memory groups | §2.2 |
| `g(l)` | int | memory group read by decoder layer `l` | §2.2 |
| `M_{<=t}^g` | growing set of (k,v) | encoder-derived cross-attention memory, group `g` | §2.2 |
| `C_t^E` | encoder KV cache | encoder's own causal cache (for incremental encoding) | §2.4 |
| `s_*` | `R^d` | **learned** initial state, part of `Theta` | §2.1, §2.3 |
| `Theta` | all params | `E_theta` encoder part, `D_phi` decoder part | §2.1 |
| `alpha` | scalar | feedback scale in the merge | §2.5 |
| `u_t` | `R^d` | merged decoder input | §2.4 |
| `z_t^l` | `R^d` | decoder layer-`l` output at token `t`; `z_t^0 = u_t` | §2.5 |
| `mu` | dist. | **behavior** policy (the sampler), distinct from `p_Theta` | §5.3 |

Note the two different "W"s: `W` the scalar window size, and `W_g`, `W_s`, `W_V`, `W_o` the weight matrices. The paper overloads the letter. We keep window size as `W` and always subscript matrices.

---

## 2. Every equation, verbatim

### 2.1 Encoder and memory

**(2.1)** `e_{1:T} = E_theta(x_{1:T})`

> **[PAPER]** §2.2: *"Positions within each encoder layer can be processed together using a causal mask. Encoder layers remain sequential."* — i.e. parallel **across positions**, sequential **across layers**. Normal Transformer encoder behaviour.

**(2.2)** `k_t^g = P_K^g(e_t, t)`, `v_t^g = W_V^g RMSNorm_E(e_t)`, `M_{<=t}^g = {(k_j^g, v_j^g)}_{j=1}^{t}`

> **[PAPER]** *"The key map includes normalization, projection, and any positional transformation."* So `P_K^g` is a composite: norm → linear → positional op. `v` gets norm + linear but **no** positional op.
> **[PAPER]** *"This global memory depends on encoder representations, not decoder states."* — this is the crucial asymmetry. `M` is a pure function of `x_{1:t}` and `theta`. It never sees `s`.
> **[PAPER]** `G = 1` shares memory across all decoder layers; `G = L_D` gives every layer its own projections.

### 2.2 The recurrence

**(2.3)** `H_0 = (s_*, {})` — empty cache, learned initial state.

**(2.4)** `u_t = Merge(e_t, s_{t-1})`

**(2.5)** `H_t = (s_t, C_t^D) = D_phi(u_t; M_{<=t}, C_{t-1}^D, t)`, for `t >= 1`

**(2.6)** `p_Theta(x_{t+1} | x_{1:t}) = softmax(W_o RMSNorm_o(s_t))_{x_{t+1}}`

**(2.7)** `H_T = F_T o F_{T-1} o ... o F_1 (H_0)` where `F_t(H) = D_phi(Merge(e_t, s); M_{<=t}, C^D, t)` for `H = (s, C^D)`

> **[PAPER]** §2.3: *"Neither component of `H_T` is reset at the serving boundary."* Both `s` **and** the SWA caches cross the prompt/response boundary.

### 2.3 Merge (the gate)

**(2.9)** `r_{t-1} = RMSNorm_s(s_{t-1})`

**(2.10)** `g_t = sigma( W_g [e_t ; r_{t-1}] + b_g )`

**(2.11)** `u_t = e_t + alpha * g_t (*) W_s r_{t-1}`

where `sigma` = logistic sigmoid, `(*)` = elementwise product, `W_g in R^{d x 2d}`, `W_s in R^{d x d}`, `b_g in R^d`, `g_t in R^d`.

> **CAREFUL — read (2.11) again.** `W_s` is applied to `r_{t-1}` (the **normalized** state), not to `s_{t-1}`. The brief's Phase-4 trace template asks to print `W_s s_{t-1}`; the paper's operand is `W_s r_{t-1}`. We implement the paper. Both will be printed in the trace so the difference is visible.
> **[PAPER]** *"`alpha` controls the feedback scale. A modest nonzero initial feedback scale is a candidate initialization, not an established stability prescription."* — the paper explicitly declines to prescribe a value.
> **[PAPER]** *"Scalar gating or a low-rank `W_s` reduces overhead."* — variants, offered, not chosen.
> **Structurally:** `alpha = 0` severs the **only** path from `s_{t-1}` into `u_t`. `g_t` still depends on `r_{t-1}` through (2.10), but that dependence is multiplied by `alpha`, so at `alpha = 0` the state contributes nothing to `u_t`. It is a clean internal ablation with identical parameter count and identical block count. **Note it does not sever the SWA path** — see §7 below.

### 2.4 Decoder block (order: SWA → cross-attention → FFN)

For `z_t^0 = u_t`:

**(2.12)** `q_t^{D,l} = P_Q^{D,l}(z_t^{l-1}, t)`

**(2.13)** `k_t^{D,l} = P_K^{D,l}(z_t^{l-1}, t)`,  `v_t^{D,l} = W_V^{D,l} RMSNorm_{S,l}(z_t^{l-1})`

**(2.14)** `b_t^l = z_t^{l-1} + Attn_l^D( q_t^{D,l}, {(k_j^{D,l}, v_j^{D,l})}_{j = max(1, t-W+1)}^{t} )`

**(2.15)** `a_t^l = b_t^l + Attn_l^M( P_Q^{M,l}(b_t^l, t), M_{<=t}^{g(l)} )`

**(2.16)** `z_t^l = a_t^l + FFN_l( RMSNorm_{D,l}(a_t^l) )`,  `s_t = z_t^{L_D}`

> **[PAPER]** *"The attention operators include their output projections; query/key maps include their respective normalizations and positional transformations."*
> **[PAPER]** *"At each layer, current KV is formed before SWA, using the layer input, so current-position attention introduces no circular dependency."* — `k_t, v_t` are computed from `z_t^{l-1}`, which is available; then the current token attends to itself plus history. No fixpoint.
> **[PAPER]** *"Historical decoder KV comes from `C_{t-1}^D`."*
> **[PAPER]** the retention rule: *"After the update, retain positions `max(1, t-W+2), ..., t` in `C_t^D`; this set is empty for `W = 1`."*
> **[PAPER]** *"Alternative sublayer orders define different variants and must be used consistently in all execution modes."*

**Invariant worth testing (Phase 8):** the read set in (2.14) is `[max(1, t-W+1), t]`, which is `<= W` entries **including** `t`. The retained set is `[max(1, t-W+2), t]`, which is `<= W-1` entries. At step `t+1` the read set is `[max(1, t-W+2), t+1]` = retained set ∪ {current}. So **retention + current KV exactly reconstructs the next read window**, with no slack and no gap. This is a hard equality our cache must satisfy at every `t` and every `W`.

### 2.5 The tied ("looped") configuration

> **[PAPER]** §2.6: `L_E = L_D = L`. Encoder self-attention at layer `l` and decoder SWA at layer `l` **share** compatible Q/K/V/O matrices **and** FFNs. Cross-attention has **separate** query/output projections plus the memory projections of (2.2). Stage-specific normalizations, the merge, and the readout are **explicit separate modules**.
> **[PAPER]** *"This is parameter reuse with different attention wiring, not activation copying. No decoder output is identified with an encoder output."*
> **[PAPER]** *"a decoder block adds cross-attention to the reused attention/FFN core; two logical passes do not imply equal per-block FLOPs."*
> **[PAPER]** An untied `E_theta, D_phi` *"preserves the complete-state recurrence while removing depth-wise parameter reuse."* — so tying is optional and orthogonal.
> **[PAPER]** *"Encoder-memory group sharing is an independent axis and does not merge or eliminate the layerwise decoder SWA caches."*

This is the sense in which the model is "looped": **one backbone, two logical passes** (once as encoder, once as decoder), 96 logical block evaluations per token at `L = 48`.

---

## 3. Inventory of states and caches

Four distinct stores. The paper is emphatic that they are not interchangeable.

| # | Object | Shape | Lifetime | Depends on | Crosses T? |
|---|---|---|---|---|---|
| 1 | `s_t` | `[d]` | one vector, overwritten each token | `x_{1:t}`, `Theta` | **yes** (§2.3) |
| 2 | `C_t^D` | `L_D` x `<= W-1` x `(k:[d], v:[d])` | bounded ring per layer | `x_{1:t}`, `Theta` | **yes** (§2.3) |
| 3 | `M_{<=t}^g` | `G` x `t` x `(k:[d], v:[d])` | grows without bound | `x_{1:t}`, `theta` **only** | yes (append-only) |
| 4 | `C_t^E` | encoder-internal causal KV | grows without bound | `x_{1:t}`, `theta` only | yes (append-only) |

> **[PAPER]** §4.2: *"Decoder layers read this memory and separately append decoder-derived KV to their bounded layerwise SWA caches. **These caches cannot be replaced by encoder memory.**"*
> **[PAPER]** App. B: *"Encoder memory provides access to past encoder information. Decoder SWA additionally exposes cached projections of recent decoder activations. **Neither is an unrestricted archive of all previous decoder states.** Evicted entries can still influence later computation through states or retained activations that previously consumed them."*

That last sentence is a directly testable claim about information flow (Phase 13): eviction from the SWA cache is **not** erasure of influence, because the evicted entry already shaped `s` at the time it was read.

---

## 4. Parameter inventory

Everything the paper names. Shapes given where stated; `?` where the paper is silent.

| Parameter | Shape | Where | Notes |
|---|---|---|---|
| token embedding | `[V, d]` | implied by (2.1) | **[UNSPECIFIED]** — never written down |
| encoder blocks | `L_E` x (attn Q/K/V/O + FFN + norms) | (2.1) | internals **[UNSPECIFIED]** |
| `W_V^g` | `[d, d]` | (2.2) | memory value projection, per group |
| `P_K^g` | norm + `[d,d]` + pos | (2.2) | memory key map, per group |
| `s_*` | `[d]` | (2.3) | **learned** initial state |
| `W_g` | `[d, 2d]` | (2.10) | gate |
| `b_g` | `[d]` | (2.10) | gate bias — the **only** bias the paper names |
| `W_s` | `[d, d]` | (2.11) | state projection; low-rank variant offered |
| `alpha` | scalar | (2.11) | **value [UNSPECIFIED]** |
| `P_Q^{D,l}`, `P_K^{D,l}` | norm + `[d,d]` + pos | (2.12)(2.13) | decoder SWA q/k maps |
| `W_V^{D,l}` | `[d, d]` | (2.13) | decoder SWA value |
| `Attn_l^D` output proj | `[d, d]` | (2.14) | "operators include their output projections" |
| `RMSNorm_{S,l}` | `[d]` gain? | (2.13) | |
| `P_Q^{M,l}` | norm + `[d,d]` + pos | (2.15) | cross-attn query |
| `Attn_l^M` output proj | `[d, d]` | (2.15) | |
| `FFN_l` | `[d, d_ff]`, `[d_ff, d]` | (2.16) | `d_ff` and activation **[UNSPECIFIED]** |
| `RMSNorm_{D,l}` | `[d]` gain? | (2.16) | pre-FFN norm |
| `RMSNorm_E` | `[d]` gain? | (2.2) | |
| `RMSNorm_s` | `[d]` gain? | (2.9) | pre-merge state norm |
| `RMSNorm_o` | `[d]` gain? | (2.6) | pre-readout norm |
| `W_o` | `[V, d]` | (2.6) | readout / unembedding |

---

## 5. Training objectives

**(5.1) Pretraining** — full next-token prediction:

`L_PT(Theta) = -E_x [ 1/(S-1) * sum_{t=1}^{S-1} log p(x_{t+1} | x_{1:t}) ]`

> **[PAPER]** *"Every non-BOS target is supervised. Encoder, memory projections, merge, and decoder train jointly."*
> **[PAPER]** *"Independent documents reset recurrent outputs, decoder SWA caches, encoder caches, and positions, and use disjoint attention masks."*
> **[PAPER]** *"a segment cut cannot silently discard state while claiming full-history likelihood."*

**(5.2) SFT** — masked to assistant targets, with `m_{t+1} in {0,1}`:

`L_SFT(Theta) = -E_x [ (1 / sum m_{t+1}) * sum_{t: m_{t+1}=1} log p(x_{t+1} | x_{1:t}) ]`

> **[PAPER]** the key sentence: *"**Loss masking does not mask state updates** or detach encoder memory, recurrent outputs, or decoder KV."* Every context token still gets a full recurrent update; gradients from assistant losses flow **through** user/tool token computations.
> **[PAPER]** *"The state is not reset at an assistant boundary, and no separate boundary-adaptation objective is needed."*

**(5.3) RL** — policy `pi_Theta(y|c) = prod_i p_Theta(y_i | c, y_{<i})`, score-function gradient (5.4), ratios

**(5.5)** `r_i(Theta) = exp( log p_Theta(y_i | c, y_{<i}) - log mu(y_i | c, y_{<i}) )`

> **[PAPER]** the replay contract: recompute encoder features **and** rebuild `s` and every decoder SWA cache over the full prompt + response prefix under **current** parameters. *"A sampler's old hidden states cannot replace current-policy replay."*
> **[PAPER]** the honest caveats, which are the substance of §5.3: exact importance sampling needs `p(.|h) >> mu(.|h)`; top-k / top-p sampling *"generally violates this condition"*; behavior log-probs *"must include temperature, truncation, and renormalization"*; *"metadata alone cannot restore missing support"*.
> **[PAPER]** *"matching forward probabilities at one parameter value is not sufficient to establish matching policy gradients"* — the trainer must differentiate the recurrent computation, not merely reproduce its numbers.

---

## 6. The two propositions

### Proposition 3.1 — Invariance to the serving split

> **[PAPER]** Fix parameters, tokens, position convention, start state. Assume equivalent causal encoder execution, identical SWA windows and cache updates, deterministic decoder ops. Then **batched prefill + recurrent decoding == fully incremental processing**, and *"Moving the prompt-response split does not change the conditional distribution for a fixed token history."*
> Proof is a one-line induction: both start at `H_0`; if states agree at `t-1`, (2.5) applies identical ops to identical inputs at `t`. *"The serving split never appears in the transition."*
> **[PAPER]** stated limits: *"This is mathematical equivalence. Different kernels and precision choices can still cause numerical discrepancies."*

**This is exactly checkable in pure Python** — we use exact float arithmetic with one code path, so the only way it can fail is if our implementation leaks the split (e.g. by resetting a cache). A failure here is a bug in *us*, which is what makes it a good test.

### Proposition B.1 — Causality

> **[PAPER]** With a causal encoder, prefix-restricted memory, and causal decoder SWA, `H_t = (s_t, C_t^D)` depends only on `x_{1:t}` and `Theta`.
> Proof by induction; the induction step notes that *"Appending current KV and evicting old entries introduce no future information."*

**Checkable by perturbation**: change `x_{t+1}` and assert `H_t` is bit-identical. This is the cleanest experiment in the whole paper.

---

## 7. Gradients and BPTT (Appendix B) — the part the brief's Phase 12 targets

**(B.1)** `J_t = dH_t / dH_{t-1}`

**(B.2)** `dH_t / dH_j = J_t J_{t-1} ... J_{j+1}`, for `j < t`

**(B.3)** the block structure — and this is the sentence that matters:

```
          [ ds_t/ds_{t-1}    ds_t/dC^D_{t-1}  ]
    J_t = [                                   ]
          [ dC^D_t/ds_{t-1}  dC^D_t/dC^D_{t-1}]
```

> **[PAPER]** *"**A product involving only `ds_t/ds_{t-1}` generally misses paths through decoder KV.** Normalization alone does not bound products of these Jacobians."*

This is a sharp, falsifiable, structural claim, and it is the single best target in the paper for a dissection: the recurrent state is **not** the only carrier between tokens. The SWA cache is a second, parallel recurrent channel. Consequences:

- Setting `alpha = 0` does **not** make the model non-recurrent. It severs `s`, leaving `C^D`.
- Zeroing `s_{t-1}` does **not** erase history. The window still holds `W-1` decoder-derived KV entries.
- A *true* non-recurrent control needs `alpha = 0` **and** `W = 1` (at `W = 1` the retained set is explicitly empty).

**[NOTE — this refines the earlier reproduction's design.]** `rlt-reproduce` used `alpha = 0` as "the" recurrence ablation. Per (B.3) that is the `s`-channel ablation only. Phase 7 must therefore be a 2x2 over (`alpha`, `W`), not a single knob.

**(B.4)** full parameter derivative, for a response loss `l(Theta, H_T(Theta), B_T^E(Theta))`:

`dl/dTheta = dl/dTheta|_fixed + (dl/dH_T)(dH_T/dTheta) + (dl/dB_T^E)(dB_T^E/dTheta)`

**(B.5)** `dl/dH_T = (dl/ds_T)(ds_T/d.) + (dl/dC_T^D)(dC_T^D/d.)`

> **[PAPER]** *"Detaching `s_T`, decoder KV, or encoder-side boundary tensors removes corresponding gradient paths even if forward probabilities are unchanged. **None of these operations is full BPTT.**"*

**(C.1)** complete detach: `H~_t = (stopgrad(s_t), stopgrad(C_t^D))`

> **[PAPER]** App. C: *"SWA eviction limits future direct access to old KV; **it is not itself a stop-gradient operation.** Full BPTT still differentiates computations that consumed those entries before eviction. The inference cache size therefore does not bound full-BPTT activation storage."*

Another crisp, testable claim: gradient reach `>` cache reach.

---

## 8. Complete list of what the paper does NOT specify

Every one of these needs an explicit choice from us. Grouped by how much the choice could matter.

### Group A — numbers the paper never gives (must be chosen; all are free variables)

| # | Quantity | Paper's constraint | **[IMPLEMENTATION ASSUMPTION]** for the tiny model |
|---|---|---|---|
| A1 | `d` | none | `8` |
| A2 | `V` (vocab) | none | `10` |
| A3 | `L_E`, `L_D` | none (48+48 is "illustrative", Fig. 2) | `1`, `1` |
| A4 | heads | never mentioned at all | `2` |
| A5 | `d_ff` | none | `16` (`2d`) |
| A6 | FFN activation | none | GELU (tanh approx.), an explicit choice |
| A7 | `W` | `W >= 1` | `3` |
| A8 | `G` | `1 <= G <= L_D` | `1` (the paper's "shares memory across layers") |
| A9 | `alpha` | "modest nonzero" | `0.1`, swept in Phase 7 |
| A10 | RMSNorm `eps` | none | `1e-5` |
| A11 | attention scale | none | `1/sqrt(d_head)`, the standard choice |

### Group B — structural choices the paper leaves open (implement BOTH where practical)

| # | Choice | Paper's words | Plan |
|---|---|---|---|
| B1 | positional transformation in `P_Q`/`P_K` | *"any positional transformation"* | implement **NoPE** (identity) and **RoPE**; default NoPE for the tiny model so positional effects don't confound state effects. Both, compared. |
| B2 | tied vs untied `E`/`D` | §2.6 gives tied as "reference"; untied *"preserves the complete-state recurrence"* | implement untied first (simpler, fewer confounds), tied as a flag |
| B3 | gate shape | vector `g_t in R^d`, or *"scalar gating"* | vector (follows `W_g in R^{d x 2d}`); scalar as a flag |
| B4 | `W_s` rank | full, or *"low-rank"* | full |
| B5 | RMSNorm learned gain | unstated | **with** learned gain (standard RMSNorm); init to 1, so the no-gain variant is the init |
| B6 | encoder block internals | only constrained via §2.6 tying | pre-norm `attn -> FFN`, mirroring the decoder core |
| B7 | biases other than `b_g` | only `b_g` is named | no other biases |

### Group C — training details, entirely absent

Optimizer, learning rate, schedule, batch size, init scheme, dropout, tokenizer, data, sequence length, `s_*` init distribution, gradient clipping. **All [UNSPECIFIED BY PAPER].** Every one becomes a logged config field in Phase 16 so no result is ever read as "the paper's".

### Group D — genuine ambiguities in the equations (flagged, not resolved yet)

1. **(2.2) `M_{<=t}` during prefill.** §2.4 says *"At decoder position `t`, attention is restricted to `M_{<=t}` even though the entire prompt memory is available."* So memory is prefix-restricted **per decoder position**, not per prefill call. Our implementation must slice, not just append. This is a correctness trap: an implementation that lets decoder position 3 see `M_{<=T}` is a *different model*, and §4.2 says so explicitly (*"a faster kernel that reads future entries changes the model"*).
2. **Multi-head in cross-attention.** (2.15) reads `M^{g(l)}` whose K/V were built at width `d` by (2.2). Whether those are split into heads, and whether the head count matches SWA's, is **[UNSPECIFIED]**. **[IMPLEMENTATION ASSUMPTION]** same head count, same split.
3. **`RMSNorm_{S,l}` appears only on the value path** (2.13), while `P_Q`/`P_K` carry their own norms. Whether all three norms are the same module or three separate ones is **[UNSPECIFIED]**. **[IMPLEMENTATION ASSUMPTION]** three separate modules (the paper writes them with different names, and §2.6 calls normalizations "stage-specific").
4. **Does the BOS token get a decoder update?** (2.3)–(2.5) say `H_0` is before BOS and the transition applies *"for every observed or sampled token"* with `t >= 1` and `x_1 = BOS`. So **yes**, `t=1` is BOS and produces `H_1`, which predicts `x_2`. Consistent with (5.1)'s sum from `t=1`.
5. **`alpha` trainable or fixed?** (2.11) calls it "the feedback scale"; §2.5 calls a nonzero value *"a candidate initialization"*, which hints trainable. **[UNSPECIFIED]**. **[IMPLEMENTATION ASSUMPTION]** fixed hyperparameter (so `alpha = 0` is an exact, clean ablation); trainable as a flag.

---

## 9. Limitations the paper states about itself

Collected because a dissection should not "discover" caveats the author already wrote down.

- §1: *"The architecture makes that path available; learning useful reasoning along it is a separate question."*
- §3.2: *"State recurrence can reduce hardware utilization even when arithmetic order matches a conventional dense Transformer."* / *"no reduced-prefill speedup is claimed."*
- §3.3: *"Gates, contraction, and learned projections may suppress the practical contribution of long paths; **structural depth alone is not a reasoning guarantee.**"*
- §4.1: *"No exact parallel scan for the general nonlinear decoder is assumed."*
- §4.2: tying *"does not by itself reduce block evaluations or guarantee lower latency."*
- §4.3: *"hardware-efficient training and inference remain engineering goals until measured."*
- §7 (Depth-wise reuse): *"Neither weight tying nor temporal recurrence alone establishes novelty or a quality improvement."*
- §7 (T2MLR): cites a result that *"localized middle-layer recurrence can outperform recurrence across the full network; a deeper recurrent block is therefore not automatically preferable"* — i.e. the paper cites evidence **against** its own design choice, and does not claim to beat it.

§3.3 is the paper's own strongest hypothesis-under-doubt, and it is the natural target of Phase 15.

---

## 10. Claims extracted for Phase 14 (to be tested, not assumed)

Numbered here, carried into `paper_claims.md` later.

| ID | Claim | Source | Testable in pure Python? |
|---|---|---|---|
| C1 | State path to `t` composes `t` transitions / `t*L_D` decoder blocks; per-token block count fixed at `L_E + L_D` | §3.3, Fig. 2 | yes — count block evaluations |
| C2 | Prompt and response use the **same** transition; moving `T` changes nothing | Prop. 3.1 | yes — exact |
| C3 | `H_t` depends only on `x_{1:t}` | Prop. B.1 | yes — exact, by perturbation |
| C4 | `s` **and** `C^D` both cross the boundary unreset | §2.3 | yes |
| C5 | Encoder memory is prefix-restricted per decoder position | §2.4 | yes — exact |
| C6 | Cached states from old parameters are not current-policy states | §5.4, App. C | yes — measure the drift |
| C7 | Full BPTT reaches through the recurrent history; detaching `s` alone is **not** full BPTT | App. B, C | yes — hand gradients |
| C8 | Retention rule `[max(1,t-W+2), t]` + current KV == next read window `[max(1,t-W+1), t]` | §2.5 | yes — exact invariant |
| C9 | A product using only `ds_t/ds_{t-1}` misses the KV paths | App. B (B.3) | yes — numerical Jacobian, 2 ways |
| C10 | SWA eviction is not a stop-gradient; gradient reach > cache reach | App. C | yes |
| C11 | `M` never depends on decoder states | §2.2 | yes — exact |

C9 and C10 are the two that would most repay a careful experiment, because both are stated without proof and both are easy to get wrong in an implementation.

---

## 11. Open questions this dissection should answer

Carried from the brief, narrowed to what this paper actually supports.

1. Does `s_t` carry information, or does the SWA cache do all the work? (B.3 says both channels exist — measure the split.)
2. At what `t` does the influence of `x_1` on `s_t` fall below float noise?
3. What does `g_t` do numerically — is it saturated, is it near-constant, is it input-dependent at all?
4. Does `||prod J_t||` grow or contract, and does the `s`-only product differ from the full product as (B.3) claims?
5. Is the `alpha = 0` control actually a non-recurrent model? (Predicted: no.)

---

*Phase 0 complete. Nothing implemented yet beyond this reading.*
