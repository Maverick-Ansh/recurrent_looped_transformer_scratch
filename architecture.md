# architecture.md — RLT as an explicit computation graph

Every operation is written as:

```
INPUT
  v OPERATION
OUTPUT                      shape        meaning
```

Shapes are given for the **tiny reference config** of Phase 3, so every number below is a number you can actually print:

```
V      = 10     vocabulary
d      = 8      residual width
L_E    = 1      encoder layers
L_D    = 1      decoder layers
H      = 2      attention heads
d_head = 4      = d / H
d_ff   = 16     FFN inner width
W      = 3      SWA window, INCLUDES current token
G      = 1      encoder-memory groups (shared across decoder layers)
alpha  = 0.1    feedback scale
```

All of these are **[IMPLEMENTATION ASSUMPTION]**s — see `research_notes.md` §8 Group A. The paper fixes none of them.

Convention: the paper uses **column vectors**, so `W x` means `[d_out, d_in] @ [d_in]`. In pure Python a "vector" is a flat `list[float]` of length `d`, and a "matrix" is `list[list[float]]` with `len(M) == d_out`, `len(M[0]) == d_in`.

---

## 0. The top-level picture

Two stacks, four stores, one loop.

```
                            x_1 .. x_S   (token ids)
                                 |
              +------------------+------------------+
              |                                     |
        CAUSAL ENCODER                        (per token t)
        parallel over t                             |
        sequential over layers                      |
              |                                     |
           e_1 .. e_S                               |
              |                                     |
        memory projections (2.2)                    |
              |                                     |
         M^g  (grows forever) --------------------->+   cross-attention reads M_{<=t}
                                                    |
   s_{t-1} ------------------> MERGE (2.9-2.11) --> u_t
      ^                                             |
      |                                    DECODER STACK (2.12-2.16)
      |                                    L_D blocks         ^
      |                                             |         |
      |                                            s_t     C^D_{t-1}  (bounded, per layer)
      +---------------------------------------------+         |
                 RECURRENT PATH                              C^D_t
                                                              |
                                                     (also recurrent!)
                                                    readout (2.6) -> logits
```

**The two recurrent channels.** The picture above has **two** backward arrows, not one:

1. `s_t -> s_{t+1}` through the merge — the obvious one.
2. `C^D_t -> C^D_{t+1}` through the SWA cache — the one Appendix B (B.3) insists on.

The paper's Jacobian `J_t` is 2x2 precisely because of this. Any diagram, ablation, or gradient argument that shows only channel 1 is wrong, and the paper says so: *"A product involving only `ds_t/ds_{t-1}` generally misses paths through decoder KV."*

---

## 1. Encoder path (per token t)

### 1.1 Token embedding

```
x_t                                          int in [0, V)
  v  embedding table lookup: E_tok[x_t]
h_t^0                                        [8]     token's initial residual stream
```

**[UNSPECIFIED BY PAPER]** — the paper never writes the embedding step; it is implied by (2.1).

### 1.2 Encoder layer l = 1 .. L_E (pre-norm block)

**[IMPLEMENTATION ASSUMPTION]** — §2.6 constrains the encoder block only by saying it shares Q/K/V/O and FFN with decoder SWA. We mirror the decoder core: pre-norm attention, then pre-norm FFN.

```
h_t^{l-1}                                    [8]
  v  RMSNorm_{E,attn,l}                              (2.9-style: x / sqrt(mean(x^2)+eps) * gain)
n_t                                          [8]     normalized residual
  v  W_Q^{E,l} n_t  /  W_K^{E,l} n_t  /  W_V^{E,l} n_t
q_t, k_t, v_t                                [8] each
  v  reshape to heads
q_t, k_t, v_t                                [2, 4]  (H, d_head)
  v  append (k_t, v_t) to encoder cache C^E   <-- STORE 4, grows forever
C^E                                          [L_E][t][2,4] x2
  v  causal attention: token t attends j = 1..t
score[h][j] = dot(q_t[h], k_j[h]) / sqrt(4)  [2, t]  pre-softmax affinity
  v  softmax over j
p[h][j]                                      [2, t]  sums to 1 along j
  v  sum_j p[h][j] * v_j[h]
o_t                                          [2, 4]
  v  flatten + output projection W_O^{E,l}
attn_out                                     [8]
  v  residual add
b_t = h_t^{l-1} + attn_out                   [8]
  v  RMSNorm_{E,ffn,l}
  v  FFN: W_2 @ gelu(W_1 @ .)                [16] inner
h_t^l = b_t + ffn_out                        [8]     encoder layer output
```

After `L_E` layers:

```
h_t^{L_E}
  v  (identity)
e_t                                          [8]     ENCODER REPRESENTATION, eq (2.1)
```

**Meaning of `e_t`:** a causal, non-recurrent feature of `x_{1:t}`. It has **never seen a decoder state**. This is the asymmetry that makes the encoder parallelizable (§4.1) and that Prop. B.1's proof leans on.

### 1.3 Memory projection — eq (2.2)

For each memory group `g = 1..G` (here `G = 1`):

```
e_t                                          [8]
  v  RMSNorm_E
n                                            [8]
  v  P_K^g = (norm, then W_K^g, then positional transform)
k_t^g                                        [8]     -> reshaped [2,4]
  v  W_V^g n     (NOTE: no positional transform on values)
v_t^g                                        [8]     -> reshaped [2,4]
  v  append to M^g
M_{<=t}^g                                    [t][2,4] x2   <-- STORE 3, grows forever
```

**Meaning:** `M` is the global, unbounded, **encoder-derived** context store. Eq (2.2) + §2.2: *"This global memory depends on encoder representations, not decoder states."* That is claim **C11**, and it is exactly checkable: perturb `s_*` and assert `M` is bit-identical.

---

## 2. The merge — eqs (2.9)-(2.11)

This is the whole recurrence for channel 1, in three lines.

```
s_{t-1}                                      [8]     previous recurrent output (s_* at t=1)
  v  RMSNorm_s:  r_i = s_i / sqrt(mean(s^2) + eps) * gain_i           (2.9)
r_{t-1}                                      [8]     normalized state
```

```
e_t, r_{t-1}
  v  concatenate                                                       (2.10)
[e_t ; r_{t-1}]                              [16]    = [2d]
  v  W_g @ .  + b_g                          W_g is [8, 16]
gate preactivation                           [8]
  v  sigma(z) = 1/(1+exp(-z))   elementwise
g_t                                          [8]     in (0,1)^8, the GATE
```

```
r_{t-1}
  v  W_s @ .                                 W_s is [8, 8]             (2.11)
W_s r_{t-1}                                  [8]     projected state
  v  elementwise multiply by g_t
g_t (*) W_s r_{t-1}                          [8]     gated feedback
  v  scale by alpha
alpha * g_t (*) W_s r_{t-1}                  [8]     FEEDBACK CONTRIBUTION
  v  add e_t
u_t = e_t + alpha * g_t (*) W_s r_{t-1}      [8]     MERGED DECODER INPUT
```

**Read the operand carefully.** `W_s` multiplies `r_{t-1}` — the **normalized** state — not `s_{t-1}`. See `research_notes.md` §2.3.

**What `alpha = 0` does:** kills the entire third line. `u_t = e_t` exactly. `g_t` is still computed (and still depends on `r_{t-1}`) but is multiplied by zero. So `alpha = 0` severs channel 1 completely while leaving parameter count, block count, and channel 2 untouched. That is what makes it a clean internal control — **and** why it is *not* a full non-recurrence control.

---

## 3. Decoder block l = 1 .. L_D — eqs (2.12)-(2.16)

Input: `z_t^0 = u_t`.

### 3.1 Sublayer A — causal SWA over decoder activations (2.12)-(2.14)

```
z_t^{l-1}                                    [8]     layer input
  v  RMSNorm (inside P_Q^{D,l}) then W_Q^{D,l} then positional transform
q_t^{D,l}                                    [2,4]                     (2.12)
  v  RMSNorm (inside P_K^{D,l}) then W_K^{D,l} then positional transform
k_t^{D,l}                                    [2,4]                     (2.13)
  v  RMSNorm_{S,l} then W_V^{D,l}     (no positional transform)
v_t^{D,l}                                    [2,4]                     (2.13)
```

Now the cache. **This is the step to watch.**

```
C^D_{t-1}[l]                                 up to W-1 = 2 entries     <-- STORE 2
  v  form read window = C^D_{t-1}[l] ++ [(k_t, v_t)]
read window                                  j in [max(1,t-W+1), t], <= 3 entries
```

The paper is explicit that current KV is formed **before** SWA from the layer input, *"so current-position attention introduces no circular dependency"* — the current token attends to itself using a key/value it just computed from `z_t^{l-1}`, which is already known. No fixpoint, no iteration.

```
q_t^{D,l}, read window
  v  score[h][j] = dot(q_t[h], k_j[h]) / sqrt(4)
scores                                       [2, <=3]
  v  softmax over j     (no mask needed: the window IS the mask)
probs                                        [2, <=3]
  v  sum_j probs[h][j] * v_j[h]
  v  flatten, output projection W_O^{D,l}
swa_out                                      [8]
  v  residual add
b_t^l = z_t^{l-1} + swa_out                  [8]                       (2.14)
```

Then the eviction:

```
read window
  v  retain positions [max(1, t-W+2), t]   (drop the oldest if full)
C^D_t[l]                                     <= W-1 = 2 entries
```

**INVARIANT C8** (Phase 8 will assert it at every `t`, every `W`):

```
C^D_t[l]  ++  [(k_{t+1}, v_{t+1})]   ==   read window at step t+1
```

i.e. retained set ∪ current == next read window, exactly. At `W = 1` the retained set is **empty**, and every token attends only to itself.

### 3.2 Sublayer B — cross-attention to encoder memory (2.15)

```
b_t^l                                        [8]
  v  RMSNorm (inside P_Q^{M,l}) then W_Q^{M,l} then positional transform
cross query                                  [2,4]
M_{<=t}^{g(l)}                               [t][2,4] x2   <-- prefix-restricted!
  v  score[h][j] = dot(q[h], k_j[h]) / sqrt(4),  j = 1..t
cross scores                                 [2, t]
  v  softmax over j
cross probs                                  [2, t]
  v  weighted sum of values, flatten, output projection W_O^{M,l}
cross_out                                    [8]
  v  residual add
a_t^l = b_t^l + cross_out                    [8]                       (2.15)
```

**The prefix restriction is a model property, not an optimization.** §2.4: *"At decoder position t, attention is restricted to `M_{<=t}` even though the entire prompt memory is available."* §4.2: *"a faster kernel that reads future entries **changes the model**."* Our implementation slices `M` to `t` entries at every decoder position; claim **C5** tests that we did.

### 3.3 Sublayer C — FFN (2.16)

```
a_t^l                                        [8]
  v  RMSNorm_{D,l}
n                                            [8]
  v  W_1 @ n                                 W_1 is [16, 8]
  v  gelu                                    [IMPLEMENTATION ASSUMPTION - activation unspecified]
  v  W_2 @ .                                 W_2 is [8, 16]
ffn_out                                      [8]
  v  residual add
z_t^l = a_t^l + ffn_out                      [8]                       (2.16)
```

After `L_D` layers:

```
z_t^{L_D}
  v  (identity)
s_t                                          [8]     THE RECURRENT STATE
```

---

## 4. Readout — eq (2.6)

```
s_t                                          [8]
  v  RMSNorm_o
n                                            [8]
  v  W_o @ n                                 W_o is [10, 8]
logits                                       [10]
  v  softmax
p(x_{t+1} | x_{1:t})                         [10]   sums to 1
  v  argmax / sample
predicted token                              int
```

---

## 5. The recurrent path, drawn explicitly

The brief asks for this not to be hidden inside an abstraction. Here it is, unrolled, with **both** channels.

```
H_0 = (s_*, {})
  |
  |  F_1 : u_1 = Merge(e_1, s_*)           -> decoder(L_D blocks) ->
  v
H_1 = (s_1, C^D_1)          C^D_1[l] = [ (k_1,v_1) ]                     (t=1, W=3)
  |
  |  F_2 : u_2 = Merge(e_2, s_1)           -> decoder ->
  v
H_2 = (s_2, C^D_2)          C^D_2[l] = [ (k_1,v_1), (k_2,v_2) ]
  |
  |  F_3 : u_3 = Merge(e_3, s_2)           -> decoder ->
  v
H_3 = (s_3, C^D_3)          C^D_3[l] = [ (k_2,v_2), (k_3,v_3) ]   <-- (k_1,v_1) EVICTED
  |
  |  F_4 : u_4 = Merge(e_4, s_3)           -> decoder ->
  v
H_4 = (s_4, C^D_4)          C^D_4[l] = [ (k_3,v_3), (k_4,v_4) ]
  |
  ...
  v
H_t = F_t o F_{t-1} o ... o F_1 (H_0)                                    (2.7)
```

**Where does `x_1` survive at `t = 4`?** Three places, and enumerating them is the point of Phase 13:

1. **Not** in `C^D_4` — `(k_1, v_1)` was evicted at `t = 3`.
2. In `M_{<=4}`, which still holds `(k_1^g, v_1^g)` — encoder memory never evicts.
3. In `s_4`, indirectly: `(k_1,v_1)` was read by the SWA at `t = 1, 2, 3`, shaping `s_1, s_2, s_3`, and `s_3` feeds `u_4`.

Route 3 is exactly App. C's *"Evicted entries can still influence later computation through states or retained activations that previously consumed them."* Route 3 is also why eviction is **not** a stop-gradient (claim C10).

---

## 6. Every cache, as a first-class object

The brief (Phase 8) asks that caches not be hidden. Four stores, all explicit data structures in our implementation:

| Store | Python type | Bound | Evicts? | Differentiable path? |
|---|---|---|---|---|
| `C^D` decoder SWA | `list[ list[ (k,v) ] ]` indexed `[layer][slot]` | `W-1` per layer | **yes** | yes — and *past* eviction (App. C) |
| `M^g` encoder memory | `list[ list[ (k,v) ] ]` indexed `[group][pos]` | unbounded | no | yes |
| `C^E` encoder causal KV | `list[ list[ (k,v) ] ]` indexed `[layer][pos]` | unbounded | no | yes |
| `s` recurrent output | `list[float]` length `d` | 1 vector | overwritten | yes |

`C_D,1 ... C_D,L_D` are separate lists — one per decoder layer, never merged. §2.6: *"Encoder-memory group sharing ... does not merge or eliminate the layerwise decoder SWA caches."*

---

## 7. Block-count accounting (claim C1)

Per processed token:

```
encoder blocks    L_E
decoder blocks    L_D
merge             1   (not counted as a block by the paper: Fig. 2 "counts exclude merge and readout")
readout           1   (same)
-------------------------
                  L_E + L_D   blocks per token       <- FIXED, independent of t
```

Along the recurrent chain from `s_0` to `s_t`:

```
t transitions  x  L_D blocks each  =  t * L_D   decoder blocks    <- GROWS with t
```

For the paper's illustrative `L_E = L_D = 48`: 96 blocks/token, `48t` along the chain — exactly Fig. 2. In our tiny config, `L_E = L_D = 1`: 2 blocks/token, `t` along the chain. Phase 14 tests this by *counting actual block invocations*, not by re-deriving the formula.

---

## 8. What the graph makes experimentally reachable

Mapping the brief's questions onto nodes above:

| Question | Node |
|---|---|
| what is in the state at `t`? | `s_t`, §3.3 output |
| what carries `t -> t+1`? | `s_t` (§2) **and** `C^D_t` (§3.1) — two channels |
| what is in the SWA cache? | §3.1 eviction step, `<= W-1` (k,v) pairs per layer |
| how does `e_t` meet `s_{t-1}`? | §2, concat at (2.10), additive at (2.11) |
| what does `g_t` do? | §2, `sigma` output in `(0,1)^8`, elementwise multiplier on the feedback |
| before/after RMSNorm? | §1.2, §2, §3.1, §3.3, §4 — seven distinct norm sites |
| what do Q/K/V look like? | §1.2, §3.1, §3.2 |
| where does early info survive? | §5, routes 1-3 |
| BPTT path | reverse of §5, through **both** channels |

---

## 9. Implementation shape contract

Written here so Phase 2's primitives have a target. Pure Python types only.

```
scalar        float
vector        list[float]                       len = d
matrix        list[list[float]]                 [rows][cols], rows = d_out
heads split   list[list[float]]                 [H][d_head]
kv entry      (pos:int, k:list[list[float]], v:list[list[float]])
swa cache     list[list[kv_entry]]              [layer][slot]
memory        list[list[kv_entry]]              [group][pos]
attn scores   list[list[float]]                 [H][n_keys]
trace         dict[str, Any]                    name -> value, for inspect.py
```

No NumPy. No framework. Every number reachable with `[i][j]`.

---

*Phase 1 complete. Next: `core_math.py` — the primitives, and nothing above them.*
