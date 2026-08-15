# SAFEFL: findings and a plan for porting its aggregation rules to hpmpc

**Status: research complete for the question "what does SAFEFL support in
MPC, and what would it take to run its rules on our hpmpc integration";
FLOD-on-Trio has a concrete phased plan (below) but no code written yet.**

**Audience note (for an agent picking this up cold):** this document assumes
you already know Flotilla's existing secure-aggregation architecture (see
`design.md`, `hpmpc_backend.md`, `threat_model.md` in this same directory)
and the multi-protocol `HpmpcBackend` work (Replicated/Trio/Tetrad). It does
NOT re-explain that. It's specifically about a *different* repo (SAFEFL)
and what it would take to bring some of its aggregation logic into this
project. All file-path claims below were verified by reading the actual
source at the time this was written — **re-verify anything load-bearing
before acting on it**, since both repos can change. SAFEFL was found at
`SAFEFL/` as a sibling directory to `flotilla/` in the user's workspace (NOT
inside this repo, NOT a submodule — just a separate clone sitting next to
it for reference).

## 1. What SAFEFL actually is

SAFEFL (`https://github.com/encryptogroup/SAFEFL`) is a research framework
for evaluating Byzantine-robust federated-learning aggregation rules
against poisoning attacks. It
implements ~14 aggregation rules total, almost all as **plaintext
simulation code** (`SAFEFL/aggregation_rules.py`) intended for evaluating
robustness against attacks (`SAFEFL/attacks.py`), not for real distributed
deployment. Only **two** of those rules have an actual MPC (distributed)
implementation, built on **MP-SPDZ** — a different C++ MPC framework than
hpmpc, with its own protocol set (Semi2k, SPDZ2k, Replicated2k,
PsReplicated2k — see `SAFEFL/README.md`'s "Multi-Party Computation"
section). MP-SPDZ is NOT used anywhere in Flotilla; this document is about
porting the *algorithms*, not the MP-SPDZ code itself.

### 1.1 The two real MPC algorithms

Located at `SAFEFL/mpspdz/Programs/Source/`:

- **`mpc_fedavg_server.mpc`** — trivial: sum all client gradients inside
  MPC, divide by a **flat worker count** (unweighted — note this is
  actually *simpler* than SAFEFL's own plaintext FedAvg, which weights by
  `data_sizes`; the MPC port is a simplification, not a faithful port).
- **`mpc_fltrust_server.mpc`** — meaningfully more complex. Per round: the
  last of the `WORKERS` inputs is a server-held "root"/trusted gradient.
  For each real client: compute a dot product against the root, compute
  every input's Euclidean (L2) norm, form a ReLU-clipped cosine-similarity
  "trust score" (`dot / (norm_i · norm_root)`), rescale each client's
  gradient by `trust_score / norm_i`, sum, then rescale the whole sum by
  `norm_root / total_trust_score`. All of this — dot products, norms,
  divisions, the ReLU clip — happens **inside** the MPC computation, not in
  plaintext after a reveal.

### 1.2 How SAFEFL's client/dispatch model works (context, not something to imitate)

`SAFEFL/main.py`'s aggregation dispatch is a single top-level branch:
```python
if args.mpspdz:
    aggregation_rules.mpspdz_aggregation(grad_list, ...)   # routes to whichever .mpc SERVER script was separately launched
elif args.aggregation == "fltrust": ...
elif args.aggregation == "krum": ...
# ...one elif per rule
```
`aggregation_rules.mpspdz_aggregation` (`SAFEFL/aggregation_rules.py:454`)
is algorithm-agnostic — it just serializes gradients and forwards them to
`mpspdz.ExternalIO.mpc_client`, which speaks MP-SPDZ's own client protocol
(`SAFEFL/mpspdz/ExternalIO/client.py`): the client connects to **every**
party server, **receives** a random masking triple from them first, checks
consistency (`triple[0]*triple[1] == triple[2]`), then sends
`value + triple[0]` as its masked input. This is architecturally different
from Flotilla's own approach, where clients compute valid shares **entirely
locally** with zero interaction before submitting (see `hpmpc_backend.md`'s
"the share-format problem, and how it was solved"). **This is not a gap —
Flotilla's existing approach already achieves the same goal more simply.**
Nothing about SAFEFL's client-transport model needs to be adopted; only the
*aggregation math itself* (section 1.1 and section 3 below) is the actual
porting target.

### 1.3 Verifying "only 2 rules are distributed" — don't take this on faith

This was independently re-verified (the user explicitly asked "are you
sure?"). Evidence, so a future reader can re-check it themselves:

1. `find SAFEFL -iname "*.mpc"` returns exactly 2 files (section 1.1's two).
2. `main.py`'s `if args.mpspdz:` branch (section 1.2) pre-empts every
   `elif args.aggregation == ...` branch — when `--mpspdz` is set, the
   `--aggregation` argument is never consulted at all. There is no
   `.mpc` server script for krum/median/flame/etc., so there is no way to
   route them through MPC even in principle.
3. Every other rule's function body in `aggregation_rules.py` (foolsgold,
   krum, median, trim_mean, flame, shieldfl, flod, divide_and_conquer,
   contra, signguard, flare, romoa) operates directly on plaintext
   `torch` tensors with zero references to `mpspdz`, `subprocess`, or
   sockets anywhere in the function body.

## 2. hpmpc's actual primitive coverage (verified against source, not assumed)

This is the load-bearing part for deciding what's portable. Every claim
below was checked by reading the file cited, inside
`flotilla/mpc_engines/hpmpc/` (the hpmpc git submodule — see `design.md`'s
"Vendored MPC libraries" note).

| Primitive | Status | Evidence |
|---|---|---|
| Secret×secret multiplication (dot products, squaring) | **Universal** — works for all 3 protocols Flotilla has wired up (Replicated=`PROTOCOL 2`, Trio=`5`, Tetrad=`8`) | `prepare_dot`/`mask_and_send_dot`/`complete_mult` are implemented via a generic functor pattern (`func_add`/`func_mul` template params) shared by every protocol's native Share class — confirmed present in Replicated/Trio/Tetrad's own share classes during this project's own Trio/Tetrad work. Also demonstrated generically in `programs/tutorials/fixed_point_tutorial.hpp`. |
| Addition / local scalar ops | **Universal** | Inherent to additive-style sharing; this is the whole basis of `backend_hpmpc.py`'s existing "sum shares in Python" design. |
| Multiplication/division by a **public** constant | **Universal** | `prepare_mult_public_fixed`/`complete_public_mult_fixed` (reciprocal-multiply trick) — see `fixed_point_tutorial.hpp` and `programs/functions/prob_div.hpp` (despite the name, `prob_div.hpp` is public-constant *truncation*, not secret/secret division — see below). |
| Secret÷secret division | **Exists as a real primitive, but not as a polished reusable library function** | `programs/benchmarks/bench_basic_primitives.hpp`'s `DIV_BENCH` implements genuine Newton-Raphson division (`1/x = lim y_n = y_{n-1}(2 - x·y_{n-1})`) built **purely from the universal multiplication primitive above** — no comparison needed, so it's protocol-agnostic. But it uses a **fixed, non-data-adaptive initial guess** (`3·e^0.5 + 0.003`), tuned for whatever value range that specific benchmark expects. `programs/tests/test_fixed_point_arithmetic.hpp`'s `division<Share>()` test similarly hardcodes a guess based on its own known test inputs. **Using this for a real algorithm means validating/re-tuning the initial guess and iteration count against the actual expected value range of your own divisor** — tractable if that range is bounded and known (see FLOD's `weight_sum` in section 4, which has a public bound), less so if the divisor's scale is unpredictable. |
| Square root | **Completely absent.** Zero occurrences of `sqrt` anywhere in the hpmpc tree (checked via `grep -rn "sqrt" --include="*.hpp" --include="*.h" .` — no hits at all, not even a stub or benchmark). | N/A — this is a real, concrete gap, not a restriction. Anything needing a genuine Euclidean norm (FLTrust, Krum, ShieldFL, SignGuard — see section 3) is blocked on this until someone implements it (e.g. Newton-Raphson for `1/sqrt(x)`, analogous to the division primitive above, built from multiplication + a data-adaptive-or-bounded initial guess). |
| Comparison (`LTZ`/`EQZ`, and therefore ReLU, sign, argmin/argmax, sorting) | **Implemented, but restricted to Trio only among our 3 wired protocols.** | `programs/functions/comparisons.hpp`'s `LTZ`/`EQZ` and `programs/functions/Relu.hpp` both depend on arithmetic-to-Boolean share conversion (`prepare_A2B_S1`/`prepare_A2B_S2`, called via `datatypes/k_bitset.hpp`'s `sbitset_t::prepare_A2B_S1/S2`, which is a thin wrapper that requires `Share::prepare_A2B_S1` to exist on the underlying protocol's own Share class — it is NOT a generic/protocol-agnostic algorithm). `grep -rln "prepare_A2B_S1" protocols/` shows this method implemented for `protocols/3-PC/ours/` (**Trio**, `PROTOCOL=5`) and `protocols/4-PC/ours/` (`PROTOCOL=7`, `OEC_MAL` — a *different* 4PC protocol, not Tetrad) and `protocols/TTP/`. It is **absent** for `protocols/3-PC/replicated/` (**Replicated**, `PROTOCOL=2`) and `protocols/4-PC/tetrad/` (**Tetrad**, `PROTOCOL=8`) — confirmed by the same grep returning no hits in either directory. |
| Clustering (KMeans, HDBSCAN) | **No evidence of any support.** Not found anywhere in the codebase. Secure clustering is a hard, protocol-heavy research problem in general MPC — treat as infeasible without a large dedicated engineering/research effort, not a normal adaptation task. | — |
| SVD / eigendecomposition | **No evidence of any support.** Same category as clustering — infeasible for a normal port. | — |
| Logarithm / exponential | **No evidence of any support** (checked as part of the sqrt grep sweep — no hits). Needed by FoolsGold's/CONTRA's `logit`, Romoa's `softmax`, and FLARE's likely-RBF-kernel MMD. | — |
| Full neural-network forward pass under MPC | **Plausible in principle** — hpmpc has `programs/functions/GEMM.hpp` (matrix multiply) and `programs/functions/log_reg.hpp` (a full logistic-regression training program already exists), and hpmpc's own README describes NN training/inference support generally. Not verified in depth for this document (out of scope — only relevant to FLARE, which also needs the missing exp/MMD primitive anyway). | — |

## 3. Per-rule portability assessment (all 12 plaintext-only rules)

Read directly from `SAFEFL/aggregation_rules.py`. Ranked easiest → hardest.
"Comparison-restricted-to-Trio" is a caveat shared by literally every rule
below except pure FedAvg (already ported) — it's called out once here
rather than repeated per row.

| Rule | Needs (beyond mult/add, which are universal) | Verdict |
|---|---|---|
| **FLOD** (`aggregation_rules.py:350`) | Sign extraction per coordinate (comparison), Hamming distance via free XOR-in-Boolean or arithmetic-XOR-formula, ReLU per **client** (not per coordinate — cheap), one final division | **Easiest of the 12.** No sqrt, no clustering, no log. See the full plan in section 4 — this is the one this document also gives an implementation plan for. |
| **Trimmed Mean** (`:151`) | Full per-coordinate sort across clients (comparison-heavy: O(n log n) comparisons × every parameter) | Tractable — only needs comparisons, no sqrt/clustering/log — but computationally heavy given real model sizes (tens of thousands of parameters, each needing an independent sort). |
| **Median** (`:180`) | Per-coordinate selection (cheaper than a full sort, same primitive family as Trimmed Mean) | Same profile as Trimmed Mean, slightly cheaper. |
| **Krum** (`:115`) | Actual Euclidean distances (**sqrt** — absent), full pairwise sort/argmin, **oblivious selection of an entire winning gradient vector by secret index** (no evidence hpmpc has this as a ready primitive) | Blocked on sqrt + needs a nontrivial new "select-by-secret-index" primitive. |
| **FLAME** (`:208`) | Cosine similarity (sqrt), **HDBSCAN clustering** | Blocked on both sqrt and clustering. |
| **ShieldFL** (`:268`) | sqrt (norm), min/max reductions, argmin-based selection | Blocked on sqrt + selection-by-index. |
| **SignGuard** (`:632`) | sqrt (norm), **KMeans clustering** | Blocked on sqrt and clustering. |
| **FoolsGold** (`:488`) | sqrt, pairwise max (comparison), a **`logit` function** (log — absent) | Blocked on sqrt and logarithm, on top of comparisons. |
| **CONTRA** (`:558`) | Same as FoolsGold: sqrt, top-k selection, `logit` (log) | Same blockers as FoolsGold. |
| **FLARE** (`:709`) | A **full model forward pass evaluated inside MPC** per client, MMD (likely an RBF/Gaussian kernel → needs `exp`), k-NN selection | Plausible in principle (hpmpc can do NN inference) but a much bigger undertaking than aggregation math, and still blocked on exp/MMD. |
| **Romoa** (`:773`) | **Two rounds of KMeans**, Pearson correlation (mean/variance/sqrt/division), cosine similarity, multiple top-k selections, **softmax (exp)** | Hardest of all 12 — stacks nearly every gap simultaneously. |

## 4. Plan: FLOD on Trio

This is the concrete next step, scoped specifically to Trio (`PROTOCOL=5`)
since that's the only one of our 3 wired protocols with the comparison
primitive FLOD needs.

### 4.1 The core architectural finding (read this before writing any code)

FLOD **cannot reuse `HpmpcBackend.run_aggregation_round()` unchanged**.
That method's design rests on summing every client's shares in Python
*before* invoking the compiled binary, so the binary only ever reveals one
already-combined value (see `hpmpc_backend.md` / `backend_hpmpc.py`'s own
module docstring). FLOD breaks this assumption: the MPC program itself
needs each client's *individual* gradient — to compute that client's own
Hamming distance against the baseline — before anything gets combined.
This needs a new backend code path, not a new C++ function slotted into
the existing `run_aggregation_round()` contract.

### 4.2 The FLOD recipe translated into hpmpc primitives

Source: `SAFEFL/aggregation_rules.py:350-398` (`def flod(...)`).

| SAFEFL step | hpmpc realization |
|---|---|
| `sign(x) == 1` per coordinate | `bool_i = LTZ(-x_i)`, reusing `comparisons.hpp`'s existing `LTZ` unmodified. `LTZ(v)=1` iff `v<0`, so `LTZ(-x)=1` iff `x>0` — exactly matches `torch.sign(x)==1` (both `x<=0` cases, sign 0 or -1, map to `False`/`0`). |
| `bitwise_xor(bool_i, baseline)` | `LTZ`'s own output is already arithmetic-shared 0/1 (not a raw XOR-shared bit exposed to the caller — see `comparisons.hpp`'s `LTZ` body, which internally does `get_msb_range` → `prepare_bit2a`/`complete_bit2a` before returning). So compute XOR via the arithmetic identity `XOR(a,b) = a + b - 2ab` (one secret multiplication per coordinate) rather than reaching into `XOR_Share`/`bit2a` internals directly — simpler, and it's just reusing the confirmed-universal multiplication primitive. |
| `hamming_distance = sum(xor)` | Free — additive shares sum locally, no communication needed. |
| `weight = ReLU(threshold - hd)` | `y = threshold - hd` (free — public constant minus a secret value is a local op). `weight = y * LTZ(-y)` — verified algebraically including the boundary case: at `y=0`, `LTZ(-0)=LTZ(0)=0` (0 is not `<0`), so `weight = 0*0 = 0`, matching `ReLU(0)=0` exactly. One more `LTZ` call plus one multiplication, **per client** (cheap — n+1 calls total, not per-coordinate). |
| `global_update = Σ(weight_i · sign_i) / Σ(weight_i)` | Numerator: P secret multiplications per client (scalar × vector, broadcasting each client's scalar weight across its P-element sign vector). Denominator: **reveal `weight_sum` as a second output and divide in plaintext**, mirroring exactly how Flotilla already reveals and divides by total dataset size for FedAvg (the `DATASET_SIZE_LAYER_NAME` pseudo-layer in `client_secure_agg_manager.py`/`aggregator_secure_mpc.py`). Deliberately avoids needing genuine secret/secret division for a first pass — `weight_sum` is a bounded, known-range value (`[0, n·threshold]`), which if a later pass wants to do this **inside** MPC instead, makes tuning the Newton-Raphson division primitive's initial guess (section 2) actually tractable, unlike an unpredictable value like a raw gradient norm. |

Cost profile: O((n+1)·P) comparisons for sign extraction (the expensive
part — same order as FLTrust's per-coordinate cost) but only O(n)
comparisons for the ReLU-weight step. No sqrt anywhere, no clustering —
this is the entire reason FLOD ranks above FLTrust/Krum/etc. in section 3.

### 4.3 Phased implementation plan

**Phase 0 — De-risk `LTZ` on Trio in isolation, before anything else.**
Nothing in Flotilla's existing hpmpc integration has ever called a
comparison primitive — `fedavg_secure_aggregation.hpp` only does sum+reveal.
Write a throwaway test program that secret-shares a few known
positive/negative/zero values on real Trio party containers and reveals
`LTZ(-x)`, confirming it matches expected signs. This is the single riskiest
new dependency in this whole plan; verify it alone first.

**Phase 1 — New hpmpc C++ program: `flod_secure_aggregation.hpp`.**
New `FUNCTION_IDENTIFIER` (e.g. 91), Trio-only (`#if PROTOCOL == 5` guard,
following the exact pattern already established for the existing two
functions in `programs/functions/`). Implements the table in 4.2: per-
element sign extraction for all n+1 inputs, arithmetic-XOR against the
baseline, per-client Hamming sum, per-client `LTZ`+multiply for the weight,
per-coordinate weighted accumulation, and a dual reveal (weighted-sum
vector + scalar `weight_sum`). Extend `scripts/build_secure_agg.sh` to
build this function too (currently hardcoded to `FUNCTION_IDENTIFIER=90`).

**Phase 2 — New backend code path in Python.**
Since the file contract is fundamentally different from
`fedavg_secure_aggregation.hpp`'s (per-client rows, not pre-summed), add a
sibling method or class in `server/secure_agg/backends/backend_hpmpc.py` —
not a modification of `run_aggregation_round()` — that writes every
selected client's own share as a separate row (plus the root gradient's
row) and reads back the two revealed outputs. Reuse the existing
peer-resolution/subprocess/timeout boilerplate rather than duplicating it.

**Phase 3 — A genuinely new concept for Flotilla: the server-side root
gradient.** Nothing today has the server train its own local update —
`aggregator_secure_mpc.py` only aggregates, it never trains. This needs: a
small trusted dataset config on the server, a training step each round
producing the root gradient, secret-sharing it with the same
`Replicated3PCScheme` clients already use (Trio's client-side sharing
scheme is the same replicated3pc scheme Replicated itself uses — see
`hpmpc_backend.md`'s Trio section), and submitting it as the distinguished
(n+1)th input. This is the single biggest net-new piece of plumbing in this
plan — comparable in scope to Tetrad's new sharing scheme was for the
Trio/Tetrad work.

**Phase 4 — New aggregator + config wiring.**
A new `aggregator_secure_flod.py` (selected via
`session_config.aggregator: secure_flod`), a `threshold` config value
(mirroring SAFEFL's own `flod_threshold`-as-fraction-of-param-count
convention — see `SAFEFL/main.py`'s `--flod_threshold` argument), and
dividing the revealed weighted sum by the revealed `weight_sum` in
plaintext, matching the FedAvg division pattern exactly.

**Phase 5 — Verification, matching this project's established standard.**
Mocked-subprocess unit tests first (fast tier, following the pattern in
`tests/unit/test_backend_hpmpc.py`), then a real Docker build/run with
hand-computed expected Hamming distances and weights for a small known
input set — not just "it runs," but exact matching against a manually
derived expectation, the same rigor already used for Trio/Tetrad's own
rollout (see `hpmpc_backend.md`'s "How this was verified" section for the
standard to match).

**Phase 6 — Docs.**
A new section (in `hpmpc_backend.md` or a dedicated file) describing FLOD
support with the same honesty applied elsewhere in this project's docs:
still Trio-only (Replicated/Tetrad lack the A2B conversion `LTZ` needs);
`weight_sum` is deliberately revealed in plaintext rather than divided
inside MPC (a documented privacy/complexity tradeoff); in-MPC division via
the existing Newton-Raphson primitive is tractable future work given
`weight_sum`'s bounded, public range, flagged as such rather than silently
left undone.

## 5. Open questions / things not yet resolved

- Whether `LTZ`/`EQZ`/`Relu.hpp` actually **compile and run correctly**
  against real Trio binaries has **not** been verified yet — everything in
  section 2's "restricted to Trio" claim is based on reading which
  `prepare_A2B_S1`-style methods exist in source, not on a real build/run.
  Phase 0 above exists specifically to close this gap before committing to
  the rest of the plan.
- Whether hpmpc's Newton-Raphson division primitive (section 2) would
  actually converge well enough for real FLOD/FLTrust value ranges within
  a reasonable iteration count has not been tested empirically.
- No attempt was made to verify whether Replicated or Tetrad could be
  *given* an A2B implementation (i.e., how much work that would actually
  be) — section 2 only established that it doesn't exist today, not how
  hard it would be to add.
