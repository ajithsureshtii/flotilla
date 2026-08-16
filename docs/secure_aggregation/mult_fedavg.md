# mult_fedavg: MPC-side product weighting mode

**Status: implemented and verified end-to-end against real compiled Trio
binaries (both a standalone C++ harness and the real `HpmpcBackend` Python
class, driving real subprocesses over real localhost sockets) — see
"How this was verified" below. Trio (`PROTOCOL=5`) only.**

## Motivation

The existing secure-aggregation design (see [`design.md`](design.md) and
[`hpmpc_backend.md`](hpmpc_backend.md)) is deliberately built so the MPC
party cluster never has to do anything more than sum shares and reveal:
each client pre-weights its own update by its own dataset size *before*
sharing, so `Sum(N_k * update_k)` is just plain addition of already-weighted
shares. This is fast and simple, but it also means the party servers never
actually have to *communicate as part of the MPC protocol itself* — summing
shares is local, free arithmetic under additive/replicated sharing.

`mult_fedavg` is a second aggregation variant that exists specifically to
exercise the case where servers genuinely must communicate: clients share
their **raw** (unweighted) update and their **raw** dataset size
separately, and the party cluster computes `weight_i * dataset_size_i` for
every client via genuine secret × secret multiplication, inside the MPC
circuit, before summing across clients. `Sum(w_i * d_i) != Sum(w_i) *
Sum(d_i)`, so presumming (like the default mode does) would be
mathematically wrong here — the multiply has to happen per-client, which is
what makes this variant a real test of inter-party communication.

Selected via a `weighting_mode` config value (`"client_side"`, the
default/original behavior, vs `"mpc_product"`, this variant) that must
agree across every client and every party in a deployment — see
"Configuration" below.

## The key design problem, and how it was solved

The obvious-looking approach — construct a raw Trio `(p1, p2)` share pair
for a client's full secret value and feed it directly into
`Additive_Share::prepare_mult` — **does not work**. It was tried during
development and gives a numerically wrong product
(`val_a*val_b - 2*r_a*r_b mod 2^64` for the specific hand-derived
share structure tried, confirmed by independent Python computation): Trio's
`prepare_mult` implements a Beaver-triple-style protocol whose masking
arithmetic depends on how a share was actually produced, not just on its
`(p1, p2)` values satisfying the reveal invariant. A raw injection that
reveals correctly does not necessarily multiply correctly.

The fix: instead of hand-constructing a share for the *full* secret, each
of the 3 parties inputs its own **already-known** `replicated3pc` share
fragment (`c_j` — the same value it already receives from the client via
`SubmitShare` for the existing reveal-only path) using hpmpc's own,
already-tested `prepare_receive_from<P_j>` mechanism — exactly what
hpmpc's own test suite (`programs/tests/test_basic_primitives.hpp`,
`test_multiplication()`) uses for its multiplication test operands. All 3
parties run all 3 of these calls (their own real fragment for their own
role, an unused placeholder for the other two), batched under one
`communicate()`. Plain, native `Additive_Share::operator+` (already
verified elsewhere in this codebase to be correct elementwise addition for
Trio) then combines the 3 fragments into a share that *does* compose
correctly with `prepare_mult`, because it was built entirely out of
hpmpc's own proven input+add primitives rather than a hand-rolled raw
construction.

This is the concrete instance of "use hpmpc's own preprocessing/native-input
mechanism instead of hand-deriving raw shares" — each party's own fragment
is exactly the kind of "material one party already has" that
`prepare_receive_from` is designed to correctly turn into a share usable by
the rest of the protocol, including multiplication.

### Why `complete_mult_without_trunc()`, not `complete_mult()`

`Additive_Share::complete_mult()` dispatches to `complete_mult_with_trunc()`
whenever `FRACTIONAL > 0` (true for every build this project uses). For
Trio specifically, `OECL0_Share::complete_mult_with_trunc()`
(`protocols/3-PC/ours/oecl-P_0_template.hpp`, and its P1/P2 siblings) is an
**empty stub** — confirmed by reading the function body. Calling it doesn't
crash; it silently leaves the share untouched, which would silently corrupt
every product.

So `mult_fedavg_secure_aggregation.hpp` always uses
`complete_mult_without_trunc()`, which gives the raw, untruncated product —
since both operands are `frac_bits`-encoded fixed-point values, their
product carries `2*frac_bits` of fractional precision. No in-MPC truncation
is needed at all: the final product-sum is revealed and decoded on the
Python side anyway (mirroring the existing `DATASET_SIZE_LAYER_NAME`
reveal-and-decode pattern), so `backend_hpmpc.py` just decodes this one
field at `2*frac_bits` instead of `frac_bits`.

## Protocol, per round

1. **Native-input round.** For every client, every weight element, and the
   client's dataset size: all 3 parties call `prepare_receive_from<P_j>`
   for `j = 0, 1, 2` (their own real fragment when `j == PARTY`, an unused
   placeholder otherwise), batched under one `communicate()`.
2. **Local combine** (no communication): `weight_full = frag_p0 + frag_p1 +
   frag_p2` (and the same for dataset size) — a genuine, multiplication-
   compatible Trio share of the client's raw value.
3. **Multiply round.** `product = weight_full.prepare_mult(dataset_size_full)`
   per client per element, one `communicate()`, then
   `complete_mult_without_trunc()`.
4. **Local sum across clients** (no communication): plain
   `Additive_Share::operator+`, both for the per-element products and for
   the dataset sizes.
5. **Reveal round.** `prepare_reveal_to_all()` for the summed product vector
   and the summed dataset size, one `communicate()`.

Three `communicate()` rounds total for the live computation — this is the
genuine inter-party communication this variant exists to exercise, as
opposed to the default mode's single reveal-only round.

## File contract

`mult_fedavg_secure_aggregation.hpp` (`FUNCTION_IDENTIFIER=91`), talked to
by `backend_hpmpc.py`'s `HpmpcBackend._run_mpc_product_round`:

**Input file** (this party's own fragments only — no other party's data is
needed in this file, since the native-input mechanism handles combining
fragments across parties inside the MPC circuit):
```
uint32 num_clients
uint32 elements_per_client
for each of num_clients clients, in a fixed (round-stable) order:
    elements_per_client x uint64   -- this party's own replicated3pc share
                                      fragment (c_j) of this client's RAW
                                      (unweighted) update, one per
                                      flattened weight element (across ALL
                                      real layers, concatenated in
                                      sorted-layer-name order, exactly like
                                      the default mode's flattening)
    1 x uint64                     -- this party's own replicated3pc share
                                      fragment (c_j) of this client's RAW
                                      dataset size
```

**Output file:**
```
uint32 elements_per_client
elements_per_client x uint64   -- revealed sum-across-clients of
                                  weight_i*dataset_size_i, RAW (2*frac_bits
                                  fractional precision -- decode
                                  accordingly, not via FixedPointCodec's
                                  single frac_bits)
1 x uint64                     -- revealed sum-across-clients of
                                  dataset_size_i (normal frac_bits -- decode
                                  like the existing DATASET_SIZE_LAYER_NAME
                                  field)
```

## Configuration

Both the client and every party must agree on `weighting_mode` for a given
deployment/round — there is no runtime negotiation.

**Client** (`secure_aggregation.weighting_mode` in `client_config.yaml`,
default `"client_side"`):
```yaml
secure_aggregation:
  enabled: true
  sharing_scheme: replicated3pc
  weighting_mode: mpc_product   # or client_side (default)
  ...
```
`client_secure_agg_manager.share_and_submit`'s `weighting_mode` param
controls whether the update is pre-multiplied by `dataset_size` before
sharing (`"client_side"`) or shared raw (`"mpc_product"`). The raw dataset
size is *always* shared under `DATASET_SIZE_LAYER_NAME`, regardless of mode
— only whether the real layers get pre-weighted changes.

**Party** (`backend.hpmpc.weighting_mode` in `secure_agg_party_config.yaml`,
or the `HPMPC_WEIGHTING_MODE` env var override, default `"client_side"`):
```yaml
backend:
  hpmpc:
    protocol: 5   # mpc_product currently requires Trio
    executable_dir: /opt/hpmpc/executables/trio_mult
    weighting_mode: mpc_product
```
`HpmpcBackend`'s constructor raises `ValueError` at construction if
`weighting_mode="mpc_product"` is paired with any `protocol` other than 5.
`_check_config_consistency()` (run at `start()`) also verifies the compiled
executables' `function_identifier` metadata field (91 for `mpc_product`, 90
for `client_side`) matches what this party is configured for — a party
pointed at the wrong executables fails loudly at startup with a clear
message, rather than mid-round with a confusing file-format mismatch (the
`mpc_product` file layout looks nothing like a pre-summed row, so the wrong
binary would misparse it in a hard-to-diagnose way).

## Building

```bash
cd mpc_engines/hpmpc
scripts/build_secure_agg.sh trio_mult [bitlength] [frac_bits]   # defaults: 64 13
```
Builds `executables/trio_mult/run-P{0,1,2}.o` and writes
`executables/trio_mult/mult_fedavg_secure_aggregation.build_metadata.json`
(`function_identifier: 91`).

## Why the aggregator needed zero changes

`aggregator_secure_mpc.py`, `party_orchestrator_client.py`, and
`party_server.py` are all unchanged by this variant — verified by tracing,
not just asserted:

- `party_server.py`'s `RunAggregationRound` forwards `shares`/`tensor_specs`
  to whatever backend is configured and pickles back whatever
  `OrderedDict` it returns — no assumptions about *how* that dict was
  produced.
- `party_orchestrator_client.py`'s `run_round` cross-checks that every
  party returned the same tensors for the same layer names — also
  backend-agnostic.
- `aggregator_secure_mpc.py` pops `DATASET_SIZE_LAYER_NAME` out of the
  returned dict and divides every remaining layer by it. Since
  `HpmpcBackend._run_mpc_product_round` returns an `OrderedDict` with
  exactly that same shape (real layers holding the revealed
  `Sum(w_i*d_i)`, plus a `DATASET_SIZE_LAYER_NAME` entry holding the
  revealed `Sum(d_i)`), dividing the two in plaintext gives the correct
  weighted average with no code changes needed anywhere in this chain.

The entire feature is scoped to `backend_hpmpc.py` (a new private method,
`_run_mpc_product_round`, dispatched from the existing
`run_aggregation_round`) plus one conditional in
`client_secure_agg_manager.py`.

## Known limitations

- **Trio (`PROTOCOL=5`) only.** The native-fragment-input +
  `prepare_mult` composition this relies on was verified empirically for
  Trio specifically. Replicated (`PROTOCOL=2`) and Tetrad (`PROTOCOL=8`)
  would each need their own from-scratch verification of the same
  composition (their own `prepare_mult`/`complete_mult` implementations,
  and whether their own `prepare_receive_from`-based native input composes
  correctly with multiplication, have not been checked) before being wired
  in. `HpmpcBackend` raises `ValueError` at construction if
  `weighting_mode="mpc_product"` is requested with any other protocol.
- **More communication than the default mode.** 3 `communicate()` rounds
  per aggregation round vs. 1 for the default (reveal-only) mode — this is
  inherent to the feature's purpose, not an inefficiency to fix.
- **No in-MPC truncation.** As explained above, this is a deliberate choice
  (Trio's truncated-multiply completion is a broken stub anyway), not a
  gap — decoding at `2*frac_bits` on the Python side is exact for this use
  case.

## How this was verified

1. **Standalone C++ harness** (scratch, not committed): a minimal hpmpc
   program performing the native-input round + multiply + reveal for
   `a=encode(3.0)=24576`, `b=encode(5.0)=40960` (both `frac_bits=13`),
   using real `Replicated3PCScheme` share fragments computed in Python.
   Built and run for real (3 processes, real Trio binaries, real sockets)
   inside a Docker container: all 3 parties revealed the raw product
   `1006632960`, exactly `24576*40960`; decoding at `2*13=26` fractional
   bits gives `1006632960 / 2**26 == 15.0 == 3.0*5.0` exactly.
2. **The real `mult_fedavg_secure_aggregation.hpp`**, built and run for
   real (3 Trio processes) against a hand-packed 2-client, 3-element input
   file (`client1: weights=[1.5,-2.0,0.25], dataset_size=100`; `client2:
   weights=[3.0,4.0,-1.0], dataset_size=50`): all 3 parties revealed
   `[300.0, 0.0, -25.0]` and total dataset size `150.0`, matching a
   plaintext computation of `Sum(w_i*d_i)` and `Sum(d_i)` exactly.
3. **The real `HpmpcBackend` Python class** (`weighting_mode="mpc_product"`),
   built via the actual committed `scripts/build_secure_agg.sh trio_mult`
   and driven end-to-end (real `asyncio.create_subprocess_exec`, real
   compiled binaries, real localhost TCP sockets, 3 concurrent party
   processes) against the same 2-client scenario as above: all 3 parties'
   `run_aggregation_round()` calls returned
   `{"w": [300.0, 0.0, -25.0], DATASET_SIZE_LAYER_NAME: [150.0]}`, matching
   the plaintext expectation exactly.
4. **Unit tests** (`tests/unit/test_backend_hpmpc.py`,
   `tests/unit/test_client_secure_agg_manager.py`): constructor validation
   (`weighting_mode`/`protocol` compatibility, unsupported
   `weighting_mode` values), the `function_identifier` config-consistency
   check, `_own_fragment`'s extraction logic, the mpc_product file-packing
   and decode-at-double-scale logic (mocked subprocess, real
   `Replicated3PCScheme` shares), and the client-side pre-weighting
   skip. All pass alongside the full existing suite (132 unit tests).
