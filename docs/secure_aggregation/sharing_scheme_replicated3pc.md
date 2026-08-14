# Sharing scheme: Replicated (2,3) additive secret sharing

Implementation: [`src/server/secure_agg/sharing_schemes/replicated3pc.py`](../../src/server/secure_agg/sharing_schemes/replicated3pc.py).

## The math

A secret `x` (a fixed-point integer, see [`fixed_point_codec.py`](../../src/server/secure_agg/fixed_point_codec.py))
is split into three random components `c0, c1, c2` such that:

```
c0 + c1 + c2 ≡ x   (mod 2**bitlength)
```

`c0` and `c1` are drawn uniformly at random from the full ring; `c2` is
computed as `x - c0 - c1`. Party `i` is handed the pair `(c_i, c_{i+1 mod 3})`:

| Party | Holds |
|---|---|
| 0 | `(c0, c1)` |
| 1 | `(c1, c2)` |
| 2 | `(c2, c0)` |

No single party's pair reveals anything about `x` (each pair is 2 of 3
uniformly random values — indistinguishable from random without the third).
Any **2** of the 3 parties' pairs together cover all three components
(`c0`, `c1`, `c2`, with one of the three duplicated) and can sum them to
recover `x`. This is the standard textbook (2,3)-replicated secret sharing
scheme (Araki et al. style) and is exactly the layout hpmpc's `PROTOCOL=2`
expects.

## The homomorphism this design relies on

Sharing is **additively homomorphic under local (no-communication) addition**:
if party `i` holds `share(a)_i` and `share(b)_i`, adding the two pairs
component-wise gives a valid share of `a + b`:

```
share(a)_i + share(b)_i == share(a + b)_i      (for every party i, no network round)
```

This is why summing N clients' secret-shared updates costs **zero network
rounds** under this scheme — each party locally sums its own share of every
client's update, and only the *final* sum needs a reveal round. FedAvg's
"sum weighted updates, then reveal" only ever needs one round-trip
regardless of N, which is what makes replicated sharing a good fit for
secure aggregation specifically (as opposed to a scheme requiring
communication per multiplication, which model averaging doesn't need at
all — averaging is public-scalar multiplication, i.e. `share(a) * public_c`
is also local/free, only used by the caller after reconstructing/decoding in
this codebase's current design, see `backends/backend_simulator.py`).

## Fixed-point rounding error vs. plaintext float arithmetic

`FixedPointCodec.encode()` rounds `plaintext * 2**frac_bits` to the nearest
integer before sharing, and `decode()` divides back down. Each individual
value therefore carries at most `2**-frac_bits / 2` of rounding error
relative to its true float value, independent of `bitlength` (as long as
the value fits — `encode()` raises `OverflowError` rather than wrapping
silently if it doesn't). Summing N clients' weighted updates accumulates at
most `N * 2**-frac_bits / 2` of error in the worst case (linear in N, not
compounding) — this bound is what test tolerances in
`tests/unit/test_backend_simulator.py` are derived from. With the default
`frac_bits: 13` (see `config/training_config.yaml`'s `aggregator_args`),
resolution is `2**-13 ≈ 1.2e-4`, comfortably below typical float32 model
weight precision for the small `frac_bits`-vs-`bitlength` tradeoffs this
project cares about.

## Why the sharing math needs no MPC library at all

`share()`/`reconstruct()`/`add()` are pure numpy — 2 random draws and a
subtraction, or a couple of additions. This is genuinely simple enough that
a client (plain Python, no C++, no hpmpc) can generate its own shares
locally and hand one to each party server; the party servers never need to
call `share()` in production (only `reconstruct()`/`add()` conceptually
happen, and even those are usually delegated to whatever
`SecureAggregationBackend` a party is configured with — see `backend_hpmpc.py`,
Phase 3 — rather than called directly, since a real backend must never
reconstruct an individual client's update, only the final aggregate).
