# Secure Multi-Party Aggregation — Design

**Status: Phase 4 complete.** Pure-Python replicated-3PC sharing (Phase 1),
the real 3-process network topology (Phase 2), a real hpmpc-backed
`SecureAggregationBackend` (Phase 3), and full rollout (Phase 4) all exist
and are verified. `docker-compose.yaml`'s default topology now builds the
`secure_agg_party0/1/2` services from the hpmpc-backed image
(`Dockerfile.secure_agg_party.hpmpc`), and a real end-to-end MNIST + LeNet5
training run has been completed successfully with `aggregator: secure_mpc`
across 3 genuine Docker containers (accuracy climbing 71.95% → 97.35% →
97.7% across 3 rounds, comparable to a plaintext `fedavg` baseline run of
the same config). Fault tolerance has also been verified live: stopping a
party mid-session produces a clean, fast per-round failure (not a hang),
and the cluster fully recovers a subsequent session once the party is
restarted. See [`hpmpc_backend.md`](hpmpc_backend.md) for the backend
adapter details, [`topology.md`](topology.md) for the process/network
picture, [`runbook.md`](runbook.md) for operating it, and
[`rollout_guide.md`](rollout_guide.md) for toggling it on/off.

See also: [`adr-0001-topology.md`](adr-0001-topology.md) (the topology decision
and why it was made), [`threat_model.md`](threat_model.md) (what is and isn't
protected), [`proto_contract.md`](proto_contract.md) (the wire contract),
[`sharing_scheme_replicated3pc.md`](sharing_scheme_replicated3pc.md) (the
sharing math), and [`hpmpc_backend.md`](hpmpc_backend.md) (the hpmpc
backend, including a hard-won share-format derivation worth reading before
touching that code).

**One design refinement made during Phase 2, worth flagging explicitly:**
the `SecureAggregationBackend.run_aggregation_round` signature originally
sketched in Phase 0 took a `weights` parameter for the backend to apply
inside the round. That was dropped — see `backends/base.py`'s docstring —
because multiplying a share by a fractional public weight requires a
correct truncation step for a fixed-point scheme (a genuine MPC primitive,
not something to improvise). Instead, clients pre-weight their update by
their own (always locally known) raw dataset size before sharing; backends
only ever sum shares (always free) and reveal; the final division by the
round's total weight happens in plaintext, post-hoc, in
`aggregator_secure_mpc.py` — exactly mirroring how `aggregator_fedavg.py`
already computes its weights today, and automatically correct under client
dropouts.

**Post-rollout enhancement: dataset sizes are also secret-shared.**
Originally (Phases 0–4), `flo_server` learned every client's dataset size
in the clear via the existing plaintext `InitBench`/`StartTraining` RPCs and
computed each client's weight fraction itself, sent to the parties on
`RunAggregationRoundRequest.client_weights`. That field has been removed
(see `proto_contract.md`) — in `secure_mpc` sessions, each client now
*also* secret-shares its raw dataset size, under a reserved
`DATASET_SIZE_LAYER_NAME` pseudo-layer (`server/secure_agg/constants.py`),
summed and revealed by the exact same generic backend mechanism used for
model-weight layers (see `backends/base.py`'s docstring) — no backend code
changed at all to support this, which is exactly the payoff of the
genericity layer being "any named tensor," not "model weights
specifically." `aggregator_secure_mpc.py` pops the revealed total and uses
it as the division denominator; neither it nor any party ever learns an
individual client's dataset size in `secure_mpc` mode. See
`threat_model.md` for the updated protected/not-protected breakdown
(including the degenerate single-client-round caveat this inherits).

**Post-rollout enhancement: `HpmpcBackend` now supports multiple hpmpc
protocols.** Originally, `HpmpcBackend` was hardcoded to hpmpc's
`PROTOCOL=2` ("Replicated 3PC"). It now takes a required `protocol`
constructor arg and internally dispatches to a small per-`(protocol,
party_index)` share-packing strategy for the only genuinely
protocol-specific pieces — share-conversion math and on-disk field
count/layout; everything else (`start()`, peer sorting, hostname
resolution, subprocess invocation, timeout/error handling) is shared,
unchanged. Protocol is a config value (`backend.hpmpc.protocol`), not a
different backend module — every hpmpc protocol shares the same
integration shape, so splitting into separate backend modules per protocol
number would triplicate the shared plumbing for no genericity benefit (see
`hpmpc_backend.md`'s "Supporting multiple protocols" section). Trio
(`PROTOCOL=5`, 3-party, semi-honest) and Tetrad (`PROTOCOL=8`, 4-party,
labeled malicious-secure upstream) have landed behind this same interface —
see `hpmpc_backend.md` for both protocols' derivations and verification
narratives. Tetrad's 4-party masking structure couldn't be derived from the
existing 3-party `Replicated3PCScheme`, so it also introduced a new
`SecretSharingScheme` (`sharing_schemes/tetrad4pc.py`, `Tetrad4PCScheme`)
that produces already-native-shaped payloads per party, plus two new
per-party env overrides (`NUM_PARTIES`, `SHARING_SCHEME`) since Tetrad is
the first protocol here that needs a party count/sharing scheme different
from the checked-in config file's defaults. **Important:** real
corruption-testing against a live Tetrad deployment found its
malicious-abort detection does not fire for this integration's actual
usage — see `hpmpc_backend.md`'s "Malicious-security caveat, found
empirically" and `threat_model.md`'s adversary model before relying on
Tetrad for anything beyond semi-honest security. Also added as part of this
work: a `local_training_disabled` client debug flag (see
`rollout_guide.md`) for fast iteration on the aggregation path without
paying real training cost, and an `hpmpc_backend.py` `log_stdout` option for
capturing hpmpc's own per-round timing/communication output (used by the
secure-aggregation overhead report, `overhead_report.md`).

**Vendored MPC libraries.** hpmpc lives at `mpc_engines/hpmpc` as a git
submodule — vendored, not committed inline, since it's forked upstream code
(`chart21/hpmpc`) with local patches (this project's `PROTOCOL=8`/Tetrad C++
branches, a real malicious-abort bug fix, and the benchmark tooling — see
`hpmpc_backend.md`) that still need to stay mergeable against upstream
updates. `mpc_engines/` is deliberately a directory, not a single hardcoded
path, so a structurally different future MPC library (see "This must not
become an hpmpc-shaped abstraction" above) has an obvious place to land as
its own submodule alongside hpmpc, without another top-level directory or a
naming collision. Cloning this repo needs
`git clone --recurse-submodules` (or `git submodule update --init` after a
plain clone) to actually get hpmpc's source — Docker builds that reference
`mpc_engines/hpmpc` will fail with a confusing "not found"-style error
without it.

## Problem

Flotilla today has exactly one central aggregation server
(`src/flo_server.py` → `FlotillaServerManager`). Each round, every selected
client trains locally and sends its **plaintext** PyTorch `state_dict` back
to that one server over insecure gRPC
(`src/client/client_grpc_manager.py`'s `InitTrainResponse.model_weights`,
pickled bytes, no TLS). The server's aggregator plugin
(`src/server/aggregation/aggregator_fedavg.py`, loaded dynamically by
`src/server/load_aggregator.py`) averages these plaintext updates, weighted
by dataset size. This one server is a single point of trust that sees every
client's raw model update each round — the thing this work eliminates.

## Approach

Instead of one server seeing plaintext updates, each client secret-shares its
update across **3 non-colluding party servers**, who jointly compute the
weighted average via MPC and only ever reveal the final aggregate — never an
individual client's update — back into the existing training-orchestration
flow. `hpmpc` (a C++ MPC framework, vendored as a git submodule at
`mpc_engines/hpmpc` -- see "Vendored MPC libraries" below) is the **first**
backend for this layer, using its Replicated (2,3) secret-sharing 3PC
protocol (`PROTOCOL=2`), semi-honest / honest-majority.

**This must not become an hpmpc-shaped abstraction.** hpmpc's integration
shape — compile-time-everything, one static executable per party, process-
per-round, file-based I/O, no Python bindings, no daemon — is deliberately
kept behind an interface (`SecureAggregationBackend`, below) that a future,
structurally different MPC library (Python bindings, a persistent daemon+RPC,
Shamir sharing instead of replicated additive sharing, a different party
count) could also implement, without Flotilla's core (session manager,
client, round loop) needing to change. The plan follows Flotilla's own
existing plugin idiom — importlib-dispatch by a config string, as already
used by `load_aggregator.py`/`aggregation/aggregator_<name>.py` and
`server_state_manager.py`/`state_manager/<loc>.py` — rather than inventing a
new mechanism.

## Architecture at a glance

- **New standalone process type**, not a mode flag on `flo_server.py`:
  `src/flo_secure_agg_party.py`, run 3 times (party 0/1/2), each its own
  process/container. `flo_server.py` stays a singleton and keeps owning
  round/session/client-selection/checkpoint/validation logic — it just stops
  being where a client's raw update is ever materialized in plaintext. See
  the ADR for why this is a separate process type rather than a flag.
- **Client dials out to all 3 party processes directly** to submit shares
  (new gRPC service), bypassing `flo_server` entirely for share transport.
  `flo_server`'s aggregator plugin dials out to all 3 parties to trigger/
  collect the round's plaintext result — it never sees a raw or shared
  client update, only the final aggregate, matching the existing
  `aggregate()` contract exactly.
- **What's protected**: only a client's per-round local model update.
  Dataset sizes, metrics, round participation, the resulting global model,
  and architecture/hyperparameters stay plaintext exactly as today. See
  `threat_model.md`.

## Core abstractions (the genericity layer)

### `SecretSharingScheme` — pure Python/numpy, zero I/O

`src/server/secure_agg/sharing_schemes/base.py`. Splits/combines/locally-adds
secret shares with no network and no MPC-library dependency, so it's testable
without any backend installed.

```python
class PartyShare:
    party_index: int
    payload: Any  # scheme-specific, must be picklable

class SecretSharingScheme(ABC):
    scheme_id: str
    num_parties: int
    reconstruction_threshold: int

    def share(self, plaintext_fixedpoint, rng) -> list[PartyShare]: ...
    def reconstruct(self, shares: dict[int, PartyShare]) -> np.ndarray: ...  # tests/debug only
    def add(self, a: PartyShare, b: PartyShare) -> PartyShare: ...           # local, no comm
```

Concrete (Phase 1): `sharing_schemes/replicated3pc.py` — 2 random shares
`r0, r1` + `r2 = x - r0 - r1`, the standard (2,3)-replicated layout hpmpc's
`PROTOCOL=2` expects. A stub `sharing_schemes/shamir_stub.py` (docs-only,
`NotImplementedError` bodies) exists specifically to prove this ABC isn't
accidentally shaped around replicated sharing's specifics.

### `FixedPointCodec` — numeric encoding, kept separate from sharing

`src/server/secure_agg/fixed_point_codec.py`. Float ⇄ fixed-point-int
encode/decode, independent of `SecretSharingScheme` (mirrors hpmpc's own
separation of `Additive_Share` from `FloatFixedConverter`). `bitlength`/
`frac_bits` must match a given backend's numeric assumptions — for hpmpc,
the compile-time `FRACTIONAL` constant baked into the party executables (a
cross-language config-consistency hazard with its own explicit check in
Phase 3).

### `SecureAggregationBackend` — the seam that absorbs hpmpc vs. anything else

`src/server/secure_agg/backends/base.py`. Runs inside each party process, one
instance per party.

```python
class SecureAggregationBackend(ABC):
    backend_id: str
    party_index: int
    num_parties: int

    async def start(self, peer_endpoints: list[PartyEndpoint]) -> None: ...
    async def run_aggregation_round(
        self, round_id: str,
        shares: dict[str, dict[str, PartyShare]],   # client_id -> layer_name -> this party's share
        tensor_specs: dict[str, TensorSpec],
        timeout_s: float,
    ) -> "OrderedDict[str, torch.Tensor]": ...       # this party's plaintext reveal of the RAW sum
    async def stop(self) -> None: ...
```
No `weights` parameter — see the design-refinement note above. Clients
pre-weight by their own dataset size before sharing; the backend only ever
sums and reveals; the caller divides by the round's total weight afterward.

This signature says nothing about *how* peer communication happens during a
round — hpmpc opens its own raw sockets between spawned binaries; a Python
simulator talks to peers over a small internal gRPC call; a hypothetical
daemon-backed library would issue one RPC to its own persistent process. That
silence is what makes the interface genuinely backend-agnostic.

Two backends: `backends/backend_simulator.py` (pure Python/numpy — Phase 1
in-process, Phase 2 networked; validates every layer above the backend with
zero MPC dependency) and `backends/backend_hpmpc.py` (Phase 3 — file-IO +
subprocess adapter over compiled hpmpc executables). Selection via
`load_backend.py`, the same importlib-dispatch idiom as `load_aggregator.py`.

**Client side** uses `FixedPointCodec.encode()` → `SecretSharingScheme.share()`.
**Party side** never calls `share()`/`reconstruct()` in production, only
deserializes incoming shares and hands them to its backend.

## Running tests

```bash
cd flotilla
pip install -r src/server/requirements.txt -r requirements-dev.txt
pytest -m "unit or integration"        # fast tier, no Docker/hpmpc, run on every push
pytest -m e2e                          # docker-compose based, run manually/locally
pytest -m slow_hpmpc_build             # requires a real hpmpc build, opt-in (Phase 3+)
```

## Phase index

| Phase | Adds | Status |
|---|---|---|
| 0 | Test infra, docs skeleton, ABC interfaces — no behavior | Done |
| 1 | Pure-Python replicated3pc sharing math + in-process simulator | Done |
| 2 | Real 3-process topology (proto, party server, docker-compose), simulator backend over network | Done |
| 3 | hpmpc backend adapter behind the same interface | Done |
| 4 | Full rollout: client toggle, real e2e, threat model finalized | Done |
| 5 (optional) | Genericity stress-test: a second, structurally different backend | Not started |

See also [`mult_fedavg.md`](mult_fedavg.md): a second aggregation variant
(`weighting_mode="mpc_product"`, Trio only) where the party cluster itself
computes `weight_i * dataset_size_i` via genuine secret × secret
multiplication, instead of clients pre-weighting before sharing — exercises
real inter-party communication, unlike the reveal-only design above.
