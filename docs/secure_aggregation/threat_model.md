# Secure Aggregation — Threat Model

**Status: Finalized (Phase 4).** The claims below are concrete and verified,
not aspirational: the data flow matches the real proto/topology (Phase 2),
the hpmpc backend exists and was verified against real compiled binaries
(Phase 3 — see `hpmpc_backend.md`), and the "no fault tolerance" residual
risk below was confirmed with an actual fault-injection test
(`tests/integration/test_secure_mpc_fault_injection.py`), not just asserted.

## Data flow (concrete as of Phase 2, dataset-size sharing added post-rollout — see `topology.md`, `proto_contract.md`)

```
flo_client            trains locally (unchanged) -> plaintext state_dict
                       |
                       | pre-weight by own dataset_size, encode, share.
                       | ALSO secret-share the raw dataset_size itself,
                       | under the reserved DATASET_SIZE_LAYER_NAME
                       | pseudo-layer (client_secure_agg_manager.py)
                       v
              SubmitShare x3  ---------------------------------+
                 |         |                                   |
                 v         v                                   v
        flo_secure_agg_party0   flo_secure_agg_party1   flo_secure_agg_party2
                 ^         ^                                   ^
                 |         |                                   |
                 +--- RunAggregationRound (triggered by flo_server, ------+
                      carries only client_ids -- no per-client weight)
                                     |
                                     v
                              flo_server (aggregator_secure_mpc.py)
                     pops the revealed DATASET_SIZE_LAYER_NAME total, divides
                     every other revealed layer by it (plaintext, post-hoc)
                       -> global_model (plaintext, same as today)
```

`flo_server` sits only at the bottom of this diagram: it triggers rounds and
receives the final plaintext aggregate (plus the round's total dataset
size, used only as a division denominator and then discarded), exactly like
`aggregate()` returns today for `fedavg`. It is never in the path a
client's share — or dataset size — travels, and it does not send any
per-client weight to the parties.

## Adversary model

- **3 parties**, semi-honest, honest-majority: at most 1 of the 3 party
  servers may be passively curious (tries to learn secrets from what it
  sees/computes) but follows the protocol correctly. No collusion between 2
  or more parties is assumed away — with 2 colluding parties, replicated
  (2,3) secret sharing is broken by construction (2 shares reconstruct the
  secret), so operationally the 3 parties must be run by genuinely
  independent, non-colluding operators for this guarantee to mean anything.
- **No protection against a maliciously/actively deviating party.** A party
  that sends incorrect protocol messages on purpose is out of scope for the
  initial implementation (hpmpc's Tetrad/4PC protocol is the documented
  upgrade path if this is later required — see `design.md`'s Phase 5).
- **Clients are trusted to correctly secret-share their own update.** A
  malicious client could submit garbage shares (garbage in, garbage out) —
  this is a data-quality/poisoning concern Flotilla already has in the
  plaintext world (a client can already submit a garbage plaintext update
  today) and is not a new attack surface introduced by this work.

## What is protected

- **A client's per-round local model update** (its trained `state_dict`)
  is never seen in the clear by any single party server, `flo_server`, or
  anyone observing the network — only secret shares of it travel from
  client to each party, and only the final weighted aggregate is ever
  revealed (to all 3 parties, then reported up to `flo_server`).
- **A client's per-round dataset size, in `secure_mpc` sessions.** The
  client secret-shares its raw dataset size the same way it shares its
  model update (under the reserved `DATASET_SIZE_LAYER_NAME` pseudo-layer —
  see `client_secure_agg_manager.py`), so no single party server or
  `flo_server` ever learns an individual client's dataset size — only the
  round's **total** across all checked-in clients is ever revealed, and
  only because `aggregator_secure_mpc.py` needs it as the division
  denominator (see `design.md`'s weighting discussion). Note this is
  `secure_mpc`-specific: a `fedavg` session still learns every client's
  dataset size in the clear via the same pre-existing `InitBench`/
  `StartTraining` RPCs it always has — this project doesn't touch that
  path at all.
  **Degenerate case:** as with the model update itself, this guarantee only
  means something when enough clients participate in a round. If only one
  client checks in, the "total" revealed IS that client's exact dataset
  size (and the "aggregate" model update IS that client's exact update) —
  an inherent property of any sum-based aggregation privacy scheme, not a
  bug in this implementation. Operators relying on this protection should
  ensure rounds have a meaningful minimum participant count.

## What is NOT protected (explicitly, so nobody assumes more than is delivered)

- **Training metrics/loss** — returned alongside the (shared) weights in
  `InitTrainResponse.metrics`, still plaintext; used by client-selection
  strategies and logging exactly as today.
- **Round participation** — which clients participated in which round is
  visible to `flo_server` and the party servers, unchanged from today.
- **The resulting global model** — revealed in the clear to all 3 parties
  and to `flo_server` at the end of every round, exactly as the plaintext
  path does today (this is required for the training loop, server-side
  validation, and checkpointing to keep working).
- **Model architecture / hyperparameters** — plaintext, unchanged.
- **The existing gRPC control-plane channels** (`flo_server` ↔ client,
  `flo_client` ↔ party for share submission, `flo_server` ↔ party for round
  triggering) are, like the rest of Flotilla today, not TLS-protected by
  default. This is a pre-existing gap in Flotilla (confirmed: today's
  `grpc.aio.insecure_channel` usage, no TLS anywhere in the current stack),
  not a regression introduced by this work — but it does mean the *shares*
  themselves, while individually meaningless without a threshold of them,
  travel in the clear over the control-plane gRPC unless that's hardened
  separately. Recommended hardening, not yet implemented: TLS on the
  `SecureAggPartyService`/`SecureAggPeerService` gRPC channels (mirroring
  whatever TLS story is eventually adopted for `EdgeService`, since none of
  Flotilla's gRPC is currently protected), and setting hpmpc's own
  `USE_SSL=1` (see `hpmpc_backend.md` — currently `0` for portability
  during development, with a real per-deployment cert instead of the
  checked-in self-signed one).

## Residual risks (tracked, not "fixed" — see `runbook.md` for living with them)

- **No fault tolerance — confirmed by test, not just asserted.**
  `tests/integration/test_secure_mpc_fault_injection.py` kills a party
  mid-round and confirms the round fails promptly (well within
  `round_timeout_s`, not hanging indefinitely) with state cleaned up so a
  later round isn't poisoned by the failure. Losing 1 of the 3 party
  processes still blocks training progress until it's back — replicated
  (2,3) sharing's "any 2 of 3 can reconstruct" property is about *offline*
  reconstruction, not the *live* protocol, which genuinely needs all 3
  parties online and participating (see `party_orchestrator_client.py`'s
  docstring). See `runbook.md`'s "Debugging a hung round" for the
  operational playbook this produces.
- **Replay/mix-up mitigation is best-effort, not a security guarantee.** The
  `round_id` field on `SubmitShare`/`RunAggregationRound` (Phase 2) guards
  against accidental cross-round mix-ups; it is not designed to resist an
  adversarial party replaying old shares.
- **`verify_party_agreement`** (a config option that cross-checks ≥2/3
  parties revealed an identical plaintext aggregate) is a correctness/
  liveness sanity check for catching bugs during development, not a
  malicious-security guarantee — a colluding or buggy majority can still
  agree on a wrong answer.
