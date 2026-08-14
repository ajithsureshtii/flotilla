# Proto contract: `secure_agg.proto`

Source: [`src/proto/secure_agg.proto`](../../src/proto/secure_agg.proto).
Regenerate with `src/proto/run_secure_agg.sh` (or `run3_secure_agg.sh`) —
see `src/proto/README.md`.

## `SecureAggPartyService` — client/flo_server ↔ party control plane

### `SubmitShare(SubmitShareRequest) -> SubmitShareAck`

Called by `flo_client` (via `client_secure_agg_manager.share_and_submit`),
once per client per round, once per party endpoint (so 3 calls per client
per round for a 3-party cluster). Carries one `TensorShare` per model layer;
each `TensorShare.share_payload` is `pickle.dumps(PartyShare.payload)` for
*that* party's share of that layer (already pre-weighted by the client's own
dataset size — see [`design.md`](design.md)).

The party rejects (`accepted=False`) if `sharing_scheme` doesn't match its
own configured scheme — a config-mismatch guard, not a security boundary
(both sides are trusted to have consistent config in this design; a
mismatched scheme would otherwise silently produce garbage on reveal).

### `RunAggregationRound(RunAggregationRoundRequest) -> RunAggregationRoundResponse`

Called by `flo_server` (via `party_orchestrator_client.run_round`), once per
round, fanned out concurrently to every configured party endpoint via a
thread pool. Requires **every** party to have already buffered a share for
every `client_id` listed — if any is missing, the party responds
`success=False` immediately rather than hanging (a live MPC round can't
proceed with a client's share missing; see `docs/secure_aggregation/
threat_model.md`'s "no fault tolerance" note). `client_weights` is carried
for potential audit/logging use, but note the returned `aggregated_model` is
the **raw, un-normalized** sum — `aggregator_secure_mpc.py` divides by
`total` in plaintext after this returns (see `design.md`'s weighting
discussion).

### `HealthCheck(HealthCheckRequest) -> HealthCheckResponse`

Reports `backend_id` (which `SecureAggregationBackend` this party is
running) — a debugging/ops aid, not currently wired into any automated
health-monitoring loop (a documented gap, candidate for `runbook.md`
in Phase 4).

## `SecureAggPeerService` — party ↔ party (backend-private)

**Not part of the `SecureAggregationBackend` ABC's contract.** Used only by
backends implemented as gRPC-native Python (`backend_simulator.py`, Phase 2).
A backend with its own wire protocol (`backend_hpmpc.py`'s raw TCP/TLS
sockets between spawned binaries, Phase 3) never calls this service at all —
it's invisible above the backend boundary.

### `GetFinalShare(GetFinalShareRequest) -> GetFinalShareResponse`

`SimulatorBackend`'s only peer RPC: "give me your locally-summed share for
this round_id." `ready=False` if the peer hasn't finished its own local sum
for that round yet — the caller polls (see `backend_simulator.py`'s
`_fetch_peer_share`) until `ready=True` or `timeout_s` elapses.

## `round_id` — replay/mix-up guard, not a security guarantee

Always `f"{session_id}:{round_idx}"`. Carried on every `SubmitShare` and
`RunAggregationRound` call so a party can key its share buffer and its
`SimulatorBackend._local_sums` cache correctly even across concurrent
sessions/rounds. This catches accidental cross-round mix-ups (a bug,
misconfiguration, or stale retry) — it is **not** designed to resist an
adversarial party replaying old shares; see `threat_model.md`.

## Wire-format notes

- `TensorShare.shape`/`dtype` are carried explicitly (not just inferred from
  the pickled payload) so that a *future* sharing scheme whose payload
  doesn't natively carry shape/dtype (e.g. a flattened byte blob) still has
  everything a backend needs to reshape/re-type the result after reveal —
  see `backends/base.py`'s `TensorSpec`.
- Everything model-related is still `pickle`-based (matching the existing
  `grpc.proto` convention for `model_wts`/`model_weights`/`metrics`), over
  plaintext gRPC (no TLS) by default — a pre-existing gap in Flotilla, not a
  regression; see `threat_model.md`'s residual-risks section for the
  recommended hardening.
- **Confirmed hazard (hit during Phase 2 e2e testing):** `PartyShare.payload`
  for `replicated3pc` is a tuple of numpy arrays, and `share_payload` is
  `pickle.dumps(payload)` — numpy's pickle format embeds version-specific
  internals (e.g. `numpy._core` on numpy>=2.0 vs `numpy.core` on numpy 1.x),
  so a client and a party process running incompatible numpy major versions
  fail to interoperate (`No module named 'numpy._core'` or similar) even
  though nothing in *this* code changed. Today's pinned `numpy==1.24.3` in
  every `requirements.txt` (client, server, and the party image, which
  reuses `src/server/requirements.txt`) keeps every process consistent, so
  this doesn't bite in the checked-in deployment — but it's a latent
  fragility if any component's numpy pin ever drifts independently.
  Hardening candidate for a future phase: serialize `PartyShare` payloads as
  raw bytes (`array.tobytes()`) plus explicit dtype/shape (already carried
  on `TensorShare` anyway) instead of `pickle.dumps(ndarray)`, removing the
  numpy-version coupling from the wire format entirely.

## `grpc.proto`'s one additive change

`InitTrainResponse.model_weights` is now `optional bytes` (was a plain
`bytes` field), plus a new `bool secure_agg_used`. Backward compatible: when
`secure_agg_used` is unset/false, behavior is byte-for-byte identical to
before this change. See `server_session_manager.py`'s
`grpc_train_callback` (`response.HasField("model_weights")`) and
`client_grpc_manager.py`'s `StartTraining` (branches on
`secure_aggregation_config["enabled"]`).
