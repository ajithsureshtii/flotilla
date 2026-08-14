# Topology

**Status: Phase 2.** Real proto, real processes, real docker-compose — using
the `simulator` backend. hpmpc (Phase 3) slots in behind the same
`SecureAggregationBackend` interface without any topology change.

## Processes

| Process | Entrypoint | Count | Role |
|---|---|---|---|
| `flo_server` | `src/flo_server.py` | 1 | Unchanged control-plane orchestrator: round/session/client-selection/checkpoint/validation logic. Never sees a client's raw or shared update. |
| `flo_client` | `src/flo_client.py` | N (many) | Trains locally (unchanged). If `secure_aggregation.enabled`, secret-shares its update and submits shares directly to the 3 party processes instead of sending plaintext weights to `flo_server`. |
| `flo_secure_agg_party` | `src/flo_secure_agg_party.py` | 3 (party 0/1/2) | **New.** One process per MPC party. Buffers incoming client shares, runs the configured `SecureAggregationBackend` once `flo_server` triggers a round, reveals the plaintext aggregate back to `flo_server`. |

## Network relationships

```
flo_client ──(gRPC EdgeService, unchanged)──> flo_server
flo_client ──(gRPC SecureAggPartyService.SubmitShare)──> flo_secure_agg_party {0,1,2}
flo_server ──(gRPC SecureAggPartyService.RunAggregationRound)──> flo_secure_agg_party {0,1,2}
flo_secure_agg_party i ──(backend-specific, e.g. SecureAggPeerService.GetFinalShare
                          for the `simulator` backend)──> flo_secure_agg_party j
```

- `flo_server` **never** talks to `flo_client` about secure-agg shares, and
  **never** talks to any `flo_secure_agg_party` process except to trigger/
  collect a round's result. It has no share-transport role at all.
- `flo_client` gains a **second** outbound gRPC relationship (to 3 party
  endpoints) in addition to its existing inbound relationship with
  `flo_server` (which still dials the client for `StartTraining` etc.,
  unchanged).
- Party-to-party traffic (`backend_port` in config) is backend-private. For
  the `simulator` backend it's a small internal gRPC service
  (`SecureAggPeerService`); hpmpc's own raw TCP/TLS sockets (Phase 3) will
  use this same port role but a completely different wire protocol — no
  proto/topology change needed above the backend boundary.

## Docker Compose (`docker/docker-compose.yaml`)

3 new services, `secure_agg_party0/1/2`, built from
`docker/Dockerfile.secure_agg_party`. Each party's control-plane port
(`bind_port`, `SecureAggPartyService`) is published and reachable on the
shared `flotilla-network` (same network `redis`/`mqtt5` use) so `flo_client`/
`flo_server` can reach it. Each party's `backend_port` additionally joins a
second, `internal: true` network (`secure-agg-network`) that only the 3
party containers are on — `flo_client`/`flo_server` have no need to reach a
party's inter-party wire port and shouldn't be able to.

`flo_server` and `flo_client` themselves are still **not** part of
docker-compose (matching today's existing pattern — see
`docker/sample_docker_server_run.sh`/`sample_docker_client_run.sh`); only the
3 party containers are defined together in compose, since they're a single
logical cluster that's naturally stood up/torn down as a unit, unlike the
server/clients which scale independently.

Per-party config differences (`party_index`, ports, `peers`) are set via env
vars (`PARTY_INDEX`, `BIND_PORT`, `BACKEND_PORT`, `PEERS_JSON`) rather than
the `sed`-patching convention `server_entrypoint.sh`/`client_entrypoint.sh`
use — `peers` is a YAML list, which `sed` can't edit safely. See
`flo_secure_agg_party.py`'s `_apply_env_overrides()` and
`docker/secure_agg_party_entrypoint.sh`.

## What doesn't change

Everything not listed above: `flo_server`'s round/session/checkpoint/
validation logic, `client_trainer.py`'s local training loop, the
`aggregate()` plugin call signature, the `StateManager` abstraction (reused
verbatim for the party's share buffer). See
[`design.md`](design.md) for the full picture.
