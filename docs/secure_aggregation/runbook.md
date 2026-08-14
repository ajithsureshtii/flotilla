# Runbook

Operational guide for the secure-aggregation party cluster. See
[`topology.md`](topology.md) for the process/network picture and
[`threat_model.md`](threat_model.md) for what a hung/failed round does and
doesn't expose.

## Bringing the cluster up

```bash
cd flotilla/docker
docker compose up -d --build secure_agg_party0 secure_agg_party1 secure_agg_party2
```

By default (`docker/Dockerfile.secure_agg_party.hpmpc`, `BACKEND_TYPE=hpmpc`
in `docker-compose.yaml`) this compiles the real hpmpc executables — the
first build takes a few minutes (C++ compile + `pytorch/pytorch` base image
pull if not already cached). Subsequent builds are fast (Docker layer
caching) unless `hpmpc/` source changes.

**Fast dev loop without a C++ build**: switch the 3 services' `dockerfile:`
back to `docker/Dockerfile.secure_agg_party` and set `BACKEND_TYPE: simulator`
(or unset it — that's the config file's own default) — see
`docker-compose.yaml`'s comment on the `secure_agg_party0` service.

### Health check

```bash
python3 -c "
import grpc
import proto.secure_agg_pb2 as pb2
import proto.secure_agg_pb2_grpc as pb2_grpc
for port in (50100, 50101, 50102):
    ch = grpc.insecure_channel(f'localhost:{port}')
    r = pb2_grpc.SecureAggPartyServiceStub(ch).HealthCheck(pb2.HealthCheckRequest(), timeout=5)
    print(port, r.ready, r.backend_id)
"
```

Expect `True hpmpc` (or `True simulator`, matching whichever backend is
configured) for all 3. If a `HealthCheck` call itself times out or refuses
the connection, the container isn't up yet or crashed — check
`docker compose logs secure_agg_party<N>`.

### Tearing down

```bash
docker compose down -v
```

## Debugging a hung round

A round can legitimately fail (not hang forever — see
`party_orchestrator_client.py`'s `run_round`, which requires all 3 parties
to respond and is bounded by `round_timeout_s` per attempt), but if training
seems stuck with no progress:

1. **Check each party's health** (above). A crashed/unreachable party is the
   most common cause — the live MPC round genuinely needs all 3 parties (no
   fault tolerance, see `threat_model.md`), so training simply cannot
   progress until it's back.
2. **Check `flo_server`'s logs** for
   `fedserver.aggregator.secure_mpc.exception` — `aggregator_secure_mpc.py`
   logs the underlying error (which party/parties failed, and why) before
   returning `None` for that round. `flo_server` will keep retrying on
   subsequent client check-ins; it does not treat one failed round as fatal.
3. **Check a party's own logs** for
   `fedparty.run_round.missing_shares` — means `flo_server` triggered
   `RunAggregationRound` before every expected client's share had arrived at
   that party. Usually a symptom of a client that's still training, or a
   client whose `SubmitShare` calls to one or more parties failed/timed out
   (check the client's own logs for
   `fedclient.secure_agg.submit_share.rejected`).
4. **hpmpc backend specifically**: `fedparty.run_round.exception` on a party
   using the `hpmpc` backend most often means either (a) a config mismatch
   caught by `HpmpcBackend._check_config_consistency()` at party *startup*
   (check logs right after the party container starts, not mid-round — see
   `hpmpc_backend.md`), or (b) the spawned executable itself failed/timed
   out — the exception message includes the executable's stdout/stderr.

## `verify_party_agreement` mismatches

`party_orchestrator_client.run_round`'s `verify_party_agreement=True`
(default) cross-checks that every party revealed the identical plaintext
before returning it, raising `RuntimeError: party disagreement on round ...`
if not. This is a **correctness/liveness sanity check for catching bugs**,
not a security mechanism (see `threat_model.md`) — if you see this in
practice, treat it as a bug report, not routine noise:

- A **fixed-point rounding mismatch** shouldn't trigger this — the check
  uses `torch.allclose(..., atol=1e-4)`, well above the
  `2**-frac_bits`-scale rounding this scheme produces per Phase 1/2's
  documented error bounds.
- Most likely causes: a **config drift** between parties (different
  `fixed_point.bitlength`/`frac_bits`, or one party still running an old
  hpmpc build after `fedavg_secure_aggregation.hpp` changed) — check
  `executables/fedavg_secure_aggregation.build_metadata.json` matches
  across all 3 party containers, or a **genuine bug** in a backend
  implementation.
- If it's operationally too strict for your setup (e.g. you've deliberately
  deployed slightly different `frac_bits` per party for some reason — not
  recommended), it can be disabled via
  `aggregator_args.verify_party_agreement: False`, but this removes a real
  bug-detection safety net with no compensating benefit — prefer fixing the
  underlying config drift instead.

## Rebuilding the hpmpc executables after a config change

If you change `fixed_point.bitlength`/`frac_bits` in the party configs, the
compiled executables must be rebuilt to match (or
`HpmpcBackend._check_config_consistency()` will refuse to start):

```bash
cd hpmpc
scripts/build_fedavg_secure_aggregation.sh <bitlength> <frac_bits>
```

Then rebuild the party image(s):
```bash
docker compose up -d --build secure_agg_party0 secure_agg_party1 secure_agg_party2
```
