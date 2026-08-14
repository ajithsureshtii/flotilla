# Rollout guide

How to turn secure aggregation on or off for a session, and how to compare
it against the existing plaintext `fedavg` path.

## The two toggles

Secure aggregation needs **both** of these set consistently — they're
independent config surfaces because they're read by different processes:

1. **Server side** — `session_config.aggregator: secure_mpc` in the
   training config submitted via `flo_session.py` (see
   `config/training_config.secure_mpc.example.yaml` for a filled-in
   example, and `config/training_config.yaml`'s inline comment for the
   `aggregator_args` shape). This selects `aggregator_secure_mpc.py`
   instead of `aggregator_fedavg.py` for that session.
2. **Client side** — `secure_aggregation.enabled: True` in each client's
   `client_config.yaml`, plus a matching `party_endpoints` list. This is
   loaded once at client process startup, not per-session.

**These must agree.** A client with `secure_aggregation.enabled: True`
always secret-shares its update and omits `model_weights` from its
response, regardless of which aggregator the session picked — pairing it
with `aggregator: fedavg` will break that session (`aggregator_fedavg.py`
expects real weights, not `None`). Conversely, a `secure_mpc` session with
`enabled: False` clients will never receive any shares.

Toggling either requires restarting that process (server picks up a new
`aggregator` per session submission automatically; a client's
`secure_aggregation.enabled` requires restarting the client container).

## Side-by-side comparison

`config/training_config.fedavg.example.yaml` and
`config/training_config.secure_mpc.example.yaml` are identical in every
respect (model, dataset, hyperparameters, number of rounds) except the
aggregator block — submit both against otherwise-identical client fleets
(toggling `secure_aggregation.enabled` between runs) to compare
accuracy/loss curves and confirm they track each other within the fixed-
point rounding tolerance documented in
[`sharing_scheme_replicated3pc.md`](sharing_scheme_replicated3pc.md).

`tests/integration/test_fedavg_vs_secure_mpc_equivalence.py` automates the
same comparison at the aggregator level (no dataset/model/docker needed) —
run it any time you want a fast confirmation that the toggle itself
introduces no regression:

```bash
pytest -m integration tests/integration/test_fedavg_vs_secure_mpc_equivalence.py
```

## Minimal steps to run the secure_mpc example session

1. Prepare data (one-time; downloads MNIST and partitions it — see
   `docker/prepare_mnist_data.py`'s docstring for the exact directory
   layout it produces):
   ```bash
   cd flotilla/docker
   python3 prepare_mnist_data.py
   ```
2. Bring up the full stack:
   ```bash
   docker compose up -d --build
   ```
   (`redis`, `mqtt5`, the 3 `secure_agg_party*` containers, `flo_server`,
   and `flo_client0/1/2` — the client/server services exist in
   `docker-compose.yaml` specifically to make this reproducible; see
   `topology.md`'s note on why they aren't normally part of compose.)
3. Wait for clients to advertise themselves to `flo_server` over MQTT
   (check `docker compose logs flo_server` for
   `fedserver.train.round.results` appearing on the first round, or
   `docker compose logs flo_client0` for `advert` activity), then submit:
   ```bash
   pip install requests pyyaml   # flo_session.py's only deps, run from the host
   python3 flo_session.py ../config/training_config.secure_mpc.example.yaml \
       --federated_server_endpoint localhost:12345
   ```
4. Watch progress via `docker compose logs -f flo_server`.

For the `fedavg` comparison run, edit each `flo_client*` service's
`client_config.yaml` (or bake a second image with
`secure_aggregation.enabled: False`) and submit
`training_config.fedavg.example.yaml` instead.

## Recommended rollout order for a new deployment

1. Run the aggregator-level equivalence test (above) — fastest signal, no
   infra needed.
2. Run the Phase 2/3 automated test suites
   (`pytest -m "unit or integration"`, and `pytest -m e2e` for the
   docker-compose party-cluster health check) to confirm the topology and
   backend are healthy in your environment.
3. Run a short real session (the steps above) with a small model/dataset
   and few rounds, comparing against the same session with `fedavg`.
4. Only then scale up round count / model size / client count for a
   production-shaped session.

Per-round overhead (secure_mpc vs fedavg) should be captured from your own
run's logs (`flo_server`'s `fedserver.train.round.client.finished` timing
lines) — this is expected to be dominated by the hpmpc subprocess spawn +
one reveal round-trip per aggregation, not by training itself; no fixed
number is asserted here since it depends on model size and deployment
network latency.
