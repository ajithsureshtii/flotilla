"""flo_server-side helper used by aggregation/aggregator_secure_mpc.py to
fan a completed round's client weights out to all configured party servers
and collect each party's own raw SHARE of the result (never plaintext --
see backend_hpmpc.py's module docstring). aggregator_secure_mpc.py
reconstructs the plaintext itself via server/secure_agg/reconstruct.py,
using every party's share collected here.

Deliberately a plain, SYNCHRONOUS function. aggregator_fedavg.py's (and
therefore aggregator_secure_mpc.py's) `aggregate()` is called synchronously
from server_session_manager.py's grpc_train_callback, which itself is
invoked without `await` from inside the async_grpc_train coroutine. Making
this async would require either running a nested event loop (unsafe/
deadlocks when already inside a running loop) or restructuring
grpc_train_callback into an async call chain — neither is warranted for 2-3
short RPCs. Concurrency across the party endpoints (there are usually only
2-4) comes from a small thread pool instead.
"""

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc

import proto.secure_agg_pb2 as secure_agg_pb2
import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc


def _call_one_party(endpoint, session_id, round_id, client_ids, timeout_s):
    channel = grpc.insecure_channel(f"{endpoint['host']}:{endpoint['port']}")
    try:
        stub = secure_agg_pb2_grpc.SecureAggPartyServiceStub(channel)
        response = stub.RunAggregationRound(
            secure_agg_pb2.RunAggregationRoundRequest(
                session_id=session_id,
                round_id=round_id,
                client_ids=client_ids,
                timeout_s=int(timeout_s),
            ),
            timeout=timeout_s,
        )
    finally:
        channel.close()

    if not response.success:
        raise RuntimeError(
            f"party {endpoint} failed round {round_id}: {response.error_message}"
        )
    return pickle.loads(response.aggregated_model)


def run_round(
    session_id,
    round_id,
    client_ids,
    party_endpoints,
    timeout_s,
):
    """Trigger RunAggregationRound on every party in `party_endpoints` and
    return (shares_by_party, tensor_specs):
      - shares_by_party: dict[party_index -> OrderedDict[layer_name ->
        PartyShare]] -- every party's own raw share of the (still-secret)
        result, keyed by `party_index` so aggregator_secure_mpc.py can pass
        it straight to server/secure_agg/reconstruct.py.
      - tensor_specs: dict[layer_name -> TensorSpec] (shape/dtype per
        layer), taken from an arbitrary one of the parties' responses --
        identical across all of them (same model), and needed by
        aggregator_secure_mpc.py to reshape/cast what it reconstructs,
        since a raw share carries no shape information of its own.

    Unlike an earlier version of this function, there is no
    `verify_party_agreement` parameter anymore: parties no longer reveal
    anything among themselves, so there is no shared plaintext for this
    function to compare across parties. The equivalent sanity check now
    happens at reconstruction time (reconstruct.py's dual-formula
    cross-check, run on the shares collected here), which needs no extra
    network round-trip since it reuses what this function already gathered.

    `client_ids`: which clients' buffered shares this round should include.
    Deliberately NOT accompanied by per-client weights — flo_server doesn't
    know any client's dataset size in secure_mpc mode (it's secret-shared,
    see client_secure_agg_manager.py), so nothing plaintext-weight-shaped is
    sent to the parties at all; round participation itself is still public
    (see docs/secure_aggregation/threat_model.md).

    Requires ALL parties to respond successfully — the live MPC round
    genuinely needs every party's participation (unlike offline
    reconstruction, which only needs a threshold of shares); see
    docs/secure_aggregation/threat_model.md's "no fault tolerance" note.
    """
    with ThreadPoolExecutor(max_workers=max(1, len(party_endpoints))) as pool:
        futures = {
            pool.submit(
                _call_one_party,
                endpoint,
                session_id,
                round_id,
                client_ids,
                timeout_s,
            ): endpoint
            for endpoint in party_endpoints
        }
        results = {}
        errors = []
        for future in as_completed(futures):
            endpoint = futures[future]
            try:
                results[endpoint["party_index"]] = future.result()
            except Exception as e:
                errors.append((endpoint, e))

    if errors:
        details = "; ".join(
            f"{endpoint['host']}:{endpoint['port']} -> {error}" for endpoint, error in errors
        )
        raise RuntimeError(
            f"secure_mpc round {round_id} failed: {len(errors)}/{len(party_endpoints)} "
            f"parties did not complete successfully ({details}). See "
            "docs/secure_aggregation/runbook.md's 'Debugging a hung round'."
        )

    shares_by_party = {party_index: result["shares"] for party_index, result in results.items()}
    tensor_specs = next(iter(results.values()))["tensor_specs"]
    return shares_by_party, tensor_specs
