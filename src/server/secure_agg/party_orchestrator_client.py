"""flo_server-side helper used by aggregation/aggregator_secure_mpc.py to
fan a completed round's client weights out to all configured party servers
and collect the revealed plaintext aggregate.

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
import torch

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
    verify_party_agreement=True,
):
    """Trigger RunAggregationRound on every party in `party_endpoints` and
    return the plaintext RAW (not yet divided by total weight — see
    backends/base.py's run_aggregation_round docstring) aggregate as an
    OrderedDict[str, torch.Tensor], including the revealed
    DATASET_SIZE_LAYER_NAME entry (see aggregator_secure_mpc.py, which pops
    and uses it as the division total).

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

    `verify_party_agreement`: if True (default), cross-checks every party
    revealed the same plaintext before returning it — a liveness/
    correctness sanity check for catching bugs, not a security guarantee
    (see aggregator_args.verify_party_agreement in
    docs/secure_aggregation/design.md).
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
                results[endpoint["host"], endpoint["port"]] = future.result()
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

    reference = next(iter(results.values()))
    if verify_party_agreement:
        for other in list(results.values())[1:]:
            for layer_name, tensor in reference.items():
                if not torch.allclose(tensor, other[layer_name], atol=1e-4):
                    raise RuntimeError(
                        f"party disagreement on round {round_id}, layer {layer_name}"
                    )

    return reference
