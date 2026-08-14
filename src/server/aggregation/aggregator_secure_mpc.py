from server.secure_agg import party_orchestrator_client
from server.secure_agg.constants import DATASET_SIZE_LAYER_NAME
from utils.logger import FedLogger


def aggregate(
    session_id,
    client_id,
    client_active,
    client_local_weights,
    client_info,
    training_state,
    training_session,
    aggregator_state,
    client_selection_state,
    args,
):
    """Secure-aggregation counterpart to aggregator_fedavg.py, loaded the
    same way via server/load_aggregator.py (session_config.aggregator:
    secure_mpc). Same call signature and same None/OrderedDict return
    contract as aggregator_fedavg.aggregate — but this function never
    touches a client's plaintext weights, and (unlike aggregator_fedavg.py)
    never learns any client's plaintext dataset size either.
    `training_state` is accepted only for call-signature parity with every
    other aggregator plugin; it is not read here.

    Clients secret-share their update AND their raw dataset size and submit
    shares directly to the party servers (see client_secure_agg_manager.py);
    by the time this function is called for a given client, that client's
    shares should already be sitting in every party's share buffer. This
    function only tracks which clients have "checked in" for the round and,
    once all expected clients have, triggers the party cluster to run the
    round and reveal the aggregate — it returns whatever that reveals,
    exactly like aggregator_fedavg.aggregate returns its own plaintext
    computation.
    """
    logger = FedLogger(id=session_id, loggername="AGGREGATOR")
    logger.info(
        "fedserver.aggregator.secure_mpc.called",
        f"client_id-active,{client_id},{client_active}",
    )

    if client_local_weights is not None:
        # Should not happen once the Phase 2 client/server changes land
        # (secure-agg clients never send model_weights up through the
        # normal training RPC) — surfaced loudly rather than silently used,
        # since using it would defeat the entire point of this aggregator.
        logger.warn(
            "fedserver.aggregator.secure_mpc.unexpected_plaintext_weights",
            f"{client_id}: received non-None client_local_weights in secure_mpc "
            "mode; ignoring them. Shares must be submitted directly to party "
            "servers, never routed through flo_server.",
        )

    if client_active:
        aggregator_state.put(f"{client_id}.checked_in", True)

    checked_in_clients = list(aggregator_state.keys())

    active_clients = [c for c in client_info.keys() if client_info.get(f"{c}.is_active")]

    selected_clients = client_selection_state.get("selected_clients")

    if client_active == False:
        try:
            selected_clients.remove(client_id)
            client_selection_state.put("selected_clients", selected_clients)
        except Exception as e:
            logger.warn(
                "fedserver.aggregator.secure_mpc.remove_dropped_client",
                f"{client_id},{e}",
            )

    clients_to_wait_for = [c for c in selected_clients if c in active_clients]

    if len(checked_in_clients) == 0 or not all(
        c in checked_in_clients for c in clients_to_wait_for
    ):
        return None

    try:
        logger.info(
            "fedserver.aggregator.secure_mpc.round_ready",
            f"clients,{checked_in_clients}",
        )
        round_no = int(training_session.get(f"{session_id}.last_round_number"))
        round_id = f"{session_id}:{round_no}"

        # Each client pre-multiplied its update by its own (raw) dataset size
        # before sharing (see client_secure_agg_manager.py) precisely so the
        # party cluster never needs to scalar-multiply a share by a
        # fractional weight — it only ever sums shares and reveals. What
        # comes back here is therefore the RAW weighted sum
        # (sum(N_k * update_k)) for every real model layer, PLUS a revealed
        # DATASET_SIZE_LAYER_NAME entry — sum(N_k) across every checked-in
        # client, summed and revealed by the same generic mechanism, since
        # clients secret-share their raw dataset size too (see
        # client_secure_agg_manager.py). Neither flo_server nor any party
        # ever learns an individual client's dataset size — only this
        # round's total, which is exactly the denominator needed to turn the
        # raw weighted sum into the final average, done here in plaintext
        # after reveal, mirroring aggregator_fedavg.py's N_k weighting.
        raw_sum = party_orchestrator_client.run_round(
            session_id=session_id,
            round_id=round_id,
            client_ids=checked_in_clients,
            party_endpoints=args["party_endpoints"],
            timeout_s=args.get("round_timeout_s", 120),
            verify_party_agreement=args.get("verify_party_agreement", True),
        )

        total_dataset_size = round(float(raw_sum.pop(DATASET_SIZE_LAYER_NAME).item()))
        if total_dataset_size <= 0:
            raise RuntimeError(
                f"round {round_id} revealed a non-positive total dataset size "
                f"({total_dataset_size}); refusing to divide by it"
            )

        aggregated_model = type(raw_sum)(
            (layer, tensor / total_dataset_size) for layer, tensor in raw_sum.items()
        )

        aggregator_state.clear()
        logger.info("fedserver.aggregator.secure_mpc.round_complete", round_id)
        logger.info(
            "fedserver.aggregator.secure_mpc.revealed_total_dataset_size",
            f"{round_id},{total_dataset_size}",
        )
        return aggregated_model
    except Exception as e:
        aggregator_state.clear()
        logger.error("fedserver.aggregator.secure_mpc.exception", f"{e}")
        return None
