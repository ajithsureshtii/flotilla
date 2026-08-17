from collections import OrderedDict

import numpy as np
import torch

from server.secure_agg import party_orchestrator_client, reconstruct
from server.secure_agg.constants import DATASET_SIZE_LAYER_NAME
from server.secure_agg.fixed_point_codec import FixedPointCodec
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
    round, THEN RECONSTRUCTS THE PLAINTEXT ITSELF from every party's raw
    share (see server/secure_agg/reconstruct.py) — the compute parties no
    longer reveal anything among themselves (see backend_hpmpc.py's module
    docstring and docs/secure_aggregation/threat_model.md for the
    resulting trust-model consequence: flo_server, not the party cluster,
    is now the one place plaintext is ever computed).

    `args` (aggregator_args in session config), for the hpmpc backend, must
    include `protocol` (2, 5, or 8 — must match every party's
    backend.hpmpc.protocol) and `fixed_point` (`{bitlength, frac_bits}` —
    must match every party's fixed_point config), in addition to the
    existing `party_endpoints`/`round_timeout_s`. `weighting_mode` (default
    `"client_side"`) must match every party's backend.hpmpc.weighting_mode
    and every client's secure_aggregation.weighting_mode; see
    docs/secure_aggregation/mult_fedavg.md.

    If `args` has no `protocol` key, this function assumes the configured
    backend still reveals internally and returns already-decoded plaintext
    layers directly (SimulatorBackend's behavior — a pure-Python reference/
    testing tool that predates, and is out of scope for, the reveal-removal
    redesign the hpmpc backend went through; see backend_simulator.py's
    module docstring) — in that case the first party's own returned value
    IS the final plaintext, taken as-is, with no reconstruct.py involved.
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

        # Each party returns its OWN raw share of the result (never
        # plaintext) plus tensor_specs (shape/dtype per layer, identical
        # across parties) -- see party_orchestrator_client.run_round's
        # docstring.
        shares_by_party, tensor_specs = party_orchestrator_client.run_round(
            session_id=session_id,
            round_id=round_id,
            client_ids=checked_in_clients,
            party_endpoints=args["party_endpoints"],
            timeout_s=args.get("round_timeout_s", 120),
        )

        if "protocol" in args:
            protocol = args["protocol"]
            bitlength = args["fixed_point"]["bitlength"]
            frac_bits = args["fixed_point"]["frac_bits"]
            weighting_mode = args.get("weighting_mode", "client_side")
            mask = np.uint64((1 << bitlength) - 1)
            codec = FixedPointCodec(bitlength=bitlength, frac_bits=frac_bits)

            # Reconstruct every layer (including DATASET_SIZE_LAYER_NAME)
            # from the shares just collected, via each protocol's own
            # reveal formula reimplemented in pure Python (reconstruct.py)
            # -- with a built-in dual-formula cross-check (the replacement
            # for the old party-side verify_party_agreement, since parties
            # can no longer agree on a plaintext they never see).
            raw_sum = OrderedDict()
            for layer_name, spec in tensor_specs.items():
                payload_by_party = {
                    party_index: layer_shares[layer_name].payload
                    for party_index, layer_shares in shares_by_party.items()
                }
                reconstructed_ring = reconstruct.reconstruct(protocol, payload_by_party, mask)
                # weighting_mode="mpc_product"'s real layers carry an
                # UNTRUNCATED product (2*frac_bits fractional precision --
                # see mult_fedavg_secure_aggregation.hpp's module
                # docstring); DATASET_SIZE_LAYER_NAME is always a plain
                # sum, decoded at the normal scale regardless of
                # weighting_mode.
                if weighting_mode == "mpc_product" and layer_name != DATASET_SIZE_LAYER_NAME:
                    decoded = reconstructed_ring.astype(np.int64).astype(np.float64) / float(
                        1 << (2 * frac_bits)
                    )
                else:
                    decoded = codec.decode(reconstructed_ring.astype(np.int64))
                layer_values = decoded.reshape(spec.shape)
                raw_sum[layer_name] = torch.from_numpy(layer_values.astype(np.dtype(spec.dtype)))
        else:
            # No "protocol" configured -- the backend already reveals
            # internally (SimulatorBackend) and every party's returned
            # value is already the same final plaintext; take any one of
            # them as-is. See this function's docstring.
            raw_sum = next(iter(shares_by_party.values()))

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
