"""Shared constants for the secure-aggregation protocol -- kept in one place
so client-side sharing code and server-side aggregation code never drift on
a magic string.
"""

# Reserved pseudo-"layer" name used to secret-share each client's raw
# dataset size alongside its model-weight layers (see
# client_secure_agg_manager.py), so the MPC party cluster sums it -- for
# free, via the exact same mechanism it already uses to sum model weights --
# and reveals only the TOTAL across a round's checked-in clients (see
# aggregator_secure_mpc.py) to flo_server. No individual client's dataset
# size is ever visible to any party or to flo_server. A leading/trailing
# double-underscore mirrors Python's own "reserved name" convention; a real
# PyTorch state_dict layer is never named this.
DATASET_SIZE_LAYER_NAME = "__secure_agg_dataset_size__"

# gRPC's own built-in default (4MB) is far too small for a real model's
# secret shares -- every SubmitShare call carries an entire model's worth of
# TensorShares in one message (see client_secure_agg_manager.py), and
# FixedPointCodec's float->int64 encoding roughly doubles per-element size
# before sharing on top of that. Every gRPC server/channel in the secure_mpc
# path (flo_secure_agg_party.py, party_orchestrator_client.py,
# client_secure_agg_manager.py, backends/backend_simulator.py) should apply
# this to both grpc.max_send_message_length and grpc.max_receive_message_length,
# matching the plaintext path's own (configurable) default -- see
# server_config.yaml's comm_config.grpc.max_message_length. Not itself
# configurable per-deployment today; if a real deployment needs a different
# value, promote this to a config field the same way the plaintext path
# already does, rather than editing this constant in place.
GRPC_MESSAGE_SIZE_LIMIT_BYTES = 1000 * 1024 * 1024  # 1000MB
