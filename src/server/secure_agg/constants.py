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
