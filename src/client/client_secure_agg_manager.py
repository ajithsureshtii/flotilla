"""Client-side counterpart to server/secure_agg: given a trained
state_dict, secret-shares it (optionally pre-weighted by the client's own
dataset size), and submits one TensorShare set to each configured party
endpoint.

Two weighting_mode values, matching backend_hpmpc.py's HpmpcBackend
weighting_mode of the same name (must agree across every client and every
party in a deployment — see docs/secure_aggregation/mult_fedavg.md):

  "client_side" (default): pre-weights the update by the client's own
  (always locally known) dataset size before sharing, rather than by a
  fraction computed later by flo_server — see backends/base.py's
  run_aggregation_round docstring and docs/secure_aggregation/design.md for
  why: it lets every backend sum shares via plain addition (always free
  under additive/replicated sharing), with the final division-by-total
  happening in plaintext, after reveal, in aggregator_secure_mpc.py.

  "mpc_product": shares the RAW (unweighted) update instead — the party
  cluster itself computes weight_i * dataset_size_i via genuine secret x
  secret multiplication before summing across clients (see
  mult_fedavg_secure_aggregation.hpp). This exists to exercise an
  aggregation where the servers genuinely have to communicate to compute
  the product, not just to reveal a sum — see
  docs/secure_aggregation/mult_fedavg.md. Currently only supported when
  every party is configured with backend.hpmpc.protocol=5 (Trio).

The client's raw (unweighted) dataset size is ALWAYS secret-shared as well,
under the reserved DATASET_SIZE_LAYER_NAME pseudo-layer, exactly like a
model-weight layer, regardless of weighting_mode — this is what lets the
party cluster sum every checked-in client's dataset size (for free, via the
same generic mechanism) and reveal only the round's TOTAL to flo_server,
instead of flo_server or any party ever learning an individual client's
dataset size. See docs/secure_aggregation/threat_model.md.
"""

import pickle

import grpc
import numpy as np

import proto.secure_agg_pb2 as secure_agg_pb2
import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc
from server.secure_agg.constants import DATASET_SIZE_LAYER_NAME, GRPC_MESSAGE_SIZE_LIMIT_BYTES
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.load_sharing_scheme import load_sharing_scheme
from utils.logger import FedLogger

_GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MESSAGE_SIZE_LIMIT_BYTES),
    ("grpc.max_receive_message_length", GRPC_MESSAGE_SIZE_LIMIT_BYTES),
]


def share_and_submit(
    client_id,
    session_id,
    round_id,
    state_dict,
    dataset_size,
    sharing_scheme_name,
    fixed_point_config,
    party_endpoints,
    submission_timeout_s,
    logger=None,
    weighting_mode="client_side",
):
    """Secret-share every layer of `state_dict` (pre-weighted by
    `dataset_size` unless `weighting_mode="mpc_product"` — see module
    docstring) and submit one share set to each entry of `party_endpoints`
    (each a dict with `party_index`, `host`, `port`). Raises RuntimeError if
    any party rejects the submission.
    """
    logger = logger or FedLogger(id=client_id, loggername="CLIENT_SECURE_AGG_MANAGER")
    if weighting_mode not in ("client_side", "mpc_product"):
        raise ValueError(
            f"client_secure_agg_manager does not support weighting_mode={weighting_mode!r} "
            "(supported: 'client_side', 'mpc_product')"
        )

    scheme_module = load_sharing_scheme(client_id, sharing_scheme_name)
    scheme = scheme_module.SCHEME_CLASS(bitlength=fixed_point_config["bitlength"])
    codec = FixedPointCodec(
        bitlength=fixed_point_config["bitlength"], frac_bits=fixed_point_config["frac_bits"]
    )
    rng = np.random.default_rng()

    shares_by_party = {endpoint["party_index"]: [] for endpoint in party_endpoints}
    for layer_name, tensor in state_dict.items():
        original_dtype = str(tensor.dtype).rsplit(".", maxsplit=1)[-1]  # "torch.float32" -> "float32"
        plaintext = tensor.detach().numpy().astype(np.float64)
        if weighting_mode == "client_side":
            plaintext = plaintext * float(dataset_size)
        fixedpoint = codec.encode(plaintext)
        for party_share in scheme.share(fixedpoint, rng):
            shares_by_party[party_share.party_index].append(
                secure_agg_pb2.TensorShare(
                    layer_name=layer_name,
                    shape=list(tensor.shape),
                    dtype=original_dtype,
                    share_payload=pickle.dumps(party_share.payload),
                )
            )

    # Share the RAW (not pre-weighted by itself) dataset size too, under the
    # reserved pseudo-layer name -- the party cluster sums it the same way
    # it sums every real layer, so only the round's TOTAL is ever revealed,
    # never this client's individual dataset size.
    dataset_size_fixedpoint = codec.encode(np.array([float(dataset_size)], dtype=np.float64))
    for party_share in scheme.share(dataset_size_fixedpoint, rng):
        shares_by_party[party_share.party_index].append(
            secure_agg_pb2.TensorShare(
                layer_name=DATASET_SIZE_LAYER_NAME,
                shape=[1],
                dtype="float64",
                share_payload=pickle.dumps(party_share.payload),
            )
        )

    for endpoint in party_endpoints:
        channel = grpc.insecure_channel(f"{endpoint['host']}:{endpoint['port']}", options=_GRPC_CHANNEL_OPTIONS)
        try:
            stub = secure_agg_pb2_grpc.SecureAggPartyServiceStub(channel)
            ack = stub.SubmitShare(
                secure_agg_pb2.SubmitShareRequest(
                    session_id=session_id,
                    round_id=round_id,
                    client_id=client_id,
                    sharing_scheme=sharing_scheme_name,
                    shares=shares_by_party[endpoint["party_index"]],
                ),
                timeout=submission_timeout_s,
            )
        finally:
            channel.close()

        if not ack.accepted:
            logger.error(
                "fedclient.secure_agg.submit_share.rejected", f"{endpoint},{ack.message}"
            )
            raise RuntimeError(f"party {endpoint} rejected share submission: {ack.message}")

    logger.info("fedclient.secure_agg.submit_share.complete", round_id)
