import asyncio
import pickle

import proto.secure_agg_pb2 as secure_agg_pb2
import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc
from server.secure_agg.backends.base import TensorSpec
from server.secure_agg.sharing_schemes.base import PartyShare
from utils.logger import FedLogger


class SecureAggPartyServicer(secure_agg_pb2_grpc.SecureAggPartyServiceServicer):
    """gRPC-facing side of one party process (run by flo_secure_agg_party.py).

    Buffers incoming client shares (keyed by round_id, then client_id) into
    a StateManager -- reusing the same pluggable state-backend abstraction
    server_state_manager.py already provides for the rest of Flotilla, so a
    party's share buffer can be backed by Redis exactly like session state
    is today, with zero new backend code. Drives the configured
    SecureAggregationBackend once flo_server calls RunAggregationRound.
    Knows nothing about how the backend actually talks to its peers -- see
    backends/base.py.
    """

    def __init__(self, party_index, sharing_scheme_name, share_state, backend, logger=None):
        self.party_index = party_index
        self.sharing_scheme_name = sharing_scheme_name
        self._share_state = share_state
        self._backend = backend
        self.logger = logger or FedLogger(
            id=f"party{party_index}", loggername="SECURE_AGG_PARTY_SERVER"
        )

    def SubmitShare(self, request, context):
        if request.sharing_scheme != self.sharing_scheme_name:
            msg = (
                f"sharing_scheme mismatch: this party is configured for "
                f"'{self.sharing_scheme_name}', client sent '{request.sharing_scheme}'"
            )
            self.logger.error("fedparty.submit_share.scheme_mismatch", msg)
            return secure_agg_pb2.SubmitShareAck(accepted=False, message=msg)

        layers = {}
        specs = {}
        for tensor_share in request.shares:
            layers[tensor_share.layer_name] = PartyShare(
                party_index=self.party_index,
                payload=pickle.loads(tensor_share.share_payload),
            )
            specs[tensor_share.layer_name] = TensorSpec(
                layer_name=tensor_share.layer_name,
                shape=tuple(tensor_share.shape),
                dtype=tensor_share.dtype,
            )

        self._share_state.put(f"{request.round_id}.shares.{request.client_id}", layers)
        # tensor_specs are identical across clients (same model), so it's
        # fine for the last submitter to "win" -- just needs to exist once
        # per round_id by the time RunAggregationRound reads it.
        self._share_state.put(f"{request.round_id}.specs", specs)

        self.logger.info(
            "fedparty.submit_share.accepted", f"{request.round_id},{request.client_id}"
        )
        return secure_agg_pb2.SubmitShareAck(accepted=True, message="ok")

    def RunAggregationRound(self, request, context):
        # Fully backend-agnostic: whatever self._backend.run_aggregation_round
        # returns gets pickled and sent back as-is. Since the reveal-removal
        # redesign (see backend_hpmpc.py's module docstring), that's this
        # party's own RAW SHARE of the result (an OrderedDict[layer_name ->
        # PartyShare]), never plaintext -- but this method doesn't need to
        # know or care, same as before the redesign. tensor_specs (shape/
        # dtype per layer) is bundled alongside it: flo_server needs this to
        # reshape/cast the plaintext it reconstructs, but no longer has any
        # other source for it now that this method returns a raw share
        # instead of an already-shaped tensor -- identical across every
        # party (same model), so any one party's copy suffices.
        buffered = self._share_state.get(f"{request.round_id}.shares") or {}
        missing = [c for c in request.client_ids if c not in buffered]
        if missing:
            msg = f"missing shares for clients {missing} in round {request.round_id}"
            self.logger.error("fedparty.run_round.missing_shares", msg)
            return secure_agg_pb2.RunAggregationRoundResponse(success=False, error_message=msg)

        shares = {c: buffered[c] for c in request.client_ids}
        tensor_specs = self._share_state.get(f"{request.round_id}.specs")

        try:
            aggregated = asyncio.run(
                self._backend.run_aggregation_round(
                    round_id=request.round_id,
                    shares=shares,
                    tensor_specs=tensor_specs,
                    timeout_s=request.timeout_s,
                )
            )
        except Exception as e:
            self.logger.error("fedparty.run_round.exception", f"{request.round_id},{e}")
            return secure_agg_pb2.RunAggregationRoundResponse(success=False, error_message=str(e))
        finally:
            self._share_state.deletebykey(f"{request.round_id}.shares")
            self._share_state.deletebykey(f"{request.round_id}.specs")

        self.logger.info("fedparty.run_round.complete", request.round_id)
        return secure_agg_pb2.RunAggregationRoundResponse(
            success=True,
            aggregated_model=pickle.dumps({"shares": aggregated, "tensor_specs": tensor_specs}),
        )

    def HealthCheck(self, request, context):
        return secure_agg_pb2.HealthCheckResponse(
            ready=True, backend_id=self._backend.backend_id
        )
