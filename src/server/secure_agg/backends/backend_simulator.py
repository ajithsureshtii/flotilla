"""Pure-Python reference backend for testing the aggregation math, control
flow, and (from Phase 2) the real 3-process network topology before any real
MPC library is wired in.

Two independent tools live in this module:

- `run_in_process()` (Phase 1): an in-process, no-sockets helper that plays
  ALL parties inside one Python process/one test — used purely to validate
  the sharing math against a plaintext FedAvg reference. It takes explicit
  per-client `weights` and does the weighting itself; see its own docstring
  for the plaintext-exposure caveat that makes it a testing tool only, never
  a real backend.
- `SimulatorBackend` (Phase 2): a real `SecureAggregationBackend` subclass,
  one instance per real party PROCESS, talking to its peers over a small
  internal gRPC service (`SecureAggPeerService`, secure_agg.proto). Unlike
  `run_in_process`, it never reconstructs an individual client's update —
  only the final sum — and takes NO `weights` parameter at all (see
  backends/base.py's run_aggregation_round docstring for why: clients
  pre-weight by their own dataset size before sharing, so summing shares is
  always free/local addition, no truncation-protocol needed).
"""

import asyncio
import pickle
import threading
import time
from collections import OrderedDict
from concurrent import futures

import grpc
import numpy as np
import torch

import proto.secure_agg_pb2 as secure_agg_pb2
import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc
from server.secure_agg.backends.base import SecureAggregationBackend
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.base import PartyShare, SecretSharingScheme


def run_in_process(
    shares: dict,
    weights: dict,
    codec: FixedPointCodec,
    scheme: SecretSharingScheme,
) -> "OrderedDict":
    """Simulate every party running inside this one process: reconstruct
    each client's update (a real backend never does this — see the
    plaintext-exposure caveat below), scale by the client's public weight,
    and sum, all in decoded float space.

    `shares`: client_id -> layer_name -> {party_index: PartyShare} — i.e.
    ALL parties' shares for every client/layer, since this helper plays
    every party at once. Contrast with SecureAggregationBackend.
    run_aggregation_round's per-party signature, where a real party only
    ever holds its OWN share of each client's update.

    `weights`: client_id -> public weight (e.g. dataset-size fraction).

    Returns layer_name -> decoded float ndarray (an OrderedDict, matching
    the plaintext state_dict shape aggregator_secure_mpc.py ultimately
    returns to server_session_manager.py).

    CAVEAT: this reconstructs each client's individual update before
    summing — exactly the plaintext exposure the whole exercise exists to
    eliminate. That is fine for a reference/testing tool that exists to
    validate the sharing math and the layers above the backend boundary,
    but a real backend (backend_hpmpc.py, Phase 3) must sum shares WITHOUT
    ever reconstructing an individual client's update, only revealing the
    final weighted aggregate.
    """
    if not shares:
        return OrderedDict()

    layer_names = next(iter(shares.values())).keys()
    result = OrderedDict()
    for layer_name in layer_names:
        accumulator = None
        for client_id, layers in shares.items():
            fixedpoint = scheme.reconstruct(layers[layer_name])
            plaintext = codec.decode(fixedpoint)
            weighted = plaintext * weights[client_id]
            accumulator = weighted if accumulator is None else accumulator + weighted
        result[layer_name] = accumulator
    return result


def _tensor_share_to_party_share(msg: "secure_agg_pb2.TensorShare", party_index: int) -> PartyShare:
    return PartyShare(party_index=party_index, payload=pickle.loads(msg.share_payload))


def _party_share_to_tensor_share(layer_name, spec, share: PartyShare) -> "secure_agg_pb2.TensorShare":
    return secure_agg_pb2.TensorShare(
        layer_name=layer_name,
        shape=list(spec.shape),
        dtype=spec.dtype,
        share_payload=pickle.dumps(share.payload),
    )


class _PeerServicer(secure_agg_pb2_grpc.SecureAggPeerServiceServicer):
    """Answers GetFinalShare for whichever rounds this party has already
    finished locally summing (see SimulatorBackend._local_sums)."""

    def __init__(self, backend: "SimulatorBackend"):
        self._backend = backend

    def GetFinalShare(self, request, context):
        with self._backend._lock:
            entry = self._backend._local_sums.get(request.round_id)
        if entry is None:
            return secure_agg_pb2.GetFinalShareResponse(ready=False)

        tensor_specs, final_share = entry
        shares_msg = [
            _party_share_to_tensor_share(
                layer_name, tensor_specs[layer_name], final_share[layer_name]
            )
            for layer_name in final_share
        ]
        return secure_agg_pb2.GetFinalShareResponse(ready=True, shares=shares_msg)


class SimulatorBackend(SecureAggregationBackend):
    """Pure Python/numpy SecureAggregationBackend, networked over a small
    internal gRPC peer service (SecureAggPeerService, secure_agg.proto).
    Exists to prove the entire stack above the backend boundary (proto,
    party_server.py, client_secure_agg_manager.py, aggregator_secure_mpc.py)
    works correctly over a real 3-process network topology BEFORE hpmpc
    (Phase 3) is wired in — see docs/secure_aggregation/design.md.

    Round algorithm (no truncation, no scalar-mul-of-shares needed — see
    backends/base.py's run_aggregation_round docstring for why):
      1. Locally sum every client's share for every layer (pure addition,
         free under additive/replicated sharing — no network needed).
      2. Ask ONE peer (the next party in ring order) for ITS locally-summed
         share of the same round, via GetFinalShare (polling until ready or
         timeout_s elapses — this is the only network round the whole
         aggregation needs, regardless of how many clients contributed).
      3. Reconstruct the plaintext sum from (my local sum, peer's local sum)
         using the configured SecretSharingScheme.

    Known limitation: `_local_sums` entries are never cleaned up (not just
    on failure -- always). Popping an entry as soon as THIS party finishes
    its own round is unsafe: a slower peer may still need to GetFinalShare
    it from us, and popping early would make that peer poll forever for an
    entry that's already gone (a real race hit during development). Since
    there's no reliable signal for "every peer that will ever ask has
    already asked," entries simply accumulate for the process's lifetime —
    acceptable for a reference backend meant to validate the topology, not
    to run indefinitely; see docs/secure_aggregation/threat_model.md's "no
    fault tolerance" note. A production-shaped backend would need a
    TTL-based sweep independent of any single round's completion.
    """

    def __init__(self, party_index, num_parties, scheme, codec, bind_host, bind_port):
        self.backend_id = "simulator"
        self.party_index = party_index
        self.num_parties = num_parties
        self._scheme = scheme
        self._codec = codec
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._peer_endpoints = []
        self._server = None
        self._lock = threading.Lock()
        self._local_sums = {}  # round_id -> (tensor_specs, {layer_name: PartyShare})

    async def start(self, peer_endpoints):
        self._peer_endpoints = list(peer_endpoints)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        secure_agg_pb2_grpc.add_SecureAggPeerServiceServicer_to_server(
            _PeerServicer(self), self._server
        )
        self._server.add_insecure_port(f"{self._bind_host}:{self._bind_port}")
        self._server.start()

    async def run_aggregation_round(self, round_id, shares, tensor_specs, timeout_s):
        local_sum = self._sum_local_shares(shares)
        with self._lock:
            self._local_sums[round_id] = (tensor_specs, local_sum)

        peer = self._pick_peer()
        peer_share = await self._fetch_peer_share(peer, round_id, timeout_s)

        result = OrderedDict()
        for layer_name, my_share in local_sum.items():
            reconstructed = self._scheme.reconstruct(
                {self.party_index: my_share, peer.party_index: peer_share[layer_name]}
            )
            plaintext = self._codec.decode(reconstructed)
            spec = tensor_specs[layer_name]
            result[layer_name] = torch.from_numpy(
                plaintext.reshape(spec.shape).astype(np.dtype(spec.dtype))
            )

        # Deliberately NOT popping self._local_sums[round_id] here: a peer
        # slower than us may still need to GetFinalShare it from us. Popping
        # eagerly (an earlier version of this method did) created a real
        # race -- whichever party finished fastest would delete its own
        # entry before a slower peer could fetch it, and that peer would
        # then poll forever for an entry that was already gone. Entries
        # accumulate for the process's lifetime instead; see the class
        # docstring's "Known limitation."
        return result

    async def stop(self):
        if self._server is not None:
            self._server.stop(grace=None)

    def _sum_local_shares(self, shares):
        summed = {}
        for layers in shares.values():
            for layer_name, share in layers.items():
                if layer_name not in summed:
                    summed[layer_name] = share
                else:
                    summed[layer_name] = self._scheme.add(summed[layer_name], share)
        return summed

    def _pick_peer(self):
        # Ring order: ask the next party index (mod num_parties). Any peer
        # works since reconstruction only ever needs 2 parties' shares.
        next_index = (self.party_index + 1) % self.num_parties
        for endpoint in self._peer_endpoints:
            if endpoint.party_index == next_index:
                return endpoint
        raise RuntimeError(f"no peer endpoint configured for party {next_index}")

    async def _fetch_peer_share(self, peer, round_id, timeout_s):
        channel = grpc.insecure_channel(f"{peer.host}:{peer.port}")
        stub = secure_agg_pb2_grpc.SecureAggPeerServiceStub(channel)
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline:
                response = stub.GetFinalShare(
                    secure_agg_pb2.GetFinalShareRequest(round_id=round_id),
                    timeout=max(1.0, deadline - time.monotonic()),
                )
                if response.ready:
                    return {
                        msg.layer_name: _tensor_share_to_party_share(msg, peer.party_index)
                        for msg in response.shares
                    }
                await asyncio.sleep(0.05)
        finally:
            channel.close()
        raise TimeoutError(
            f"peer party {peer.party_index} did not finish round {round_id} in time"
        )


BACKEND_CLASS = SimulatorBackend
