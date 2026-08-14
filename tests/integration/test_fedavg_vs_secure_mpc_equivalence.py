"""Config-toggle regression test: aggregator: fedavg vs aggregator:
secure_mpc must produce statistically equivalent results for the same
underlying client updates -- proving the secure_mpc rollout toggle
(session_config.aggregator + client secure_aggregation.enabled) introduces
no functional regression on its own. See
docs/secure_aggregation/rollout_guide.md.

This drives BOTH real aggregator plugins (server/aggregation/aggregator_fedavg.py
and aggregator_secure_mpc.py) exactly as server_session_manager.py would,
using the real 3-party networked SimulatorBackend topology for the secure_mpc
side (same infrastructure as test_full_secure_agg_round.py) -- not a
simplified stand-in.
"""

import asyncio
import socket
from concurrent import futures

import grpc
import pytest
import torch

import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc
from client import client_secure_agg_manager
from server.aggregation import aggregator_fedavg, aggregator_secure_mpc
from server.secure_agg.backends.backend_simulator import SimulatorBackend
from server.secure_agg.backends.base import PartyEndpoint
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.party_server import SecureAggPartyServicer
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme
from server.server_state_manager import StateManager

pytestmark = pytest.mark.integration

BITLENGTH = 64
FRAC_BITS = 13


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _PartyProcessStandIn:
    def __init__(self, party_index, num_parties):
        self.party_index = party_index
        self.bind_port = _free_port()
        self.backend_port = _free_port()
        self.backend = SimulatorBackend(
            party_index=party_index,
            num_parties=num_parties,
            scheme=Replicated3PCScheme(bitlength=BITLENGTH),
            codec=FixedPointCodec(bitlength=BITLENGTH, frac_bits=FRAC_BITS),
            bind_host="localhost",
            bind_port=self.backend_port,
        )
        self.share_state = StateManager(
            loc="inmemory", name=f"equiv_party{party_index}_shares", host=None, port=None
        )
        self.servicer = SecureAggPartyServicer(
            party_index=party_index,
            sharing_scheme_name="replicated3pc",
            share_state=self.share_state,
            backend=self.backend,
        )
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        secure_agg_pb2_grpc.add_SecureAggPartyServiceServicer_to_server(
            self.servicer, self.server
        )
        self.server.add_insecure_port(f"localhost:{self.bind_port}")

    async def start(self, peer_endpoints):
        await self.backend.start(peer_endpoints)
        self.server.start()

    def stop(self):
        self.server.stop(grace=None)
        asyncio.run(self.backend.stop())


@pytest.fixture
def three_party_cluster():
    parties = [_PartyProcessStandIn(party_index=i, num_parties=3) for i in range(3)]
    for party in parties:
        peers = [
            PartyEndpoint(party_index=other.party_index, host="localhost", port=other.backend_port)
            for other in parties
            if other.party_index != party.party_index
        ]
        asyncio.run(party.start(peers))

    yield parties

    for party in parties:
        party.stop()


def _run_fedavg(client_weights, dataset_sizes):
    aggregator_state = StateManager(loc="inmemory", name="cmp_fedavg_agg", host=None, port=None)
    client_selection_state = StateManager(loc="inmemory", name="cmp_fedavg_cs", host=None, port=None)
    training_state = StateManager(loc="inmemory", name="cmp_fedavg_ts", host=None, port=None)
    training_session = StateManager(loc="inmemory", name="cmp_fedavg_tsess", host=None, port=None)
    client_info = StateManager(loc="inmemory", name="cmp_fedavg_ci", host=None, port=None)

    client_selection_state.put("selected_clients", list(client_weights.keys()))
    for client_id in client_weights:
        client_info.put(f"{client_id}.is_active", True)
        training_state.put(
            f"{client_id}.current_dataset_detail",
            {"metadata": {"num_items": dataset_sizes[client_id]}},
        )

    result = None
    for client_id, state_dict in client_weights.items():
        result = aggregator_fedavg.aggregate(
            session_id="cmp-session",
            client_id=client_id,
            client_active=True,
            client_local_weights=state_dict,
            client_info=client_info,
            training_state=training_state,
            training_session=training_session,
            aggregator_state=aggregator_state,
            client_selection_state=client_selection_state,
            args=None,
        )
    return result


def _run_secure_mpc(client_weights, dataset_sizes, party_endpoints):
    session_id = "cmp-session-secure"
    round_id = f"{session_id}:0"
    for client_id, state_dict in client_weights.items():
        client_secure_agg_manager.share_and_submit(
            client_id=client_id,
            session_id=session_id,
            round_id=round_id,
            state_dict=state_dict,
            dataset_size=dataset_sizes[client_id],
            sharing_scheme_name="replicated3pc",
            fixed_point_config={"bitlength": BITLENGTH, "frac_bits": FRAC_BITS},
            party_endpoints=party_endpoints,
            submission_timeout_s=5,
        )

    aggregator_state = StateManager(loc="inmemory", name="cmp_secure_agg", host=None, port=None)
    client_selection_state = StateManager(loc="inmemory", name="cmp_secure_cs", host=None, port=None)
    training_state = StateManager(loc="inmemory", name="cmp_secure_ts", host=None, port=None)
    training_session = StateManager(loc="inmemory", name="cmp_secure_tsess", host=None, port=None)
    client_info = StateManager(loc="inmemory", name="cmp_secure_ci", host=None, port=None)

    client_selection_state.put("selected_clients", list(client_weights.keys()))
    training_session.put(f"{session_id}.last_round_number", 0)
    for client_id in client_weights:
        client_info.put(f"{client_id}.is_active", True)
    # NOTE: unlike _run_fedavg above, aggregator_secure_mpc.py never reads
    # current_dataset_detail -- each client secret-shared its own dataset
    # size above, and the round's total is revealed by the real MPC round.

    result = None
    for client_id in client_weights:
        result = aggregator_secure_mpc.aggregate(
            session_id=session_id,
            client_id=client_id,
            client_active=True,
            client_local_weights=None,
            client_info=client_info,
            training_state=training_state,
            training_session=training_session,
            aggregator_state=aggregator_state,
            client_selection_state=client_selection_state,
            args={"party_endpoints": party_endpoints, "round_timeout_s": 10},
        )
    return result


def test_fedavg_and_secure_mpc_produce_equivalent_results_for_the_same_updates(
    three_party_cluster, make_client_weights
):
    parties = three_party_cluster
    party_endpoints = [
        {"party_index": p.party_index, "host": "localhost", "port": p.bind_port} for p in parties
    ]

    client_weights = {
        "clientA": make_client_weights(10),
        "clientB": make_client_weights(20),
        "clientC": make_client_weights(30),
    }
    dataset_sizes = {"clientA": 120, "clientB": 45, "clientC": 300}

    plaintext_result = _run_fedavg(client_weights, dataset_sizes)
    secure_result = _run_secure_mpc(client_weights, dataset_sizes, party_endpoints)

    assert plaintext_result is not None
    assert secure_result is not None
    assert set(plaintext_result.keys()) == set(secure_result.keys())

    tolerance = len(client_weights) * (2**-FRAC_BITS) / 2 + 1e-3
    for layer_name, plaintext_tensor in plaintext_result.items():
        secure_tensor = secure_result[layer_name]
        assert torch.max(torch.abs(plaintext_tensor - secure_tensor)) <= tolerance, (
            f"layer {layer_name} diverged beyond fixed-point tolerance: "
            f"fedavg={plaintext_tensor}, secure_mpc={secure_tensor}"
        )
