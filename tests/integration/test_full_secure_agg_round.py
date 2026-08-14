import asyncio
import socket
from concurrent import futures

import grpc
import pytest
import torch

import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc
from client import client_secure_agg_manager
from server.aggregation import aggregator_secure_mpc
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
    """Spins up ONE real SecureAggPartyServicer (control-plane gRPC server)
    plus its SimulatorBackend's own peer gRPC server, on ephemeral localhost
    ports -- exactly what flo_secure_agg_party.py does for a real process,
    just in-thread for a fast test loop instead of a separate OS process."""

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
            loc="inmemory", name=f"party{party_index}_shares", host=None, port=None
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


def test_full_round_through_real_party_processes_matches_plaintext_fedavg(
    three_party_cluster, make_client_weights, fedavg_reference
):
    parties = three_party_cluster
    party_endpoints = [
        {"party_index": p.party_index, "host": "localhost", "port": p.bind_port} for p in parties
    ]

    session_id = "session-int-1"
    round_id = f"{session_id}:7"

    client_weights = {
        "clientA": make_client_weights(1),
        "clientB": make_client_weights(2),
        "clientC": make_client_weights(3),
    }
    dataset_sizes = {"clientA": 100, "clientB": 50, "clientC": 25}

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

    aggregator_state = StateManager(loc="inmemory", name="agg", host=None, port=None)
    client_selection_state = StateManager(loc="inmemory", name="cs", host=None, port=None)
    training_state = StateManager(loc="inmemory", name="ts", host=None, port=None)
    training_session = StateManager(loc="inmemory", name="tsess", host=None, port=None)
    client_info = StateManager(loc="inmemory", name="ci", host=None, port=None)

    client_selection_state.put("selected_clients", list(client_weights.keys()))
    training_session.put(f"{session_id}.last_round_number", 7)
    for client_id in client_weights:
        client_info.put(f"{client_id}.is_active", True)
        training_state.put(
            f"{client_id}.current_dataset_detail",
            {"metadata": {"num_items": dataset_sizes[client_id]}},
        )

    args = {"party_endpoints": party_endpoints, "round_timeout_s": 10}

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
            args=args,
        )

    assert result is not None
    reference = fedavg_reference(client_weights, dataset_sizes)

    tolerance = len(client_weights) * (2**-FRAC_BITS) / 2 + 1e-3
    for layer_name, expected in reference.items():
        assert torch.max(torch.abs(result[layer_name] - expected)) <= tolerance, layer_name


def test_round_fails_cleanly_when_a_client_never_submitted_a_share(three_party_cluster):
    parties = three_party_cluster
    party_endpoints = [
        {"party_index": p.party_index, "host": "localhost", "port": p.bind_port} for p in parties
    ]
    session_id = "session-int-2"

    aggregator_state = StateManager(loc="inmemory", name="agg2", host=None, port=None)
    client_selection_state = StateManager(loc="inmemory", name="cs2", host=None, port=None)
    training_state = StateManager(loc="inmemory", name="ts2", host=None, port=None)
    training_session = StateManager(loc="inmemory", name="tsess2", host=None, port=None)
    client_info = StateManager(loc="inmemory", name="ci2", host=None, port=None)

    client_selection_state.put("selected_clients", ["ghost-client"])
    training_session.put(f"{session_id}.last_round_number", 1)
    client_info.put("ghost-client.is_active", True)
    training_state.put(
        "ghost-client.current_dataset_detail", {"metadata": {"num_items": 10}}
    )

    result = aggregator_secure_mpc.aggregate(
        session_id=session_id,
        client_id="ghost-client",
        client_active=True,
        client_local_weights=None,
        client_info=client_info,
        training_state=training_state,
        training_session=training_session,
        aggregator_state=aggregator_state,
        client_selection_state=client_selection_state,
        args={"party_endpoints": party_endpoints, "round_timeout_s": 5},
    )

    # Party never received a share for "ghost-client" -> RunAggregationRound
    # fails on every party -> run_round raises -> aggregate() swallows it.
    assert result is None
