"""Fault-injection test: what happens to a round when a party goes away?

See docs/secure_aggregation/threat_model.md's "no fault tolerance" note --
this test exists to PROVE that claim concretely (a round fails cleanly and
promptly, it does not hang forever) rather than leave it as an assumption,
and to give aggregator_secure_mpc.py's error-handling path real exercise.
"""

import asyncio
import socket
import time
from concurrent import futures

import grpc
import pytest

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
            loc="inmemory", name=f"fault_party{party_index}_shares", host=None, port=None
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
        try:
            party.stop()
        except Exception:
            pass  # a party may already be stopped by the fault-injection test itself


def _submit_share(client_id, state_dict, dataset_size, session_id, round_id, party_endpoints):
    client_secure_agg_manager.share_and_submit(
        client_id=client_id,
        session_id=session_id,
        round_id=round_id,
        state_dict=state_dict,
        dataset_size=dataset_size,
        sharing_scheme_name="replicated3pc",
        fixed_point_config={"bitlength": BITLENGTH, "frac_bits": FRAC_BITS},
        party_endpoints=party_endpoints,
        submission_timeout_s=5,
    )


def test_round_fails_promptly_and_cleanly_when_a_party_is_killed_mid_round(
    three_party_cluster, make_client_weights
):
    parties = three_party_cluster
    party_endpoints = [
        {"party_index": p.party_index, "host": "localhost", "port": p.bind_port} for p in parties
    ]

    session_id = "fault-session-1"
    round_id = f"{session_id}:0"
    client_id = "client1"
    state_dict = make_client_weights(1)

    _submit_share(client_id, state_dict, 100, session_id, round_id, party_endpoints)

    # Kill party 2 AFTER shares were submitted (matching a party crashing
    # mid-round, not before it ever came up) but BEFORE the round is
    # triggered -- this is the realistic "one party goes away" scenario the
    # live MPC protocol has no fault tolerance for (see threat_model.md).
    parties[2].stop()

    aggregator_state = StateManager(loc="inmemory", name="fault_agg", host=None, port=None)
    client_selection_state = StateManager(loc="inmemory", name="fault_cs", host=None, port=None)
    training_state = StateManager(loc="inmemory", name="fault_ts", host=None, port=None)
    training_session = StateManager(loc="inmemory", name="fault_tsess", host=None, port=None)
    client_info = StateManager(loc="inmemory", name="fault_ci", host=None, port=None)

    client_selection_state.put("selected_clients", [client_id])
    training_session.put(f"{session_id}.last_round_number", 0)
    client_info.put(f"{client_id}.is_active", True)

    round_timeout_s = 5
    start = time.monotonic()
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
        args={"party_endpoints": party_endpoints, "round_timeout_s": round_timeout_s},
    )
    elapsed = time.monotonic() - start

    # The round must fail cleanly (None, matching aggregator_fedavg.py's own
    # "something went wrong" contract), not hang -- bounded well within the
    # configured round_timeout_s plus normal RPC overhead, not indefinitely.
    assert result is None
    assert elapsed < round_timeout_s + 5, (
        f"round took {elapsed:.1f}s to fail -- should fail well within round_timeout_s, not hang"
    )

    # State is cleared so a subsequent round attempt isn't poisoned by this
    # failed one (aggregator_secure_mpc.py's except-clause contract).
    assert list(aggregator_state.keys()) == []


def test_a_subsequent_round_can_still_be_attempted_after_a_party_recovers(
    three_party_cluster, make_client_weights
):
    """Confirms the failure in the test above doesn't leave any process-wide
    state poisoned -- a fresh round with all 3 parties healthy still works
    normally. (The "party recovers" here is simulated by simply not killing
    anyone in this test; the point is that aggregator_secure_mpc.py/
    party_server.py don't have any lingering per-round state that would
    break a LATER, different round_id.)
    """
    parties = three_party_cluster
    party_endpoints = [
        {"party_index": p.party_index, "host": "localhost", "port": p.bind_port} for p in parties
    ]

    session_id = "fault-session-2"
    round_id = f"{session_id}:0"
    client_id = "client1"
    state_dict = make_client_weights(2)

    _submit_share(client_id, state_dict, 50, session_id, round_id, party_endpoints)

    aggregator_state = StateManager(loc="inmemory", name="recover_agg", host=None, port=None)
    client_selection_state = StateManager(loc="inmemory", name="recover_cs", host=None, port=None)
    training_state = StateManager(loc="inmemory", name="recover_ts", host=None, port=None)
    training_session = StateManager(loc="inmemory", name="recover_tsess", host=None, port=None)
    client_info = StateManager(loc="inmemory", name="recover_ci", host=None, port=None)

    client_selection_state.put("selected_clients", [client_id])
    training_session.put(f"{session_id}.last_round_number", 0)
    client_info.put(f"{client_id}.is_active", True)

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

    assert result is not None
