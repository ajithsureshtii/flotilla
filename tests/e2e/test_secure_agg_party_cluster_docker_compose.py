"""End-to-end test for the containerized party cluster (docker-compose),
as distinct from tests/integration/test_full_secure_agg_round.py which
proves the same protocol logic in-process/in-thread on ephemeral ports.

This tier is deliberately NOT run on every commit (building the
pytorch-based images is slow) -- see docs/secure_aggregation/design.md's
"Running tests" section. Run manually with:

    pytest -m e2e tests/e2e/test_secure_agg_party_cluster_docker_compose.py

A full real-model training session through flo_server/flo_client with
aggregator: secure_mpc vs fedavg (statistically equivalent curves) is Phase
4's job, once the hpmpc backend exists -- flo_server/flo_client aren't part
of docker-compose at all today (see topology.md), so that test also needs
docker/sample_docker_server_run.sh / sample_docker_client_run.sh wiring that
doesn't exist yet. This test's scope is narrower and already achievable
now: prove the 3 real containers actually come up, are individually
reachable, and can run one real round end-to-end against real Docker
networking (not just in-process threads).
"""

import json
import subprocess
import time
from pathlib import Path

import grpc
import pytest

import proto.secure_agg_pb2 as secure_agg_pb2
import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc
from client import client_secure_agg_manager
from server.secure_agg.constants import DATASET_SIZE_LAYER_NAME
from server.secure_agg.party_orchestrator_client import run_round

pytestmark = pytest.mark.e2e

DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
PARTY_ENDPOINTS = [
    {"party_index": 0, "host": "localhost", "port": 50100},
    {"party_index": 1, "host": "localhost", "port": 50101},
    {"party_index": 2, "host": "localhost", "port": 50102},
]


def _compose(*args):
    return subprocess.run(
        ["docker", "compose", *args], cwd=DOCKER_DIR, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def party_cluster():
    up = _compose(
        "up", "-d", "--build", "secure_agg_party0", "secure_agg_party1", "secure_agg_party2"
    )
    assert up.returncode == 0, up.stderr

    deadline = time.monotonic() + 120
    last_error = None
    while time.monotonic() < deadline:
        try:
            for endpoint in PARTY_ENDPOINTS:
                channel = grpc.insecure_channel(f"{endpoint['host']}:{endpoint['port']}")
                stub = secure_agg_pb2_grpc.SecureAggPartyServiceStub(channel)
                response = stub.HealthCheck(secure_agg_pb2.HealthCheckRequest(), timeout=2)
                assert response.ready
                channel.close()
            break
        except Exception as e:
            last_error = e
            time.sleep(2)
    else:
        _compose("logs", "secure_agg_party0", "secure_agg_party1", "secure_agg_party2")
        pytest.fail(f"party cluster never became healthy: {last_error}")

    yield

    _compose("down", "-v")


def test_three_containers_come_up_and_report_healthy(party_cluster):
    for endpoint in PARTY_ENDPOINTS:
        channel = grpc.insecure_channel(f"{endpoint['host']}:{endpoint['port']}")
        stub = secure_agg_pb2_grpc.SecureAggPartyServiceStub(channel)
        response = stub.HealthCheck(secure_agg_pb2.HealthCheckRequest(), timeout=5)
        assert response.ready
        assert response.backend_id == "simulator"
        channel.close()


def test_one_real_round_through_the_containerized_cluster(party_cluster):
    import numpy as np
    import torch
    from collections import OrderedDict

    session_id = "e2e-docker-1"
    round_id = f"{session_id}:0"
    state_dict = OrderedDict({"w": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)})
    dataset_size = 10

    client_secure_agg_manager.share_and_submit(
        client_id="e2e-client",
        session_id=session_id,
        round_id=round_id,
        state_dict=state_dict,
        dataset_size=dataset_size,
        sharing_scheme_name="replicated3pc",
        fixed_point_config={"bitlength": 64, "frac_bits": 13},
        party_endpoints=PARTY_ENDPOINTS,
        submission_timeout_s=10,
    )

    # run_round now returns (shares_by_party, tensor_specs) -- the
    # simulator backend still reveals internally (unchanged, out of scope
    # for the hpmpc reveal-removal redesign -- see backend_simulator.py's
    # module docstring), so every party's own returned value is already
    # the same final plaintext; take any one of them, matching
    # aggregator_secure_mpc.py's own fallback for backends with no
    # "protocol" configured.
    shares_by_party, _tensor_specs = run_round(
        session_id=session_id,
        round_id=round_id,
        client_ids=["e2e-client"],
        party_endpoints=PARTY_ENDPOINTS,
        timeout_s=15,
    )
    raw_sum = next(iter(shares_by_party.values()))

    expected = state_dict["w"].numpy() * dataset_size
    assert np.allclose(raw_sum["w"].numpy(), expected, atol=1e-2)
    # Real, containerized proof that the dataset size itself was secret-
    # shared and revealed (never sent to the parties in the clear) by real
    # Docker containers, not just in-process test doubles.
    assert np.allclose(
        raw_sum[DATASET_SIZE_LAYER_NAME].numpy(), [dataset_size], atol=1e-2
    )
