import pickle

import pytest

import proto.secure_agg_pb2 as secure_agg_pb2
from server.secure_agg.party_server import SecureAggPartyServicer
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme
from server.server_state_manager import StateManager

pytestmark = pytest.mark.unit


class _FakeBackend:
    """Minimal SecureAggregationBackend stand-in -- party_server.py's own
    logic (share buffering, round_id/client_id bookkeeping, buffer cleanup)
    is what's under test here, not any real MPC math."""

    backend_id = "fake"

    def __init__(self, result=None, raise_on_run=None):
        self._result = result if result is not None else {}
        self._raise_on_run = raise_on_run
        self.calls = []

    async def run_aggregation_round(self, round_id, shares, tensor_specs, timeout_s):
        self.calls.append({"round_id": round_id, "shares": shares, "tensor_specs": tensor_specs})
        if self._raise_on_run is not None:
            raise self._raise_on_run
        return self._result


def _make_servicer(backend=None, sharing_scheme_name="replicated3pc"):
    share_state = StateManager(loc="inmemory", name="test_party_shares", host=None, port=None)
    return SecureAggPartyServicer(
        party_index=0,
        sharing_scheme_name=sharing_scheme_name,
        share_state=share_state,
        backend=backend or _FakeBackend(),
    ), share_state


def _tensor_share(layer_name="w", payload=b"share-bytes"):
    return secure_agg_pb2.TensorShare(
        layer_name=layer_name, shape=[1], dtype="float32", share_payload=pickle.dumps(payload)
    )


def test_submit_share_rejects_scheme_mismatch():
    servicer, _ = _make_servicer(sharing_scheme_name="replicated3pc")

    ack = servicer.SubmitShare(
        secure_agg_pb2.SubmitShareRequest(
            session_id="s1",
            round_id="s1:0",
            client_id="c1",
            sharing_scheme="shamir_stub",
            shares=[_tensor_share()],
        ),
        context=None,
    )

    assert ack.accepted is False
    assert "mismatch" in ack.message


def test_submit_share_accepts_and_buffers():
    servicer, share_state = _make_servicer()

    ack = servicer.SubmitShare(
        secure_agg_pb2.SubmitShareRequest(
            session_id="s1",
            round_id="s1:0",
            client_id="c1",
            sharing_scheme="replicated3pc",
            shares=[_tensor_share()],
        ),
        context=None,
    )

    assert ack.accepted is True
    buffered = share_state.get("s1:0.shares")
    assert "c1" in buffered
    assert "w" in buffered["c1"]


def test_resubmitting_a_share_for_the_same_client_and_round_overwrites_not_rejects():
    # Deliberate: a client retrying after a transient failure should be able
    # to resubmit and have the retry win, not be rejected as a duplicate.
    servicer, share_state = _make_servicer()
    request_kwargs = dict(session_id="s1", round_id="s1:0", client_id="c1", sharing_scheme="replicated3pc")

    first = servicer.SubmitShare(
        secure_agg_pb2.SubmitShareRequest(shares=[_tensor_share(payload=b"first")], **request_kwargs),
        context=None,
    )
    second = servicer.SubmitShare(
        secure_agg_pb2.SubmitShareRequest(shares=[_tensor_share(payload=b"second")], **request_kwargs),
        context=None,
    )

    assert first.accepted and second.accepted
    buffered = share_state.get("s1:0.shares")
    # party_server.py already unpickles share_payload into PartyShare.payload
    # when buffering, so no second pickle.loads() here.
    assert buffered["c1"]["w"].payload == b"second"


def test_run_aggregation_round_fails_when_a_clients_share_never_arrived():
    servicer, _ = _make_servicer()
    servicer.SubmitShare(
        secure_agg_pb2.SubmitShareRequest(
            session_id="s1",
            round_id="s1:0",
            client_id="c1",
            sharing_scheme="replicated3pc",
            shares=[_tensor_share()],
        ),
        context=None,
    )

    response = servicer.RunAggregationRound(
        secure_agg_pb2.RunAggregationRoundRequest(
            session_id="s1", round_id="s1:0", client_ids=["c1", "c2"], timeout_s=5
        ),
        context=None,
    )

    assert response.success is False
    assert "c2" in response.error_message


def test_run_aggregation_round_succeeds_and_clears_the_buffer_afterward():
    backend = _FakeBackend(result={"w": "aggregated-value"})
    servicer, share_state = _make_servicer(backend=backend)
    servicer.SubmitShare(
        secure_agg_pb2.SubmitShareRequest(
            session_id="s1",
            round_id="s1:0",
            client_id="c1",
            sharing_scheme="replicated3pc",
            shares=[_tensor_share()],
        ),
        context=None,
    )

    response = servicer.RunAggregationRound(
        secure_agg_pb2.RunAggregationRoundRequest(
            session_id="s1", round_id="s1:0", client_ids=["c1"], timeout_s=5
        ),
        context=None,
    )

    assert response.success is True
    # Bundled with tensor_specs alongside the backend's raw-share result --
    # see party_server.py's RunAggregationRound docstring for why flo_server
    # needs this now (it no longer receives an already-shaped tensor).
    unpickled = pickle.loads(response.aggregated_model)
    assert unpickled["shares"] == {"w": "aggregated-value"}
    assert "w" in unpickled["tensor_specs"]
    assert share_state.get("s1:0.shares") is None
    assert share_state.get("s1:0.specs") is None
    assert backend.calls[0]["round_id"] == "s1:0"


def test_run_aggregation_round_surfaces_backend_exceptions_as_failure():
    backend = _FakeBackend(raise_on_run=RuntimeError("peer unreachable"))
    servicer, _ = _make_servicer(backend=backend)
    servicer.SubmitShare(
        secure_agg_pb2.SubmitShareRequest(
            session_id="s1",
            round_id="s1:0",
            client_id="c1",
            sharing_scheme="replicated3pc",
            shares=[_tensor_share()],
        ),
        context=None,
    )

    response = servicer.RunAggregationRound(
        secure_agg_pb2.RunAggregationRoundRequest(
            session_id="s1", round_id="s1:0", client_ids=["c1"], timeout_s=5
        ),
        context=None,
    )

    assert response.success is False
    assert "peer unreachable" in response.error_message


def test_healthcheck_reports_backend_id():
    servicer, _ = _make_servicer(backend=_FakeBackend())

    response = servicer.HealthCheck(secure_agg_pb2.HealthCheckRequest(), context=None)

    assert response.ready is True
    assert response.backend_id == "fake"


def test_replicated3pc_is_a_valid_sharing_scheme_for_wire_smoke_test():
    # Sanity: the scheme name used throughout these tests is real, not a typo.
    assert Replicated3PCScheme.scheme_id == "replicated3pc"
