import pytest

from server.aggregation import aggregator_secure_mpc
from server.secure_agg import party_orchestrator_client
from server.server_state_manager import StateManager

pytestmark = pytest.mark.unit


def _state():
    return StateManager(loc="inmemory", name="test", host=None, port=None)


def _make_state_managers():
    return (
        _state(),  # aggregator_state
        _state(),  # client_selection_state
        _state(),  # training_state
        _state(),  # training_session
        _state(),  # client_info
    )


def _setup_round(
    client_selection_state,
    training_state,
    training_session,
    client_info,
    client_ids,
    dataset_sizes,
    round_no=3,
    session_id="s1",
):
    client_selection_state.put("selected_clients", list(client_ids))
    training_session.put(f"{session_id}.last_round_number", round_no)
    for c in client_ids:
        client_info.put(f"{c}.is_active", True)
        training_state.put(
            f"{c}.current_dataset_detail", {"metadata": {"num_items": dataset_sizes[c]}}
        )


def test_returns_none_until_all_selected_clients_check_in(monkeypatch):
    calls = []
    monkeypatch.setattr(
        party_orchestrator_client, "run_round", lambda **kw: calls.append(kw) or {"sentinel": True}
    )

    aggregator_state, client_selection_state, training_state, training_session, client_info = (
        _make_state_managers()
    )
    _setup_round(
        client_selection_state,
        training_state,
        training_session,
        client_info,
        ["c1", "c2"],
        {"c1": 100, "c2": 50},
    )

    result = aggregator_secure_mpc.aggregate(
        session_id="s1",
        client_id="c1",
        client_active=True,
        client_local_weights=None,
        client_info=client_info,
        training_state=training_state,
        training_session=training_session,
        aggregator_state=aggregator_state,
        client_selection_state=client_selection_state,
        args={"party_endpoints": []},
    )

    assert result is None
    assert calls == []


def test_triggers_round_once_all_selected_clients_check_in(monkeypatch):
    calls = []
    monkeypatch.setattr(
        party_orchestrator_client,
        "run_round",
        # run_round returns the RAW (client-pre-weighted-by-N_k) sum, not the
        # final average — aggregate() divides by `total` (150) itself.
        lambda **kw: (calls.append(kw), {"layer": 150.0})[1],
    )

    aggregator_state, client_selection_state, training_state, training_session, client_info = (
        _make_state_managers()
    )
    _setup_round(
        client_selection_state,
        training_state,
        training_session,
        client_info,
        ["c1", "c2"],
        {"c1": 100, "c2": 50},
    )

    common_kwargs = dict(
        session_id="s1",
        client_local_weights=None,
        client_info=client_info,
        training_state=training_state,
        training_session=training_session,
        aggregator_state=aggregator_state,
        client_selection_state=client_selection_state,
        args={"party_endpoints": [{"host": "p0", "port": 1}], "round_timeout_s": 30},
    )

    first = aggregator_secure_mpc.aggregate(client_id="c1", client_active=True, **common_kwargs)
    assert first is None

    second = aggregator_secure_mpc.aggregate(client_id="c2", client_active=True, **common_kwargs)

    assert second == {"layer": 1.0}
    assert len(calls) == 1
    call = calls[0]
    assert call["session_id"] == "s1"
    assert call["round_id"] == "s1:3"
    assert call["client_weights"] == {"c1": 100 / 150, "c2": 50 / 150}
    assert call["party_endpoints"] == [{"host": "p0", "port": 1}]
    assert call["timeout_s"] == 30


def test_dropped_client_is_removed_from_wait_list(monkeypatch):
    calls = []
    monkeypatch.setattr(
        party_orchestrator_client,
        "run_round",
        # Only c1 (N=100) remains after c2 drops, so total == 100 and the raw
        # sum equals the final average directly.
        lambda **kw: (calls.append(kw), {"layer": 100.0})[1],
    )

    aggregator_state, client_selection_state, training_state, training_session, client_info = (
        _make_state_managers()
    )
    _setup_round(
        client_selection_state,
        training_state,
        training_session,
        client_info,
        ["c1", "c2"],
        {"c1": 100, "c2": 50},
    )
    client_info.put("c2.is_active", False)  # c2 has dropped

    common_kwargs = dict(
        session_id="s1",
        client_local_weights=None,
        client_info=client_info,
        training_state=training_state,
        training_session=training_session,
        aggregator_state=aggregator_state,
        client_selection_state=client_selection_state,
        args={"party_endpoints": []},
    )

    result = aggregator_secure_mpc.aggregate(client_id="c2", client_active=False, **common_kwargs)
    assert result is None  # still waiting on c1

    result = aggregator_secure_mpc.aggregate(client_id="c1", client_active=True, **common_kwargs)

    assert result == {"layer": 1.0}
    assert len(calls) == 1
    assert calls[0]["client_weights"] == {"c1": 1.0}


def test_ignores_unexpected_plaintext_weights_without_using_them(monkeypatch):
    monkeypatch.setattr(party_orchestrator_client, "run_round", lambda **kw: {"layer": 100.0})

    aggregator_state, client_selection_state, training_state, training_session, client_info = (
        _make_state_managers()
    )
    _setup_round(
        client_selection_state, training_state, training_session, client_info, ["c1"], {"c1": 100}
    )

    result = aggregator_secure_mpc.aggregate(
        session_id="s1",
        client_id="c1",
        client_active=True,
        client_local_weights={"should": "be ignored"},
        client_info=client_info,
        training_state=training_state,
        training_session=training_session,
        aggregator_state=aggregator_state,
        client_selection_state=client_selection_state,
        args={"party_endpoints": []},
    )

    assert result == {"layer": 1.0}


def test_run_round_exception_clears_state_and_returns_none(monkeypatch):
    def boom(**kw):
        raise RuntimeError("party unreachable")

    monkeypatch.setattr(party_orchestrator_client, "run_round", boom)

    aggregator_state, client_selection_state, training_state, training_session, client_info = (
        _make_state_managers()
    )
    _setup_round(
        client_selection_state, training_state, training_session, client_info, ["c1"], {"c1": 100}
    )

    result = aggregator_secure_mpc.aggregate(
        session_id="s1",
        client_id="c1",
        client_active=True,
        client_local_weights=None,
        client_info=client_info,
        training_state=training_state,
        training_session=training_session,
        aggregator_state=aggregator_state,
        client_selection_state=client_selection_state,
        args={"party_endpoints": []},
    )

    assert result is None
    assert list(aggregator_state.keys()) == []


def test_unmocked_stub_raises_not_implemented_and_is_handled_gracefully():
    # Exercises the REAL (Phase 1) party_orchestrator_client.run_round stub,
    # not a mock — confirms aggregate() swallows it the same way it would
    # swallow a real network failure, matching aggregator_fedavg.py's
    # existing try/except/clear/return-None convention.
    aggregator_state, client_selection_state, training_state, training_session, client_info = (
        _make_state_managers()
    )
    _setup_round(
        client_selection_state, training_state, training_session, client_info, ["c1"], {"c1": 100}
    )

    result = aggregator_secure_mpc.aggregate(
        session_id="s1",
        client_id="c1",
        client_active=True,
        client_local_weights=None,
        client_info=client_info,
        training_state=training_state,
        training_session=training_session,
        aggregator_state=aggregator_state,
        client_selection_state=client_selection_state,
        args={"party_endpoints": []},
    )

    assert result is None
