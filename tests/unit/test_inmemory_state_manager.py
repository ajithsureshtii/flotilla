import pytest

from server.state_manager.inmemory import StateManager

pytestmark = pytest.mark.unit


def test_deletebykey_removes_a_single_segment_key():
    # Regression test: deletebykey() used to iterate over the *characters*
    # of the last path segment (`for k in keys[-1]`) instead of the parent
    # path segments (`for k in keys[:-1]`), matching put()'s own pattern.
    # That made it a silent no-op for every existing caller
    # (aggregator_fedat.py, aggregator_fedasync.py) whenever the inmemory
    # backend was in use.
    state = StateManager(name="test")
    state.put("client1", {"weights": "x"})

    state.deletebykey("client1")

    assert "client1" not in state.state


def test_deletebykey_removes_a_nested_dotted_key_without_touching_siblings():
    state = StateManager(name="test")
    state.put("round1.shares.clientA", "share-a")
    state.put("round1.shares.clientB", "share-b")

    state.deletebykey("round1.shares.clientA")

    assert state.get("round1.shares.clientA") is None
    assert state.get("round1.shares.clientB") == "share-b"


def test_deletebykey_on_missing_key_does_not_raise():
    state = StateManager(name="test")
    state.deletebykey("does.not.exist")  # should return quietly, not raise
