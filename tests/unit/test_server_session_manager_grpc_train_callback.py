from unittest.mock import MagicMock

import pytest

from server.server_session_manager import FloSessionManager

pytestmark = pytest.mark.unit


def _make_bare_session_manager(server_validation_interval=1, checkpoint_interval=0):
    """FloSessionManager's real __init__ needs a full session_config, model
    setup, etc. -- far more than grpc_train_callback itself touches. Building
    a bare instance with just the attributes that method reads/writes is the
    standard lightweight way to unit test one method in isolation."""
    manager = object.__new__(FloSessionManager)
    manager.id = "session-1"
    manager.logger = MagicMock()
    manager.training_state = MagicMock()
    manager.client_info = MagicMock()
    manager.training_session = MagicMock()

    def _training_session_get(key):
        if key.endswith(".last_round_number"):
            return 0
        if key.endswith(".global_validation_metrics"):
            return {}
        return MagicMock()

    manager.training_session.get.side_effect = _training_session_get
    manager.aggregator_state = MagicMock()
    manager.client_selection_state = MagicMock()
    manager.aggregator_args = {}
    manager.model_util = MagicMock()
    manager.model_util.validate_model.return_value = {"accuracy": 1.0}
    manager.server_validation_interval = server_validation_interval
    manager.checkpoint_interval = checkpoint_interval
    manager.round_start_time = 0.0
    return manager


def test_grpc_train_callback_does_not_raise_when_a_dropped_clients_checkin_completes_the_round():
    # Regression test: aggregate_start_time was only ever assigned in the
    # `if response:` branch, never the `elif response == None:` (client
    # dropped) branch -- if THAT call happened to be the one completing the
    # round (aggregated_model truthy) and the validation-interval condition
    # was hit, `aggregate_end_time = time() - aggregate_start_time` raised
    # UnboundLocalError. A dropped client's check-in can genuinely be the
    # one that completes a round (e.g. aggregator_fedavg.py removes it from
    # `selected_clients` and then finds every remaining selected client has
    # already reported in).
    manager = _make_bare_session_manager()
    manager.aggregate = MagicMock(return_value={"layer": "fake-aggregated-model"})

    # Must not raise UnboundLocalError (or anything else).
    manager.grpc_train_callback(client_id="c1", start_time=0.0, response=None)


def test_grpc_train_callback_logs_aggregate_time_on_dropped_client_completion():
    manager = _make_bare_session_manager()
    manager.aggregate = MagicMock(return_value={"layer": "fake-aggregated-model"})

    manager.grpc_train_callback(client_id="c1", start_time=0.0, response=None)

    logged_events = [call.args[0] for call in manager.logger.info.call_args_list]
    assert "fedserver.train_callback.aggregate_time" in logged_events
