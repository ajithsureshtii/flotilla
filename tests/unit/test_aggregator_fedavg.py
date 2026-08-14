import pytest
import torch

from server.aggregation import aggregator_fedavg
from server.server_state_manager import StateManager

pytestmark = pytest.mark.unit


def _state():
    return StateManager(loc="inmemory", name="fedavg_test", host=None, port=None)


def test_aggregate_with_multiple_clients_works_under_the_inmemory_backend(make_client_weights):
    # Regression test: aggregator_fedavg.py used `aggregator_state.keys()`
    # directly and indexed it (`finished_clients[0]`) -- fine under the
    # redis backend (.keys() returns a list) but a dict_keys view under the
    # inmemory backend, which isn't subscriptable. Only surfaces with >=1
    # client actually reaching the aggregation branch under `inmemory`.
    aggregator_state = _state()
    client_selection_state = _state()
    training_state = _state()
    training_session = _state()
    client_info = _state()

    client_ids = ["clientA", "clientB", "clientC"]
    dataset_sizes = {"clientA": 100, "clientB": 50, "clientC": 25}
    client_selection_state.put("selected_clients", list(client_ids))
    for client_id in client_ids:
        client_info.put(f"{client_id}.is_active", True)
        training_state.put(
            f"{client_id}.current_dataset_detail",
            {"metadata": {"num_items": dataset_sizes[client_id]}},
        )

    result = None
    for client_id in client_ids:
        result = aggregator_fedavg.aggregate(
            session_id="s1",
            client_id=client_id,
            client_active=True,
            client_local_weights=make_client_weights(hash(client_id) % 1000),
            client_info=client_info,
            training_state=training_state,
            training_session=training_session,
            aggregator_state=aggregator_state,
            client_selection_state=client_selection_state,
            args=None,
        )

    assert result is not None
    assert isinstance(next(iter(result.values())), torch.Tensor)
