import pickle
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

import proto.grpc_pb2 as grpc_pb2
from client.client_grpc_manager import ClientGRPCManager

pytestmark = pytest.mark.unit


def _make_request(model_wts):
    return grpc_pb2.InitTrainRequest(
        session_id="s1",
        model_id="m1",
        model_class="LeNet5",
        model_config=pickle.dumps({}),
        dataset_id="d1",
        model_wts=pickle.dumps(model_wts),
        batch_size=32,
        learning_rate=0.01,
        num_epochs=1,
        round_idx=0,
        loss_function=pickle.dumps(None),
        optimizer=pickle.dumps(None),
        timeout_duration_s=10.0,
    )


def _make_manager(local_training_disabled):
    with patch("client.client_grpc_manager.Client") as MockClient:
        manager = ClientGRPCManager(
            client_id="c1",
            temp_dir_path="/tmp/whatever",
            torch_device="cpu",
            dataset_paths={"d1": "/tmp/whatever/d1"},
            client_info={},
            local_training_disabled=local_training_disabled,
        )
    return manager, MockClient.return_value


def test_local_training_disabled_skips_client_train_and_returns_model_wts_unchanged():
    original_wts = OrderedDict({"w": torch.tensor([1.0, 2.0, 3.0])})
    manager, mock_client = _make_manager(local_training_disabled=True)

    request = _make_request(original_wts)
    context = MagicMock()
    context.is_active.return_value = True

    response = manager.StartTraining(request, context)

    mock_client.Train.assert_not_called()
    returned_wts = pickle.loads(response.model_weights)
    assert torch.equal(returned_wts["w"], original_wts["w"])


def test_local_training_disabled_false_still_calls_client_train():
    original_wts = OrderedDict({"w": torch.tensor([1.0, 2.0, 3.0])})
    trained_wts = OrderedDict({"w": torch.tensor([9.0, 9.0, 9.0])})
    manager, mock_client = _make_manager(local_training_disabled=False)
    mock_client.Train.return_value = ({"loss": 0.1, "accuracy": 0.9}, trained_wts)

    request = _make_request(original_wts)
    context = MagicMock()
    context.is_active.return_value = True

    response = manager.StartTraining(request, context)

    mock_client.Train.assert_called_once()
    returned_wts = pickle.loads(response.model_weights)
    assert torch.equal(returned_wts["w"], trained_wts["w"])
