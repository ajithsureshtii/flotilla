"""Regression tests for a real, pre-existing bug in StartTraining's 3-way
training-bound dispatch (client_grpc_manager.py): InitTrainRequest.request
is a oneof of timeout_duration_s / max_mini_batch_count / max_epochs (see
grpc.proto), but the handler used to read `request.max_epochs` (didn't
exist on the message at all -- AttributeError) and
`request.max_mini_batches` (should have been max_mini_batch_count). Only
the timeout_duration_s path ever worked in practice, since every real
caller (server_session_manager.py) always sets it -- these tests cover the
other two, previously-broken paths directly.
"""

import pickle
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

import proto.grpc_pb2 as grpc_pb2
from client.client_grpc_manager import ClientGRPCManager

pytestmark = pytest.mark.unit


def _make_request(**oneof_kwargs):
    return grpc_pb2.InitTrainRequest(
        session_id="s1",
        model_id="m1",
        model_class="LeNet5",
        model_config=pickle.dumps({}),
        dataset_id="d1",
        model_wts=pickle.dumps(OrderedDict({"w": torch.tensor([1.0])})),
        batch_size=32,
        learning_rate=0.01,
        num_epochs=1,
        round_idx=0,
        loss_function=pickle.dumps(None),
        optimizer=pickle.dumps(None),
        **oneof_kwargs,
    )


def _make_manager():
    with patch("client.client_grpc_manager.Client") as MockClient:
        manager = ClientGRPCManager(
            client_id="c1",
            temp_dir_path="/tmp/whatever",
            torch_device="cpu",
            dataset_paths={"d1": "/tmp/whatever/d1"},
            client_info={},
        )
    mock_client = MockClient.return_value
    mock_client.Train.return_value = ({"loss": 0.1, "accuracy": 0.9}, OrderedDict())
    return manager, mock_client


def _run(request):
    manager, mock_client = _make_manager()
    context = MagicMock()
    context.is_active.return_value = True
    manager.StartTraining(request, context)
    return mock_client.Train.call_args.kwargs


def test_timeout_duration_s_dispatches_only_timeout():
    kwargs = _run(_make_request(timeout_duration_s=10.0))

    assert kwargs["timeout_duration_s"] == 10.0
    assert kwargs["max_epochs"] is None
    assert kwargs["max_mini_batches"] is None


def test_max_epochs_dispatches_only_max_epochs():
    # Previously crashed: request.max_epochs didn't exist on the message at
    # all, so any request that didn't set timeout_duration_s raised
    # AttributeError before even reaching Client.Train.
    kwargs = _run(_make_request(max_epochs=5))

    assert kwargs["max_epochs"] == 5
    assert kwargs["timeout_duration_s"] is None
    assert kwargs["max_mini_batches"] is None


def test_max_mini_batch_count_dispatches_only_max_mini_batches():
    # Previously broken: the handler read request.max_mini_batches (wrong
    # field name -- the proto field is max_mini_batch_count), which also
    # raised AttributeError.
    kwargs = _run(_make_request(max_mini_batch_count=42))

    assert kwargs["max_mini_batches"] == 42
    assert kwargs["timeout_duration_s"] is None
    assert kwargs["max_epochs"] is None


def test_no_bound_set_leaves_all_three_none():
    # No oneof member set at all (WhichOneof returns None) -- degrades to
    # unbounded training (num_epochs' own range() is the only bound left),
    # rather than crashing or guessing which field was "meant".
    kwargs = _run(_make_request())

    assert kwargs["timeout_duration_s"] is None
    assert kwargs["max_epochs"] is None
    assert kwargs["max_mini_batches"] is None
