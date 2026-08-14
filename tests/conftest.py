import os
import sys
from collections import OrderedDict

# Flotilla's own code imports as `server.foo` / `client.foo` / `utils.foo`
# relative to flotilla/src (see docker/Dockerfile.server's WORKDIR /src). Tests
# import the same modules the same way, so src must be on sys.path.
_SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1234)


@pytest.fixture
def toy_state_dict():
    """A tiny 2-layer state_dict, shaped like a real PyTorch model's, for
    testing sharing/aggregation math without needing an actual model. Only
    tests that use this fixture (or make_client_weights) need torch
    installed — the rest of the Phase 0 scaffolding does not."""
    import torch

    torch.manual_seed(0)
    return OrderedDict(
        {
            "fc1.weight": torch.randn(8, 4, dtype=torch.float32),
            "fc1.bias": torch.randn(8, dtype=torch.float32),
            "fc2.weight": torch.randn(2, 8, dtype=torch.float32),
            "fc2.bias": torch.randn(2, dtype=torch.float32),
        }
    )


def _make_client_weights(seed: int, base_state_dict: OrderedDict) -> OrderedDict:
    """Perturb a base state_dict deterministically to simulate a distinct
    client's locally-trained update."""
    import torch

    torch.manual_seed(seed)
    return OrderedDict(
        {
            name: tensor + torch.randn_like(tensor) * 0.01
            for name, tensor in base_state_dict.items()
        }
    )


def _plaintext_fedavg_reference(client_weights: dict, dataset_sizes: dict) -> OrderedDict:
    """Reference dataset-size-weighted average, mirroring
    server/aggregation/aggregator_fedavg.py's math, for tests to diff against
    without depending on Flotilla's server runtime."""
    total = sum(dataset_sizes.values())
    layer_names = next(iter(client_weights.values())).keys()
    result = OrderedDict()
    for layer in layer_names:
        acc = None
        for client_id, weights in client_weights.items():
            term = weights[layer] * (dataset_sizes[client_id] / total)
            acc = term if acc is None else acc + term
        result[layer] = acc
    return result


@pytest.fixture
def make_client_weights(toy_state_dict):
    """Factory fixture: make_client_weights(seed) -> OrderedDict, perturbed
    from the shared toy_state_dict so multiple "clients" in one test have
    distinct-but-comparable updates."""

    def _make(seed: int) -> OrderedDict:
        return _make_client_weights(seed, toy_state_dict)

    return _make


@pytest.fixture
def fedavg_reference():
    return _plaintext_fedavg_reference
