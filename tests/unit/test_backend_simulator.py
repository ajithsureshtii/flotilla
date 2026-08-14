import numpy as np
import pytest

from server.secure_agg.backends.backend_simulator import run_in_process
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme

pytestmark = pytest.mark.unit


def _share_all_clients(client_state_dicts, codec, scheme, rng):
    """client_state_dicts: client_id -> OrderedDict[layer_name -> torch.Tensor].
    Returns client_id -> layer_name -> {party_index: PartyShare}."""
    shares = {}
    for client_id, state_dict in client_state_dicts.items():
        shares[client_id] = {}
        for layer_name, tensor in state_dict.items():
            fixedpoint = codec.encode(tensor.numpy())
            party_shares = scheme.share(fixedpoint, rng)
            shares[client_id][layer_name] = {s.party_index: s for s in party_shares}
    return shares


def test_run_in_process_matches_plaintext_fedavg(make_client_weights, fedavg_reference):
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(seed=42)

    client_weights = {
        "client-a": make_client_weights(1),
        "client-b": make_client_weights(2),
        "client-c": make_client_weights(3),
    }
    dataset_sizes = {"client-a": 100, "client-b": 50, "client-c": 25}
    total = sum(dataset_sizes.values())
    weights = {c: n / total for c, n in dataset_sizes.items()}

    shares = _share_all_clients(client_weights, codec, scheme, rng)
    result = run_in_process(shares, weights, codec, scheme)

    reference = fedavg_reference(client_weights, dataset_sizes)

    # Worst-case rounding error bound from docs/secure_aggregation/
    # sharing_scheme_replicated3pc.md: N clients * half a fixed-point tick.
    tolerance = len(client_weights) * (2**-13) / 2 + 1e-6
    for layer_name in reference:
        actual = result[layer_name]
        expected = reference[layer_name].numpy()
        assert np.max(np.abs(actual - expected)) <= tolerance, layer_name


def test_run_in_process_with_single_client_returns_that_clients_update(make_client_weights):
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(seed=7)

    client_weights = {"only-client": make_client_weights(1)}
    shares = _share_all_clients(client_weights, codec, scheme, rng)

    result = run_in_process(shares, {"only-client": 1.0}, codec, scheme)

    for layer_name, tensor in client_weights["only-client"].items():
        assert np.max(np.abs(result[layer_name] - tensor.numpy())) <= 2**-13 / 2 + 1e-6


def test_run_in_process_with_no_clients_returns_empty():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)

    assert run_in_process({}, {}, codec, scheme) == {}
