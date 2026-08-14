import pickle

import pytest

from server.secure_agg.backends.base import (
    PartyEndpoint,
    SecureAggregationBackend,
    TensorSpec,
)
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.base import PartyShare, SecretSharingScheme

pytestmark = pytest.mark.unit


def test_party_share_is_a_plain_picklable_dataclass():
    share = PartyShare(party_index=1, payload={"a": 1})
    assert pickle.loads(pickle.dumps(share)) == share


def test_secret_sharing_scheme_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        SecretSharingScheme()


def test_secret_sharing_scheme_concrete_subclass_satisfies_interface():
    class _Trivial1PartyScheme(SecretSharingScheme):
        scheme_id = "trivial1"
        num_parties = 1
        reconstruction_threshold = 1

        def share(self, plaintext_fixedpoint, rng):
            return [PartyShare(party_index=0, payload=plaintext_fixedpoint)]

        def reconstruct(self, shares):
            return shares[0].payload

        def add(self, a, b):
            return PartyShare(party_index=a.party_index, payload=a.payload + b.payload)

    scheme = _Trivial1PartyScheme()
    shares = scheme.share(5, rng=None)
    assert scheme.reconstruct({0: shares[0]}) == 5
    assert scheme.add(shares[0], shares[0]).payload == 10


def test_party_endpoint_and_tensor_spec_construct():
    endpoint = PartyEndpoint(party_index=0, host="localhost", port=50100)
    spec = TensorSpec(layer_name="fc1.weight", shape=(8, 4), dtype="float32")
    assert endpoint.party_index == 0
    assert spec.shape == (8, 4)


def test_secure_aggregation_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        SecureAggregationBackend()


@pytest.mark.asyncio
async def test_secure_aggregation_backend_concrete_subclass_satisfies_interface():
    class _NoopBackend(SecureAggregationBackend):
        backend_id = "noop"
        party_index = 0
        num_parties = 1

        async def start(self, peer_endpoints):
            self.started = True

        async def run_aggregation_round(
            self, round_id, shares, tensor_specs, timeout_s
        ):
            return {}

        async def stop(self):
            self.stopped = True

    backend = _NoopBackend()
    await backend.start(peer_endpoints=[])
    assert backend.started

    result = await backend.run_aggregation_round(
        round_id="session1:0",
        shares={},
        tensor_specs={},
        timeout_s=1.0,
    )
    assert result == {}

    await backend.stop()
    assert backend.stopped


def test_fixed_point_codec_validates_params():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    assert codec.bitlength == 64
    assert codec.frac_bits == 13

    with pytest.raises(ValueError):
        FixedPointCodec(bitlength=8, frac_bits=8)


# FixedPointCodec.encode()/decode() behavioral tests now live in
# tests/unit/test_fixed_point_codec.py (Phase 1).
