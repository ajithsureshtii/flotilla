import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from server.secure_agg.sharing_schemes.base import PartyShare
from server.secure_agg.sharing_schemes.tetrad4pc import Tetrad4PCScheme

pytestmark = pytest.mark.unit


def test_share_returns_one_share_per_party_in_order():
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(0)

    shares = scheme.share(np.array([42], dtype=np.int64), rng)

    assert [s.party_index for s in shares] == [0, 1, 2, 3]


def test_share_payload_shapes_match_each_roles_native_fields():
    # party 0/1/2 hold (mv, lambda_a, lambda_b); party 3 holds
    # (lambda1, lambda2, lambda3), no mv -- see hpmpc_backend.md.
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(1)

    shares = {s.party_index: s for s in scheme.share(np.array([1, 2], dtype=np.int64), rng)}

    for p in (0, 1, 2, 3):
        assert len(shares[p].payload) == 3

    # party 0's mv, party 1's mv, party 2's mv must all be identical --
    # they all hold the SAME masked value.
    mv0, _, _ = shares[0].payload
    mv1, _, _ = shares[1].payload
    mv2, _, _ = shares[2].payload
    assert mv0.tolist() == mv1.tolist() == mv2.tolist()


def test_reconstruct_recovers_the_secret_with_all_4_shares():
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(2)
    x = np.array([12345, -6789, 0], dtype=np.int64)

    shares = {s.party_index: s for s in scheme.share(x, rng)}
    reconstructed = scheme.reconstruct(shares)

    assert reconstructed.tolist() == x.tolist()


def test_reconstruct_rejects_fewer_than_4_shares():
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(3)
    shares = {s.party_index: s for s in scheme.share(np.array([1], dtype=np.int64), rng)}

    with pytest.raises(ValueError):
        scheme.reconstruct({0: shares[0], 1: shares[1], 2: shares[2]})


def test_a_single_party_share_looks_uniformly_random_not_the_secret():
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(4)
    x = np.array([999999], dtype=np.int64)

    shares = scheme.share(x, rng)
    for s in shares:
        for field in s.payload:
            assert field.tolist() != x.tolist()


def test_add_is_additively_homomorphic_per_party():
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(5)
    a = np.array([10, -20, 30], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)

    shares_a = {s.party_index: s for s in scheme.share(a, rng)}
    shares_b = {s.party_index: s for s in scheme.share(b, rng)}

    summed_shares = {p: scheme.add(shares_a[p], shares_b[p]) for p in range(4)}
    reconstructed_sum = scheme.reconstruct(summed_shares)

    assert reconstructed_sum.tolist() == (a + b).tolist()


def test_add_rejects_mismatched_party_indices():
    scheme = Tetrad4PCScheme(bitlength=64)
    a = PartyShare(party_index=0, payload=(np.uint64(1), np.uint64(2), np.uint64(3)))
    b = PartyShare(party_index=1, payload=(np.uint64(4), np.uint64(5), np.uint64(6)))

    with pytest.raises(ValueError):
        scheme.add(a, b)


def test_constructor_rejects_invalid_bitlength():
    with pytest.raises(ValueError):
        Tetrad4PCScheme(bitlength=0)
    with pytest.raises(ValueError):
        Tetrad4PCScheme(bitlength=65)


@given(
    st.lists(
        st.integers(min_value=-(2**31), max_value=2**31 - 1),
        min_size=1,
        max_size=16,
    )
)
@settings(max_examples=50)
def test_property_share_reconstruct_round_trip_for_random_integers(values):
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(123)
    x = np.array(values, dtype=np.int64)

    shares = {s.party_index: s for s in scheme.share(x, rng)}
    reconstructed = scheme.reconstruct(shares)

    assert reconstructed.tolist() == x.tolist()


@given(
    st.lists(
        st.integers(min_value=-(2**20), max_value=2**20 - 1),
        min_size=1,
        max_size=8,
    ),
    st.lists(
        st.integers(min_value=-(2**20), max_value=2**20 - 1),
        min_size=1,
        max_size=8,
    ),
)
@settings(max_examples=50)
def test_property_additive_homomorphism_holds_for_random_pairs(values_a, values_b):
    n = min(len(values_a), len(values_b))
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(456)
    a = np.array(values_a[:n], dtype=np.int64)
    b = np.array(values_b[:n], dtype=np.int64)

    shares_a = {s.party_index: s for s in scheme.share(a, rng)}
    shares_b = {s.party_index: s for s in scheme.share(b, rng)}
    summed = {p: scheme.add(shares_a[p], shares_b[p]) for p in range(4)}
    reconstructed = scheme.reconstruct(summed)

    assert reconstructed.tolist() == (a + b).tolist()
