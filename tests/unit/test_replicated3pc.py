import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from server.secure_agg.sharing_schemes.base import PartyShare
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme

pytestmark = pytest.mark.unit


def test_int64_to_uint64_astype_is_a_twos_complement_reinterpretation():
    # The whole scheme depends on this exact numpy behavior: casting a
    # negative int64 to uint64 must reinterpret the bit pattern (giving
    # 2**64 + x), not clip or raise. This test exists to catch a numpy
    # behavior change before it silently corrupts every other test here.
    arr = np.array([-1, -2, 0, 5], dtype=np.int64)
    result = arr.astype(np.uint64)
    assert result.tolist() == [2**64 - 1, 2**64 - 2, 0, 5]


def test_share_returns_one_share_per_party_in_order():
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(0)

    shares = scheme.share(np.array([42], dtype=np.int64), rng)

    assert [s.party_index for s in shares] == [0, 1, 2]


def test_reconstruct_with_any_two_of_three_parties_recovers_the_secret():
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(1)
    x = np.array([12345, -6789, 0], dtype=np.int64)

    shares = scheme.share(x, rng)
    by_party = {s.party_index: s for s in shares}

    for pair in [(0, 1), (1, 2), (0, 2)]:
        reconstructed = scheme.reconstruct({p: by_party[p] for p in pair})
        assert reconstructed.tolist() == x.tolist(), f"failed for party pair {pair}"


def test_reconstruct_with_all_three_parties_also_works():
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(2)
    x = np.array([7], dtype=np.int64)

    shares = scheme.share(x, rng)
    reconstructed = scheme.reconstruct({s.party_index: s for s in shares})

    assert reconstructed.tolist() == x.tolist()


def test_reconstruct_rejects_fewer_than_threshold_shares():
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(3)
    shares = scheme.share(np.array([1], dtype=np.int64), rng)

    with pytest.raises(ValueError):
        scheme.reconstruct({0: shares[0]})


def test_a_single_party_share_looks_uniformly_random_not_the_secret():
    # Not a rigorous statistical test — just a sanity check that share()
    # doesn't leak the plaintext directly in any component.
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(4)
    x = np.array([999999], dtype=np.int64)

    shares = scheme.share(x, rng)
    for s in shares:
        c_first, c_second = s.payload
        assert c_first.tolist() != x.tolist()
        assert c_second.tolist() != x.tolist()


def test_add_is_additively_homomorphic_per_party():
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(5)
    a = np.array([10, -20, 30], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)

    shares_a = {s.party_index: s for s in scheme.share(a, rng)}
    shares_b = {s.party_index: s for s in scheme.share(b, rng)}

    summed_shares = {
        p: scheme.add(shares_a[p], shares_b[p]) for p in range(3)
    }
    reconstructed_sum = scheme.reconstruct({0: summed_shares[0], 1: summed_shares[1]})

    assert reconstructed_sum.tolist() == (a + b).tolist()


def test_add_rejects_mismatched_party_indices():
    scheme = Replicated3PCScheme(bitlength=64)
    a = PartyShare(party_index=0, payload=(np.uint64(1), np.uint64(2)))
    b = PartyShare(party_index=1, payload=(np.uint64(3), np.uint64(4)))

    with pytest.raises(ValueError):
        scheme.add(a, b)


def test_constructor_rejects_invalid_bitlength():
    with pytest.raises(ValueError):
        Replicated3PCScheme(bitlength=0)
    with pytest.raises(ValueError):
        Replicated3PCScheme(bitlength=65)


@given(
    st.lists(
        st.integers(min_value=-(2**31), max_value=2**31 - 1),
        min_size=1,
        max_size=16,
    )
)
@settings(max_examples=50)
def test_property_share_reconstruct_round_trip_for_random_integers(values):
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(123)
    x = np.array(values, dtype=np.int64)

    shares = scheme.share(x, rng)
    by_party = {s.party_index: s for s in shares}
    reconstructed = scheme.reconstruct({0: by_party[0], 1: by_party[1]})

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
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(456)
    a = np.array(values_a[:n], dtype=np.int64)
    b = np.array(values_b[:n], dtype=np.int64)

    shares_a = {s.party_index: s for s in scheme.share(a, rng)}
    shares_b = {s.party_index: s for s in scheme.share(b, rng)}
    summed = {p: scheme.add(shares_a[p], shares_b[p]) for p in range(3)}
    reconstructed = scheme.reconstruct({0: summed[0], 1: summed[1]})

    assert reconstructed.tolist() == (a + b).tolist()
