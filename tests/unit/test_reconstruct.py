import numpy as np
import pytest

from server.secure_agg import reconstruct
from server.secure_agg.backends.backend_hpmpc import _to_hpmpc_tetrad, _to_hpmpc_trio, _to_hpmpc_xa
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme
from server.secure_agg.sharing_schemes.tetrad4pc import Tetrad4PCScheme

pytestmark = pytest.mark.unit

RING_MASK = np.uint64((1 << 64) - 1)


def test_reconstruct_xa_matches_plaintext():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(1)
    secret = codec.encode(np.array([-4.25, 2.0]))
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    shares_by_party = {}
    for j in range(3):
        c_j, c_j1 = shares[j].payload
        x, a = _to_hpmpc_xa(np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), RING_MASK)
        shares_by_party[j] = (x, a)

    result = reconstruct.reconstruct(2, shares_by_party, RING_MASK)
    decoded = codec.decode(result.astype(np.int64))
    assert decoded.tolist() == pytest.approx([-4.25, 2.0], abs=1e-3)


def test_reconstruct_trio_matches_plaintext():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(2)
    secret = codec.encode(np.array([7.5, -1.0]))
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    shares_by_party = {}
    for j in range(3):
        c_j, c_j1 = shares[j].payload
        p1, p2 = _to_hpmpc_trio(np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), j, RING_MASK)
        shares_by_party[j] = (p1, p2)

    result = reconstruct.reconstruct(5, shares_by_party, RING_MASK)
    decoded = codec.decode(result.astype(np.int64))
    assert decoded.tolist() == pytest.approx([7.5, -1.0], abs=1e-3)


def test_reconstruct_tetrad_matches_plaintext():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(3)
    secret = codec.encode(np.array([9.0, -3.5]))
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    shares_by_party = {j: _to_hpmpc_tetrad(shares[j].payload, RING_MASK) for j in range(4)}

    result = reconstruct.reconstruct(8, shares_by_party, RING_MASK)
    decoded = codec.decode(result.astype(np.int64))
    assert decoded.tolist() == pytest.approx([9.0, -3.5], abs=1e-3)


def test_reconstruct_rejects_unsupported_protocol():
    with pytest.raises(ValueError, match="does not support protocol"):
        reconstruct.reconstruct(99, {}, RING_MASK)


def test_reconstruct_rejects_wrong_party_count():
    with pytest.raises(ValueError, match="needs exactly 3 parties"):
        reconstruct.reconstruct(5, {0: (np.array([1]), np.array([2]))}, RING_MASK)


def test_reconstruct_cross_check_catches_inconsistent_shares():
    # Corrupt one party's share so the two independent reveal formulas
    # disagree -- this is the exact failure mode the cross-check exists to
    # catch, now that parties can't verify agreement on plaintext directly.
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(4)
    secret = codec.encode(np.array([1.0]))
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    shares_by_party = {}
    for j in range(3):
        c_j, c_j1 = shares[j].payload
        p1, p2 = _to_hpmpc_trio(np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), j, RING_MASK)
        shares_by_party[j] = (p1, p2)

    # Corrupt party 1's p1 field only -- reconstruct_trio (uses P2.p1/P0.p2)
    # is unaffected, but reconstruct_trio_alt (uses P1.p1/P0.p1) changes.
    corrupted_p1 = shares_by_party[1][0] + np.uint64(1)
    shares_by_party[1] = (corrupted_p1, shares_by_party[1][1])

    with pytest.raises(RuntimeError, match="cross-check failed"):
        reconstruct.reconstruct(5, shares_by_party, RING_MASK)


def test_reconstruct_cross_check_can_be_disabled():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(5)
    secret = codec.encode(np.array([1.0]))
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    shares_by_party = {}
    for j in range(3):
        c_j, c_j1 = shares[j].payload
        p1, p2 = _to_hpmpc_trio(np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), j, RING_MASK)
        shares_by_party[j] = (p1, p2)
    shares_by_party[1] = (shares_by_party[1][0] + np.uint64(1), shares_by_party[1][1])

    # Should not raise -- cross_check=False skips the alt-formula check.
    reconstruct.reconstruct(5, shares_by_party, RING_MASK, cross_check=False)
