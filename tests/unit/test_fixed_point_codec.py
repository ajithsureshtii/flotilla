import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from server.secure_agg.fixed_point_codec import FixedPointCodec

pytestmark = pytest.mark.unit


def test_encode_decode_round_trip_within_half_a_tick():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    values = np.array([0.0, 1.5, -1.5, 3.14159, -100.25])

    decoded = codec.decode(codec.encode(values))

    assert np.max(np.abs(decoded - values)) <= 2**-13 / 2 + 1e-12


@given(
    st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    )
)
def test_round_trip_error_never_exceeds_half_a_tick(values):
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    arr = np.array(values)

    decoded = codec.decode(codec.encode(arr))

    assert np.max(np.abs(decoded - arr)) <= 2**-13 / 2 + 1e-9


def test_encode_raises_overflow_error_instead_of_wrapping():
    codec = FixedPointCodec(bitlength=16, frac_bits=8)
    # max representable magnitude here is 2**7 == 128
    with pytest.raises(OverflowError):
        codec.encode(np.array([200.0]))


def test_encode_accepts_values_at_the_boundary():
    codec = FixedPointCodec(bitlength=16, frac_bits=8)
    max_value = (2**7) - 2**-8  # just under the boundary after rounding
    codec.encode(np.array([max_value]))  # should not raise


def test_decode_matches_hand_computed_value():
    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    fixedpoint = np.array([2**13], dtype=np.int64)  # encodes 1.0

    assert codec.decode(fixedpoint)[0] == pytest.approx(1.0)
