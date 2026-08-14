import numpy as np


class FixedPointCodec:
    """Encodes/decodes between plaintext floats and the fixed-point integers
    that secret-sharing schemes operate on. Deliberately independent of any
    SecretSharingScheme — mirrors hpmpc's own separation of Additive_Share
    from FloatFixedConverter, so the numeric representation is a choice, not
    baked into the sharing math.

    `bitlength`/`frac_bits` must match whatever a given backend's numeric
    assumptions are; for the hpmpc backend, they must equal the compile-time
    FRACTIONAL/bit-width constants baked into the party executables (checked
    explicitly at party startup — see backends/backend_hpmpc.py, Phase 3).

    encode() raises OverflowError rather than silently clipping/wrapping a
    value that doesn't fit — a silently corrupted model weight is a much
    worse failure mode than a loud one during development.
    """

    def __init__(self, bitlength: int, frac_bits: int):
        if bitlength <= 0 or frac_bits < 0 or frac_bits >= bitlength:
            raise ValueError("bitlength must be > 0 and > frac_bits >= 0")
        self.bitlength = bitlength
        self.frac_bits = frac_bits
        self._scale = float(1 << frac_bits)
        self._min_value = -(1 << (bitlength - 1))
        self._max_value = (1 << (bitlength - 1)) - 1

    def encode(self, plaintext: np.ndarray) -> np.ndarray:
        scaled = np.round(np.asarray(plaintext, dtype=np.float64) * self._scale)
        if np.any(scaled < self._min_value) or np.any(scaled > self._max_value):
            raise OverflowError(
                f"value(s) do not fit in a signed {self.bitlength}-bit fixed-point "
                f"representation with {self.frac_bits} fractional bits"
            )
        return scaled.astype(np.int64)

    def decode(self, fixedpoint: np.ndarray) -> np.ndarray:
        return np.asarray(fixedpoint, dtype=np.int64).astype(np.float64) / self._scale
