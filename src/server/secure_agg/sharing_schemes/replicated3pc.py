import numpy as np

from server.secure_agg.sharing_schemes.base import PartyShare, SecretSharingScheme


class Replicated3PCScheme(SecretSharingScheme):
    """Textbook (2,3)-replicated additive secret sharing, semi-honest,
    honest-majority — the scheme hpmpc's PROTOCOL=2 expects.

    A secret x is split into three random "components" c0, c1, c2 with
    c0 + c1 + c2 == x (mod 2**bitlength). Party i is given the pair
    (c_i, c_{i+1 mod 3}) — i.e. party 0 holds (c0, c1), party 1 holds
    (c1, c2), party 2 holds (c2, c0). Any 2 of the 3 parties' pairs together
    cover all three components (with one duplicate) and can reconstruct x;
    no single party's pair reveals x. See docs/secure_aggregation/
    sharing_scheme_replicated3pc.md for the homomorphism argument this
    relies on.

    Arithmetic is over the ring Z_(2**bitlength) (wraparound, matching how
    fixed-point-encoded model weights — see FixedPointCodec — wrap on real
    hardware and in hpmpc). Internally, components are represented as
    numpy uint64 arrays masked to `bitlength` bits.
    """

    scheme_id = "replicated3pc"
    num_parties = 3
    reconstruction_threshold = 2

    def __init__(self, bitlength: int = 64):
        if not (0 < bitlength <= 64):
            raise ValueError("replicated3pc supports bitlength in (0, 64]")
        self.bitlength = bitlength
        self._mask = np.uint64((1 << bitlength) - 1)

    def _wrap(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.uint64) & self._mask

    def _random_ring_element(self, shape, rng: np.random.Generator) -> np.ndarray:
        # np.random.Generator.integers cannot directly draw a uniform value
        # across the full [0, 2**64) range, so build one from two
        # independent 32-bit draws instead.
        hi = rng.integers(0, 2**32, size=shape, dtype=np.uint64, endpoint=False)
        lo = rng.integers(0, 2**32, size=shape, dtype=np.uint64, endpoint=False)
        return self._wrap((hi << np.uint64(32)) | lo)

    def share(self, plaintext_fixedpoint, rng):
        x = self._wrap(np.asarray(plaintext_fixedpoint))
        c0 = self._random_ring_element(x.shape, rng)
        c1 = self._random_ring_element(x.shape, rng)
        c2 = self._wrap(x - c0 - c1)
        return [
            PartyShare(party_index=0, payload=(c0, c1)),
            PartyShare(party_index=1, payload=(c1, c2)),
            PartyShare(party_index=2, payload=(c2, c0)),
        ]

    def reconstruct(self, shares):
        if len(shares) < self.reconstruction_threshold:
            raise ValueError(
                f"replicated3pc needs shares from >= {self.reconstruction_threshold} "
                f"distinct parties to reconstruct, got {len(shares)}"
            )
        components = {}
        for party_index, share in shares.items():
            c_first, c_second = share.payload
            components[party_index % 3] = c_first
            components[(party_index + 1) % 3] = c_second
        total = np.uint64(0)
        for component in components.values():
            total = self._wrap(total + component)
        return total.astype(np.int64)

    def add(self, a, b):
        if a.party_index != b.party_index:
            raise ValueError("can only locally add two shares held by the same party")
        a0, a1 = a.payload
        b0, b1 = b.payload
        return PartyShare(
            party_index=a.party_index,
            payload=(self._wrap(a0 + b0), self._wrap(a1 + b1)),
        )


# Convention followed by every sharing-scheme module (see load_sharing_scheme.py):
# expose the concrete class as SCHEME_CLASS so the loader can instantiate it
# with backend-specific constructor args (here, just `bitlength`).
SCHEME_CLASS = Replicated3PCScheme
