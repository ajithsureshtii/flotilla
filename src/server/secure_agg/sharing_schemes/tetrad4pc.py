import numpy as np

from server.secure_agg.sharing_schemes.base import PartyShare, SecretSharingScheme


class Tetrad4PCScheme(SecretSharingScheme):
    """Additive secret sharing matching hpmpc's Tetrad (PROTOCOL=8) native
    share structure directly — see docs/secure_aggregation/hpmpc_backend.md's
    Tetrad section for the full derivation.

    Unlike `Replicated3PCScheme` (a 3-party intermediate format that
    `backend_hpmpc.py` then converts, per-party-locally, into whatever a
    given hpmpc protocol's native layout needs), this scheme produces
    ALREADY-NATIVE-SHAPED payloads per party — Tetrad's masking structure
    genuinely needs 4 independent pieces of randomness, not 3, so there is
    no way to derive a 4th party's share from a 3-party replicated3pc
    share the way Replicated/Trio's per-party-local conversions do.

    A secret `x` is masked as `mv = x + lambda1 + lambda2 + lambda3`
    (mod 2**bitlength), with `lambda1`/`lambda2`/`lambda3` drawn uniformly
    at random. Party `i`'s payload:
        party 0: (mv, lambda1, lambda2)
        party 1: (mv, lambda1, lambda3)
        party 2: (mv, lambda2, lambda3)
        party 3: (lambda1, lambda2, lambda3)   -- no mv; party 3 never
                                                   sees the masked value
    Numerically verified (both the single-secret reveal across all 4 roles
    and cross-client elementwise-summed reveal) against Tetrad's real
    reveal invariants traced from `Tetrad-P_{0,1,2,3}_template.hpp` — see
    hpmpc_backend.md.
    """

    scheme_id = "tetrad4pc"
    num_parties = 4
    # reconstruct() below is a test/debug-only helper (see
    # SecretSharingScheme's docstring) and simply requires every party's
    # share for simplicity; the real live protocol's actual minimum
    # requirement (mv from any one of P0-2, plus all 3 lambdas) is a
    # live-network detail this pure-Python helper doesn't model.
    reconstruction_threshold = 4

    def __init__(self, bitlength: int = 64):
        if not (0 < bitlength <= 64):
            raise ValueError("tetrad4pc supports bitlength in (0, 64]")
        self.bitlength = bitlength
        self._mask = np.uint64((1 << bitlength) - 1)

    def _wrap(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.uint64) & self._mask

    def _random_ring_element(self, shape, rng: np.random.Generator) -> np.ndarray:
        hi = rng.integers(0, 2**32, size=shape, dtype=np.uint64, endpoint=False)
        lo = rng.integers(0, 2**32, size=shape, dtype=np.uint64, endpoint=False)
        return self._wrap((hi << np.uint64(32)) | lo)

    def share(self, plaintext_fixedpoint, rng):
        x = self._wrap(np.asarray(plaintext_fixedpoint))
        lambda1 = self._random_ring_element(x.shape, rng)
        lambda2 = self._random_ring_element(x.shape, rng)
        lambda3 = self._random_ring_element(x.shape, rng)
        mv = self._wrap(x + lambda1 + lambda2 + lambda3)
        return [
            PartyShare(party_index=0, payload=(mv, lambda1, lambda2)),
            PartyShare(party_index=1, payload=(mv, lambda1, lambda3)),
            PartyShare(party_index=2, payload=(mv, lambda2, lambda3)),
            PartyShare(party_index=3, payload=(lambda1, lambda2, lambda3)),
        ]

    def reconstruct(self, shares):
        if len(shares) < self.reconstruction_threshold:
            raise ValueError(
                f"tetrad4pc needs shares from all {self.reconstruction_threshold} "
                f"parties to reconstruct (test/debug helper only), got {len(shares)}"
            )
        mv, _, _ = shares[0].payload  # any of P0/P1/P2 holds mv; P0's used here
        lambda1, lambda2, lambda3 = shares[3].payload  # P3 holds all three
        total = self._wrap(mv - lambda1 - lambda2 - lambda3)
        return total.astype(np.int64)

    def add(self, a, b):
        if a.party_index != b.party_index:
            raise ValueError("can only locally add two shares held by the same party")
        return PartyShare(
            party_index=a.party_index,
            payload=tuple(self._wrap(x + y) for x, y in zip(a.payload, b.payload)),
        )


# Convention followed by every sharing-scheme module (see load_sharing_scheme.py):
# expose the concrete class as SCHEME_CLASS so the loader can instantiate it
# with backend-specific constructor args (here, just `bitlength`).
SCHEME_CLASS = Tetrad4PCScheme
