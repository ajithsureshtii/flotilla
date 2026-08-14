from server.secure_agg.sharing_schemes.base import SecretSharingScheme


class ShamirStubScheme(SecretSharingScheme):
    """Documentation-only stub — deliberately NOT implemented.

    Exists to prove SecretSharingScheme's ABC contract isn't accidentally
    shaped around replicated3pc's specifics (a fixed num_parties == 3, or
    "2 random values + one subtraction" sharing math). Shamir secret sharing
    uses a genuine (t, n) polynomial-evaluation/Lagrange-interpolation
    scheme with a party count and threshold that are independent of each
    other and of 3 — if this class can subclass SecretSharingScheme without
    needing any change to that ABC, the abstraction is generic enough.

    Not wired into any config or loader; not intended to run. A real
    implementation is a documented future upgrade path — see
    docs/secure_aggregation/design.md's Phase 5 (genericity stress-test).
    """

    scheme_id = "shamir_stub"
    num_parties = 5  # deliberately different from replicated3pc's 3
    reconstruction_threshold = 3  # deliberately a genuine (t, n) threshold

    def share(self, plaintext_fixedpoint, rng):
        raise NotImplementedError(
            "Shamir secret sharing is a documented future backend, not "
            "implemented — see docs/secure_aggregation/design.md."
        )

    def reconstruct(self, shares):
        raise NotImplementedError(
            "Shamir secret sharing is a documented future backend, not implemented."
        )

    def add(self, a, b):
        raise NotImplementedError(
            "Shamir secret sharing is a documented future backend, not implemented."
        )


SCHEME_CLASS = ShamirStubScheme
