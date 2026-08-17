"""Server-side (flo_server) reconstruction of hpmpc-native raw shares into
plaintext ring values, per protocol.

Exists because compute parties no longer reveal aggregates among
themselves (see fedavg_secure_aggregation.hpp's and
mult_fedavg_secure_aggregation.hpp's module docstrings, and
docs/secure_aggregation/threat_model.md's trust-model note): each party now
exports its OWN raw hpmpc share of the (still-secret) aggregate, and
flo_server -- the only place plaintext is ever computed -- combines them
using each protocol's own reveal formula, reimplemented here in pure
Python. These formulas are not new derivations; they are the exact
formulas hpmpc's own `complete_reveal_to_all()` implementations used to run
inside the compute parties, verified against real compiled hpmpc binaries
(see the docstrings below for the concrete numeric checks).

All functions operate on raw uint64 ring values (already-summed/multiplied
shares, `& mask` where a mask is passed) and return a plaintext ring value
(still signed-ambiguous -- caller decodes via FixedPointCodec separately).
No decoding, no MPC-library dependency -- pure integer arithmetic, same
spirit as sharing_schemes/*.py's own reconstruct() methods.
"""

import numpy as np


def reconstruct_xa(shares_by_party: dict, mask: np.uint64) -> np.ndarray:
    """PROTOCOL=2 (Replicated 3PC). Each party's raw share is an (x, a)
    pair (see fedavg_secure_aggregation.hpp's PROTOCOL==2 branch).
    Reveal invariant, traced from Replicated_Share::complete_reveal_to_all
    and verified against a real compiled binary post-refactor:

        secret = x_{(p-1) mod 3} - a_p   -- holds for ANY p in {0, 1, 2}

    `shares_by_party`: {0: (x0, a0), 1: (x1, a1), 2: (x2, a2)}, each a
    numpy array. Uses p=0 (secret = x_2 - a_0) -- an arbitrary but fixed
    choice; `reconstruct_xa_alt` below uses a different p for cross-
    checking (see reconstruct_trio's docstring for why this redundancy is
    useful)."""
    x2 = np.asarray(shares_by_party[2][0]).astype(np.uint64)
    a0 = np.asarray(shares_by_party[0][1]).astype(np.uint64)
    return (x2 - a0) & mask


def reconstruct_xa_alt(shares_by_party: dict, mask: np.uint64) -> np.ndarray:
    """Same as reconstruct_xa but using p=1 (secret = x_0 - a_1) -- an
    independent reconstruction path from the SAME 3 shares, for the
    dual-formula cross-check that replaces the old party-side
    verify_party_agreement (see aggregator_secure_mpc.py)."""
    x0 = np.asarray(shares_by_party[0][0]).astype(np.uint64)
    a1 = np.asarray(shares_by_party[1][1]).astype(np.uint64)
    return (x0 - a1) & mask


def reconstruct_trio(shares_by_party: dict, mask: np.uint64) -> np.ndarray:
    """PROTOCOL=5 (Trio). Each party's raw share is a (p1, p2) pair (see
    fedavg_secure_aggregation.hpp's and mult_fedavg_secure_aggregation.hpp's
    PROTOCOL==5 branches). Reveal invariant, traced from
    OECL{0,1,2}_Share::complete_reveal_to_all and verified against real
    compiled binaries post-refactor, for both a plain summed share
    (fedavg) and a summed-product share (mult_fedavg):

        secret = P2.p1 - P0.p2 = P1.p1 - P0.p1

    `shares_by_party`: {0: (p1_0, p2_0), 1: (p1_1, p2_1), 2: (p1_2, p2_2)}.
    Uses the first formula; `reconstruct_trio_alt` uses the second, for
    cross-checking against the SAME 3 collected shares."""
    p1_2 = np.asarray(shares_by_party[2][0]).astype(np.uint64)
    p2_0 = np.asarray(shares_by_party[0][1]).astype(np.uint64)
    return (p1_2 - p2_0) & mask


def reconstruct_trio_alt(shares_by_party: dict, mask: np.uint64) -> np.ndarray:
    """Same as reconstruct_trio but using the P1.p1 - P0.p1 formula -- an
    independent reconstruction path from the SAME 3 shares, for the
    dual-formula cross-check that replaces the old party-side
    verify_party_agreement."""
    p1_1 = np.asarray(shares_by_party[1][0]).astype(np.uint64)
    p1_0 = np.asarray(shares_by_party[0][0]).astype(np.uint64)
    return (p1_1 - p1_0) & mask


def reconstruct_tetrad(shares_by_party: dict, mask: np.uint64) -> np.ndarray:
    """PROTOCOL=8 (Tetrad). Parties 0/1/2 each hold a raw (mv, l0, l1)
    triple; party 3 holds (l1, l2, l3) -- see
    fedavg_secure_aggregation.hpp's PROTOCOL==8 branch and
    Tetrad-P_{0,1,2,3}_template.hpp. Reveal invariant, traced from
    Tetrad0_Share::complete_reveal_to_all and verified against a real
    compiled binary post-refactor:

        secret = mv - l1 - l2 - l3   -- mv from ANY of P0/1/2 (identical
                                        across all three), l1/l2/l3 from P3

    `shares_by_party`: {0/1/2: (mv, l0, l1), 3: (l1, l2, l3)}. Uses P0's
    mv; `reconstruct_tetrad_alt` uses P1's, for cross-checking."""
    mv = np.asarray(shares_by_party[0][0]).astype(np.uint64)
    l1, l2, l3 = (np.asarray(f).astype(np.uint64) for f in shares_by_party[3])
    return (mv - l1 - l2 - l3) & mask


def reconstruct_tetrad_alt(shares_by_party: dict, mask: np.uint64) -> np.ndarray:
    """Same as reconstruct_tetrad but using P1's mv instead of P0's -- an
    independent reconstruction path from the SAME shares (mv is supposed
    to be identical across P0/1/2; if it isn't, this cross-check catches
    it), for the dual-formula cross-check that replaces the old
    party-side verify_party_agreement."""
    mv = np.asarray(shares_by_party[1][0]).astype(np.uint64)
    l1, l2, l3 = (np.asarray(f).astype(np.uint64) for f in shares_by_party[3])
    return (mv - l1 - l2 - l3) & mask


# protocol number -> (primary reconstruction fn, cross-check fn, num_parties)
_RECONSTRUCTORS = {
    2: (reconstruct_xa, reconstruct_xa_alt, 3),
    5: (reconstruct_trio, reconstruct_trio_alt, 3),
    8: (reconstruct_tetrad, reconstruct_tetrad_alt, 4),
}


def reconstruct(protocol: int, shares_by_party: dict, mask: np.uint64, cross_check: bool = True) -> np.ndarray:
    """Dispatches to the right protocol's reconstruction formula(s) and, if
    `cross_check` (default True), verifies the two independent formulas
    agree -- raises RuntimeError if they don't (this is the flo_server-side
    replacement for the old party_orchestrator_client.run_round's
    verify_party_agreement: since compute parties can no longer see
    plaintext to agree on, the same redundancy the sharing scheme already
    provides is used here instead, from shares already collected in one
    round -- no extra network round-trip needed)."""
    if protocol not in _RECONSTRUCTORS:
        raise ValueError(f"reconstruct() does not support protocol={protocol} (supported: {sorted(_RECONSTRUCTORS)})")
    primary_fn, alt_fn, expected_num_parties = _RECONSTRUCTORS[protocol]
    if len(shares_by_party) != expected_num_parties:
        raise ValueError(
            f"reconstruct() for protocol={protocol} needs exactly {expected_num_parties} parties' "
            f"shares, got {len(shares_by_party)}"
        )
    primary = primary_fn(shares_by_party, mask)
    if cross_check:
        alt = alt_fn(shares_by_party, mask)
        if not np.array_equal(primary, alt):
            raise RuntimeError(
                f"reconstruct() cross-check failed for protocol={protocol}: the two independent "
                "reveal formulas disagree -- one or more parties' shares are inconsistent "
                "(bug, or in principle a misbehaving party)"
            )
    return primary
