from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PartyShare:
    """One party's share of one secret tensor. `payload` is scheme-specific
    and must be picklable, since it travels over the wire (see
    secure_agg.proto's TensorShare.share_payload)."""

    party_index: int
    payload: Any


class SecretSharingScheme(ABC):
    """Pure math: splits/combines/locally-adds secret shares. No I/O, no
    network, no dependency on any MPC library, so implementations are
    testable without any backend (hpmpc or otherwise) installed.

    Concrete implementations: sharing_schemes/replicated3pc.py (Phase 1),
    sharing_schemes/shamir_stub.py (Phase 1, documentation-only stub that
    proves this ABC isn't accidentally shaped around replicated sharing's
    specifics, e.g. it must not assume num_parties == 3 anywhere).
    """

    scheme_id: str
    num_parties: int
    reconstruction_threshold: int

    @abstractmethod
    def share(
        self, plaintext_fixedpoint: np.ndarray, rng: np.random.Generator
    ) -> list[PartyShare]:
        """Split a fixed-point-encoded integer tensor into `num_parties`
        PartyShare objects, one per party index (result[i].party_index == i).
        Pure function: no network, no side effects."""

    @abstractmethod
    def reconstruct(self, shares: dict[int, PartyShare]) -> np.ndarray:
        """Combine enough parties' shares back into fixed-point plaintext.
        Used by tests, the in-process simulator backend, and debug tooling
        only — real MPC parties never call this in production; they run the
        protocol via a SecureAggregationBackend instead."""

    @abstractmethod
    def add(self, a: PartyShare, b: PartyShare) -> PartyShare:
        """Locally (no communication) add two same-party shares. Used by the
        in-process simulator and by any backend implementing a reference
        "sum shares" step."""
