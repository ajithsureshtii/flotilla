from abc import ABC, abstractmethod
from dataclasses import dataclass

from server.secure_agg.sharing_schemes.base import PartyShare


@dataclass(frozen=True)
class PartyEndpoint:
    party_index: int
    host: str
    port: int


@dataclass(frozen=True)
class TensorSpec:
    layer_name: str
    shape: tuple
    dtype: str


class SecureAggregationBackend(ABC):
    """Runs inside one party process, one instance per party. Deliberately
    silent about *how* peer communication happens during a round — that is
    each backend's private business (hpmpc opens its own raw TCP/TLS sockets
    between spawned per-round binaries; a Python-native backend might talk to
    peers over a small internal gRPC call; a hypothetical daemon-backed
    library might issue one RPC to its own persistent process). That silence
    is what lets multiple, structurally different MPC libraries implement
    this same interface without Flotilla's core (party server, aggregator
    plugin, proto contracts) needing to know or care which shape a given
    backend uses.

    Concrete implementations: backends/backend_simulator.py (Phase 1/2, pure
    Python/numpy reference), backends/backend_hpmpc.py (Phase 3, subprocess +
    file-IO adapter over compiled hpmpc executables).
    """

    backend_id: str
    party_index: int
    num_parties: int

    @abstractmethod
    async def start(self, peer_endpoints: list[PartyEndpoint]) -> None:
        """One-time setup after construction, before any round runs. For a
        process-per-round backend like hpmpc this may just remember peer
        endpoints; for a daemon-backed backend this is where a persistent
        connection would be opened."""

    @abstractmethod
    async def run_aggregation_round(
        self,
        round_id: str,
        shares: dict[str, dict[str, PartyShare]],
        tensor_specs: dict[str, TensorSpec],
        timeout_s: float,
    ):
        """Run one round of the underlying MPC protocol against the other
        `num_parties - 1` peers and return this party's PLAINTEXT reveal of
        the SUM of the given shares (not a weighted average — see below), as
        an OrderedDict[str, torch.Tensor] keyed by layer_name. Every honest
        party must return an identical result.

        `shares`: client_id -> layer_name -> this party's share of that
        client's update.

        Deliberately no `weights` parameter. An earlier draft of this
        interface had the backend compute a weighted sum directly (client_id
        -> public weight fraction), but that requires multiplying a share by
        a fractional public scalar, which for a fixed-point scheme needs a
        correct truncation step afterward (rescaling from 2x fractional bits
        back down to 1x) — a genuine MPC primitive (see hpmpc's
        prob_truncation.hpp) that a reference backend has no business
        reimplementing, and getting subtly wrong would silently corrupt
        every aggregate.

        Sidestepped instead by moving weighting to where it's free: the
        CLIENT already knows its own update and its own dataset size with no
        dependency on any other party, so it pre-multiplies its update by
        that (its-own-data) scalar BEFORE encoding/sharing (see
        client_secure_agg_manager.py). Summing shares is then always pure
        addition — free under additive/replicated sharing, no truncation, no
        per-backend special-casing. The caller (aggregator_secure_mpc.py)
        divides the returned raw sum by the total dataset size of the
        round's checked-in clients in PLAINTEXT after reveal — trivial
        arithmetic, computed post-hoc so it's automatically correct under
        client dropouts, exactly mirroring how aggregator_fedavg.py already
        computes its weights today.

        That total is itself a REVEALED value, not something flo_server
        already knows: each client also secret-shares its raw dataset size
        under the reserved DATASET_SIZE_LAYER_NAME pseudo-layer (see
        constants.py), summed and revealed by this exact same mechanism
        alongside every real model layer — this backend never needs to know
        or care that one of its "layers" happens to be a dataset-size total
        rather than a model weight, which is exactly what this interface's
        genericity is for.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Release any resources acquired in start()."""
