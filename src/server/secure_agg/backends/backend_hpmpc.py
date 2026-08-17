"""Subprocess + file-IO adapter over compiled hpmpc executables (see
docs/secure_aggregation/hpmpc_backend.md for the full design, the
per-protocol share-format derivations, and how each was verified). One
compiled executable per party, built ahead of time (see
hpmpc/scripts/build_secure_agg.sh); this backend spawns a FRESH process per
round — hpmpc's own execution model, each run does one fixed computation and
exits (see protocol_executer.hpp) — rather than keeping one running.

Supports multiple hpmpc protocols behind this one class (currently
PROTOCOL=2, "Replicated 3PC" — see docs/secure_aggregation/hpmpc_backend.md
for which others have landed) via a `protocol` constructor arg and a small
per-(protocol, party_index) share-packing strategy — the only genuinely
protocol-specific pieces are the share-conversion math and the on-disk field
count/layout. Everything else here (start(), peer sorting, hostname
resolution, subprocess invocation, timeout/error handling) is shared across
every protocol, unchanged. Protocol is a config value, not a different
backend module, since these are all still hpmpc, still subprocess-per-round,
still file-based I/O via the same env-var convention -- see
load_backend.py's docstring for what *does* warrant a separate backend
module (a structurally different MPC library, not a protocol-number
variant of this one).

Deliberately does NOT sum shares across clients using hpmpc's own
Additive_Share::operator+ for PROTOCOL=2 — see fedavg_secure_aggregation.hpp's
module docstring for why: empirically confirmed during development that
hpmpc's operator+ (PROTOCOL==2 specifically) reveals the DIFFERENCE, not the
sum, of two independently-shared values (it's specialized for combining a
running share with a locally-derived delta, not for adding two unrelated
secrets). Instead, this backend sums every selected client's share fields
itself (plain add-mod-2**bitlength, local/free under each scheme's
homomorphism) before ever invoking the compiled binary. (Other protocols'
operator+ has been verified to have no such pitfall — see hpmpc_backend.md
— but shares are still pre-summed in Python for every protocol, for
consistency with this one convention.)

IMPORTANT: neither round method reveals the aggregate among the compute
parties anymore. Each returns this party's own RAW hpmpc share of the
(still-secret) result, as an OrderedDict[layer_name -> PartyShare] — never
a decoded plaintext tensor. flo_server (party_orchestrator_client.py /
aggregator_secure_mpc.py) collects every party's raw share and reconstructs
the plaintext itself via server/secure_agg/reconstruct.py. See
fedavg_secure_aggregation.hpp's and mult_fedavg_secure_aggregation.hpp's
module docstrings for why the reveal step was removed, and
docs/secure_aggregation/threat_model.md for the resulting trust-model
change (flo_server, not the compute-party cluster, is now the one place
plaintext is ever computed).
"""

import asyncio
import json
import os
import socket
import struct
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np

from server.secure_agg.backends.base import SecureAggregationBackend
from server.secure_agg.constants import DATASET_SIZE_LAYER_NAME
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.base import PartyShare
from utils.logger import FedLogger

# weighting_mode -> (build-metadata filename, hpmpc FUNCTION_IDENTIFIER,
# whether this mode needs Trio's per-client native-input+multiply path
# instead of the original presummed-reveal-only path). "client_side" is the
# original, still-default behavior (client pre-weights by its own dataset
# size -- see client_secure_agg_manager.py -- so this backend only ever
# sums already-weighted shares and reveals). "mpc_product" is the new
# variant (see mult_fedavg_secure_aggregation.hpp): the client shares its
# RAW update and RAW dataset size separately, and the party cluster itself
# computes weight_i * dataset_size_i via genuine secret x secret
# multiplication before summing across clients -- Trio (PROTOCOL=5) only,
# see that file's module docstring for why.
_WEIGHTING_MODE_METADATA = {
    "client_side": ("fedavg_secure_aggregation.build_metadata.json", 90),
    "mpc_product": ("mult_fedavg_secure_aggregation.build_metadata.json", 91),
}


def _own_fragment(payload, mask: np.uint64) -> np.ndarray:
    """Extracts just this party's raw replicated3pc share fragment (c_j --
    the first field of a replicated3pc.py PartyShare payload), with NO
    protocol-specific re-derivation (contrast _to_hpmpc_trio, which derives
    a raw Trio (p1, p2) pair for the OLD presummed-reveal-only path). Used
    ONLY by the "mpc_product" weighting-mode round (see
    mult_fedavg_secure_aggregation.hpp's module docstring for why this
    simpler extraction is correct there): each party inputs this raw
    fragment directly into hpmpc's own prepare_receive_from<P_j> mechanism,
    so no per-protocol share-shape conversion is needed or wanted here --
    doing one would just be undone by that native-input step."""
    c_j, _c_j1 = payload
    return np.asarray(c_j).reshape(-1).astype(np.uint64) & mask


def _to_hpmpc_xa(c_j: np.ndarray, c_j1: np.ndarray, mask: np.uint64):
    """Per-party-local conversion from this scheme's (c_j, c_{j+1}) pair
    (sharing_schemes/replicated3pc.py) to hpmpc's native
    Replicated_Share(x, a) layout (protocols/3-PC/replicated/
    replicated_template.hpp): x_j = c_j, a_j = -(c_j + c_{j+1}) mod ring.
    Needs no coordination with other parties — see
    docs/secure_aggregation/hpmpc_backend.md for the derivation and its
    numeric verification."""
    c_j = c_j.astype(np.uint64) & mask
    c_j1 = c_j1.astype(np.uint64) & mask
    x = c_j
    a = (np.uint64(0) - ((c_j + c_j1) & mask)) & mask
    return x, a


def _to_hpmpc_trio(c_j: np.ndarray, c_j1: np.ndarray, party_index: int, mask: np.uint64):
    """Per-party-local conversion from this scheme's (c_j, c_{j+1}) pair
    (sharing_schemes/replicated3pc.py) to hpmpc's native Trio
    OECL{0,1,2}_Share(p1, p2) layout (protocols/3-PC/ours/
    oecl-P_{0,1,2}_template.hpp). Derived from Trio's reveal invariants
    (secret = P2.p1 - P0.p2 = P1.p1 - P0.p1 — traced from
    prepare_reveal_to_all/complete_Reveal in all 3 role classes, see
    docs/secure_aggregation/hpmpc_backend.md):

        party 0: p1 = -c_j, p2 = -c_j1
        party 1: p1 = c_j + c_j1, p2 unused by reveal (0)
        party 2: p1 = c_j + c_j1, p2 unused by reveal (0)

    Needs no coordination with other parties, and no knowledge of the
    secret — only this party's own (c_j, c_j1) pair, exactly like
    _to_hpmpc_xa. Numerically verified (both single-client reveal and
    cross-client elementwise-summed reveal) — see hpmpc_backend.md."""
    c_j = c_j.astype(np.uint64) & mask
    c_j1 = c_j1.astype(np.uint64) & mask
    if party_index == 0:
        p1 = (np.uint64(0) - c_j) & mask
        p2 = (np.uint64(0) - c_j1) & mask
    else:
        p1 = (c_j + c_j1) & mask
        p2 = np.zeros_like(c_j)
    return p1, p2


def _to_hpmpc_tetrad(payload, mask: np.uint64):
    """Flattens + ring-masks the already-native-shaped 3-field payload
    sharing_schemes/tetrad4pc.py produces for this party (mv, lambda_a,
    lambda_b for party 0/1/2; lambda1, lambda2, lambda3 for party 3 — see
    that module's docstring). Unlike Replicated/Trio, no per-party
    re-derivation math is needed here: tetrad4pc.py's share() already
    outputs values shaped for hpmpc's native Tetrad{0,1,2,3}_Share
    constructors directly, since Tetrad's masking structure doesn't map
    onto a 3-party replicated3pc intermediate the way Replicated/Trio's do.
    Raises ValueError with a clear, actionable message (rather than a
    confusing downstream shape error) if the payload doesn't have exactly 3
    fields — the most likely cause being a sharing_scheme/protocol config
    mismatch (e.g. protocol=8 paired with sharing_scheme=replicated3pc)."""
    if len(payload) != 3:
        raise ValueError(
            f"HpmpcBackend protocol=8 (Tetrad) requires a 3-field share payload "
            f"(from sharing_scheme=tetrad4pc), got {len(payload)} field(s) -- "
            "check this party's and the client's sharing_scheme config agree."
        )
    return tuple(np.asarray(field).reshape(-1).astype(np.uint64) & mask for field in payload)


class _SharePackingStrategy(ABC):
    """Protocol- and (for role-asymmetric protocols) party-specific share
    conversion and on-disk field layout. One instance per (protocol,
    party_index) — constructed once in HpmpcBackend.__init__, not per round.
    """

    field_names: tuple

    @abstractmethod
    def convert(self, payload, mask: np.uint64) -> tuple:
        """Per-party-local conversion from this party's PartyShare.payload
        (whatever shape the configured sharing scheme produces for this
        party — a replicated3pc (c_j, c_j1) pair for Replicated/Trio, an
        already-native-shaped tuple for Tetrad's own scheme) to this
        protocol's native per-element field tuple, in the same order as
        `field_names`. Pure, no I/O, no coordination with other parties."""


class _ReplicatedStrategy(_SharePackingStrategy):
    """PROTOCOL=2, Replicated 3PC. Symmetric across all 3 parties — see
    _to_hpmpc_xa's docstring. Expects a replicated3pc.py (c_j, c_j1) pair."""

    field_names = ("x", "a")

    def convert(self, payload, mask):
        c_j, c_j1 = payload
        return _to_hpmpc_xa(np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), mask)


class _TrioStrategy(_SharePackingStrategy):
    """PROTOCOL=5, Trio. Role-asymmetric (party 0's conversion differs from
    parties 1/2's — see _to_hpmpc_trio's docstring), though the on-disk
    field count/order is uniform across all 3 roles. Also expects a
    replicated3pc.py (c_j, c_j1) pair."""

    field_names = ("p1", "p2")

    def __init__(self, party_index: int):
        self._party_index = party_index

    def convert(self, payload, mask):
        c_j, c_j1 = payload
        return _to_hpmpc_trio(
            np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), self._party_index, mask
        )


class _TetradStrategy(_SharePackingStrategy):
    """PROTOCOL=8, Tetrad. Expects an ALREADY-NATIVE-SHAPED 3-field payload
    from sharing_schemes/tetrad4pc.py (NOT a replicated3pc pair — Tetrad's
    masking structure genuinely needs 4 independent random values, so it
    cannot be derived from a 3-party replicated3pc share the way
    Replicated/Trio's conversions can). See _to_hpmpc_tetrad's docstring;
    field semantics differ by role (mv,l0,l1 for P0/1/2; l1,l2,l3 for P3)
    but the field COUNT (3) and conversion (flatten + ring-mask, no
    per-party math) are uniform, so party_index isn't even needed here."""

    field_names = ("f0", "f1", "f2")  # semantics differ by role -- see hpmpc_backend.md

    def convert(self, payload, mask):
        return _to_hpmpc_tetrad(payload, mask)


# protocol number -> factory(party_index) -> _SharePackingStrategy. See
# docs/secure_aggregation/hpmpc_backend.md's per-protocol sections.
_STRATEGY_FACTORY = {
    2: lambda party_index: _ReplicatedStrategy(),
    5: lambda party_index: _TrioStrategy(party_index),
    8: lambda party_index: _TetradStrategy(),
}

# protocol number -> exactly how many parties that protocol needs. Kept
# alongside _STRATEGY_FACTORY (same keys) rather than merged into it, since
# this table is meaningful even before a given protocol's strategy exists
# (e.g. for a clearer "not yet supported" error than a bare KeyError).
_EXPECTED_NUM_PARTIES = {
    2: 3,
    5: 3,
    8: 4,
}


class HpmpcBackend(SecureAggregationBackend):
    backend_id = "hpmpc"

    def __init__(
        self,
        party_index: int,
        num_parties: int,
        codec: FixedPointCodec,
        protocol: int,
        executable_dir: str,
        tmp_dir: str,
        run_timeout_margin_s: float = 10.0,
        log_stdout: bool = False,
        weighting_mode: str = "client_side",
    ):
        if protocol not in _EXPECTED_NUM_PARTIES:
            raise ValueError(
                f"backend_hpmpc does not support protocol={protocol} "
                f"(supported: {sorted(_EXPECTED_NUM_PARTIES)})"
            )
        expected_num_parties = _EXPECTED_NUM_PARTIES[protocol]
        if num_parties != expected_num_parties:
            raise ValueError(
                f"backend_hpmpc protocol={protocol} requires exactly "
                f"{expected_num_parties} parties, got num_parties={num_parties}"
            )
        if weighting_mode not in _WEIGHTING_MODE_METADATA:
            raise ValueError(
                f"backend_hpmpc does not support weighting_mode={weighting_mode!r} "
                f"(supported: {sorted(_WEIGHTING_MODE_METADATA)})"
            )
        if weighting_mode == "mpc_product" and protocol != 5:
            raise ValueError(
                "backend_hpmpc weighting_mode='mpc_product' is currently only supported for "
                "protocol=5 (Trio) -- see mult_fedavg_secure_aggregation.hpp's module docstring "
                "for why Replicated/Tetrad aren't wired in yet"
            )
        self.backend_id = "hpmpc"
        self.party_index = party_index
        self.num_parties = num_parties
        self._codec = codec
        self._protocol = protocol
        self._weighting_mode = weighting_mode
        self._strategy = _STRATEGY_FACTORY[protocol](party_index)
        self._ring_mask = np.uint64((1 << codec.bitlength) - 1)
        self._executable_dir = Path(executable_dir)
        self._tmp_dir = Path(tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._run_timeout_margin_s = run_timeout_margin_s
        self._log_stdout = log_stdout
        self._logger = FedLogger(id=f"party{party_index}", loggername="HPMPC_BACKEND")
        self._peer_endpoints = []

    def _executable_path(self) -> Path:
        return self._executable_dir / f"run-P{self.party_index}.o"

    def _metadata_path(self) -> Path:
        metadata_filename, _function_identifier = _WEIGHTING_MODE_METADATA[self._weighting_mode]
        return self._executable_dir / metadata_filename

    def _check_config_consistency(self):
        """Fails fast, at start(), rather than mid-round, if this party's
        configured protocol/fixed-point/weighting_mode don't match what the
        compiled executable was actually built with. See hpmpc_backend.md's
        "config-mismatch hazard" — protocol/bitlength/frac_bits live in two
        places (this party's YAML config, and hpmpc's compile-time
        PROTOCOL/BITLENGTH/FRACTIONAL macros) with no automatic way to keep
        them in sync; build_secure_agg.sh writes the metadata file this
        checks against. function_identifier is checked too -- since
        weighting_mode='mpc_product' needs a DIFFERENT compiled program
        (mult_fedavg_secure_aggregation.hpp, FUNCTION_IDENTIFIER=91) than the
        default (fedavg_secure_aggregation.hpp, FUNCTION_IDENTIFIER=90),
        pointing a 'mpc_product'-configured party at the wrong executable_dir
        would otherwise only be caught by a confusing runtime file-format
        mismatch (this backend would write the mult-fedavg per-client file
        layout, but the fedavg binary would try to parse it as one
        pre-summed row)."""
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            raise RuntimeError(
                f"missing build metadata at {metadata_path} -- rebuild via "
                "hpmpc/scripts/build_secure_agg.sh"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("protocol") != self._protocol:
            raise RuntimeError(
                f"protocol config mismatch: this party is configured for "
                f"protocol={self._protocol}, but the compiled executables at "
                f"{self._executable_dir} were built with protocol={metadata.get('protocol')}. "
                "Point this party's backend.hpmpc.executable_dir at the correctly-compiled "
                "executables, or rebuild them for the configured protocol via "
                "hpmpc/scripts/build_secure_agg.sh."
            )
        expected_function_identifier = _WEIGHTING_MODE_METADATA[self._weighting_mode][1]
        if metadata.get("function_identifier") != expected_function_identifier:
            raise RuntimeError(
                f"weighting_mode config mismatch: this party is configured for "
                f"weighting_mode={self._weighting_mode!r} (expects FUNCTION_IDENTIFIER="
                f"{expected_function_identifier}), but the compiled executables at "
                f"{self._executable_dir} were built with FUNCTION_IDENTIFIER="
                f"{metadata.get('function_identifier')}. Point this party's "
                "backend.hpmpc.executable_dir at the correctly-compiled executables, or rebuild "
                "them for the configured weighting_mode via hpmpc/scripts/build_secure_agg.sh."
            )
        if metadata.get("bitlength") != self._codec.bitlength or metadata.get("frac_bits") != self._codec.frac_bits:
            raise RuntimeError(
                f"fixed_point config mismatch: this party is configured for "
                f"bitlength={self._codec.bitlength}, frac_bits={self._codec.frac_bits}, but the "
                f"compiled executables were built with bitlength={metadata.get('bitlength')}, "
                f"frac_bits={metadata.get('frac_bits')}. Rebuild the executables or fix this "
                "party's fixed_point config."
            )
        if not self._executable_path().exists():
            raise RuntimeError(f"missing compiled executable at {self._executable_path()}")

    async def start(self, peer_endpoints):
        self._check_config_consistency()
        # CLI arg order hpmpc expects: the OTHER parties' IPs, sorted by
        # their own party_index ascending (self excluded) -- derived from
        # protocol_executer.hpp's per-PARTY slot relabeling table, the same
        # for every protocol hpmpc_backend.md documents; see that doc.
        self._peer_endpoints = sorted(peer_endpoints, key=lambda p: p.party_index)

    async def run_aggregation_round(self, round_id, shares, tensor_specs, timeout_s):
        if self._weighting_mode == "mpc_product":
            return await self._run_mpc_product_round(round_id, shares, tensor_specs, timeout_s)
        return await self._run_client_side_round(round_id, shares, tensor_specs, timeout_s)

    async def _invoke_executable(self, round_id, input_path, output_path, timeout_s):
        """Shared subprocess-invocation plumbing for both weighting modes --
        peer sorting, hostname resolution, timeout/error handling are
        identical regardless of which program/file-format is in play; only
        the file contents differ (see the two round methods below)."""
        # hpmpc's own C++ socket layer (core/networking/socket.hpp) parses
        # its peer-IP CLI args as literal dotted-quad addresses -- it has no
        # DNS resolution of its own, so a Docker Compose service name like
        # "secure_agg_party1" (which gRPC/Python resolve transparently
        # everywhere else in this codebase) crashes it outright
        # ("Invalid address: secure_agg_party1", confirmed during the Phase
        # 4 real end-to-end run). Resolved fresh per round (not cached in
        # start()) so a peer container restarting mid-deployment with a new
        # IP doesn't leave this backend pointed at a stale address.
        peer_ips = [socket.gethostbyname(peer.host) for peer in self._peer_endpoints]
        env = os.environ.copy()
        env["SECURE_AGG_INPUT_FILE"] = str(input_path)
        env["SECURE_AGG_OUTPUT_FILE"] = str(output_path)

        process = await asyncio.create_subprocess_exec(
            str(self._executable_path()),
            *peer_ips,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s + self._run_timeout_margin_s
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"hpmpc executable for round {round_id} did not finish within {timeout_s}s"
            )

        if process.returncode != 0:
            raise RuntimeError(
                f"hpmpc executable for round {round_id} exited with code {process.returncode}: "
                f"{stdout.decode(errors='replace') if stdout else ''}"
            )

        if self._log_stdout:
            # Off by default (log spam in normal deployments) -- turned on
            # for benchmarking, since hpmpc's own stdout already contains
            # per-phase communication/timing lines, unconditionally, with no
            # extra build flags (see docs/secure_aggregation/hpmpc_backend.md
            # and the overhead report). round_id first so a benchmarking
            # script can grep/attribute this line to a specific round.
            self._logger.info(
                "fedparty.hpmpc.round_stdout",
                f"{round_id},{stdout.decode(errors='replace') if stdout else ''}",
            )

    async def _run_client_side_round(self, round_id, shares, tensor_specs, timeout_s):
        """Original, still-default path: clients already pre-weighted their
        update by their own dataset size (see client_secure_agg_manager.py),
        so this backend only ever needs to SUM already-weighted shares
        across clients (in Python, before ever invoking the binary -- see
        this module's docstring for why). Talks to
        fedavg_secure_aggregation.hpp, which no longer reveals -- it exports
        this party's own raw share of the sum, in the SAME on-disk field
        layout as the input (see that file's module docstring)."""
        layer_names = sorted(tensor_specs.keys())
        safe_round_id = round_id.replace(":", "_").replace("/", "_")
        input_path = self._tmp_dir / f"input_{safe_round_id}.bin"
        output_path = self._tmp_dir / f"output_{safe_round_id}.bin"

        num_fields = len(self._strategy.field_names)
        flat_field_parts = [[] for _ in range(num_fields)]
        for layer_name in layer_names:
            spec = tensor_specs[layer_name]
            num_elements = int(np.prod(spec.shape)) if spec.shape else 1
            sums = [np.zeros(num_elements, dtype=np.uint64) for _ in range(num_fields)]
            for client_shares in shares.values():
                fields = self._strategy.convert(
                    client_shares[layer_name].payload, self._ring_mask
                )
                for i, value in enumerate(fields):
                    sums[i] = (sums[i] + value) & self._ring_mask
            for i in range(num_fields):
                flat_field_parts[i].append(sums[i])

        flat_fields = [
            np.concatenate(parts) if parts else np.array([], dtype=np.uint64)
            for parts in flat_field_parts
        ]
        num_total_elements = len(flat_fields[0]) if flat_fields else 0

        row_format = "<" + "Q" * num_fields
        with open(input_path, "wb") as f:
            f.write(struct.pack("<I", num_total_elements))
            for row in zip(*(arr.tolist() for arr in flat_fields)):
                f.write(struct.pack(row_format, *row))

        await self._invoke_executable(round_id, input_path, output_path, timeout_s)

        # Output is this party's own raw share of the sum, in the SAME
        # per-element field layout as the input (num_fields uint64s per
        # element) -- no reveal, no decode here; see module docstring.
        with open(output_path, "rb") as f:
            (num_output_elements,) = struct.unpack("<I", f.read(4))
            row_format = "<" + "Q" * num_fields
            rows = [
                struct.unpack(row_format, f.read(8 * num_fields)) for _ in range(num_output_elements)
            ]
        output_fields = [np.array(field, dtype=np.uint64) for field in zip(*rows)] if rows else [
            np.array([], dtype=np.uint64) for _ in range(num_fields)
        ]

        result = OrderedDict()
        offset = 0
        for layer_name in layer_names:
            spec = tensor_specs[layer_name]
            num_elements = int(np.prod(spec.shape)) if spec.shape else 1
            payload = tuple(field[offset : offset + num_elements] for field in output_fields)
            result[layer_name] = PartyShare(party_index=self.party_index, payload=payload)
            offset += num_elements

        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result

    async def _run_mpc_product_round(self, round_id, shares, tensor_specs, timeout_s):
        """New path (weighting_mode='mpc_product'): clients share their RAW
        update and RAW dataset size separately (no client-side pre-weighting
        -- see client_secure_agg_manager.py's weighting_mode branch), so
        weight_i * dataset_size_i must be computed HERE, inside the MPC
        circuit, via genuine secret x secret multiplication, for every
        client, before summing across clients -- Sum(w_i*d_i) != Sum(w_i) *
        Sum(d_i), so presumming (like the client_side path does) would be
        mathematically wrong here. Talks to
        mult_fedavg_secure_aggregation.hpp; see that file's module docstring
        for the on-disk file contract and why only this party's OWN raw
        replicated3pc share fragment (not a protocol-specific re-derivation)
        is written per client per element. No reveal -- this party's own
        raw Trio (p1, p2) share of the result is returned, not plaintext."""
        layer_names = sorted(name for name in tensor_specs.keys() if name != DATASET_SIZE_LAYER_NAME)
        client_ids = sorted(shares.keys())
        safe_round_id = round_id.replace(":", "_").replace("/", "_")
        input_path = self._tmp_dir / f"input_{safe_round_id}.bin"
        output_path = self._tmp_dir / f"output_{safe_round_id}.bin"

        elements_per_client = 0
        for layer_name in layer_names:
            spec = tensor_specs[layer_name]
            elements_per_client += int(np.prod(spec.shape)) if spec.shape else 1

        with open(input_path, "wb") as f:
            f.write(struct.pack("<I", len(client_ids)))
            f.write(struct.pack("<I", elements_per_client))
            for client_id in client_ids:
                client_shares = shares[client_id]
                for layer_name in layer_names:
                    fragment = _own_fragment(client_shares[layer_name].payload, self._ring_mask)
                    f.write(struct.pack(f"<{len(fragment)}Q", *fragment.tolist()))
                ds_fragment = _own_fragment(
                    client_shares[DATASET_SIZE_LAYER_NAME].payload, self._ring_mask
                )
                f.write(struct.pack("<Q", int(ds_fragment[0])))

        await self._invoke_executable(round_id, input_path, output_path, timeout_s)

        # Output is this party's own raw Trio (p1, p2) share of the summed
        # product (elements_per_client rows) followed by one more (p1, p2)
        # row for the summed dataset size -- no reveal, no decode here; see
        # module docstring. The product share reconstructs to an
        # UNTRUNCATED value (2*frac_bits fractional precision) once
        # flo_server calls reconstruct() -- that decode step, and the
        # dataset-size field's normal-frac_bits decode, both happen at
        # flo_server, not here.
        with open(output_path, "rb") as f:
            (num_output_elements,) = struct.unpack("<I", f.read(4))
            rows = [struct.unpack("<QQ", f.read(16)) for _ in range(num_output_elements)]
            ds_p1, ds_p2 = struct.unpack("<QQ", f.read(16))
        p1_field = np.array([row[0] for row in rows], dtype=np.uint64)
        p2_field = np.array([row[1] for row in rows], dtype=np.uint64)

        result = OrderedDict()
        offset = 0
        for layer_name in layer_names:
            spec = tensor_specs[layer_name]
            num_elements = int(np.prod(spec.shape)) if spec.shape else 1
            payload = (p1_field[offset : offset + num_elements], p2_field[offset : offset + num_elements])
            result[layer_name] = PartyShare(party_index=self.party_index, payload=payload)
            offset += num_elements
        result[DATASET_SIZE_LAYER_NAME] = PartyShare(
            party_index=self.party_index,
            payload=(np.array([ds_p1], dtype=np.uint64), np.array([ds_p2], dtype=np.uint64)),
        )

        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result

    async def stop(self):
        pass


BACKEND_CLASS = HpmpcBackend
