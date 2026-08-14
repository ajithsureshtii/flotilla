"""Subprocess + file-IO adapter over compiled hpmpc executables (PROTOCOL=2,
Replicated 3PC — see docs/secure_aggregation/hpmpc_backend.md for the full
design, the share-format derivation, and how this was verified). One
compiled executable per party, built ahead of time (see
hpmpc/scripts/build_fedavg_secure_aggregation.sh); this backend spawns a
FRESH process per round — hpmpc's own execution model, each run does one
fixed computation and exits (see protocol_executer.hpp) — rather than
keeping one running.

Deliberately does NOT sum shares across clients using hpmpc's own
Additive_Share::operator+ — see fedavg_secure_aggregation.hpp's module
docstring for why: empirically confirmed during development that hpmpc's
operator+ (PROTOCOL==2) reveals the DIFFERENCE, not the sum, of two
independently-shared values (it's specialized for combining a running share
with a locally-derived delta, not for adding two unrelated secrets).
Instead, this backend sums every selected client's (x, a) pair itself
(plain add-mod-2**bitlength, local/free under this sharing scheme's
homomorphism — see sharing_scheme_replicated3pc.md) before ever invoking the
compiled binary, which therefore only ever has to reveal a single,
already-combined pair per model-weight element.
"""

import asyncio
import json
import os
import socket
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from server.secure_agg.backends.base import SecureAggregationBackend
from server.secure_agg.fixed_point_codec import FixedPointCodec


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


class HpmpcBackend(SecureAggregationBackend):
    backend_id = "hpmpc"

    def __init__(
        self,
        party_index: int,
        num_parties: int,
        codec: FixedPointCodec,
        executable_dir: str,
        tmp_dir: str,
        run_timeout_margin_s: float = 10.0,
    ):
        if num_parties != 3:
            raise ValueError(
                "backend_hpmpc currently only supports the 3-party Replicated protocol (PROTOCOL=2)"
            )
        self.backend_id = "hpmpc"
        self.party_index = party_index
        self.num_parties = num_parties
        self._codec = codec
        self._ring_mask = np.uint64((1 << codec.bitlength) - 1)
        self._executable_dir = Path(executable_dir)
        self._tmp_dir = Path(tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._run_timeout_margin_s = run_timeout_margin_s
        self._peer_endpoints = []

    def _executable_path(self) -> Path:
        return self._executable_dir / f"run-P{self.party_index}.o"

    def _metadata_path(self) -> Path:
        return self._executable_dir / "fedavg_secure_aggregation.build_metadata.json"

    def _check_config_consistency(self):
        """Fails fast, at start(), rather than mid-round, if this party's
        configured fixed-point parameters don't match what the compiled
        executable was actually built with. See hpmpc_backend.md's
        "config-mismatch hazard" — bitlength/frac_bits live in two places
        (this party's YAML config, and hpmpc's compile-time BITLENGTH/
        FRACTIONAL macros) with no automatic way to keep them in sync;
        build_fedavg_secure_aggregation.sh writes the metadata file this
        checks against."""
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            raise RuntimeError(
                f"missing build metadata at {metadata_path} -- rebuild via "
                "hpmpc/scripts/build_fedavg_secure_aggregation.sh"
            )
        metadata = json.loads(metadata_path.read_text())
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
        # protocol_executer.hpp's per-PARTY P_0/P_1/P_2 slot relabeling; see
        # hpmpc_backend.md.
        self._peer_endpoints = sorted(peer_endpoints, key=lambda p: p.party_index)

    async def run_aggregation_round(self, round_id, shares, tensor_specs, timeout_s):
        layer_names = sorted(tensor_specs.keys())
        safe_round_id = round_id.replace(":", "_").replace("/", "_")
        input_path = self._tmp_dir / f"input_{safe_round_id}.bin"
        output_path = self._tmp_dir / f"output_{safe_round_id}.bin"

        flat_x_parts = []
        flat_a_parts = []
        for layer_name in layer_names:
            spec = tensor_specs[layer_name]
            num_elements = int(np.prod(spec.shape)) if spec.shape else 1
            sum_x = np.zeros(num_elements, dtype=np.uint64)
            sum_a = np.zeros(num_elements, dtype=np.uint64)
            for client_shares in shares.values():
                c_j, c_j1 = client_shares[layer_name].payload
                x, a = _to_hpmpc_xa(
                    np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), self._ring_mask
                )
                sum_x = (sum_x + x) & self._ring_mask
                sum_a = (sum_a + a) & self._ring_mask
            flat_x_parts.append(sum_x)
            flat_a_parts.append(sum_a)

        flat_x = np.concatenate(flat_x_parts) if flat_x_parts else np.array([], dtype=np.uint64)
        flat_a = np.concatenate(flat_a_parts) if flat_a_parts else np.array([], dtype=np.uint64)

        with open(input_path, "wb") as f:
            f.write(struct.pack("<I", len(flat_x)))
            for x_val, a_val in zip(flat_x.tolist(), flat_a.tolist()):
                f.write(struct.pack("<QQ", x_val, a_val))

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

        with open(output_path, "rb") as f:
            (num_output_elements,) = struct.unpack("<I", f.read(4))
            raw_values = struct.unpack(f"<{num_output_elements}Q", f.read(8 * num_output_elements))

        raw_array = np.array(raw_values, dtype=np.uint64).astype(np.int64)
        decoded = self._codec.decode(raw_array)

        result = OrderedDict()
        offset = 0
        for layer_name in layer_names:
            spec = tensor_specs[layer_name]
            num_elements = int(np.prod(spec.shape)) if spec.shape else 1
            layer_values = decoded[offset : offset + num_elements].reshape(spec.shape)
            result[layer_name] = torch.from_numpy(layer_values.astype(np.dtype(spec.dtype)))
            offset += num_elements

        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result

    async def stop(self):
        pass


BACKEND_CLASS = HpmpcBackend
