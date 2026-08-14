import json
import struct
from collections import OrderedDict
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from server.secure_agg.backends.backend_hpmpc import HpmpcBackend, _to_hpmpc_xa
from server.secure_agg.backends.base import PartyEndpoint, TensorSpec
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme

pytestmark = pytest.mark.unit

RING_MASK = np.uint64((1 << 64) - 1)


def _write_metadata(executable_dir, bitlength=64, frac_bits=13):
    executable_dir.mkdir(parents=True, exist_ok=True)
    (executable_dir / "fedavg_secure_aggregation.build_metadata.json").write_text(
        json.dumps({"bitlength": bitlength, "frac_bits": frac_bits, "protocol": 2, "function_identifier": 90})
    )
    for party in range(3):
        (executable_dir / f"run-P{party}.o").touch()


def _make_backend(tmp_path, party_index=0, bitlength=64, frac_bits=13):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir, bitlength=bitlength, frac_bits=frac_bits)
    return HpmpcBackend(
        party_index=party_index,
        num_parties=3,
        codec=FixedPointCodec(bitlength=bitlength, frac_bits=frac_bits),
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )


def test_rejects_non_three_party_configurations(tmp_path):
    with pytest.raises(ValueError):
        HpmpcBackend(
            party_index=0,
            num_parties=4,
            codec=FixedPointCodec(bitlength=64, frac_bits=13),
            executable_dir=str(tmp_path),
            tmp_dir=str(tmp_path / "tmp"),
        )


def test_to_hpmpc_xa_matches_hand_derived_formula():
    # x_j = c_j, a_j = -(c_j + c_{j+1}) mod ring -- see
    # docs/secure_aggregation/hpmpc_backend.md for the derivation.
    c_j = np.array([3, 100], dtype=np.uint64)
    c_j1 = np.array([4, 5], dtype=np.uint64)

    x, a = _to_hpmpc_xa(c_j, c_j1, RING_MASK)

    assert x.tolist() == [3, 100]
    expected_a = [(-(3 + 4)) & int(RING_MASK), (-(100 + 5)) & int(RING_MASK)]
    assert a.tolist() == expected_a


def test_to_hpmpc_xa_round_trips_through_replicated3pc_reconstruct():
    # Sanity: converting all 3 parties' shares of the SAME secret and
    # re-deriving via the reveal invariant (x_{p-1} - a_p) recovers it --
    # the same relationship verified against the real hpmpc binary during
    # development (see hpmpc_backend.md).
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(0)
    secret = np.array([12345], dtype=np.int64)
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    xa = {}
    for j in range(3):
        c_j, c_j1 = shares[j].payload
        x, a = _to_hpmpc_xa(c_j, c_j1, RING_MASK)
        xa[j] = (x, a)

    for p in range(3):
        x_prev = xa[(p - 1) % 3][0]
        a_p = xa[p][1]
        reconstructed = (x_prev.astype(np.uint64) - a_p.astype(np.uint64)) & RING_MASK
        assert reconstructed.astype(np.int64).tolist() == secret.tolist()


def test_start_raises_when_metadata_file_missing(tmp_path):
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        executable_dir=str(tmp_path / "executables"),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="missing build metadata"):
        import asyncio

        asyncio.run(backend.start([]))


def test_start_raises_on_bitlength_mismatch(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir, bitlength=32, frac_bits=13)
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="fixed_point config mismatch"):
        import asyncio

        asyncio.run(backend.start([]))


def test_start_raises_on_frac_bits_mismatch(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir, bitlength=64, frac_bits=8)
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="fixed_point config mismatch"):
        import asyncio

        asyncio.run(backend.start([]))


def test_start_raises_when_executable_missing(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir)
    (executable_dir / "run-P0.o").unlink()
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="missing compiled executable"):
        import asyncio

        asyncio.run(backend.start([]))


def test_start_succeeds_and_sorts_peers_by_party_index(tmp_path):
    import asyncio

    backend = _make_backend(tmp_path)
    peers = [
        PartyEndpoint(party_index=2, host="party2", port=1),
        PartyEndpoint(party_index=1, host="party1", port=1),
    ]
    asyncio.run(backend.start(peers))
    assert [p.host for p in backend._peer_endpoints] == ["party1", "party2"]


@pytest.mark.asyncio
async def test_run_aggregation_round_resolves_peer_hostnames_to_ip_addresses(tmp_path):
    # Regression test: hpmpc's own C++ socket layer parses its peer-IP CLI
    # args as literal dotted-quad addresses with no DNS resolution of its
    # own -- a Docker Compose service name like "secure_agg_party1" crashes
    # it outright ("Invalid address: secure_agg_party1", hit during the
    # Phase 4 real end-to-end run). backend_hpmpc.py must resolve peer
    # hostnames to IPs itself before invoking the binary.
    backend = _make_backend(tmp_path)
    await backend.start(
        [
            PartyEndpoint(party_index=1, host="some-service-name", port=1),
            PartyEndpoint(party_index=2, host="another-service-name", port=1),
        ]
    )

    captured_args = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_args["args"] = args
        env = kwargs["env"]
        with open(env["SECURE_AGG_OUTPUT_FILE"], "wb") as f:
            f.write(struct.pack("<I", 0))
        return _FakeProcess()

    with patch("socket.gethostbyname", side_effect=lambda host: f"10.0.0.{hash(host) % 250}"):
        with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            await backend.run_aggregation_round(
                round_id="session1:0", shares={}, tensor_specs={}, timeout_s=5
            )

    executable_path, *peer_ip_args = captured_args["args"]
    for ip in peer_ip_args:
        assert ip.startswith("10.0.0."), f"expected a resolved IP, got {ip!r}"
    assert "some-service-name" not in peer_ip_args
    assert "another-service-name" not in peer_ip_args


@pytest.mark.asyncio
async def test_run_aggregation_round_writes_correct_input_file_and_decodes_output(tmp_path):
    backend = _make_backend(tmp_path)
    await backend.start([])

    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(1)

    plaintext = np.array([1.5, -2.0], dtype=np.float64)
    fixedpoint = codec.encode(plaintext)
    party_shares = {s.party_index: s for s in scheme.share(fixedpoint, rng)}
    shares = {"client1": {"w": party_shares[0]}}
    tensor_specs = {"w": TensorSpec(layer_name="w", shape=(2,), dtype="float32")}

    written_input_path = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        env = kwargs["env"]
        written_input_path["path"] = env["SECURE_AGG_INPUT_FILE"]
        output_path = env["SECURE_AGG_OUTPUT_FILE"]

        # Simulate the real binary: read the (already-summed, single-client
        # here) x/a pair per element and "reveal" it by just decoding what a
        # single-share reconstruction would give -- since there's only one
        # client, the party's own share IS the full reveal input for this
        # test's purposes; write back the known correct plaintext instead of
        # re-deriving cryptographically, since this test is about the FILE
        # FORMAT glue, not the cross-process protocol (covered by the
        # slow_hpmpc_build e2e tier).
        with open(written_input_path["path"], "rb") as f:
            (num_elements,) = struct.unpack("<I", f.read(4))
            f.read(16 * num_elements)  # consume the (x, a) pairs

        raw_fixedpoint = codec.encode(plaintext)
        with open(output_path, "wb") as f:
            f.write(struct.pack("<I", len(raw_fixedpoint)))
            f.write(struct.pack(f"<{len(raw_fixedpoint)}Q", *(int(v) & 0xFFFFFFFFFFFFFFFF for v in raw_fixedpoint)))

        return _FakeProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        result = await backend.run_aggregation_round(
            round_id="session1:0", shares=shares, tensor_specs=tensor_specs, timeout_s=5
        )

    assert set(result.keys()) == {"w"}
    assert np.allclose(result["w"].numpy(), plaintext, atol=1e-3)
    # input/output files get cleaned up after a successful round
    assert not tmp_path.joinpath("tmp").exists() or not any(tmp_path.joinpath("tmp").iterdir())


@pytest.mark.asyncio
async def test_run_aggregation_round_raises_on_nonzero_exit_code(tmp_path):
    backend = _make_backend(tmp_path)
    await backend.start([])

    class _FailingProcess:
        returncode = 1

        async def communicate(self):
            return b"party crashed", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FailingProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        with pytest.raises(RuntimeError, match="exited with code 1"):
            await backend.run_aggregation_round(
                round_id="session1:0",
                shares={},
                tensor_specs={},
                timeout_s=5,
            )


@pytest.mark.asyncio
async def test_run_aggregation_round_raises_timeout_error_and_kills_process(tmp_path):
    import asyncio as asyncio_module

    backend = _make_backend(tmp_path)
    await backend.start([])

    killed = {"called": False}

    class _HangingProcess:
        returncode = None

        async def communicate(self):
            await asyncio_module.sleep(100)

        def kill(self):
            killed["called"] = True

        async def wait(self):
            return None

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _HangingProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        with pytest.raises(TimeoutError):
            await backend.run_aggregation_round(
                round_id="session1:0",
                shares={},
                tensor_specs={},
                timeout_s=0.05,
            )

    assert killed["called"]
