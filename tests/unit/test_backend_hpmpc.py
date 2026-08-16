import asyncio
import json
import struct
from unittest.mock import patch

import numpy as np
import pytest

from server.secure_agg.backends.backend_hpmpc import (
    HpmpcBackend,
    _own_fragment,
    _to_hpmpc_tetrad,
    _to_hpmpc_trio,
    _to_hpmpc_xa,
)
from server.secure_agg.backends.base import PartyEndpoint, TensorSpec
from server.secure_agg.constants import DATASET_SIZE_LAYER_NAME
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme
from server.secure_agg.sharing_schemes.tetrad4pc import Tetrad4PCScheme

pytestmark = pytest.mark.unit

RING_MASK = np.uint64((1 << 64) - 1)


def _write_metadata(executable_dir, bitlength=64, frac_bits=13, protocol=2, num_parties=3):
    executable_dir.mkdir(parents=True, exist_ok=True)
    (executable_dir / "fedavg_secure_aggregation.build_metadata.json").write_text(
        json.dumps(
            {
                "bitlength": bitlength,
                "frac_bits": frac_bits,
                "protocol": protocol,
                "function_identifier": 90,
            }
        )
    )
    for party in range(num_parties):
        (executable_dir / f"run-P{party}.o").touch()


def _make_backend(tmp_path, party_index=0, bitlength=64, frac_bits=13, protocol=2, num_parties=3):
    executable_dir = tmp_path / "executables"
    _write_metadata(
        executable_dir, bitlength=bitlength, frac_bits=frac_bits, protocol=protocol, num_parties=num_parties
    )
    return HpmpcBackend(
        party_index=party_index,
        num_parties=num_parties,
        codec=FixedPointCodec(bitlength=bitlength, frac_bits=frac_bits),
        protocol=protocol,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )


@pytest.mark.parametrize(
    "protocol,num_parties,should_raise",
    [
        (2, 3, False),  # correct: Replicated 3PC needs exactly 3 parties
        (2, 4, True),
        (2, 2, True),
        (5, 3, False),  # correct: Trio also needs exactly 3 parties
        (5, 4, True),
        (5, 2, True),
        (8, 4, False),  # correct: Tetrad needs exactly 4 parties
        (8, 3, True),
        (8, 5, True),
    ],
)
def test_num_parties_validation_is_protocol_aware(tmp_path, protocol, num_parties, should_raise):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir, protocol=protocol, num_parties=max(num_parties, 3))
    kwargs = dict(
        party_index=0,
        num_parties=num_parties,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=protocol,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    if should_raise:
        with pytest.raises(ValueError):
            HpmpcBackend(**kwargs)
    else:
        HpmpcBackend(**kwargs)  # should not raise


def test_rejects_unsupported_protocol_number(tmp_path):
    with pytest.raises(ValueError, match="does not support protocol"):
        HpmpcBackend(
            party_index=0,
            num_parties=3,
            codec=FixedPointCodec(bitlength=64, frac_bits=13),
            protocol=99,
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


def test_to_hpmpc_trio_matches_hand_derived_formula():
    # party 0: p1 = -c_j, p2 = -c_j1; parties 1/2: p1 = c_j + c_j1, p2 = 0 --
    # see docs/secure_aggregation/hpmpc_backend.md for the derivation.
    c_j = np.array([3, 100], dtype=np.uint64)
    c_j1 = np.array([4, 5], dtype=np.uint64)

    p1_0, p2_0 = _to_hpmpc_trio(c_j, c_j1, party_index=0, mask=RING_MASK)
    assert p1_0.tolist() == [(-3) & int(RING_MASK), (-100) & int(RING_MASK)]
    assert p2_0.tolist() == [(-4) & int(RING_MASK), (-5) & int(RING_MASK)]

    for party_index in (1, 2):
        p1, p2 = _to_hpmpc_trio(c_j, c_j1, party_index=party_index, mask=RING_MASK)
        assert p1.tolist() == [3 + 4, 100 + 5]
        assert p2.tolist() == [0, 0]


def test_to_hpmpc_trio_round_trips_through_replicated3pc_reconstruct():
    # Sanity: converting all 3 parties' shares of the SAME secret and
    # re-deriving via Trio's reveal invariants (secret = P2.p1 - P0.p2 =
    # P1.p1 - P0.p1) recovers it -- traced directly from
    # oecl-P_{0,1,2}_template.hpp's prepare_reveal_to_all/complete_Reveal,
    # see hpmpc_backend.md.
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(0)
    secret = np.array([12345], dtype=np.int64)
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    trio = {}
    for j in range(3):
        c_j, c_j1 = shares[j].payload
        trio[j] = _to_hpmpc_trio(c_j, c_j1, party_index=j, mask=RING_MASK)

    reveal_via_p2_p0 = (trio[2][0].astype(np.uint64) - trio[0][1].astype(np.uint64)) & RING_MASK
    reveal_via_p1_p0 = (trio[1][0].astype(np.uint64) - trio[0][0].astype(np.uint64)) & RING_MASK
    assert reveal_via_p2_p0.astype(np.int64).tolist() == secret.tolist()
    assert reveal_via_p1_p0.astype(np.int64).tolist() == secret.tolist()


def test_to_hpmpc_trio_cross_client_summing_is_homomorphic():
    # The property backend_hpmpc.py's run_aggregation_round actually relies
    # on: summing two independent clients' converted (p1, p2) pairs
    # elementwise (mod ring) yields a valid Trio share of the SUM of their
    # secrets -- confirmed numerically during development, see
    # hpmpc_backend.md.
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(1)
    secret_a = np.array([10], dtype=np.int64)
    secret_b = np.array([20], dtype=np.int64)
    shares_a = {s.party_index: s for s in scheme.share(secret_a, rng)}
    shares_b = {s.party_index: s for s in scheme.share(secret_b, rng)}

    summed = {}
    for j in range(3):
        ca_j, ca_j1 = shares_a[j].payload
        cb_j, cb_j1 = shares_b[j].payload
        p1_a, p2_a = _to_hpmpc_trio(ca_j, ca_j1, party_index=j, mask=RING_MASK)
        p1_b, p2_b = _to_hpmpc_trio(cb_j, cb_j1, party_index=j, mask=RING_MASK)
        summed[j] = (
            (p1_a.astype(np.uint64) + p1_b.astype(np.uint64)) & RING_MASK,
            (p2_a.astype(np.uint64) + p2_b.astype(np.uint64)) & RING_MASK,
        )

    reveal = (summed[2][0] - summed[0][1]) & RING_MASK
    assert reveal.astype(np.int64).tolist() == (secret_a + secret_b).tolist()


def test_to_hpmpc_tetrad_matches_hand_derived_formula():
    # _to_hpmpc_tetrad is a near-passthrough: tetrad4pc.py's share() already
    # produces values shaped for hpmpc's native Tetrad{0,1,2,3}_Share
    # constructors directly (mv, lambda_a, lambda_b for P0/1/2; lambda1,
    # lambda2, lambda3 for P3) -- see docs/secure_aggregation/hpmpc_backend.md.
    f0 = np.array([3, 100], dtype=np.uint64)
    f1 = np.array([4, 5], dtype=np.uint64)
    f2 = np.array([6, 7], dtype=np.uint64)

    out0, out1, out2 = _to_hpmpc_tetrad((f0, f1, f2), RING_MASK)

    assert out0.tolist() == [3, 100]
    assert out1.tolist() == [4, 5]
    assert out2.tolist() == [6, 7]


def test_to_hpmpc_tetrad_raises_on_wrong_field_count():
    # The most likely real-world cause of a malformed payload here is a
    # sharing_scheme/protocol config mismatch (e.g. protocol=8 paired with
    # sharing_scheme=replicated3pc, which produces 2-field payloads) -- this
    # should fail with a clear, actionable message, not a confusing shape
    # error several lines later.
    with pytest.raises(ValueError, match="requires a 3-field share payload"):
        _to_hpmpc_tetrad((np.array([1]), np.array([2])), RING_MASK)


def test_to_hpmpc_tetrad_round_trips_through_tetrad4pc_reconstruct():
    # Sanity: converting all 4 parties' shares of the SAME secret and
    # re-deriving via Tetrad's reveal invariant (mv - lambda1 - lambda2 -
    # lambda3, mv from any of P0/1/2, all 3 lambdas from P3) recovers it --
    # traced directly from Tetrad-P_{0,1,2,3}_template.hpp, see
    # hpmpc_backend.md.
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(0)
    secret = np.array([12345], dtype=np.int64)
    shares = {s.party_index: s for s in scheme.share(secret, rng)}

    tetrad = {j: _to_hpmpc_tetrad(shares[j].payload, RING_MASK) for j in range(4)}

    mv = tetrad[0][0]
    lambda1, lambda2, lambda3 = tetrad[3]
    reconstructed = (mv.astype(np.uint64) - lambda1.astype(np.uint64) - lambda2.astype(np.uint64) - lambda3.astype(np.uint64)) & RING_MASK
    assert reconstructed.astype(np.int64).tolist() == secret.tolist()


def test_to_hpmpc_tetrad_cross_client_summing_is_homomorphic():
    # The property backend_hpmpc.py's run_aggregation_round actually relies
    # on: summing two independent clients' converted 3-field tuples
    # elementwise (mod ring) yields a valid Tetrad share of the SUM of their
    # secrets -- confirmed numerically during development, see
    # hpmpc_backend.md.
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(1)
    secret_a = np.array([10], dtype=np.int64)
    secret_b = np.array([20], dtype=np.int64)
    shares_a = {s.party_index: s for s in scheme.share(secret_a, rng)}
    shares_b = {s.party_index: s for s in scheme.share(secret_b, rng)}

    summed = {}
    for j in range(4):
        fields_a = _to_hpmpc_tetrad(shares_a[j].payload, RING_MASK)
        fields_b = _to_hpmpc_tetrad(shares_b[j].payload, RING_MASK)
        summed[j] = tuple((a.astype(np.uint64) + b.astype(np.uint64)) & RING_MASK for a, b in zip(fields_a, fields_b))

    mv = summed[0][0]
    lambda1, lambda2, lambda3 = summed[3]
    reveal = (mv - lambda1 - lambda2 - lambda3) & RING_MASK
    assert reveal.astype(np.int64).tolist() == (secret_a + secret_b).tolist()


def test_start_raises_when_metadata_file_missing(tmp_path):
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=2,
        executable_dir=str(tmp_path / "executables"),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="missing build metadata"):
        asyncio.run(backend.start([]))


def test_start_raises_on_protocol_mismatch(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir, protocol=5)  # compiled for a different protocol
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=2,  # this party is configured for protocol 2
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="protocol config mismatch"):
        asyncio.run(backend.start([]))


def test_start_raises_on_bitlength_mismatch(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir, bitlength=32, frac_bits=13)
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=2,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="fixed_point config mismatch"):
        asyncio.run(backend.start([]))


def test_start_raises_on_frac_bits_mismatch(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir, bitlength=64, frac_bits=8)
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=2,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="fixed_point config mismatch"):
        asyncio.run(backend.start([]))


def test_start_raises_when_executable_missing(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir)
    (executable_dir / "run-P0.o").unlink()
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=2,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
    )
    with pytest.raises(RuntimeError, match="missing compiled executable"):
        asyncio.run(backend.start([]))


def test_start_succeeds_and_sorts_peers_by_party_index(tmp_path):
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
async def test_run_aggregation_round_writes_correct_trio_input_file_for_party_0(tmp_path):
    # Verifies the packed input file's bytes actually match _to_hpmpc_trio's
    # per-party-0 conversion (p1=-c_j, p2=-c_j1) -- not just that it's the
    # right byte count, since Trio's asymmetric-per-role layout is exactly
    # the new thing this test needs to catch a copy-paste mistake in.
    backend = _make_backend(tmp_path, party_index=0, protocol=5)
    await backend.start([])

    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(2)

    plaintext = np.array([1.5, -2.0], dtype=np.float64)
    fixedpoint = codec.encode(plaintext)
    party_shares = {s.party_index: s for s in scheme.share(fixedpoint, rng)}
    shares = {"client1": {"w": party_shares[0]}}
    tensor_specs = {"w": TensorSpec(layer_name="w", shape=(2,), dtype="float32")}

    c_j, c_j1 = party_shares[0].payload
    expected_p1, expected_p2 = _to_hpmpc_trio(
        np.asarray(c_j).reshape(-1), np.asarray(c_j1).reshape(-1), party_index=0, mask=RING_MASK
    )

    captured = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        env = kwargs["env"]
        with open(env["SECURE_AGG_INPUT_FILE"], "rb") as f:
            (num_elements,) = struct.unpack("<I", f.read(4))
            rows = [struct.unpack("<QQ", f.read(16)) for _ in range(num_elements)]
        captured["rows"] = rows
        # Output element count must match tensor_specs ("w" has 2 elements)
        # -- this test only cares about the INPUT file's bytes, so the
        # output values themselves are arbitrary placeholders.
        with open(env["SECURE_AGG_OUTPUT_FILE"], "wb") as f:
            f.write(struct.pack("<I", num_elements))
            f.write(struct.pack(f"<{num_elements}Q", *([0] * num_elements)))
        return _FakeProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        await backend.run_aggregation_round(
            round_id="session1:0", shares=shares, tensor_specs=tensor_specs, timeout_s=5
        )

    assert captured["rows"] == list(zip(expected_p1.tolist(), expected_p2.tolist()))


@pytest.mark.asyncio
async def test_run_aggregation_round_writes_correct_tetrad_input_file_for_party_0(tmp_path):
    # Verifies the packed input file's bytes actually match _to_hpmpc_tetrad's
    # 3-field passthrough for party 0's (mv, lambda_a, lambda_b) role -- not
    # just that it's the right byte count, mirroring the Trio file-packing
    # test above.
    backend = _make_backend(tmp_path, party_index=0, protocol=8, num_parties=4)
    await backend.start([])

    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Tetrad4PCScheme(bitlength=64)
    rng = np.random.default_rng(2)

    plaintext = np.array([1.5, -2.0], dtype=np.float64)
    fixedpoint = codec.encode(plaintext)
    party_shares = {s.party_index: s for s in scheme.share(fixedpoint, rng)}
    shares = {"client1": {"w": party_shares[0]}}
    tensor_specs = {"w": TensorSpec(layer_name="w", shape=(2,), dtype="float32")}

    expected_f0, expected_f1, expected_f2 = _to_hpmpc_tetrad(party_shares[0].payload, RING_MASK)

    captured = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        env = kwargs["env"]
        with open(env["SECURE_AGG_INPUT_FILE"], "rb") as f:
            (num_elements,) = struct.unpack("<I", f.read(4))
            rows = [struct.unpack("<QQQ", f.read(24)) for _ in range(num_elements)]
        captured["rows"] = rows
        with open(env["SECURE_AGG_OUTPUT_FILE"], "wb") as f:
            f.write(struct.pack("<I", num_elements))
            f.write(struct.pack(f"<{num_elements}Q", *([0] * num_elements)))
        return _FakeProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        await backend.run_aggregation_round(
            round_id="session1:0", shares=shares, tensor_specs=tensor_specs, timeout_s=5
        )

    assert captured["rows"] == list(zip(expected_f0.tolist(), expected_f1.tolist(), expected_f2.tolist()))


@pytest.mark.asyncio
async def test_run_aggregation_round_raises_on_malicious_abort_detected_by_hpmpc(tmp_path):
    # This tests HpmpcBackend's own error-surfacing logic ONLY: a nonzero
    # exit code is already surfaced as a RuntimeError regardless of *why*
    # the process exited nonzero, so if/when hpmpc's compare_views()
    # (live_protocol_base.hpp, patched to exit(1) on a detected cheat -- see
    # that file's comment) does detect and abort, this confirms
    # HpmpcBackend correctly reports it as a failed round with zero
    # special-casing needed. It does NOT confirm compare_views() actually
    # fires for real corruptions in this integration -- real corruption
    # testing against a live 4-party Tetrad deployment found it does NOT
    # currently catch at least two real corruption scenarios (see
    # hpmpc_backend.md's "Malicious-security caveat, found empirically").
    backend = _make_backend(tmp_path, protocol=8, num_parties=4)
    await backend.start([])

    class _CheatDetectedProcess:
        returncode = 1

        async def communicate(self):
            return b"Compareviews failed! Aborting.", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _CheatDetectedProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        with pytest.raises(RuntimeError, match="Compareviews failed"):
            await backend.run_aggregation_round(
                round_id="session1:0", shares={}, tensor_specs={}, timeout_s=5
            )


@pytest.mark.asyncio
async def test_run_aggregation_round_logs_stdout_when_log_stdout_enabled(tmp_path):
    executable_dir = tmp_path / "executables"
    _write_metadata(executable_dir)
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=2,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
        log_stdout=True,
    )
    await backend.start([])

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"Time measured to perform computation clock: 0.01s", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        env = kwargs["env"]
        with open(env["SECURE_AGG_OUTPUT_FILE"], "wb") as f:
            f.write(struct.pack("<I", 0))
        return _FakeProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        with patch.object(backend._logger, "info") as mock_info:
            await backend.run_aggregation_round(
                round_id="session1:0", shares={}, tensor_specs={}, timeout_s=5
            )

    mock_info.assert_called_once()
    event_name, message = mock_info.call_args[0]
    assert event_name == "fedparty.hpmpc.round_stdout"
    assert message.startswith("session1:0,")
    assert "Time measured to perform computation clock" in message


@pytest.mark.asyncio
async def test_run_aggregation_round_does_not_log_stdout_by_default(tmp_path):
    backend = _make_backend(tmp_path)  # log_stdout defaults to False
    await backend.start([])

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"some stdout", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        env = kwargs["env"]
        with open(env["SECURE_AGG_OUTPUT_FILE"], "wb") as f:
            f.write(struct.pack("<I", 0))
        return _FakeProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        with patch.object(backend._logger, "info") as mock_info:
            await backend.run_aggregation_round(
                round_id="session1:0", shares={}, tensor_specs={}, timeout_s=5
            )

    mock_info.assert_not_called()


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
    backend = _make_backend(tmp_path)
    await backend.start([])

    killed = {"called": False}

    class _HangingProcess:
        returncode = None

        async def communicate(self):
            await asyncio.sleep(100)

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


# --- weighting_mode="mpc_product" (mult_fedavg) ---------------------------


def _write_mult_metadata(executable_dir, bitlength=64, frac_bits=13, protocol=5, num_parties=3):
    executable_dir.mkdir(parents=True, exist_ok=True)
    (executable_dir / "mult_fedavg_secure_aggregation.build_metadata.json").write_text(
        json.dumps(
            {
                "bitlength": bitlength,
                "frac_bits": frac_bits,
                "protocol": protocol,
                "function_identifier": 91,
            }
        )
    )
    for party in range(num_parties):
        (executable_dir / f"run-P{party}.o").touch()


def _make_mult_backend(tmp_path, party_index=0, bitlength=64, frac_bits=13):
    executable_dir = tmp_path / "executables"
    _write_mult_metadata(executable_dir, bitlength=bitlength, frac_bits=frac_bits)
    return HpmpcBackend(
        party_index=party_index,
        num_parties=3,
        codec=FixedPointCodec(bitlength=bitlength, frac_bits=frac_bits),
        protocol=5,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
        weighting_mode="mpc_product",
    )


def test_rejects_unsupported_weighting_mode(tmp_path):
    with pytest.raises(ValueError, match="does not support weighting_mode"):
        HpmpcBackend(
            party_index=0,
            num_parties=3,
            codec=FixedPointCodec(bitlength=64, frac_bits=13),
            protocol=2,
            executable_dir=str(tmp_path),
            tmp_dir=str(tmp_path / "tmp"),
            weighting_mode="bogus",
        )


@pytest.mark.parametrize("protocol,num_parties", [(2, 3), (8, 4)])
def test_mpc_product_weighting_mode_requires_trio(tmp_path, protocol, num_parties):
    with pytest.raises(ValueError, match="only supported for protocol=5"):
        HpmpcBackend(
            party_index=0,
            num_parties=num_parties,
            codec=FixedPointCodec(bitlength=64, frac_bits=13),
            protocol=protocol,
            executable_dir=str(tmp_path),
            tmp_dir=str(tmp_path / "tmp"),
            weighting_mode="mpc_product",
        )


def test_start_raises_on_function_identifier_mismatch(tmp_path):
    # A party configured for weighting_mode="mpc_product" (expects
    # FUNCTION_IDENTIFIER=91) but pointed at a
    # mult_fedavg_secure_aggregation.build_metadata.json that was
    # (mistakenly) written with the default fedavg program's
    # FUNCTION_IDENTIFIER=90 must fail loudly at start(), not with a
    # confusing file-format mismatch mid-round.
    executable_dir = tmp_path / "executables"
    executable_dir.mkdir(parents=True)
    (executable_dir / "mult_fedavg_secure_aggregation.build_metadata.json").write_text(
        json.dumps({"bitlength": 64, "frac_bits": 13, "protocol": 5, "function_identifier": 90})
    )
    for party in range(3):
        (executable_dir / f"run-P{party}.o").touch()
    backend = HpmpcBackend(
        party_index=0,
        num_parties=3,
        codec=FixedPointCodec(bitlength=64, frac_bits=13),
        protocol=5,
        executable_dir=str(executable_dir),
        tmp_dir=str(tmp_path / "tmp"),
        weighting_mode="mpc_product",
    )
    with pytest.raises(RuntimeError, match="weighting_mode config mismatch"):
        asyncio.run(backend.start([]))


def test_own_fragment_extracts_first_field_only():
    c_j = np.array([3, 100], dtype=np.uint64)
    c_j1 = np.array([999, 999], dtype=np.uint64)  # must be ignored
    result = _own_fragment((c_j, c_j1), RING_MASK)
    assert result.tolist() == [3, 100]


@pytest.mark.asyncio
async def test_run_mpc_product_round_writes_per_client_fragments_and_decodes_product(tmp_path):
    # End-to-end (mocked subprocess) check of the mpc_product file format:
    # 2 clients, one real weight layer + the dataset-size pseudo-layer, RAW
    # (unweighted) values. The fake subprocess reads back party 0's own
    # fragments, asserts they match _own_fragment's extraction directly from
    # the SAME Replicated3PCScheme shares, then writes back a "revealed"
    # output computed independently in plaintext (sum of weight_i *
    # dataset_size_i, encoded at 2x frac_bits -- see
    # mult_fedavg_secure_aggregation.hpp's module docstring for why) so this
    # test also exercises HpmpcBackend's own decode-at-double-scale logic.
    backend = _make_mult_backend(tmp_path, party_index=0)
    await backend.start([])

    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)
    rng = np.random.default_rng(3)

    clients = {
        "client1": {"weights": np.array([1.5, -2.0]), "dataset_size": 100.0},
        "client2": {"weights": np.array([3.0, 4.0]), "dataset_size": 50.0},
    }
    shares = {}
    expected_party0_weight_fragments = {}
    expected_party0_ds_fragment = {}
    for client_id, data in clients.items():
        w_fp = codec.encode(data["weights"])
        ds_fp = codec.encode(np.array([data["dataset_size"]]))
        w_shares = {s.party_index: s for s in scheme.share(w_fp, rng)}
        ds_shares = {s.party_index: s for s in scheme.share(ds_fp, rng)}
        shares[client_id] = {
            "w": w_shares[0],
            DATASET_SIZE_LAYER_NAME: ds_shares[0],
        }
        expected_party0_weight_fragments[client_id] = _own_fragment(w_shares[0].payload, RING_MASK)
        expected_party0_ds_fragment[client_id] = _own_fragment(ds_shares[0].payload, RING_MASK)

    tensor_specs = {
        "w": TensorSpec(layer_name="w", shape=(2,), dtype="float32"),
        DATASET_SIZE_LAYER_NAME: TensorSpec(layer_name=DATASET_SIZE_LAYER_NAME, shape=(1,), dtype="float64"),
    }

    expected_raw_product = np.zeros(2, dtype=np.int64)
    expected_total_ds_fp = 0
    for data in clients.values():
        w_fp = codec.encode(data["weights"]).astype(np.int64)
        ds_fp = int(codec.encode(np.array([data["dataset_size"]]))[0])
        expected_raw_product += w_fp * ds_fp
        expected_total_ds_fp += ds_fp

    captured = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        env = kwargs["env"]
        with open(env["SECURE_AGG_INPUT_FILE"], "rb") as f:
            (num_clients,) = struct.unpack("<I", f.read(4))
            (elements_per_client,) = struct.unpack("<I", f.read(4))
            captured["num_clients"] = num_clients
            captured["elements_per_client"] = elements_per_client
            rows = {}
            for client_id in sorted(clients.keys()):
                weight_fragment = struct.unpack(f"<{elements_per_client}Q", f.read(8 * elements_per_client))
                (ds_fragment,) = struct.unpack("<Q", f.read(8))
                rows[client_id] = (weight_fragment, ds_fragment)
        captured["rows"] = rows

        with open(env["SECURE_AGG_OUTPUT_FILE"], "wb") as f:
            f.write(struct.pack("<I", elements_per_client))
            f.write(
                struct.pack(
                    f"<{elements_per_client}Q",
                    *(int(v) & 0xFFFFFFFFFFFFFFFF for v in expected_raw_product.tolist()),
                )
            )
            f.write(struct.pack("<Q", expected_total_ds_fp & 0xFFFFFFFFFFFFFFFF))
        return _FakeProcess()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        result = await backend.run_aggregation_round(
            round_id="session1:0", shares=shares, tensor_specs=tensor_specs, timeout_s=5
        )

    assert captured["num_clients"] == 2
    assert captured["elements_per_client"] == 2
    for client_id, (weight_fragment, ds_fragment) in captured["rows"].items():
        assert list(weight_fragment) == expected_party0_weight_fragments[client_id].tolist()
        assert ds_fragment == int(expected_party0_ds_fragment[client_id][0])

    assert set(result.keys()) == {"w", DATASET_SIZE_LAYER_NAME}
    expected_product_plaintext = expected_raw_product.astype(np.float64) / float(1 << (2 * 13))
    assert np.allclose(result["w"].numpy(), expected_product_plaintext, atol=1e-3)
    assert result[DATASET_SIZE_LAYER_NAME].item() == pytest.approx(150.0)
    # input/output files get cleaned up after a successful round
    assert not tmp_path.joinpath("tmp").exists() or not any(tmp_path.joinpath("tmp").iterdir())
