import pickle
from unittest.mock import MagicMock, patch

import torch

import pytest

from client import client_secure_agg_manager
from server.secure_agg.constants import DATASET_SIZE_LAYER_NAME
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.sharing_schemes.base import PartyShare
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme

pytestmark = pytest.mark.unit

PARTY_ENDPOINTS = [
    {"party_index": 0, "host": "party0", "port": 1},
    {"party_index": 1, "host": "party1", "port": 1},
    {"party_index": 2, "host": "party2", "port": 1},
]


def _fake_stub_capturing(captured):
    class _FakeStub:
        def __init__(self, channel):
            pass

        def SubmitShare(self, request, timeout):
            captured.append(request)
            ack = MagicMock()
            ack.accepted = True
            return ack

    return _FakeStub


def _share_and_submit(dataset_size, captured, weighting_mode="client_side"):
    state_dict = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    with patch("grpc.insecure_channel", return_value=MagicMock(close=lambda: None)):
        with patch(
            "proto.secure_agg_pb2_grpc.SecureAggPartyServiceStub",
            side_effect=_fake_stub_capturing(captured),
        ):
            client_secure_agg_manager.share_and_submit(
                client_id="c1",
                session_id="s1",
                round_id="s1:0",
                state_dict=state_dict,
                dataset_size=dataset_size,
                sharing_scheme_name="replicated3pc",
                fixed_point_config={"bitlength": 64, "frac_bits": 13},
                party_endpoints=PARTY_ENDPOINTS,
                submission_timeout_s=5,
                weighting_mode=weighting_mode,
            )


def test_share_and_submit_shares_the_dataset_size_as_a_reserved_pseudo_layer():
    captured = []
    _share_and_submit(dataset_size=42, captured=captured)

    assert len(captured) == 3  # one SubmitShareRequest per party
    for request in captured:
        layer_names = {share.layer_name for share in request.shares}
        assert "w" in layer_names
        assert DATASET_SIZE_LAYER_NAME in layer_names

        dataset_size_share = next(
            s for s in request.shares if s.layer_name == DATASET_SIZE_LAYER_NAME
        )
        assert list(dataset_size_share.shape) == [1]
        assert dataset_size_share.dtype == "float64"


def test_share_and_submit_dataset_size_shares_reconstruct_to_the_raw_value():
    captured = []
    _share_and_submit(dataset_size=777, captured=captured)

    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)

    shares_by_party = {}
    for party_index, request in enumerate(captured):
        dataset_size_share = next(
            s for s in request.shares if s.layer_name == DATASET_SIZE_LAYER_NAME
        )
        shares_by_party[party_index] = PartyShare(
            party_index=party_index,
            payload=pickle.loads(dataset_size_share.share_payload),
        )

    reconstructed = scheme.reconstruct(shares_by_party)
    decoded = codec.decode(reconstructed)

    # This is the RAW dataset size, not pre-weighted by itself -- unlike
    # every real model layer, which IS pre-weighted (see "w"'s share below).
    assert round(float(decoded[0])) == 777


def test_share_and_submit_real_layers_are_still_pre_weighted_by_dataset_size():
    captured = []
    _share_and_submit(dataset_size=10, captured=captured)

    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)

    shares_by_party = {}
    for party_index, request in enumerate(captured):
        w_share = next(s for s in request.shares if s.layer_name == "w")
        shares_by_party[party_index] = PartyShare(
            party_index=party_index, payload=pickle.loads(w_share.share_payload)
        )

    reconstructed = scheme.reconstruct(shares_by_party)
    decoded = codec.decode(reconstructed)

    # state_dict["w"] == [1.0, 2.0], dataset_size == 10 -> pre-weighted == [10.0, 20.0]
    assert decoded.tolist() == pytest.approx([10.0, 20.0], abs=1e-3)


def test_share_and_submit_mpc_product_mode_does_not_pre_weight_real_layers():
    # weighting_mode="mpc_product": the party cluster computes
    # weight_i*dataset_size_i itself (see mult_fedavg_secure_aggregation.hpp
    # and backend_hpmpc.py's _run_mpc_product_round), so the client must
    # share the RAW update, unlike the default "client_side" mode.
    captured = []
    _share_and_submit(dataset_size=10, captured=captured, weighting_mode="mpc_product")

    codec = FixedPointCodec(bitlength=64, frac_bits=13)
    scheme = Replicated3PCScheme(bitlength=64)

    shares_by_party = {}
    for party_index, request in enumerate(captured):
        w_share = next(s for s in request.shares if s.layer_name == "w")
        shares_by_party[party_index] = PartyShare(
            party_index=party_index, payload=pickle.loads(w_share.share_payload)
        )

    reconstructed = scheme.reconstruct(shares_by_party)
    decoded = codec.decode(reconstructed)

    # state_dict["w"] == [1.0, 2.0], RAW (not pre-weighted by dataset_size=10)
    assert decoded.tolist() == pytest.approx([1.0, 2.0], abs=1e-3)


def test_share_and_submit_rejects_unsupported_weighting_mode():
    with pytest.raises(ValueError, match="does not support weighting_mode"):
        _share_and_submit(dataset_size=10, captured=[], weighting_mode="bogus")
