import pytest

from server.secure_agg.load_sharing_scheme import load_sharing_scheme
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme

pytestmark = pytest.mark.unit


def test_loads_replicated3pc_by_name():
    module = load_sharing_scheme("test-session", "replicated3pc")
    assert module.SCHEME_CLASS is Replicated3PCScheme


def test_loads_shamir_stub_by_name():
    module = load_sharing_scheme("test-session", "shamir_stub")
    assert module.SCHEME_CLASS.scheme_id == "shamir_stub"


def test_loads_tetrad4pc_by_name():
    module = load_sharing_scheme("test-session", "tetrad4pc")
    assert module.SCHEME_CLASS.scheme_id == "tetrad4pc"
    assert module.SCHEME_CLASS.num_parties == 4


def test_unknown_scheme_name_returns_none_without_raising():
    result = load_sharing_scheme("test-session", "does_not_exist")
    assert result is None
