import pytest

import flo_secure_agg_party as party_entrypoint

pytestmark = pytest.mark.unit


class _FakeSimulatorBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeHpmpcBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeModule:
    def __init__(self, backend_class):
        self.BACKEND_CLASS = backend_class


def test_construct_backend_simulator_uses_bind_host_and_bind_port():
    module = _FakeModule(_FakeSimulatorBackend)

    backend = party_entrypoint._construct_backend(
        module,
        "simulator",
        {"simulator": {"bind_host": "1.2.3.4"}},
        party_index=0,
        num_parties=3,
        scheme="scheme-obj",
        codec="codec-obj",
        bind_host="0.0.0.0",
        backend_port=51100,
    )

    assert isinstance(backend, _FakeSimulatorBackend)
    assert backend.kwargs == {
        "party_index": 0,
        "num_parties": 3,
        "scheme": "scheme-obj",
        "codec": "codec-obj",
        "bind_host": "1.2.3.4",
        "bind_port": 51100,
    }


def test_construct_backend_simulator_falls_back_to_top_level_bind_host():
    module = _FakeModule(_FakeSimulatorBackend)

    backend = party_entrypoint._construct_backend(
        module,
        "simulator",
        {},  # no per-backend simulator config at all
        party_index=1,
        num_parties=3,
        scheme="scheme-obj",
        codec="codec-obj",
        bind_host="0.0.0.0",
        backend_port=51101,
    )

    assert backend.kwargs["bind_host"] == "0.0.0.0"


def test_construct_backend_hpmpc_uses_executable_dir_and_tmp_dir():
    module = _FakeModule(_FakeHpmpcBackend)

    backend = party_entrypoint._construct_backend(
        module,
        "hpmpc",
        {"hpmpc": {"executable_dir": "/opt/hpmpc/executables", "tmp_dir": "/tmp/secure_agg"}},
        party_index=2,
        num_parties=3,
        scheme="scheme-obj",
        codec="codec-obj",
        bind_host="0.0.0.0",
        backend_port=51102,
    )

    assert isinstance(backend, _FakeHpmpcBackend)
    assert backend.kwargs == {
        "party_index": 2,
        "num_parties": 3,
        "codec": "codec-obj",
        "executable_dir": "/opt/hpmpc/executables",
        "tmp_dir": "/tmp/secure_agg",
    }
    # hpmpc doesn't manage its own bind_host/bind_port -- must not be passed
    assert "bind_host" not in backend.kwargs
    assert "bind_port" not in backend.kwargs


def test_construct_backend_hpmpc_defaults_tmp_dir():
    module = _FakeModule(_FakeHpmpcBackend)

    backend = party_entrypoint._construct_backend(
        module,
        "hpmpc",
        {"hpmpc": {"executable_dir": "/opt/hpmpc/executables"}},
        party_index=0,
        num_parties=3,
        scheme="scheme-obj",
        codec="codec-obj",
        bind_host="0.0.0.0",
        backend_port=51100,
    )

    assert backend.kwargs["tmp_dir"] == "/tmp/secure_agg"


def test_construct_backend_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown backend type"):
        party_entrypoint._construct_backend(
            _FakeModule(_FakeSimulatorBackend),
            "not_a_real_backend",
            {},
            party_index=0,
            num_parties=3,
            scheme=None,
            codec=None,
            bind_host="0.0.0.0",
            backend_port=0,
        )


def test_apply_env_overrides_sets_backend_type(monkeypatch):
    monkeypatch.setenv("BACKEND_TYPE", "hpmpc")
    config = {"backend": {"type": "simulator"}}

    result = party_entrypoint._apply_env_overrides(config)

    assert result["backend"]["type"] == "hpmpc"


def test_apply_env_overrides_leaves_backend_type_untouched_when_unset(monkeypatch):
    monkeypatch.delenv("BACKEND_TYPE", raising=False)
    config = {"backend": {"type": "simulator"}}

    result = party_entrypoint._apply_env_overrides(config)

    assert result["backend"]["type"] == "simulator"
