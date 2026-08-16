import asyncio
import json
import os
from argparse import ArgumentParser
from concurrent import futures

import grpc

import proto.secure_agg_pb2_grpc as secure_agg_pb2_grpc
from server.secure_agg.backends.base import PartyEndpoint
from server.secure_agg.fixed_point_codec import FixedPointCodec
from server.secure_agg.load_backend import load_backend
from server.secure_agg.load_sharing_scheme import load_sharing_scheme
from server.secure_agg.party_server import SecureAggPartyServicer
from server.server_file_manager import OpenYaML
from server.server_state_manager import StateManager


def _apply_env_overrides(config: dict) -> dict:
    """Lets docker-compose (or any process supervisor) run the SAME checked-
    in config file for all N party containers and just vary a handful of
    env vars per container, mirroring the pattern docker/server_entrypoint.sh
    and docker/client_entrypoint.sh already use -- except done in Python
    rather than sed, since `peers` is a list and sed doesn't edit YAML lists
    cleanly. Unset env vars leave the checked-in config value untouched."""
    if os.environ.get("PARTY_INDEX") is not None:
        config["party_index"] = int(os.environ["PARTY_INDEX"])
    if os.environ.get("NUM_PARTIES") is not None:
        config["num_parties"] = int(os.environ["NUM_PARTIES"])
    if os.environ.get("SHARING_SCHEME") is not None:
        config["sharing_scheme"] = os.environ["SHARING_SCHEME"]
    if os.environ.get("BIND_PORT") is not None:
        config["bind_port"] = int(os.environ["BIND_PORT"])
    if os.environ.get("BACKEND_PORT") is not None:
        config["backend_port"] = int(os.environ["BACKEND_PORT"])
    if os.environ.get("PEERS_JSON") is not None:
        config["peers"] = json.loads(os.environ["PEERS_JSON"])
    if os.environ.get("BACKEND_TYPE") is not None:
        config["backend"]["type"] = os.environ["BACKEND_TYPE"]
    if os.environ.get("HPMPC_PROTOCOL") is not None:
        config["backend"]["hpmpc"]["protocol"] = int(os.environ["HPMPC_PROTOCOL"])
    if os.environ.get("HPMPC_EXECUTABLE_DIR") is not None:
        config["backend"]["hpmpc"]["executable_dir"] = os.environ["HPMPC_EXECUTABLE_DIR"]
    if os.environ.get("HPMPC_WEIGHTING_MODE") is not None:
        config["backend"]["hpmpc"]["weighting_mode"] = os.environ["HPMPC_WEIGHTING_MODE"]
    return config


def _construct_backend(backend_module, backend_type, backend_config, party_index, num_parties, scheme, codec, bind_host, backend_port):
    """Each backend genuinely needs different constructor arguments (the
    simulator manages its own bind_host/bind_port for its peer gRPC
    service; hpmpc has no such thing -- it manages sockets internally and
    instead needs an executable_dir/tmp_dir) -- see backends/base.py's
    docstring on why the ABC's *methods* are uniform but construction isn't
    forced to be. This is the one place that dispatch lives; everything
    above the backend boundary (party_server.py, aggregator_secure_mpc.py,
    the client) stays backend-agnostic. Adding a third backend means adding
    one more branch here, not touching anything else.
    """
    per_backend_config = backend_config.get(backend_type, {})
    if backend_type == "simulator":
        return backend_module.BACKEND_CLASS(
            party_index=party_index,
            num_parties=num_parties,
            scheme=scheme,
            codec=codec,
            bind_host=per_backend_config.get("bind_host", bind_host),
            bind_port=backend_port,
        )
    if backend_type == "hpmpc":
        return backend_module.BACKEND_CLASS(
            party_index=party_index,
            num_parties=num_parties,
            codec=codec,
            # Required, no silent default -- see docs/secure_aggregation/
            # hpmpc_backend.md. A missing value is a config bug, not
            # something to paper over with protocol=2.
            protocol=per_backend_config["protocol"],
            executable_dir=per_backend_config["executable_dir"],
            tmp_dir=per_backend_config.get("tmp_dir", "/tmp/secure_agg"),
            log_stdout=per_backend_config.get("log_stdout", False),
            weighting_mode=per_backend_config.get("weighting_mode", "client_side"),
        )
    raise ValueError(f"unknown backend type: {backend_type}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./config/secure_agg_party_config.yaml",
        help="Path to this party's config YAML.",
    )
    args = parser.parse_args()

    config = _apply_env_overrides(OpenYaML(args.config))

    party_index = config["party_index"]
    num_parties = config["num_parties"]
    bind_host = config.get("bind_host", "0.0.0.0")

    scheme_module = load_sharing_scheme(f"party{party_index}", config["sharing_scheme"])
    scheme = scheme_module.SCHEME_CLASS(bitlength=config["fixed_point"]["bitlength"])
    codec = FixedPointCodec(
        bitlength=config["fixed_point"]["bitlength"],
        frac_bits=config["fixed_point"]["frac_bits"],
    )

    backend_type = config["backend"]["type"]
    backend_module = load_backend(f"party{party_index}", backend_type)
    backend = _construct_backend(
        backend_module,
        backend_type,
        config["backend"],
        party_index,
        num_parties,
        scheme,
        codec,
        bind_host,
        config["backend_port"],
    )

    peer_endpoints = [
        PartyEndpoint(party_index=peer["party_index"], host=peer["host"], port=peer["backend_port"])
        for peer in config["peers"]
    ]
    asyncio.run(backend.start(peer_endpoints))

    share_state = StateManager(
        loc=config["state"]["state_location"],
        name=f"secure_agg_party{party_index}_shares",
        host=config["state"].get("state_hostname"),
        port=config["state"].get("state_port"),
    )

    servicer = SecureAggPartyServicer(
        party_index=party_index,
        sharing_scheme_name=config["sharing_scheme"],
        share_state=share_state,
        backend=backend,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.get("grpc_workers", 8)))
    secure_agg_pb2_grpc.add_SecureAggPartyServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{bind_host}:{config['bind_port']}")
    server.start()
    print(
        f"flo_secure_agg_party:: party {party_index}/{num_parties} listening on "
        f"{bind_host}:{config['bind_port']} (backend={config['backend']['type']}, "
        f"backend_port={config['backend_port']})"
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()
