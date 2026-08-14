import importlib

from utils.logger import FedLogger


def load_backend(id, backend):
    """Dynamically loads server/secure_agg/backends/backend_<backend>.py, the
    same importlib-dispatch-by-config-string idiom as
    server/load_aggregator.py, server_state_manager.py's backend loader, and
    load_sharing_scheme.py. Every backend module must expose its concrete
    SecureAggregationBackend subclass as a module-level `BACKEND_CLASS`
    attribute (see backends/backend_simulator.py) so callers can do
    `load_backend(id, "simulator").BACKEND_CLASS(party_index=0, ...)`.
    """
    logger = FedLogger(id=id, loggername="BACKEND_LOADER")

    module_name = f"server.secure_agg.backends.backend_{backend}"
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "fedserver.secure_agg.backend.module",
            f"Backend module name:,{module_name}",
        )
        return module
    except ImportError:
        logger.error(
            "fedserver.secure_agg.backend.invalid_module",
            f"Could not import the module ,{module_name}",
        )
