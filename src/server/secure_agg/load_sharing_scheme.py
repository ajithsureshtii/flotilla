import importlib

from utils.logger import FedLogger


def load_sharing_scheme(id, scheme):
    """Dynamically loads server/secure_agg/sharing_schemes/<scheme>.py, the
    same importlib-dispatch-by-config-string idiom as
    server/load_aggregator.py and server_state_manager.py's backend loader.
    Every sharing-scheme module must expose its concrete
    SecretSharingScheme subclass as a module-level `SCHEME_CLASS` attribute
    (see sharing_schemes/replicated3pc.py) so callers can do
    `load_sharing_scheme(id, "replicated3pc").SCHEME_CLASS(bitlength=64)`.
    """
    logger = FedLogger(id=id, loggername="SHARING_SCHEME_LOADER")

    module_name = f"server.secure_agg.sharing_schemes.{scheme}"
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "fedserver.secure_agg.sharing_scheme.module",
            f"Sharing scheme module name:,{module_name}",
        )
        return module
    except ImportError:
        logger.error(
            "fedserver.secure_agg.sharing_scheme.invalid_module",
            f"Could not import the module ,{module_name}",
        )
