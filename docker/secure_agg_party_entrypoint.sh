# Unlike server_entrypoint.sh / client_entrypoint.sh, this does NOT sed-patch
# config/secure_agg_party_config.yaml: the fields that vary per party
# (party_index, ports, peers) include a YAML list (`peers`), which sed
# can't edit safely. flo_secure_agg_party.py instead reads PARTY_INDEX /
# BIND_PORT / BACKEND_PORT / PEERS_JSON env vars directly and overrides the
# checked-in config with them -- see _apply_env_overrides() there.
cat config/secure_agg_party_config.yaml
python3 ./flo_secure_agg_party.py
