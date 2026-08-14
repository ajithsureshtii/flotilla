# Sample script for running the 3 secure-aggregation party containers
# outside docker-compose (e.g. on 3 genuinely separate machines) -- mirrors
# sample_docker_server_run.sh / sample_docker_client_run.sh's convention of
# a template with <placeholder> values to fill in per deployment. See
# docs/secure_aggregation/topology.md and runbook.md.
#
# All 3 parties use the SAME image; only PARTY_INDEX/BIND_PORT/BACKEND_PORT/
# PEERS_JSON differ -- see flo_secure_agg_party.py's _apply_env_overrides
# and secure_agg_party_config.yaml.

for PARTY_INDEX in 0 1 2; do
    docker run --network flotilla-network \
        --name="secure_agg_party_$PARTY_INDEX" \
        --env PARTY_INDEX=$PARTY_INDEX \
        --env BIND_PORT=<this_party_control_plane_port> \
        --env BACKEND_PORT=<this_party_backend_wire_port> \
        --env BACKEND_TYPE=hpmpc \
        --env PEERS_JSON='[{"party_index":<peer_a_index>,"host":"<peer_a_host>","backend_port":<peer_a_backend_port>},{"party_index":<peer_b_index>,"host":"<peer_b_host>","backend_port":<peer_b_backend_port>}]' \
        --memory 4096m \
        -p <this_party_control_plane_port>:<this_party_control_plane_port> \
        -dti secure-agg-party:latest
done
