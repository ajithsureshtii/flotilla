# Proto

The `Proto` directory contains all the necessary files to generate the protocol buffers for the gRPC calls. The protocol buffer files can be generated using the run.sh files.

If python is installed on the system as `python`, then run:
```bash
bash run.sh
```

If python is installed on the system as `python3`, then run:
```bash
bash run3.sh
```

`grpc.proto` defines the client/server training RPCs. `secure_agg.proto`
(added for secure aggregation, see docs/secure_aggregation/proto_contract.md)
defines the party-server RPCs and is regenerated separately with
`run_secure_agg.sh` / `run3_secure_agg.sh`, using the same convention.

