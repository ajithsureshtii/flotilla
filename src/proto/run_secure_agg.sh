python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./secure_agg.proto
sed -i 's/secure_agg_pb2/proto.secure_agg_pb2/1' secure_agg_pb2_grpc.py
