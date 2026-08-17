# Chunked model transfer — plan (not yet implemented)

**Status: planned, not started.** This document captures the plan as agreed
before implementation began, so the work can be picked up later without
re-deriving the design. See [`docs/secure_aggregation/`](secure_aggregation)
for the secure-aggregation-specific docs this plan's Phases 3-5 extend.

## Motivation

Flotilla does not currently support models beyond roughly 1GB on the
plaintext path, and only ~4MB on the secure_mpc path (now raised to
~1000MB after a follow-up fix — see "Current state" below). Even under
those limits, every transfer holds the *entire* serialized model in memory
as one `bytes` object on both ends, and hpmpc's own per-round file I/O
materializes the entire flattened model as one array/file. None of this
scales gracefully to genuinely large (multi-GB) models regardless of the
configured gRPC message-size limit — that requires actually chunking the
transfer, not just raising a ceiling.

## Current state (confirmed by reading the code)

- **Server→client weights** (`InitTrainRequest.model_wts`,
  `InitValidationRequest.model_wts`): single `bytes` field,
  `pickle.dumps(model_wts)` — [`grpc.proto:62`](../src/proto/grpc.proto),
  [`server_session_manager.py:594`](../src/server/server_session_manager.py).
- **Client→server weights** (`InitTrainResponse.model_weights`): single
  `bytes` field, returned as part of `StartTraining`'s unary response —
  [`grpc.proto:97`](../src/proto/grpc.proto),
  [`client_grpc_manager.py`](../src/client/client_grpc_manager.py).
- **secure_mpc share submission** (`SubmitShareRequest.shares`): a
  `repeated TensorShare`, but still sent as one unary message carrying
  every layer at once —
  [`secure_agg.proto:29`](../src/proto/secure_agg.proto).
- **secure_mpc aggregated result** (`RunAggregationRoundResponse.aggregated_model`):
  single `bytes` field — [`secure_agg.proto:47`](../src/proto/secure_agg.proto).
- **hpmpc file I/O** (`backend_hpmpc.py`): reads/writes an entire round's
  flattened model as one `np.array` and one flat `.bin` file, no batching.
- **The only existing chunking precedent**, `StreamFile`
  (`grpc.proto:11`, `stream_file_chunk` in
  `server_session_manager.py:1177`), only pushes model *code* files, and
  even it fully buffers the reassembled file in a `bytearray` client-side
  before writing — it solves the wire message-size limit but not peak
  memory.
- There is **no gRPC service running on `flo_server`** for clients to call
  into — `flo_server` always initiates, and gets results back as the
  *response* of that same call. This shapes what's possible for the
  client→server leg (see Phase 2).
- (Separately fixed, not part of this plan): a typo in
  `server_session_manager.py`'s `grpc_opts`
  (`"grpc.max_send_message_lenght"`) meant the server's outbound message
  limit was silently stuck at gRPC's 4MB default; and the secure_mpc path
  had no message-size options anywhere, also defaulting to 4MB. Both are
  now raised to a shared 1000MB constant
  (`server/secure_agg/constants.py`'s `GRPC_MESSAGE_SIZE_LIMIT_BYTES`).
  This plan is what actually removes the ceiling rather than just raising it.

## Phase 0 — Shared chunking primitives

New `src/proto/common.proto` (imported by both `grpc.proto` and
`secure_agg.proto`, which don't share anything today):
```protobuf
message DataChunk {
  oneof payload {
    ChunkMetadata metadata = 1;  // always first
    bytes data = 2;              // one or more, in order
  }
}
message ChunkMetadata {
  string transfer_id = 1;   // e.g. f"{session_id}:{round_idx}"
  int64 total_bytes = 2;    // for progress logging / a sanity check on reassembly
}
```
New `src/utils/chunking.py`: a
`write_chunks_to_stream(data: bytes, chunk_size: int) -> Iterator[DataChunk]`
generator and a `reassemble_chunks_to_file(request_iterator, dest_path)`
consumer that **writes each chunk straight to a temp file as it arrives**,
never accumulating the full payload in RAM — the actual fix over
`StreamFile`'s bytearray approach. Both take `chunk_size`/paths as
arguments, not globals, so tests can use tiny chunk sizes against tiny
payloads.

**Tests**: pure unit tests against `chunking.py` directly — round-trip a
small in-memory payload through `write_chunks_to_stream` →
`reassemble_chunks_to_file` with `chunk_size=8` bytes against a ~200-byte
payload (forces ~25 chunks through the exact same code path a 2GB transfer
would use, with no huge fixtures needed), assert byte-for-byte equality,
assert chunk count matches `ceil(len(data)/chunk_size)`, and a
metadata-mismatch/truncated-stream error case.

## Phase 1 — Server→client weights (`StartTraining`/`StartValidation`)

Mirror `grpc_send_model`'s existing "stream once, reference by id
thereafter" pattern instead of restructuring `StartTraining` itself:
- New RPC on `EdgeService`:
  `rpc StreamModelWeights(stream DataChunk) returns (StringResponse)`.
- `server_session_manager.py`: before each `StartTraining`/
  `StartValidation` call, stream the current global weights via
  `StreamModelWeights` (writing directly from
  `model_util.get_model_weights()` through `chunking.py`, never building
  one giant `pickle.dumps` bytes object first — stream the pickler's
  output instead, e.g. via a `BytesIO`-backed incremental writer, or
  simplest: pickle to a temp file with `pickle.dump(obj, fh)` and stream
  that file).
- `InitTrainRequest`/`InitValidationRequest`: replace `bytes model_wts`
  with `string weights_transfer_id`.
- `client_grpc_manager.py`: `StreamModelWeights` writes chunks straight to
  a per-round temp file (via `chunking.py`); `StartTraining` loads weights
  via `pickle.load(open(path, "rb"))` — streamed off disk, not a second
  in-memory copy — keyed by `weights_transfer_id`.

**Tests**: unit test `StreamModelWeights` server-and-client handlers with a
tiny chunk size and a small fake state_dict, asserting the reconstructed
weights match exactly; integration test reusing `tests/integration`'s
existing real-gRPC pattern end-to-end with `chunk_size_bytes` set small
enough to force dozens of chunks for a normal-sized test model; a
regression test asserting `StartTraining` fails clearly if called with a
`weights_transfer_id` that was never streamed.

## Phase 2 — Client→server weights (trained result)

Convert `StartTraining`'s return type:
`rpc StartTraining(InitTrainRequest) returns (stream DataChunk)`. The
client writes its trained `state_dict` to a temp file (`pickle.dump`) and
streams it back via `chunking.py`; a final small metadata chunk (or a
first chunk, matching `UploadFile`'s convention) carries
`metrics`/`secure_agg_used` alongside the transfer id.
`server_session_manager.py`'s `stub.StartTraining(...)` becomes an
async-iterator consumer, reassembling to a temp file and loading from disk.

**Tests**: same shape as Phase 1 — tiny-chunk-size round-trip test, plus
updating the existing `grpc_train_callback`/`StartTraining` integration
tests to consume a stream instead of one response object (these currently
mock a single `InitTrainResponse`; they'll need a fake async iterator
yielding chunks instead).

## Phase 3 — secure_mpc `SubmitShare`

`rpc SubmitShare(stream DataChunk) returns (SubmitShareAck)` — the client
streams one `TensorShare` (or, for a single huge layer, its own
sub-chunked `share_payload`) per message instead of one
`repeated TensorShare` blob. `party_server.py`'s handler accumulates
directly into `share_state` per layer as chunks arrive, rather than
waiting for one giant message.

**Tests**: unit test with a fake multi-chunk request iterator and a small
multi-layer share set (tiny chunk size), asserting `share_state` ends up
identical to today's non-chunked path; extend the existing SubmitShare
rejection tests (scheme mismatch, etc.) to the streaming shape.

## Phase 4 — secure_mpc `RunAggregationRound` response

`rpc RunAggregationRound(RunAggregationRoundRequest) returns (stream DataChunk)`
— mirrors Phase 3 in the return direction. `party_orchestrator_client.py`'s
`_call_one_party` becomes a stream consumer.

**Tests**: same pattern; also re-verify `reconstruct.py`'s dual-formula
cross-check still works unchanged (it operates on the reassembled
`PartyShare` objects, so this should need zero changes — worth asserting
explicitly with a test, not just assuming).

## Phase 5 — `backend_hpmpc.py` file I/O batching

Independent of gRPC entirely: rewrite the per-round input/output file
read/write loops to process elements in fixed-size batches (e.g. 1M
elements at a time) instead of materializing one `np.array`/`struct.pack`
call over the whole flattened model. This bounds peak memory regardless of
model size.

**Tests**: unit test with a small `batch_size` override against a moderate
element count, asserting output is byte-identical to the current
non-batched path; the existing real-Docker verification (already used
throughout the secure-aggregation work) re-run once to confirm hpmpc
binaries still produce correct results against batch-written input files.

## Phase 6 — real large-payload smoke test + docs

One test, marked `slow`/manual (mirroring the existing `slow_hpmpc_build`
convention), that actually round-trips a multi-hundred-MB synthetic model
end-to-end with default (not tiny) chunk sizes, as a sanity check the
tiny-chunk unit tests can't fully substitute for. Update
`docs/secure_aggregation/` and any main design docs with the new chunked
transfer contract and the chunk-size config knob.

## Test strategy summary

The throughline across every phase: every chunking code path takes
`chunk_size` as a parameter, so unit/integration tests use tiny chunk
sizes (8–64 bytes) against small payloads to force many-chunk behavior
deterministically and fast — never requiring multi-GB test fixtures. Only
Phase 6's smoke test uses a genuinely large payload, and it's opt-in/
manual, not part of the default suite.

## Open design decisions to confirm before implementation starts

- **Backward compatibility**: this plan assumes `model_wts`/`model_weights`
  bytes fields are *replaced* outright (no dual old/new path kept), matching
  this project's existing convention of not carrying backwards-compatibility
  shims. Confirm this is still desired before Phase 1 lands, since it is a
  breaking proto change for any external caller.
- **Pickle vs `torch.save`/`torch.load`**: this plan keeps `pickle` for
  consistency with the current codebase; `torch.save`/`torch.load` against
  a file object would give the same disk-streaming benefit with less
  custom chunking code on the serialization side, at the cost of diverging
  from how every other payload in this codebase is (de)serialized. Worth a
  final look before Phase 1, not decided here.
- **Chunk size default**: reuse the existing
  `comm_config.grpc.chunk_size_bytes` config field (already used by
  `stream_file_chunk`) rather than introducing a second chunk-size knob.
