# Secure aggregation overhead report

**Status: complete for the 3 protocols currently wired up (`fedavg` plaintext,
hpmpc `PROTOCOL=2` Replicated 3PC, `PROTOCOL=5` Trio, `PROTOCOL=8` Tetrad).**
All numbers below are from real runs — a controlled micro-benchmark against
real compiled hpmpc binaries, and 12 real end-to-end Docker/Flotilla
sessions (4 configurations × 3 repeats) with real MNIST + LeNet5 training —
not estimated or simulated. See "Method" for exactly how, and "Known
limitations" for what these numbers do and don't tell you.

**Read this before trusting Tetrad's numbers as "malicious-secure
performance":** real corruption-testing during development
(`hpmpc_backend.md`'s "Malicious-security caveat, found empirically") found
Tetrad's own cheat-detection does not fire for this integration's actual
usage pattern. The Tetrad numbers below measure real, working aggregation —
just not with the stronger security guarantee its protocol number implies.

## Scope

Four configurations, all training the same model (LeNet5, 61,706 parameters)
on the same MNIST partition (3 clients, ~19,333 images each), for the same
3 rounds, 1 epoch/round, batch size 64:

1. **`fedavg`** — today's plaintext single-server aggregation (no MPC at
   all; the pre-existing baseline this whole project set out to replace).
2. **`secure_mpc` / hpmpc `PROTOCOL=2`** ("Replicated 3PC", semi-honest,
   3-party).
3. **`secure_mpc` / hpmpc `PROTOCOL=5`** ("Trio", semi-honest, 3-party).
4. **`secure_mpc` / hpmpc `PROTOCOL=8`** ("Tetrad", labeled malicious-secure
   upstream, 4-party — see the caveat above).

## Method

Two complementary measurements, per the plan this report closes out:

- **Micro-benchmark** (`mpc_engines/hpmpc/measurements/run_fedavg_bench.py` +
  `run_plaintext_bench.py`): isolates the MPC protocol's own per-round
  timing/communication from Docker cross-container network jitter, by
  running all of a protocol's party processes on loopback inside ONE
  container, using the real production `HpmpcBackend.run_aggregation_round`
  code path (imported directly, not reimplemented). 30 rounds per protocol,
  a 61,706-element payload matching LeNet5's real parameter count exactly
  (`sum(p.numel() for p in state_dict.values())`, computed once and checked
  in as a constant — not guessed). Communication/timing figures are parsed
  from hpmpc's own stdout (`core/utils/print.hpp`'s `print_communication()`
  and the per-round `"Time measured to perform..."` lines, both emitted
  unconditionally, no extra build flags) via hpmpc's own
  `measurements/parse_logs.py`, reused unmodified rather than
  reimplemented. The plaintext baseline mirrors `aggregator_fedavg.py`'s
  actual weighted-sum arithmetic over the same 3-client, same-size update,
  1000 iterations, plus the real client→server wire payload size
  (`len(pickle.dumps(state_dict))`, the literal serialization
  `client_grpc_manager.py` performs today).
- **End-to-end** (real `docker compose` deployments, real `flo_session.py`
  submissions against real containers — not mocked): for each of the 4
  configurations, 3 independent full session runs (`docker compose up`
  freshly each time client/party images needed rebuilding for that
  configuration), 3 real training rounds each. Per-round aggregation
  overhead comes from `flo_server`'s own
  `fedserver.train_callback.aggregate_time` log line (brackets the
  `aggregate()` call directly — see `rollout_guide.md` for why this is the
  right metric and `fedserver.train.round.client.finished` is not).
  Final-round accuracy is recorded as a correctness tripwire, not an
  overhead metric. Round 1 (cold start — first gRPC handshake, first
  container-to-container connections) is reported separately from the
  rounds-2-3 mean (steady state), not blended in, since they measure
  genuinely different things.

Raw session logs (all 12 runs) are checked in under
`overhead_report_raw/*.log`; raw micro-benchmark output is
`mpc_engines/hpmpc/measurements/fedavg_bench_results.json` and
`plaintext_bench_results.json`.

**Environment note:** all runs (both micro-benchmark and end-to-end) were
executed via Docker on an Apple Silicon (ARM64) host running `linux/amd64`
images under emulation (Docker reported `"the requested image's platform
(linux/amd64) does not match the detected host platform (linux/arm64/v8)"`
for every container). Absolute wall-clock numbers below should not be taken
as representative of a native deployment's absolute performance — but since
every configuration ran under the identical emulated environment, the
*relative* comparisons between configurations (which is what this report is
actually for) should still be informative.

## Micro-benchmark results (30 rounds/protocol, 61,706-element payload)

| Protocol | Wall time/round (mean) | Wall time/round (p95) | hpmpc ONLINE comm (mean, sent=received) | hpmpc-reported ONLINE compute time (mean) |
|---|---|---|---|---|
| Plaintext (`fedavg`) | 0.46 ms | 0.55 ms | n/a — no party-to-party tier at all | n/a |
| Replicated 3PC (`PROTOCOL=2`) | 74.6 ms | 78.5 ms | 1.48 MB | 2.89 ms |
| Trio (`PROTOCOL=5`) | 72.3 ms | 76.9 ms | 1.48 MB | 2.95 ms |
| Tetrad (`PROTOCOL=8`) | 115.4 ms | 117.7 ms | 1.97 MB | 8.85 ms |

"Wall time/round" is the full subprocess-per-round cost as
`HpmpcBackend.run_aggregation_round` actually experiences it (spawn + socket
handshake + reveal + file I/O), all on loopback. "hpmpc ONLINE comm" and
"compute time" are hpmpc's own self-reported figures for the reveal
operation itself (a subset of the wall time above — socket
handshake/process-spawn overhead isn't included in hpmpc's own numbers).
Tetrad's higher communication (1.97 MB vs. 1.48 MB) is expected: it writes
3 `uint64` fields per element instead of 2 (see `hpmpc_backend.md`'s file
contract), on top of running 4 parties instead of 3.

Client count does not appear as a dimension here — `backend_hpmpc.py` sums
every selected client's shares in Python before ever invoking the compiled
binary (see that module's docstring), so the binary's own cost is
independent of client count; see `run_fedavg_bench.py`'s docstring for the
full reasoning.

## End-to-end results (real Docker deployment, 3 real rounds/session, 3 sessions/config)

**`fedserver.train_callback.aggregate_time` per round (seconds):**

| Config | Round 1 (cold start) | Rounds 2-3 (steady state, mean) |
|---|---|---|
| `fedavg` (plaintext) | 0.0230 | 0.0139 |
| `secure_mpc` / Replicated (`PROTOCOL=2`) | 0.0834 | 0.0578 |
| `secure_mpc` / Trio (`PROTOCOL=5`) | 0.0677 | 0.0562 |
| `secure_mpc` / Tetrad (`PROTOCOL=8`) | 0.0802 | 0.0768 |

**Session wall time (submission to completion) and final-round accuracy, mean of 3 repeats:**

| Config | Session wall time (mean) | Final-round accuracy (mean) |
|---|---|---|
| `fedavg` | 79.3 s | 98.13% |
| Replicated (`PROTOCOL=2`) | 74.7 s | 97.75% |
| Trio (`PROTOCOL=5`) | 66.3 s | 98.05% |
| Tetrad (`PROTOCOL=8`) | 81.3 s | 97.98% |

All 4 configurations converge to within ~0.4 percentage points of each
other by round 3 — confirms `secure_mpc` (any of the 3 protocols) does not
harm convergence relative to plaintext `fedavg`, consistent with the
fixed-point rounding tolerance already documented in
`sharing_scheme_replicated3pc.md`.

**Session wall time is dominated by real training (~18-30s/round for this
dataset/model), not aggregation** (aggregation is 15-85ms/round, roughly
1000x smaller) — the session-level numbers above vary by tens of seconds
between repeats of the *same* configuration (container scheduling jitter
under Docker emulation, not a real difference), and should not be read as a
clean "MPC overhead" signal. `aggregate_time` (previous table) is the
metric that actually isolates it, and its variance across repeats of the
same config is much smaller (single-digit milliseconds).

**One real transient, not hidden:** `overhead_report_raw/tetrad_run1.log`
contains a handful of `fedserver_gRPC.train.invalid_channel` /
`Connection refused` errors
immediately before that session's first round — a stale client-registry
entry left over from rebuilding the client containers for Tetrad's
`sharing_scheme`/`party_endpoints` change, resolved before round 0 actually
started (all 3 rounds completed normally with all 3 clients and correct
accuracy). Included here for transparency, not because it affected the
reported numbers.

## Normalized overhead multiplier (secure_mpc vs. plaintext `fedavg`)

Using the end-to-end steady-state (rounds 2-3 mean) `aggregate_time` as the
primary, real-deployment number — the micro-benchmark's own multiplier vs.
plaintext is technically computable but not shown as a headline figure,
since plaintext addition is microseconds and any network round-trip is
inherently orders of magnitude slower in isolation; presenting that ratio
alone would read as far more dramatic than what actually matters
operationally (the absolute end-to-end numbers above, which are all under
100ms):

| Config | Steady-state `aggregate_time` | Multiplier vs. `fedavg` |
|---|---|---|
| `fedavg` | 0.0139 s | 1.0x (baseline) |
| Replicated (`PROTOCOL=2`) | 0.0578 s | **4.16x** |
| Trio (`PROTOCOL=5`) | 0.0562 s | **4.05x** |
| Tetrad (`PROTOCOL=8`) | 0.0768 s | **5.53x** |

Trio and Replicated cost essentially the same (both 3-party, semi-honest,
same on-disk field count); Tetrad costs about 33% more than either — the
extra party, extra field per element, and (nominally) stronger security
model all add up, even though (per the caveat above) the stronger security
model isn't currently confirmed to actually hold for this integration.

Communication has no clean per-round end-to-end number (see "Known
limitations") — the micro-benchmark table above is the correctly-attributed
figure for this same model size, and is what should be cited for
communication overhead specifically.

## Known limitations

- **Statistical power.** 3 session repeats per configuration (30 rounds per
  protocol for the micro-benchmark) is enough to see a clear, consistent
  signal here (steady-state `aggregate_time` varied by single-digit
  milliseconds across repeats of the same config), but is not a rigorous
  statistical study — treat the specific multipliers above as "roughly
  4-5.5x," not precise-to-the-percent figures.
- **Docker-emulation environment.** See the environment note above — all
  runs used `linux/amd64` images under ARM64 emulation. Absolute numbers on
  native hardware (either architecture) would likely be lower; relative
  comparisons should hold.
- **No per-round, per-config communication number for the end-to-end runs.**
  `docker stats`' per-container network counters are cumulative-since-start,
  not attributable to a specific round or phase, so they weren't used here
  — the micro-benchmark's correctly-attributed number is cited instead for
  communication overhead. No coarse `docker stats` snapshot appendix was
  taken either, since the micro-benchmark already answers this more
  precisely for the same model size.
- **Only 3 hpmpc protocols are covered** (Replicated, Trio, Tetrad) — the
  originally-considered `PROTOCOL=6` ("Trusted Third Party") was
  deliberately excluded from this whole effort (see `design.md`) since it
  provides no privacy guarantee at all; it would not be a meaningful,
  honest comparison point here.
- **Tetrad's malicious-security guarantee was not confirmed to hold for this
  integration** (repeated from the top of this report deliberately, given
  how easy it would be to miss) — see `hpmpc_backend.md`'s "Malicious-
  security caveat, found empirically" and `threat_model.md`'s adversary
  model. The overhead numbers above are real and correct for what Tetrad
  *does* provide here (a working, correct 4-party aggregation, currently
  equivalent in practice to Replicated/Trio's semi-honest guarantee at
  higher cost) — not for hpmpc's usual malicious-security claim.
- **`TUTORIAL.md`/its PDF are not updated** to reflect the multi-protocol
  work or this report — flagged here explicitly as a known, deliberately
  out-of-scope follow-up (see the original plan), not a silent gap.

## Appendix

- Raw end-to-end session logs: `overhead_report_raw/*.log` (12 files, one
  per session — `fedavg_run{1,2,3}.log`, `secure_mpc_p2_run{1,2,3}.log`,
  `trio_run{1,2,3}.log`, `tetrad_run{1,2,3}.log`). Grep for
  `aggregate_time` or `'accuracy'` to find the lines this report's tables
  were computed from.
- Raw micro-benchmark output: `mpc_engines/hpmpc/measurements/fedavg_bench_results.json`
  (per-round wall time + hpmpc-parsed stats for all 3 protocols, 30 rounds
  each) and `mpc_engines/hpmpc/measurements/plaintext_bench_results.json`.
- Micro-benchmark tooling (reusable, re-runnable):
  `mpc_engines/hpmpc/measurements/run_fedavg_bench.py`,
  `mpc_engines/hpmpc/measurements/run_plaintext_bench.py`,
  `mpc_engines/hpmpc/measurements/Dockerfile.bench` (builds all 3 protocols + a Python
  environment for the benchmark script in one image — see that file's own
  comments for build/run instructions).
