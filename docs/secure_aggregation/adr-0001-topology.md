# ADR-0001: Secure-aggregation topology and locked-in parameters

## Status

Accepted (Phase 0).

## Context

Flotilla's single aggregation server (`flo_server.py`) is the one place a
client's raw model update is ever seen in plaintext. We're replacing that
plaintext aggregation step with an MPC-based secure aggregation among
multiple non-colluding party servers, starting with the `hpmpc` library. Four
questions had to be settled before any implementation could start:

1. How many MPC parties, and which protocol?
2. What adversary/threat model?
3. Where does the new multi-party topology physically live — a mode of
   `flo_server.py`, or something else?
4. Where is this first validated during development?

## Decision

1. **3 parties, hpmpc Replicated (2,3) secret sharing (`PROTOCOL=2`).** This
   is the textbook honest-majority replicated-SS scheme most secure-
   aggregation FL literature assumes, the smallest/most-auditable
   implementation in hpmpc, and requires no offline preprocessing phase —
   simplest to operate as a per-round job. (Alternative considered: hpmpc's
   4-party malicious-secure Tetrad protocol — rejected for now due to added
   preprocessing/operational complexity; documented as a future upgrade
   path, not blocked by anything in this design.)

2. **Semi-honest / honest-majority threat model.** Parties are assumed to
   follow the protocol correctly but might try to passively learn secrets;
   no protection against a party that actively deviates. (Alternative:
   malicious-secure from day one — rejected as unnecessary added complexity
   before the basic pipeline is even proven; the abstraction is designed so
   swapping to a malicious-secure backend later should be mostly a
   config/backend change, not a redesign.)

3. **New standalone process type — `src/flo_secure_agg_party.py`, run 3
   times — not a mode flag on `flo_server.py`.** Considered and rejected: a
   `--secure-agg-party=N` flag on the existing `flo_server.py` entrypoint.
   Rejected because:
   - `flo_server.py`'s responsibilities (REST session intake, round
     sequencing, client selection, checkpointing, server-side validation)
     have nothing to do with running an MPC party's share-buffering/protocol-
     execution loop; conflating them would make both harder to reason about
     and test independently.
   - The three party processes need a different lifecycle and network
     exposure than `flo_server` (they need to be reachable by every client
     for share submission, and by each other for the MPC wire protocol, but
     should not expose `flo_server`'s REST session-intake endpoint).
   - Keeping party processes structurally uninvolved in round/session logic
     is exactly what makes the `SecureAggregationBackend` interface (see
     `design.md`) clean to test in isolation, and what makes hpmpc
     swappable for a different backend later without touching
     `flo_server.py` at all.
   `flo_server.py` remains a singleton, unchanged in role — it stops being
   where plaintext client updates are materialized, but keeps owning
   everything else it owns today.

4. **Single-machine, multi-process/docker-compose for dev/test.** Real TCP/
   TLS sockets between party processes, just co-located on one host — this
   matches how hpmpc's own scripts (`scripts/run_locally.sh`) are set up for
   local dev, and gives a realistic-enough topology (genuine separate
   processes and network hops) without requiring multi-machine
   provisioning before Phase 1 can even start. Multi-machine deployment is
   not blocked by anything in this design; it's a config/inventory change
   (different `host` values in `party_endpoints`), not an architecture change.

## Consequences

- `flo_client` gains a second outbound gRPC relationship (to 3 party
  endpoints) in addition to its existing inbound relationship with
  `flo_server`. Documented in `topology.md` (Phase 2).
- Operators must now run and monitor 3 additional long-lived-ish processes
  per deployment (up from 1 server); this is accepted as the cost of
  removing the single point of trust, and is exactly what
  `docs/secure_aggregation/runbook.md` (Phase 4) exists to make manageable.
- Because the party processes are structurally separate from `flo_server`,
  adding a second/future MPC backend (Phase 5) requires touching only
  `src/server/secure_agg/backends/` and config — never `flo_server.py`,
  `server_session_manager.py`, or the client's training loop.
