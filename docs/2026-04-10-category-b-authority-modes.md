# Category B Authority Modes

Date: 2026-04-10
Status: frozen authority-policy boundary for Category B A3C9 paradigm comparisons

## Scope

This note freezes the organizational-paradigm layer added on top of the existing Category B inter-cell DES.

It does not change:

- Category A
- machine dispatch substrate
- transport/setup/blocking physics
- the Plant / Area / Cell object boundary

It changes only the routing / commitment / reroute / escalation authority distribution inside the existing Category B runtime.

## Runtime Contract

All four modes run on the same DES/runtime boundary:

- one shared inter-cell DES world
- one Plant / Area / Cell object model
- the same route grammar, backlog visibility, setup semantics, transport semantics, and blocking semantics
- the same machine-dispatch substrate

The comparison axis is therefore organizational authority, not plant physics and not machine-level dispatch heuristics.

## Mode Surface

### Frozen fields

The authority-policy layer freezes these switchable rights:

- plant direct award right
- area-local reroute right
- cell accept/reject right
- reject escalation destination
- mediated bid/award round enablement

## centralized

- routing authority: plant/global controller directly selects the next execution cell
- local autonomy: cell executes only and does not own a real reject right
- reroute policy: plant reselects directly rather than running area-local reroute
- escalation policy: plant remains the final and immediate reroute authority

## hierarchical

- routing authority: fixed Plant -> Area -> Cell chain
- local autonomy: cell can report but is effectively accept-only on contract commitment
- reroute policy: no area-local sibling reroute after reject or infeasibility
- escalation policy: reject goes upward and plant revises routing

## mediated_heterarchical_cnp

- routing authority: allocation is awarded from a tree-mediated CFP / bid / award round
- local autonomy: cell can bid and can reject, but still inside the current tree runtime
- reroute policy: mediated competition replaces fixed top-down award; no area-local sibling reroute
- escalation policy: plant revises routing only after the mediated path cannot settle

## holonic_hybrid

- routing authority: stable Plant / Area / Cell holarchy with bounded local autonomy
- local autonomy: cell has a real accept/reject right
- reroute policy: area reroutes locally first across eligible sibling cells
- escalation policy: plant fallback happens only after local reroute exhaustion

## Honest Boundary

- `mediated_heterarchical_cnp` is a restricted heterarchical baseline under the current tree runtime, not a pure peer-to-peer heterarchy
- `holonic_hybrid` is the closest match to the current DMS runtime because it preserves the stable holarchy while giving cell-level contract refusal and area-first recovery
- the current contract still has no true public `defer` action; do not claim one unless the contract is explicitly extended

## A3C9 Benchmark Overlay

The current benchmark overlay is intentionally narrow:

- authority-expanded mainline suites: `intercell_a3c9_wide`, `intercell_a3c9_dynamics`
- default authority set: `centralized`, `hierarchical`, `mediated_heterarchical_cnp`, `holonic_hybrid`
- authority-expanded case identity: `__authority=<mode>`

Historical checked release bundles from `2026-04-07` remain valid as the pre-overlay baseline. The authority-mode comparison is a new benchmark overlay on the same runtime, not a rewrite of that checked release.

## Source Paths

- `fms_bench/v2/authority_modes.py`
- `fms_bench/v2/category_b_controller_families.py`
- `fms_bench/v2/intercell_tree_runtime_env.py`
- `fms_bench/experiment_runner.py`
- `examples/run_tree_runtime_v2_framework_native.py`
