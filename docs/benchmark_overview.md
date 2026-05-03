# Benchmark Overview

DESBench separates coordination research questions from implementation details.
For a new user, the workflow is: choose a reported instance, connect a MAS
runner to the public decision payload, optionally register a new paradigm id,
run the public entry point, and inspect the emitted metrics and audit traces.

## Core Layers

| Layer | Meaning | Typical Change |
| --- | --- | --- |
| Suite/profile | The scheduling instance family, topology, profile knobs, and seed protocol. | Add A5C12-like or easier/harder instance families. |
| Paradigm | The coordination protocol `Gamma`: authority distribution, legal action semantics, and recovery path. | Add or revise centralized, hierarchical, heterarchical, holonic, or hybrid paradigms. |
| MAS runner | The user-provided system that consumes public decision payloads and returns legal actions. | Connect a new MAS architecture, LLM controller, learned policy, or external service. |
| Implementation carrier | Optional orchestration implementation carrying the same benchmark-visible controller payload. | Add a LangGraph-like or AgentScope-like adapter only when needed. |
| Seed | The locked stochastic replicate for arrivals, disruptions, and related randomized surfaces. | Use train/dev/test splits from the seed protocol. |

## Why This Boundary Matters

The benchmark compares coordination design under a common dynamic substrate. If a result changes suite profile, topology, route grammar, or seed split at the same time as it changes paradigm, it is no longer a pure paradigm comparison.

Implementation carriers are deliberately treated as realization layers. A MAS
runner or adapter may keep controller-side memory or local orchestration state,
but it must not read simulator-private state or alter the shared event-driven
environment.

## Runtime Flow

1. A suite builder creates benchmark cases and seed lists.
2. The shared event-driven simulator advances until a decision epoch.
3. The event interpreter activates Plant, Area, or Cell decision roles under the selected authority mode.
4. The controller payload builder exposes an agent-local JSON-style payload.
5. The MAS runner returns one legal integer action for every active agent.
6. The runtime applies the action, updates the world state, and records metrics plus decision audit fields.

## Release Surfaces

The headline release surface is A5C12:

A5C12 is the headline release instance family, not the whole benchmark
abstraction. For a new MAS design, start from one reported A5C12 instance and
expand only after the runner passes validity checks. New suites should preserve
the public identifiers below so results remain comparable:

```text
suite_id, profile_id, paradigm_id, runner/controller, model/backend, seed
```

The source package also includes A3C9 as a standard smaller surface:

A3C9 is a separate suite, not a profile inside A5C12. Use `suite_id` when reporting or comparing results across these surfaces.

## Reported-Instance Boundary

The paper-facing comparison unit is more specific than `suite_id` alone. For the public appendix, DESBench uses four reported instances:

| Paper ID | `suite_id` | `profile_id` | Meaning |
| --- | --- | --- | --- |
| `A3C9-1` | `intercell_a3c9_wide` | `cross_area_tradeoff` | Smaller-topology transfer with cross-area congestion regret. |
| `A5C12-1` | `intercell_a5c12_harder` | `branch_pressure_k4_tm115_bk4` | Branch-pressure anchor on the harder topology. |
| `A5C12-2` | `intercell_a5c12_harder` | `cluster_pull_k4_tm120_bk4` | Cluster-pull anchor exposing greedy-routing lure. |
| `A5C12-3` | `intercell_a5c12_harder` | `late_commit_soft_k4_tm125_bk4` | Late-commit anchor emphasizing long-horizon regret. |

This is why the public guidance says not to collapse a result to a profile-only label. Changing suite changes topology and route surface, not just difficulty knobs.

## Public Run Modes

DESBench exposes two public run modes:

1. `examples/run_suite.py` for full suite surfaces such as the full A5C12 or A3C9 matrix.
2. `examples/run_case.py` for explicit case reruns when you want one reported instance or a small selected bundle.
3. `examples/run_mas.py` for a user-provided MAS process that reads public decision payloads and returns action mappings.

The full-suite runner is the default path for standard evaluation. The case-shard runner is the paper/debug path for exact instance-level inspection and reruns.
