# Benchmark Overview

DESBench separates coordination research questions from implementation details. The benchmark unit is:

```text
model/controller x framework implementation x coordination paradigm x suite/profile x seed
```

## Core Layers

| Layer | Meaning | Typical Change |
| --- | --- | --- |
| Suite/profile | The scheduling instance family, topology, profile knobs, and seed protocol. | Add A5C12-like or easier/harder instance families. |
| Paradigm | The coordination protocol `Gamma`: authority distribution, legal action semantics, and recovery path. | Add or revise centralized, hierarchical, heterarchical, holonic, or hybrid authority modes. |
| Framework | The orchestration implementation carrying the same benchmark-visible controller payload. | Add a LangGraph-like or AgentScope-like adapter. |
| Controller/model | The action selector behind the framework or native policy. | Swap deterministic policy, LLM, learned policy, or external service. |
| Seed | The locked stochastic replicate for arrivals, disruptions, and related randomized surfaces. | Use train/dev/test splits from the seed protocol. |

## Why This Boundary Matters

The benchmark compares coordination design under a common dynamic substrate. If a result changes suite profile, topology, route grammar, or seed split at the same time as it changes paradigm, it is no longer a pure paradigm comparison.

Frameworks are deliberately treated as realization layers. A framework adapter may keep controller-side memory or framework-local state, but it must not read simulator-private state or alter the shared event-driven environment.

## Runtime Flow

1. A suite builder creates benchmark cases and seed lists.
2. The shared event-driven simulator advances until a decision epoch.
3. The event interpreter activates Plant, Area, or Cell decision roles under the selected authority mode.
4. The controller payload builder exposes an agent-local JSON-style payload.
5. The framework/controller returns one legal integer action.
6. The runtime applies the action, updates the world state, and records metrics plus decision audit fields.

## Release Surfaces

The headline release surface is A5C12:

```text
suite = intercell_a5c12_harder
profiles = 3
paradigms = 4
frameworks = 2
seeds per case = 10
```

A5C12 is the headline release instance family, not the whole benchmark abstraction. New suites should preserve the public axis names so results remain comparable:

```text
suite_id, profile_id, authority_mode_id, framework_name, model/controller, seed
```

The source package also includes A3C9 as a standard smaller surface:

```text
suite = intercell_a3c9_wide
profiles = 3
paradigms = 4
frameworks = 2
seeds per case = 10
```

A3C9 is a separate suite, not a profile inside A5C12. Use `suite_id` when reporting or comparing results across these surfaces.
