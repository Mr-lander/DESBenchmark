# Tree Runtime Category B A3C9 Suite Matrix v1

Date: 2026-04-07
Status: contract freeze and checked release surface for the A3C9 family

## Scope

A3C9 is the `tree_a3_c9` extension of Category B.

It keeps the same Category B physical semantics:

- one shared DES world
- multi-stage inter-cell routing
- non-free setup and transport
- finite inbound admission
- explicit downstream blocking

It adds a non-degenerate three-area, nine-cell decision surface.

## Frozen Control Contract

Plant uses two sequential epochs on the same public role:

1. `backlog_selection`
2. `area_selection`

Then control continues with:

1. area-local `cell_selection`
2. cell-local `dispatch`

Plant accountability remains a plant epoch and is explicitly marked as `accountability_reroute`.

## Frozen Route Grammar

Topology: `tree_a3_c9`

Canonical route grammar:

- stage 0: `(0, 1, 3, 4, 6, 7)`
- stage 1: `(1, 2, 4, 5, 7, 8)`
- stage 2: `(0, 2, 3, 5, 6, 8)`

Contract id:

- `a3c9_canonical_non_degenerate_v1`

Every stage spans all three areas and exposes at least six candidate cells.

## Branching Audit Surface

The A3C9 summary schema must expose:

- `valid_backlog_count_mean`
- `backlog_choice_count_mean`
- `plant_valid_area_count_mean`
- `area_valid_cell_count_mean`
- `dispatch_option_count_mean`
- `backlog_multi_action_epoch_rate`
- `plant_multi_action_epoch_rate`
- `area_multi_action_epoch_rate`
- `cell_multi_action_epoch_rate`

Plant payload and trace must expose:

- `plant_decision_kind`
- `valid_backlog_ids`
- `visible_backlog_candidates`
- `selected_backlog_id`
- `selected_backlog_stage_id`
- `selected_backlog_valid_areas`
- `selected_backlog_valid_cells_by_area`

## Suite Family

### `intercell_a3c9_wide`

Role:
- official standard dynamic surface

Profiles:
- `balanced_branching_anchor`
- `cross_area_tradeoff`
- `late_stage_reversal`

Frozen visibility:
- backlog top-k = 3

### `intercell_a3c9_dynamics`

Role:
- deprecated semantic-source alias during migration

Profiles:
- `arrival_setup_exponential`
- `arrival_setup_weibull`
- `arrival_setup_load_dependent`

Frozen visibility:
- backlog top-k = 3

Required nonzero checked coverage target:
- `arrival_events_*`
- `setup_time_*`
- `breakdown_count_*`

### `intercell_a3c9_nightmare`

Role:
- official hard-tier dynamic surface

Profiles:
- `blocking_reroute_choke`
- `night_shift_breakdown_pressure`

Frozen visibility:
- backlog top-k = 5

## Authority-Expanded Overlay

The checked `2026-04-07` release bundle above remains the historical baseline.

The current research overlay reuses the same DES/runtime boundary and fans out the mainline A3C9 suites across four authority modes:

- `centralized`
- `hierarchical`
- `mediated_heterarchical_cnp`
- `holonic_hybrid`

Overlay rule:

- `intercell_a3c9_wide` and `intercell_a3c9_dynamics` expand each frozen profile into four authority-tagged cases on the same DES/runtime boundary
- authority differences are restricted to routing authority, local autonomy, reroute policy, and escalation policy
- machine dispatch, setup, transport, blocking, and route grammar stay frozen

Authority-expanded case identity uses:

- `__authority=<mode>` in `benchmark_item_id`
- `__authority_<mode>` in `case_id`

## Release Boundary

Official public Category B A3C9 surfaces:

- `intercell_a3c9_wide`
- `intercell_a3c9_nightmare`

`intercell_a3c9_dynamics` remains only as a deprecated internal / compatibility surface during migration.

Main-release framework lines:

- `LangGraph-native`
- `LangGraph-LLM`
- `AgentScope-native`
- `AgentScope-LLM`

Appendix / extension surfaces:

- `intercell_a3c9_dynamics_budget_sweep`

Checked 2026-04-07 release surface:

- main release bundle: `docs/sample_results/tree_runtime_category_b_a3c9_main_release_bundle_20260407/`
- appendix bundle: `docs/sample_results/tree_runtime_category_b_a3c9_appendix_bundle_20260407/`
- canonical manifest: `docs/sample_results/tree_runtime_category_b_a3c9_canonical_release_20260407/`
- checked `2026-04-07` main-release rows: `24` historical pre-unification rows
- checked `2026-04-07` appendix rows: `8` historical pre-unification rows
- post-unification public contract is `wide + nightmare`; historical `wide + dynamics` bundles remain archived references
