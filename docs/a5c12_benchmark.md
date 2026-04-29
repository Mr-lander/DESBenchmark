# A5C12 Benchmark

`A5C12` is the harder Category B inter-cell benchmark surface in this release. Its code path is `build_tree_runtime_v2_intercell_a5c12_harder_suite(...)` in `fms_bench/experiment_runner.py`.

## Core Shape

- suite id: `intercell_a5c12_harder`
- internal suite tier: `tree_runtime_v2_intercell_a5c12_harder`
- composition: `brandimarte_ordertrap_twelve_cell`
- topology: `tree_a5_c12`
- regime: `R0`
- area count: 5
- cell count: 12
- route grammar contract: `a5c12_asymmetric_long_horizon_v1`
- arrival profile: `arrival_nhpp_random_batch_overlap_a3c9_v2`
- setup profile: `setup_family_change_nominal_v1`
- seed protocol: `docs/benchmark_seed_protocol_v1.json`

The source cell envelope is intentionally asymmetric before profile-specific knobs are applied. The baseline cell labels distinguish strong, ordinary, and weak clusters.

## Official Profiles

The official suite selects the k=4 family from the larger design grid. The code comment records why: k=5 and k=6 variants often reduced tardiness or makespan instead of increasing decision difficulty, so those variants are not part of the release matrix.

| Profile | Intent From Code | Main Knobs |
| --- | --- | --- |
| `branch_pressure_k4_tm115_bk4` | Keep the anchor profile schedulable while making late weak cells expensive enough that early branch choices matter. | `max_in_flight_jobs=4`, `backlog_top_k=4`, `transport_time_multiplier=1.15`, exponential nominal failure profile. |
| `cluster_pull_k4_tm120_bk4` | Pull jobs into stronger early clusters so controllers must reason about later congestion instead of greedily following fast starts. | `max_in_flight_jobs=4`, `backlog_top_k=4`, strong-cluster processing speedups, `transport_time_multiplier=1.20`, Weibull aging failure profile. |
| `late_commit_soft_k4_tm125_bk4` | Emphasize long-horizon consequence without zero-buffer collapse; the late path is slower but still runnable. | `max_in_flight_jobs=4`, `backlog_top_k=4`, late-path processing slowdowns, `transport_time_multiplier=1.25`, load-dependent failure profile. |

## Formal Matrix

For one LLM:

```text
3 profiles x 4 paradigms x 2 frameworks = 24 cases
```

With the locked `test` seed split:

```text
24 cases x 10 seeds = 240 episodes
```

The benchmark unit should be reported as:

```text
model x framework implementation x paradigm x benchmark profile x seed
```

The suite builder expands profiles and authority modes in `fms_bench/experiment_runner.py`. Case ids include both the selected profile and the authority tag, for example:

```text
tree_runtime_v2_intercell_a5c12_harder__...__branch_pressure_k4_tm115_bk4__authority_centralized
```

Framework selection happens outside the suite builder through the runner's `--framework` argument. This keeps profile/paradigm semantics separate from framework implementation.

## Result Validity

A reported A5C12 result should come from a completed suite summary with the expected profile, authority mode, framework, model, and seed coverage. For real-provider runs, inspect decision audit fields before reporting results:

- `framework_llm_decision_count`
- `framework_fallback_contract_decision_count`
- `framework_no_legal_action_count`
- `framework_decision_failure_trace`

Incomplete runs and partially recovered output directories should be treated as local artifacts, not benchmark results.
