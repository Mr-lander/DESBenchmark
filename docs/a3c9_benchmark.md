# A3C9 Benchmark

`A3C9` is the included standard inter-cell suite for DESBench. It is smaller than the A5C12 headline surface, but it keeps the same benchmark axes:

```text
model/controller x framework implementation x coordination paradigm x benchmark profile x seed
```

Its code path is `build_tree_runtime_v2_intercell_a3c9_wide_suite(...)` in `fms_bench/experiment_runner.py`.

## Identity

- suite id: `intercell_a3c9_wide`
- internal suite tier: `tree_runtime_v2_intercell_a3c9_wide`
- topology: `tree_a3_c9`
- composition: `brandimarte_ordertrap_nine_cell`
- areas: 3
- cells: 9
- route grammar contract: `a3c9_canonical_non_degenerate_v1`
- arrival profile: `arrival_nhpp_random_batch_overlap_a3c9_v2`
- setup profile: `setup_family_change_nominal_v1`

The suite is useful as a standard, lighter-weight instance family when the full A5C12 surface is too expensive for early iteration. It should be reported as `intercell_a3c9_wide`, not as an A5C12 result.

## Profiles

| Profile | Purpose | Main knobs |
| --- | --- | --- |
| `balanced_branching_anchor` | Baseline balanced branching surface for checking paradigm behavior without extreme bottlenecks. | `max_in_flight_jobs=3`, `backlog_top_k=3`, mild late-cell processing slowdowns, exponential nominal failure profile. |
| `cross_area_tradeoff` | Introduces cross-area regret where early local gains can increase downstream congestion. | `max_in_flight_jobs=4`, `backlog_top_k=3`, targeted area processing asymmetry, `transport_time_multiplier=1.15`, Weibull aging failure profile. |
| `late_stage_reversal` | Makes late-stage route choice matter through narrow buffers and stronger late reversal pressure. | `max_in_flight_jobs=3`, `backlog_top_k=3`, selected zero inbound buffers, late processing slowdowns, `transport_time_multiplier=1.30`, load-dependent failure profile. |

## Formal Shape

For one controller/model line and the two release framework implementations:

```text
3 profiles x 4 paradigms x 2 frameworks = 24 cases
```

With 10 test seeds per case:

```text
24 cases x 10 seeds = 240 episodes
```

The four official paradigms are `centralized`, `hierarchical`, `mediated_heterarchical_cnp`, and `holonic_hybrid`. The release framework implementations are `agentscope` and `langgraph`.

## Relationship To A5C12

A3C9 and A5C12 are different benchmark instance families. They can both be used to evaluate the same paradigm/framework/controller axes, but their results should not be mixed into one table unless the table explicitly includes `suite_id` or `topology`.

A5C12 remains the headline paper surface in this source release. A3C9 is included to make the benchmark easier to extend, smoke test, and compare across a smaller standard topology.
