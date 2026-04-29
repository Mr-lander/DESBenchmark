# Suites And Profiles

A suite is the replaceable benchmark instance layer. It defines which cases exist, which profiles are official, and which seed split should be used for reporting.

## Headline Release Surface

The current headline release suite is:

```text
intercell_a5c12_harder
```

It is built by:

```text
fms_bench/experiment_runner.py::build_tree_runtime_v2_intercell_a5c12_harder_suite(...)
```

The headline release suite expands three official profiles across the four official authority modes. Framework selection is not part of suite construction; it is selected by runner arguments such as `--framework langgraph`.

## A5C12 Profiles

| Profile | Purpose |
| --- | --- |
| `branch_pressure_k4_tm115_bk4` | Emphasizes early branch choice under visible late weak-cell cost. |
| `cluster_pull_k4_tm120_bk4` | Pulls work toward strong early clusters and tests later congestion regret. |
| `late_commit_soft_k4_tm125_bk4` | Emphasizes long-horizon commitment cost without turning the instance into a collapse case. |

Detailed profile knobs are in [a5c12_benchmark.md](a5c12_benchmark.md).

## Included Standard Suite

DESBench also includes a smaller standard inter-cell suite:

```text
intercell_a3c9_wide
```

It is built by:

```text
fms_bench/experiment_runner.py::build_tree_runtime_v2_intercell_a3c9_wide_suite(...)
```

A3C9 uses `tree_a3_c9` over `brandimarte_ordertrap_nine_cell`. It keeps the same paradigm/framework/controller axes as A5C12 but should be reported under its own `suite_id`.

## A3C9 Profiles

| Profile | Purpose |
| --- | --- |
| `balanced_branching_anchor` | Balanced branching anchor for standard smaller-suite comparisons. |
| `cross_area_tradeoff` | Cross-area routing tradeoff with downstream congestion regret. |
| `late_stage_reversal` | Late-stage reversal pressure through narrow buffers and dynamic disturbance. |

Detailed profile knobs are in [a3c9_benchmark.md](a3c9_benchmark.md).

## Other Source-Level Suites

The codebase retains additional smaller or historical suite builders for smoke tests, regression coverage, and extension work. Public comparisons should use `intercell_a5c12_harder` or `intercell_a3c9_wide` unless a paper or report explicitly declares a different suite surface.

Common smoke/debug surfaces include:

- `small`
- `intercell_small`
- `intercell_brandimarte_probe`
- `intercell_blocking_probe`

These are useful for validating code paths quickly, but they are not substitutes for the documented A5C12 or A3C9 suite surfaces.

## Add A Suite

Add a suite when you want to change the benchmark instance layer: topology, composition, route grammar, profile knobs, disturbance family, due-date/numeric profile, or seed protocol.

Minimum checklist:

1. Add a suite builder in `fms_bench/experiment_runner.py`.
2. Define explicit `suite_id`, `profile_id`, and semantic metadata fields.
3. Keep authority modes as an argument where possible, rather than hardcoding one paradigm.
4. Register the suite in `examples/run_tree_runtime_v2_framework_native.py`.
5. Add tests that assert case count, profile ids, authority-mode expansion, and seed split.
6. Document the suite and distinguish release/reporting suites from smoke/debug suites.

Do not overload A5C12 profile names for a different topology or route grammar. Use a new suite id when the instance family changes.
