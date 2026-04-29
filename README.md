# DESBench

DESBench is a multi-agent coordination benchmark for shared discrete-event manufacturing scheduling. It evaluates how different coordination paradigms organize Plant, Area, and Cell decision makers over the same evolving factory state.

The benchmark is designed around swappable axes:

```text
model/controller x framework implementation x coordination paradigm x benchmark suite/profile x seed
```

The main experimental variable is the coordination paradigm, also called the authority mode or protocol `Gamma` in the paper. Frameworks such as AgentScope and LangGraph are implementation carriers over the same benchmark-visible controller payload. A benchmark suite such as A5C12 is an instance family; it can be replaced or extended without changing the paradigm/framework abstraction.

## What Is Included

This source package includes:

- A shared event-driven scheduling runtime with Plant, Area, and Cell decision roles.
- The headline A5C12 inter-cell suite and an included A3C9 standard suite.
- Four official coordination paradigms: `centralized`, `hierarchical`, `mediated_heterarchical_cnp`, and `holonic_hybrid`.
- Framework-backed controller adapters for the formal A5C12 matrix: `agentscope` and `langgraph`.
- Focused tests, no-key smoke runs, extension docs, and minimal public fixture data.

This repository does not include paper result CSVs, raw provider traces, long run logs, virtual environments, or private orchestration state. Generated run artifacts should stay outside git until reviewed.

## Design Reference

The README structure follows common agent/MAS benchmark conventions:

- AgentBench-style multi-environment framing and quick-start entry points.
- Tau-bench-style provider configuration through environment variables and runnable task examples.
- AutoGenBench-style separation between running, tabulating/reviewing, and trace/result artifacts.
- MultiAgentBench and BattleAgentBench-style emphasis on multi-agent coordination protocols, fine-grained task axes, and collaboration/competition diagnostics.

In this repository those conventions map to `suite/profile`, `paradigm`, `framework`, `controller/model`, and `seed`.

## Repository Map

| Goal | Start here |
| --- | --- |
| Understand the benchmark abstraction | [docs/benchmark_overview.md](docs/benchmark_overview.md) |
| Understand available suites/profiles | [docs/suites.md](docs/suites.md) |
| Understand A5C12 | [docs/a5c12_benchmark.md](docs/a5c12_benchmark.md) |
| Understand A3C9 | [docs/a3c9_benchmark.md](docs/a3c9_benchmark.md) |
| Understand paradigms / authority modes | [docs/paradigms.md](docs/paradigms.md), `fms_bench/v2/authority_modes.py` |
| Understand framework adapters | [docs/framework_adapters.md](docs/framework_adapters.md), `fms_bench/v2/framework_drivers/base.py` |
| Add a suite, paradigm, framework, or controller | [docs/extension_guide.md](docs/extension_guide.md) |
| Install, smoke test, and run experiments | [docs/reproducibility.md](docs/reproducibility.md) |

Main runnable entry points:

- `examples/run_tree_runtime_v2_framework_native.py`: run a suite through one framework-backed controller line.
- `examples/run_tree_runtime_v2_framework_case_shard.py`: run explicit case shards into a shared output root.
- `examples/run_tree_runtime_v2_category_b.py`: Category B utilities retained for compatibility and review.

## Installation

Use Python 3.12 or newer. `uv` is recommended because optional framework dependencies can conflict and should be installed in isolated environments.

Install the core package and test dependencies:

```bash
uv sync --extra test
```

Pip fallback:

```bash
python -m pip install -e ".[test]"
```

Optional native framework packages are exposed as extras. Install only the framework you are testing in a given environment:

```bash
uv sync --extra test --extra framework_langgraph
uv sync --extra test --extra framework_agentscope
```

The no-key smoke path works without native framework packages by using contract-stub behavior.

## Quick Start

Run focused tests:

```bash
uv run pytest tests/test_tree_runtime_env.py -q
uv run pytest tests/test_experiment_runner.py -q
```

Run a no-key A5C12 smoke case:

```bash
uv run python examples/run_tree_runtime_v2_framework_native.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode holonic_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

The default `--llm-backend none` path does not call any provider.

## Run A Real Provider

Real LLM runs use an OpenAI-compatible chat backend. Keep credentials in environment variables, not checked-in files.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "<provider-token>"
uv run python examples/run_tree_runtime_v2_framework_native.py `
  --framework langgraph `
  --suite intercell_a5c12_harder `
  --authority-mode centralized `
  --llm-backend openai_compat `
  --llm-model "<model-id>" `
  --llm-base-url "https://<provider-host>/v1" `
  --llm-timeout 90 `
  --llm-retries 7 `
  --llm-retry-backoff-seconds 1 `
  --llm-retry-backoff-max-seconds 60
```

For full reporting, remove debug limits such as `--case-limit` and `--seed-limit`, then review the exported suite summary and framework decision audit fields.

## Headline Release Surface: A5C12

`intercell_a5c12_harder` is the headline release suite in this source package:

- topology: `tree_a5_c12`
- composition: `brandimarte_ordertrap_twelve_cell`
- profiles: `branch_pressure_k4_tm115_bk4`, `cluster_pull_k4_tm120_bk4`, `late_commit_soft_k4_tm125_bk4`
- paradigms: `centralized`, `hierarchical`, `mediated_heterarchical_cnp`, `holonic_hybrid`
- formal frameworks: `agentscope`, `langgraph`

For one LLM/controller line:

```text
3 profiles x 4 paradigms x 2 frameworks = 24 cases
```

With 10 test seeds per case:

```text
24 cases x 10 seeds = 240 episodes
```

## Included Standard Suite: A3C9

`intercell_a3c9_wide` is also included as a standard smaller suite:

- topology: `tree_a3_c9`
- composition: `brandimarte_ordertrap_nine_cell`
- profiles: `balanced_branching_anchor`, `cross_area_tradeoff`, `late_stage_reversal`
- paradigms: `centralized`, `hierarchical`, `mediated_heterarchical_cnp`, `holonic_hybrid`
- formal frameworks: `agentscope`, `langgraph`

It has the same 24-case per-model shape as A5C12, but it is a separate suite and should be reported with `suite_id=intercell_a3c9_wide`.

See [docs/suites.md](docs/suites.md) for how A5C12 and A3C9 fit into the broader suite abstraction.

## Extending The Benchmark

Most extensions belong to one of four layers:

- Add a suite/profile when the environment instance, topology, disturbance profile, or difficulty surface changes.
- Add a paradigm when the research question changes authority distribution or coordination protocol `Gamma`.
- Add a framework when the same controller contract should be realized through a new orchestration library.
- Add a controller/model when only the action-selection policy changes.

See [docs/extension_guide.md](docs/extension_guide.md) for registration points, tests, and validity checks.

## Result Boundary

Report only completed, reviewed runs. For real-provider runs, inspect:

- expected case and seed coverage
- `framework_llm_decision_count`
- `framework_fallback_contract_decision_count`
- `framework_no_legal_action_count`
- `framework_decision_failure_trace`

Raw traces and run directories are intentionally gitignored. Promote only reviewed summaries into paper or release materials.

## Citation

If you use this benchmark before a formal citation file is added, cite the accompanying DESBench paper draft and include the repository commit hash used for your experiments.

## License

MIT. See [LICENSE](LICENSE).
