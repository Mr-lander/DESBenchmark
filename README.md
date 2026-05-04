# DESBench

DESBench is a multi-agent coordination benchmark for shared discrete-event manufacturing scheduling. It evaluates how different coordination paradigms organize Plant, Area, and Cell decision makers over the same evolving factory state.

The public use path is linear:

1. Choose a reported instance such as `A5C12-1` or `A3C9-1`.
2. Connect a MAS runner that consumes the public decision payload.
3. Return one legal action for every active agent at each decision epoch.
4. Optionally register a new coordination paradigm id.
5. Run the same instance and seed split for the official paradigms and for your method.

The main experimental variable is the coordination paradigm, also called protocol `Gamma` in the paper. Frameworks such as AgentScope and LangGraph are optional implementation carriers over the same benchmark-visible controller payload. A benchmark suite such as A5C12 is an instance family supplied by DESBench; ordinary users do not need to create new instances to compare a MAS design.

## What Is Included

This source package includes:

- A shared event-driven scheduling runtime with Plant, Area, and Cell decision roles.
- The headline A5C12 inter-cell suite and an included A3C9 standard suite.
- The paper-level reported-instance mapping used by the public evaluation appendix.
- Four official coordination paradigms: `centralized`, `hierarchical`, `heterarchical_cnp`, and `holonic_hybrid`.
- Framework-backed controller adapters for compatibility with the reported runs: `agentscope` and `langgraph`.
- Focused tests, no-key smoke runs, extension docs, and minimal public fixture data.

This repository does not include paper result CSVs, raw provider traces, long run logs, virtual environments, or private orchestration state. Generated run artifacts should stay outside git until reviewed.

## Design Reference

The README structure follows common agent/MAS benchmark conventions:

- AgentBench-style multi-environment framing and quick-start entry points.
- Tau-bench-style provider configuration through environment variables and runnable task examples.
- AutoGenBench-style separation between running, tabulating/reviewing, and trace/result artifacts.
- MultiAgentBench and BattleAgentBench-style emphasis on multi-agent coordination protocols, fine-grained task axes, and collaboration/competition diagnostics.

In this repository those conventions are exposed through reported instances, paradigm ids, public controller payloads, and reviewed output artifacts.

## Repository Map

| Goal | Start here |
| --- | --- |
| Understand the benchmark abstraction | [docs/benchmark_overview.md](docs/benchmark_overview.md) |
| Understand available suites/profiles | [docs/suites.md](docs/suites.md) |
| Understand A5C12 | [docs/a5c12_benchmark.md](docs/a5c12_benchmark.md) |
| Understand A3C9 | [docs/a3c9_benchmark.md](docs/a3c9_benchmark.md) |
| Understand paradigms | [docs/paradigms.md](docs/paradigms.md) |
| Understand implementation carriers | [docs/framework_adapters.md](docs/framework_adapters.md) |
| Connect a MAS runner or add a paradigm | [docs/extension_guide.md](docs/extension_guide.md) |
| Install, smoke test, and run experiments | [docs/reproducibility.md](docs/reproducibility.md) |

Main runnable entry points:

- `examples/run_suite.py`: run one full suite surface through one MAS runner or controller line.
- `examples/run_case.py`: rerun explicit reported cases into a shared output root and recover the partial suite summary.
- `examples/run_mas.py`: run a user-provided MAS or controller process over the public JSONL decision interface.
- `examples/custom_paradigm.py`: minimal extension module showing how to register a new coordination paradigm.

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
uv run python examples/run_suite.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode holonic_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

The default `--llm-backend none` path does not call any provider.

Connect a MAS process through the public decision payload:

```bash
uv run python examples/run_mas.py \
  --suite small \
  --case-limit 1 \
  --seed-limit 1 \
  --controller-command uv run python examples/subprocess_mask_first_controller.py
```

## Run A Real Provider

Real LLM runs use an OpenAI-compatible chat backend. Keep credentials in environment variables, not checked-in files.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "<provider-token>"
uv run python examples/run_suite.py `
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

For full reporting, remove debug limits such as `--case-limit` and `--seed-limit`, then review the exported suite summary and decision audit fields.

## Headline Release Surface: A5C12

`intercell_a5c12_harder` is the headline release suite in this source package:

- topology: `tree_a5_c12`
- composition: `brandimarte_ordertrap_twelve_cell`
- profiles: `branch_pressure_k4_tm115_bk4`, `cluster_pull_k4_tm120_bk4`, `late_commit_soft_k4_tm125_bk4`
- paradigms: `centralized`, `hierarchical`, `heterarchical_cnp`, `holonic_hybrid`
- compatibility carriers used in the paper runs: `agentscope`, `langgraph`

For a new MAS design, start with one reported A5C12 instance, run it against the official paradigms and your proposed paradigm, then expand to the remaining reported instances after the runner passes audit checks.

## Included Standard Suite: A3C9

`intercell_a3c9_wide` is also included as a standard smaller suite:

- topology: `tree_a3_c9`
- composition: `brandimarte_ordertrap_nine_cell`
- profiles: `balanced_branching_anchor`, `cross_area_tradeoff`, `late_stage_reversal`
- paradigms: `centralized`, `hierarchical`, `heterarchical_cnp`, `holonic_hybrid`
- compatibility carriers used in the paper runs: `agentscope`, `langgraph`

It is a separate suite and should be reported with `suite_id=intercell_a3c9_wide`.

See [docs/suites.md](docs/suites.md) for how A5C12 and A3C9 fit into the broader suite abstraction.

## Paper Reported Instances

The paper does not use a profile name alone as the public benchmark unit. The paper-facing unit is a reported instance: a fixed `suite_id` plus `profile_id` under the locked test seed split.

| Paper ID | Public source id | `suite_id` | `profile_id` | Coordination stress |
| --- | --- | --- | --- | --- |
| `A3C9-1` | `intercell_a3c9_cross_area_tradeoff` | `intercell_a3c9_wide` | `cross_area_tradeoff` | Smaller-topology transfer with cross-area routing regret. |
| `A5C12-1` | `intercell_a5c12_branch_pressure` | `intercell_a5c12_harder` | `branch_pressure_k4_tm115_bk4` | High routing ambiguity and early branch pressure. |
| `A5C12-2` | `intercell_a5c12_cluster_pull` | `intercell_a5c12_harder` | `cluster_pull_k4_tm120_bk4` | Strong cluster pull and greedy-routing regret. |
| `A5C12-3` | `intercell_a5c12_late_commit` | `intercell_a5c12_harder` | `late_commit_soft_k4_tm125_bk4` | Long-horizon commitment cost and late regret. |

If you want paper-grade comparisons, keep the reported instance, runner boundary, backend/model setting, and locked seed split fixed inside the comparison block.

## Extending The Benchmark

Most users should choose one of two paths.

Path A connects an existing MAS. Use this when you already have an agent graph,
message-passing system, learned policy, LLM controller, or custom framework.
Run it through `examples/run_mas.py` or implement `select_actions(decision)`.
DESBench supplies the active-agent payload and legal-action masks; your MAS
returns one legal action for each active agent.

Path B configures a new DESBench paradigm. Use this when the research question
changes the coordination protocol `Gamma`: authority distribution, rejection
rights, reroute behavior, mediated bidding, or escalation. Start with
`examples/custom_paradigm.py`, give the paradigm a new `mode_id`, connect a MAS
runner to the public decision payload, and run the same reported instance,
backend/model, and seed split used for the official modes.

Only add a suite/profile when the environment instance, topology, disturbance
profile, or difficulty surface changes. Only add an implementation carrier when
the same controller contract must be realized through a new orchestration
library. Do not change topology, route grammar, profile knobs, or metric
definitions while claiming a pure paradigm comparison.

See [docs/extension_guide.md](docs/extension_guide.md) for registration points, tests, and validity checks.

## Result Boundary

Report only completed, reviewed runs. For real-provider runs, inspect:

- expected case and seed coverage
- `framework_llm_decision_count`
- `framework_fallback_contract_decision_count`
- `framework_no_legal_action_count`
- `framework_decision_failure_trace`

Raw traces and run directories are intentionally gitignored. Promote only reviewed summaries into paper or release materials.

## Expected Output Artifacts

Every completed run should leave enough evidence to reconstruct the reported row and diagnose anomalies:

- `suite_summary.csv`: suite-level summary table for a full suite run or a recovered case-shard bundle.
- `<case_id>/summary.csv`: per-case metric summary for one reported instance under one runner/controller line.
- `<case_id>/artifacts.csv`: manifest pointing to the exact trace files produced for that case.
- `*_activation_trace.json`, `*_protocol_trace.json`, `*_settlement_trace.json`, `*_world_event_trace.json`: runtime traces for activation, protocol messages, settlement, and world events.
- `framework_llm_decision_count`, `framework_fallback_contract_decision_count`, `framework_no_legal_action_count`, and `framework_decision_failure_trace`: decision-audit fields that distinguish model-backed behavior from fallback or execution failure.
- `progress_log.jsonl` and `progress_snapshot.json`: progress and failure bookkeeping for interrupted or partially recovered runs.

## Citation

If you use this benchmark before a formal citation file is added, cite the accompanying DESBench paper draft and include the repository commit hash used for your experiments.

## License

MIT. See [LICENSE](LICENSE).
