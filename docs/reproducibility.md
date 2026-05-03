# Reproducibility

This repository is designed for code review, smoke testing, and benchmark extension. It is not a bundled result archive.

## Environment

Use Python 3.12 or newer.

Recommended:

```bash
uv sync --extra test
```

Optional native framework environments:

```bash
uv sync --extra test --extra framework_langgraph
uv sync --extra test --extra framework_agentscope
```

Pip fallback:

```bash
python -m pip install -e ".[test]"
```

Optional native framework dependencies are declared as extras in `pyproject.toml`. The default no-key smoke path uses `--llm-backend none`.

## Smoke Tests

```bash
uv run pytest tests/test_tree_runtime_env.py -q
uv run pytest tests/test_experiment_runner.py -q
```

Run a one-case, one-seed no-key benchmark smoke:

```bash
uv run python examples/run_suite.py --framework langgraph --suite intercell_a5c12_harder --case-limit 1 --seed-limit 1
```

Run a one-case MAS subprocess smoke:

```bash
uv run python examples/run_mas.py --suite small --case-limit 1 --seed-limit 1 --controller-command uv run python examples/subprocess_mask_first_controller.py
```

## One A5C12 Case

```bash
uv run python examples/run_suite.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode holonic_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

Replace `langgraph` with `agentscope` to exercise the other formal implementation carrier.

## One A3C9 Case

Use A3C9 for a smaller standard surface with the same paradigm/framework axes:

```bash
uv run python examples/run_suite.py \
  --framework langgraph \
  --suite intercell_a3c9_wide \
  --authority-mode holonic_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

Report A3C9 results with `suite_id=intercell_a3c9_wide`; do not merge them into A5C12 tables without an explicit suite column.

## Paper Reported Instances

If you want to reproduce the paper-facing public comparison surface, use these four reported instances:

| Paper ID | `suite_id` | `profile_id` |
| --- | --- | --- |
| `A3C9-1` | `intercell_a3c9_wide` | `cross_area_tradeoff` |
| `A5C12-1` | `intercell_a5c12_harder` | `branch_pressure_k4_tm115_bk4` |
| `A5C12-2` | `intercell_a5c12_harder` | `cluster_pull_k4_tm120_bk4` |
| `A5C12-3` | `intercell_a5c12_harder` | `late_commit_soft_k4_tm125_bk4` |

Use the full-suite runner when you want the complete matrix for a suite surface. Use `examples/run_case.py` when you want one exact reported instance or a small selected bundle under a shared output root.

## Real Provider Runs

Use environment variables for credentials. Do not write real keys into files or command history artifacts.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "<provider-token>"
uv run python examples/run_suite.py `
  --framework agentscope `
  --suite intercell_a5c12_harder `
  --authority-mode heterarchical_cnp `
  --case-limit 1 `
  --seed-limit 1 `
  --llm-backend openai_compat `
  --llm-model "<model-id>" `
  --llm-base-url "https://<provider-host>/v1"
```

Before treating a real-provider run as valid, inspect:

- completion of all expected cases and seeds
- `framework_llm_decision_count`
- `framework_fallback_contract_decision_count`
- `framework_no_legal_action_count`
- decision failure traces

For OpenAI-compatible endpoints, the runner exposes bounded retry/backoff controls:

```powershell
$env:OPENAI_API_KEY = "<provider-token>"
uv run python examples/run_suite.py `
  --framework langgraph `
  --suite intercell_a5c12_harder `
  --authority-mode holonic_hybrid `
  --llm-backend openai_compat `
  --llm-model "<model-id>" `
  --llm-base-url "https://<provider-host>/v1" `
  --llm-timeout 90 `
  --llm-retries 7 `
  --llm-retry-backoff-seconds 1 `
  --llm-retry-backoff-max-seconds 60 `
  --real-llm-response-backed
```

AgentScope and LangGraph use exponential backoff capped at `--llm-retry-backoff-max-seconds`. Runs should be reported only after all expected cases and seeds have completed and the decision audit fields have been reviewed.

## Formal A5C12 Matrix

For one LLM, run both formal frameworks and all four official authority modes over the `intercell_a5c12_harder` suite:

```text
framework in {agentscope, langgraph}
authority-mode in {centralized, hierarchical, heterarchical_cnp, holonic_hybrid}
profiles expanded by suite builder = branch_pressure, cluster_pull, late_commit_soft
test seeds from docs/benchmark_seed_protocol_v1.json
```

Use `--case-limit` and `--seed-limit` only for debugging or smoke tests. Do not report limited runs as the full 24-case matrix.

## Standard A3C9 Matrix

The included A3C9 suite has the same per-model matrix shape:

```text
framework in {agentscope, langgraph}
authority-mode in {centralized, hierarchical, heterarchical_cnp, holonic_hybrid}
profiles expanded by suite builder = balanced_branching_anchor, cross_area_tradeoff, late_stage_reversal
test seeds from docs/benchmark_seed_protocol_v1.json
```

## Output Artifact Contract

For reviewed runs, expect at least these public artifacts:

- `suite_summary.csv` for the recovered suite-level summary.
- `<case_id>/summary.csv` for the per-case summary row.
- `<case_id>/artifacts.csv` for the trace manifest.
- `*_activation_trace.json`, `*_protocol_trace.json`, `*_settlement_trace.json`, and `*_world_event_trace.json` for runtime diagnosis.
- decision-audit fields including `framework_llm_decision_count`, `framework_fallback_contract_decision_count`, `framework_no_legal_action_count`, and `framework_decision_failure_trace`.
- `progress_log.jsonl` and `progress_snapshot.json` for interrupted or partial runs.

## Result Boundary

Generated benchmark outputs, raw traces, and paper result CSVs are intentionally outside this source repository. Store run artifacts under a separate output directory, then promote only reviewed summaries into paper or release materials.
