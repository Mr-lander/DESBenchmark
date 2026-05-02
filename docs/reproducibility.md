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
uv run python examples/run_tree_runtime_v2_framework_native.py --framework langgraph --suite intercell_a5c12_harder --case-limit 1 --seed-limit 1
```

## One A5C12 Case

```bash
uv run python examples/run_tree_runtime_v2_framework_native.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode holonic_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

Replace `langgraph` with `agentscope` to exercise the other formal framework implementation.

## One A3C9 Case

Use A3C9 for a smaller standard surface with the same paradigm/framework axes:

```bash
uv run python examples/run_tree_runtime_v2_framework_native.py \
  --framework langgraph \
  --suite intercell_a3c9_wide \
  --authority-mode holonic_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

Report A3C9 results with `suite_id=intercell_a3c9_wide`; do not merge them into A5C12 tables without an explicit suite column.

## Real Provider Runs

Use environment variables for credentials. Do not write real keys into files or command history artifacts.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "<provider-token>"
uv run python examples/run_tree_runtime_v2_framework_native.py `
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
uv run python examples/run_tree_runtime_v2_framework_native.py `
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

## Result Boundary

Generated benchmark outputs, raw traces, and paper result CSVs are intentionally outside this source repository. Store run artifacts under a separate output directory, then promote only reviewed summaries into paper or release materials.
