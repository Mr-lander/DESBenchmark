# Extension Guide

This repository is organized around one experimental separation:

```text
coordination paradigm Gamma = benchmark variable
framework driver = implementation carrier
controller backend = deterministic, LLM-backed, or user-provided action selector
```

The paper comparison is about the coordination paradigm, not about making one framework "win." A framework adapter should therefore preserve the same benchmark-visible controller contract: receive the active agent's local payload and return exactly one legal action.

## Public Release Paths

| Concept | Repository path |
| --- | --- |
| Suite/profile construction | `fms_bench/experiment_runner.py` |
| Paradigm registry | `fms_bench/v2/authority_modes.py` |
| Runtime use of authority switches | `fms_bench/v2/tree_runtime_env.py`, `fms_bench/v2/area_runtime.py`, `fms_bench/v2/cell_runtime.py` |
| Controller payload construction | `fms_bench/v2/controller_payloads.py` |
| Framework base contract | `fms_bench/v2/framework_drivers/base.py` |
| Formal framework implementations | `fms_bench/v2/framework_drivers/agentscope_driver.py`, `fms_bench/v2/framework_drivers/langgraph_driver.py` |
| Shared LLM prompt surface | `fms_bench/v2/framework_drivers/llm_prompting.py` |
| Main one-line runner | `examples/run_tree_runtime_v2_framework_native.py` |
| Explicit case-shard runner | `examples/run_tree_runtime_v2_framework_case_shard.py` |

## Add A New Suite Or Profile

A suite/profile is the benchmark instance layer. Add one when you want to change topology, composition, route grammar, arrival profile, setup profile, disturbance family, numeric profile, due-date profile, or difficulty knobs.

Minimum implementation checklist:

1. Add a suite builder in `fms_bench/experiment_runner.py`.
2. Give the suite a stable public `suite_id` and each case a stable `profile_id`.
3. Keep `authority_modes` as a builder argument where possible, so the same suite can evaluate multiple paradigms.
4. Register the suite in `SUITE_BUILDERS` in `examples/run_tree_runtime_v2_framework_native.py`.
5. Add tests for case count, profile ids, authority-mode expansion, and seed split.
6. Document the suite in `docs/suites.md` and distinguish release/reporting suites from smoke/debug suites.

Do not reuse A5C12 or A3C9 profile names for a different topology or route grammar. A changed instance family should receive a new suite id.

## Add A New Paradigm

A paradigm is an instantiation of the paper's coordination protocol `Gamma`. In this codebase, the release-facing paradigms are represented by `AuthorityMode` fields:

```text
routing_authority_mode
cell_contract_actions
area_local_reroute_enabled
plant_reroute_required
mediated_bid_round_enabled
```

Use the narrow path when the new paradigm can be expressed by these fields:

1. Add a new `AuthorityMode` entry in `_AUTHORITY_MODE_REGISTRY` in `fms_bench/v2/authority_modes.py`.
2. Add the new id to `MAINLINE_CATEGORY_B_AUTHORITY_MODES` in `fms_bench/experiment_runner.py` only if it should be part of an expanded public matrix.
3. Run or add tests that call `get_authority_mode(...)`, build the target suite with `authority_modes=(...)`, and execute a one-seed no-key smoke.
4. Document the new mode in `docs/paradigms.md` and label it as an extension, not one of the four formal A5C12 paper modes, unless the paper matrix is deliberately revised.

Use the wider path when the paradigm needs new activation, message, or action semantics:

1. Add the registry entry in `authority_modes.py`.
2. Update the runtime semantics where the switches are consumed, especially `tree_runtime_env.py`, `area_runtime.py`, `cell_runtime.py`, and `controller_payloads.py`.
3. Add regression tests for legal action sets, activation order, reroute/escalation behavior, and exported audit fields.
4. Add a small no-key fixture or smoke case before running real-provider experiments.

Do not change topology, route grammar, profile knobs, seed split, or metrics while claiming the result is a pure paradigm comparison. Those are different experimental axes.

## Add A New Framework Adapter

A framework adapter is a controller realization layer. It must not read simulator-private state or mutate benchmark semantics.

Minimum implementation checklist:

1. Create `fms_bench/v2/framework_drivers/<framework>_driver.py`.
2. Subclass `FrameworkBackedTreeDriver` from `fms_bench/v2/framework_drivers/base.py`.
3. Set `framework_name`, `python_package`, and `controller_name`.
4. Implement `_select_framework_action(...)` so every active agent returns one legal integer action.
5. If the adapter is LLM-backed, use `llm_prompting.py` or prove that the prompt surface is equivalent.
6. Record decision sources through `_record_decision_source(...)`; record exhausted failures through `_record_decision_failure(...)`.
7. Export the driver in `fms_bench/v2/framework_drivers/__init__.py`.
8. Add a `FrameworkEnvSpec` in `fms_bench/v2/framework_envs.py` if the framework has an optional native package or separate uv environment.
9. Register the key in `FRAMEWORK_CHOICES`, `_build_driver(...)`, and `_build_llm_driver(...)` in `examples/run_tree_runtime_v2_framework_native.py`.
10. Register the same key in `examples/run_tree_runtime_v2_framework_case_shard.py` if case-sharded runs should support it.
11. Add tests that cover no-key execution, strict-framework dependency behavior, LLM argument threading, decision-source audit counts, and invalid-response fallback behavior.

The formal A5C12 matrix uses only `agentscope` and `langgraph`. Additional adapters can be useful for reproducibility and external comparison, but they should be reported as extension experiments unless the evaluation matrix is explicitly updated.

## Add A New Controller Without A New Framework

If the new method is just a policy over the existing payload, do not create a framework adapter. Implement a controller that consumes the same payload and produces a legal action, then wire it through a runner or test harness as a policy object. This keeps the benchmark variable clean: the paradigm remains `Gamma`, and the controller backend remains a separate axis.

## Run The Formal Suite Shape

For a single LLM, the A5C12 headline release matrix is:

```text
3 profiles x 4 paradigms x 2 frameworks = 24 cases
```

The A3C9 standard suite has the same 24-case per-model shape under `intercell_a3c9_wide`.

The suite builders expand profile and authority-mode combinations in `build_tree_runtime_v2_intercell_a5c12_harder_suite(...)` and `build_tree_runtime_v2_intercell_a3c9_wide_suite(...)`. The runner selects frameworks through `--framework` and authority modes through repeated `--authority-mode`.

No-key smoke:

```bash
uv run python examples/run_tree_runtime_v2_framework_native.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode holonic_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

Real-provider run with bounded retry/backoff:

```bash
uv run python examples/run_tree_runtime_v2_framework_native.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode holonic_hybrid \
  --llm-backend openai_compat \
  --llm-model "<model-id>" \
  --llm-base-url "https://<provider-host>/v1" \
  --llm-timeout 90 \
  --llm-retries 7 \
  --llm-retry-backoff-seconds 1 \
  --llm-retry-backoff-max-seconds 60 \
  --real-llm-response-backed
```

Keep credentials in environment variables only. Do not commit provider keys, raw run logs, or interrupted partial runs.

## Validity Checklist For Extensions

Before reporting an extension result, check:

- The suite id, profile id, authority mode, framework key, model id, and seed split are present in the exported summary.
- The expected number of cases and seeds completed.
- `framework_llm_decision_count` is nonzero for real-provider runs.
- `framework_fallback_contract_decision_count` and `framework_no_legal_action_count` are reviewed and explained.
- `decision_failure_trace` has no unreviewed `empty_response` or `exception` entries for paper-grade response-backed runs.
- Any changed paradigm semantics are documented separately from framework implementation details.
