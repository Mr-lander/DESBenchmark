# Framework Adapters

Framework adapters are implementation carriers for the same benchmark-facing controller contract. In the formal A5C12 matrix, the supported framework implementations are `agentscope` and `langgraph`.

## Base Contract

The base driver is `FrameworkBackedTreeDriver` in `fms_bench/v2/framework_drivers/base.py`.

Important contract points:

- The driver receives per-agent controller payloads only.
- Each active agent must return one legal integer action.
- The driver records decision source traces.
- The driver exposes framework runtime metadata, including whether the optional framework package is available.
- The no-key path can run with contract-stub behavior when optional native framework packages are absent.

Decision source audit fields distinguish:

- `llm`
- `fallback_contract`
- `no_legal_action`

Paper-grade real LLM runs must review these counts. A run with unexpected fallback decisions should not be treated as directly comparable to a fully response-backed LLM run without review.

## AgentScope

Implementation path:

```text
fms_bench/v2/framework_drivers/agentscope_driver.py
```

Main classes:

- `AgentScopeTreeDriver`
- `AgentScopeLLMTreeDriver`

## LangGraph

Implementation path:

```text
fms_bench/v2/framework_drivers/langgraph_driver.py
```

Main classes:

- `LangGraphTreeDriver`
- `LangGraphLLMTreeDriver`

## Prompt Surface

The LLM-backed AgentScope and LangGraph drivers share the OpenAI-compatible prompt builder in `fms_bench/v2/framework_drivers/llm_prompting.py`. Framework differences should therefore come from implementation/runtime behavior and not from divergent prompt wording.

## Adding A Framework

A new framework adapter should be treated as another realization of the same controller contract, not as a new benchmark task.

Required registration points:

- Implement `fms_bench/v2/framework_drivers/<framework>_driver.py`.
- Subclass `FrameworkBackedTreeDriver`.
- Export the new driver in `fms_bench/v2/framework_drivers/__init__.py`.
- Add an optional dependency/environment entry in `fms_bench/v2/framework_envs.py` if the framework has a native package.
- Add the framework key to `FRAMEWORK_CHOICES`, `_build_driver(...)`, and `_build_llm_driver(...)` in `examples/run_tree_runtime_v2_framework_native.py`.
- Add the key to `examples/run_tree_runtime_v2_framework_case_shard.py` if explicit case shards should support it.
- Add tests for no-key execution, strict dependency behavior, LLM argument threading, and decision audit fields.

For LLM-backed adapters, keep the same prompt and output schema unless the experiment explicitly studies prompt variation. The current valid response surface is one JSON object:

```json
{"action": 0}
```

The action value must be one of the legal integer action indices in the active agent's payload.

## Provider Retry Boundary

The AgentScope and LangGraph LLM drivers support per-decision retry with exponential backoff:

```bash
--llm-retries 7
--llm-retry-backoff-seconds 1
--llm-retry-backoff-max-seconds 60
```

Recovered transient failures do not enter `decision_failure_trace`. A failure is recorded only after retries are exhausted, and the driver then falls back according to the selected validation mode. Paper-grade runs should still inspect fallback counts and failure traces before inclusion.
