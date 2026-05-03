# Implementation Carriers

Implementation carriers realize the same benchmark-facing controller contract. In the formal A5C12 matrix, the supported carriers are `agentscope` and `langgraph`.

## Base Contract

The current carrier drivers live under the framework-driver package in the source tree.

Important contract points:

- The driver receives per-agent controller payloads only.
- Each active agent must return one legal integer action.
- The driver records decision source traces.
- The driver exposes runtime metadata, including whether the optional carrier package is available.
- The no-key path can run with contract-stub behavior when optional native framework packages are absent.

Decision source audit fields distinguish:

- `llm`
- `fallback_contract`
- `no_legal_action`

Paper-grade real LLM runs must review these counts. A run with unexpected fallback decisions should not be treated as directly comparable to a fully response-backed LLM run without review.

## AgentScope

Main classes:

- `AgentScopeTreeDriver`
- `AgentScopeLLMTreeDriver`

## LangGraph

Main classes:

- `LangGraphTreeDriver`
- `LangGraphLLMTreeDriver`

## Prompt Surface

The LLM-backed AgentScope and LangGraph drivers share one OpenAI-compatible prompt builder. Carrier differences should therefore come from implementation/runtime behavior and not from divergent prompt wording.

## Adding A Carrier

A new carrier should be treated as another realization of the same controller contract, not as a new benchmark task.

Required registration points:

- Implement a driver under the source tree's framework-driver package.
- Subclass `FrameworkBackedTreeDriver`.
- Export the new driver from the framework-driver package.
- Add an optional dependency/environment entry if the framework has a native package.
- Add the carrier key to the public suite and case runners.
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
