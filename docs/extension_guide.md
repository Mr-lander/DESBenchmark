# Extension Guide

DESBench is organized around one practical extension boundary: the benchmark
supplies the reported instance, event-driven simulator, legal-action masks,
traces, and metrics; the user supplies a MAS runner that receives public
decision payloads and returns legal actions.

The paper comparison is about coordination paradigm `Gamma`, not about making
one framework win. A runner may be implemented in any MAS framework or custom
orchestration stack, but it must receive the active agents' benchmark-visible
payloads and return exactly one legal action for each active agent.

## Public Release Paths

| Concept | Start here |
| --- | --- |
| Run a suite | `examples/run_suite.py` |
| Run selected reported cases | `examples/run_case.py` |
| Connect a MAS runner | `examples/run_mas.py` or implement `select_actions(decision)` |
| Example user paradigm | `examples/custom_paradigm.py` |
| Suite/profile construction | `fms_bench/experiment_runner.py` |
| Paradigm extension API | `fms_bench/paradigms.py` |
| Controller payload construction | runtime controller payload builder |
| Optional implementation carriers | framework-driver package |

Some source paths retain historical package names for compatibility. The
paper-facing public paths are `examples/run_suite.py`,
`examples/run_case.py`, `examples/run_mas.py`, and
`examples/custom_paradigm.py`.

## Connect A MAS Runner

Use this path when you already have a MAS architecture and want to evaluate it
on DESBench. Implement a runner with a `select_actions(decision)` method. The
`decision` object contains the current time, active agent ids, supported
execution mode, and one local public view per active Plant, Area, or Cell
agent. Each view includes an `action_mask`. Return a mapping from active agent
id to integer action id.

Minimal shape:

```python
class MyMASRunner:
    def select_actions(self, decision):
        actions = {}
        for agent_id, view in decision["active_agents"].items():
            actions[agent_id] = choose_legal_action(view, view["action_mask"])
        return actions
```

The runner may use message passing, role-specific agents, LLM calls, learned
policies, or an external process. It must not read simulator-private state,
submit actions for inactive agents, submit masked actions, mutate the reported
instance, or change metrics.

For a subprocess MAS, read one JSON decision object per line from stdin and
write one JSON action mapping per line to stdout. The public runner handles the
event loop:

```bash
uv run python examples/run_mas.py \
  --suite small \
  --case-limit 1 \
  --seed-limit 1 \
  --controller-command uv run python examples/subprocess_mask_first_controller.py
```

## Add A New Paradigm

Use this path when a researcher wants to compare a new MAS coordination design
against the official DESBench paradigms.

1. Create a standalone Python file, for example `examples/my_paradigm.py`.
2. Import `AuthorityMode` and `register_authority_mode`.
3. Register a new `mode_id` that describes the public paradigm name.
4. Choose the authority switches that define information flow, rejection rights, reroute authority, and mediated bidding.
5. Connect your MAS runner to the public payload.
6. Run `examples/run_suite.py` or `examples/run_mas.py` with `--paradigm-module examples/my_paradigm.py` and `--authority-mode <mode_id>`.
7. Keep the reported instance, seed split, runner boundary, backend/model, simulator, and metrics fixed when comparing against official modes.
8. Review decision audit fields before reporting the result.

Minimal example:

```python
from fms_bench.paradigms import AuthorityMode, register_authority_mode

register_authority_mode(
    AuthorityMode(
        mode_id="my_hybrid",
        routing_authority_mode="holonic_hybrid",
        cell_contract_actions="accept_reject",
        area_local_reroute_enabled=True,
        plant_reroute_required=False,
        mediated_bid_round_enabled=True,
    )
)
```

Smoke command:

```bash
uv run python examples/run_suite.py \
  --paradigm-module examples/my_paradigm.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode my_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

If the new paradigm needs new activation, message, or action semantics, the
registry file is not enough. Update the runtime consumers in
`tree_runtime_env.py`, `area_runtime.py`, `cell_runtime.py`, and
`controller_payloads.py`, then add regression tests for legal action masks,
activation order, reroute/escalation behavior, and exported audit fields.

## Add A New Suite Or Profile

A suite/profile is the benchmark instance layer. Add one when changing topology,
composition, route grammar, arrival profile, setup profile, disturbance family,
numeric profile, due-date profile, or difficulty knobs.

Minimum checklist:

1. Add a suite builder in `fms_bench/experiment_runner.py`.
2. Give the suite a stable public `suite_id` and each case a stable `profile_id`.
3. Keep paradigms as a builder argument where possible.
4. Register the suite in the public runner's `SUITE_BUILDERS`.
5. Add tests for case count, profile ids, paradigm expansion, and seed split.
6. Document the suite in `docs/suites.md`.

Do not reuse A5C12 or A3C9 profile names for a different topology or route
grammar. A changed instance family should receive a new suite id.

## Add A New Implementation Carrier

Add a carrier only when the same runner contract must be realized through a
new orchestration library or adapter. Most users do not need this: they can
connect their MAS through `select_actions` directly. A carrier must not read
simulator-private state or mutate benchmark semantics.

Minimum checklist:

1. Implement a driver under the framework-driver package.
2. Subclass the existing framework-backed driver base or provide the same external-session contract.
3. Return exactly one legal integer action for each active payload.
4. Reuse the shared LLM prompt surface unless the paper explicitly studies prompt differences.
5. Record decision sources and failures through the existing audit hooks.
6. Add no-key execution and invalid-response fallback tests.

Additional carriers should be reported as extension experiments unless the
evaluation matrix is explicitly updated.

## Add A New Controller Without A New Carrier

If the new method is only a policy over the existing payload, do not create a
new carrier. Implement a runner or controller that consumes the payload and
produces legal actions. This keeps the benchmark variable clean: the paradigm
remains `Gamma`, and the controller backend remains a separate implementation
detail.

## Validity Checklist

Before reporting an extension result, check:

- The suite id, profile id, paradigm id, runner or implementation name, model id, and seed split are present in the exported summary.
- The expected number of cases and seeds completed.
- Real-provider runs have nonzero `framework_llm_decision_count`.
- `framework_fallback_contract_decision_count` and `framework_no_legal_action_count` are reviewed and explained.
- `decision_failure_trace` has no unreviewed `empty_response` or `exception` entries for paper-grade response-backed runs.
- Any changed paradigm semantics are documented separately from implementation-carrier details.
