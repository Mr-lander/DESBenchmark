# A3C9 Benchmark Public Export

This repository is a clean first public export of the runnable A3C9 benchmark slice from the larger internal benchmark workspace.

Current public scope:

- shared DES manufacturing benchmark code under `fms_bench/`
- runnable A3C9 Category B benchmark surface
- native framework runner entrypoints for `LangGraph` and `AgentScope`
- minimal benchmark instance files required by the shipped code and focused tests
- focused tests for A3C9 runtime contracts and the core runtime substrate

Not included in this export:

- result artifacts
- checked-in experiment outputs
- AI orchestration state
- worktrees, logs, scratch analysis files
- paper bundles and release packaging history

## A3C9

`A3C9` is the Category B inter-cell benchmark family with:

- `tree_a3_c9` topology
- `3` areas
- `9` cells
- one shared DES world
- explicit setup, transport, finite inbound admission, and downstream blocking

The main public runnable surface in this export is `intercell_a3c9_wide`.

That surface expands into:

- `3` scenario profiles
- `4` authority modes
- `12` benchmark cases total

Suite and authority contracts:

- [docs/tree_runtime_category_b_a3c9_suite_matrix_v1.md](docs/tree_runtime_category_b_a3c9_suite_matrix_v1.md)
- [docs/2026-04-10-category-b-authority-modes.md](docs/2026-04-10-category-b-authority-modes.md)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Category B wrapper:

```bash
python examples/run_tree_runtime_v2_category_b.py --framework langgraph --suite intercell_a3c9_wide
```

Run the native framework runner directly:

```bash
python examples/run_tree_runtime_v2_framework_native.py --framework agentscope --suite intercell_a3c9_wide
```

Run only one authority mode:

```bash
python examples/run_tree_runtime_v2_framework_native.py --framework langgraph --suite intercell_a3c9_wide --authority-mode hierarchical
```

Debug with fewer cases or seeds:

```bash
python examples/run_tree_runtime_v2_framework_native.py --framework langgraph --suite intercell_a3c9_wide --case-limit 1 --seed-limit 1
```

## Repository Tree

```text
FMS-Benchmark/
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- uv.lock
|-- docs/
|   |-- tree_runtime_category_b_a3c9_suite_matrix_v1.md
|   |-- 2026-04-10-category-b-authority-modes.md
|   `-- benchmark_seed_protocol_v1.json
|-- examples/
|   |-- run_tree_runtime_v2_category_b.py
|   `-- run_tree_runtime_v2_framework_native.py
|-- fms_bench/
|   |-- core benchmark package
|   `-- v2/
|       `-- runtime truth layer
|-- tests/
|   `-- focused A3C9 and runtime tests
`-- external/
    `-- benchmarks/flexible jobshop/instances/
```

## `fms_bench/` File Guide

### Package root

- `__init__.py`: public package exports.
- `README.md`: package-level quick start and module summary.
- `config.py`: benchmark defaults, profile IDs, and top-level configuration objects.
- `models.py`: core scheduling dataclasses such as jobs, operations, and options.
- `holonic_types.py`: plant/cell config objects, typed contracts, events, messages, and summaries.
- `data_loader.py`: parsers for public JSP/FJSP instance files.
- `instance_catalog.py`: local catalog of supported benchmark families and their source directories.
- `suite_builder.py`: builds multi-cell plant assemblies from public instance families.
- `tree_builder.py`: constructs stable Plant/Area/Cell topologies such as `tree_a3_c9`.
- `distributions.py`: shared stochastic sampling helpers.
- `simulator.py`: shared DES kernel and event queue progression logic.
- `constraints.py`: energy, tariff, carbon, and resource-accounting helpers.
- `rewards.py`: reward and interval cost decomposition.
- `env.py`: flat Gym-style environment wrapper.
- `plant_env.py`: plant-level environment over the shared DES world.
- `shared_backlog_env.py`: shared-backlog assignment environment.
- `tree_env.py`: earlier tree benchmark environment path.
- `tree_shared_backlog_env.py`: tree plus shared-backlog environment path.
- `tree_intercell_env.py`: Category B inter-cell environment substrate with setup, transport, buffer, and blocking semantics.
- `evaluation.py`: flat-environment evaluation helpers.
- `holonic_evaluation.py`: multi-seed benchmark evaluation, summaries, and result serialization.
- `experiment_runner.py`: suite builders, benchmark item identity generation, and batch runner entrypoints.
- `baselines.py`: non-MAS heuristic baselines.
- `mas_baselines.py`: MAS-style baselines such as Contract-Net variants.
- `tree_policies.py`: tree-runtime policy helpers and hand-written policy logic.
- `arrival_profiles.py`: named arrival-process profiles.
- `due_date_profiles.py`: named due-date profiles.
- `setup_profiles.py`: named setup/changeover profiles.
- `failure_profiles.py`: named disturbance/failure profiles.
- `authority_profiles.py`: named numeric/energy/carbon profile overlays.
- `paper_artifacts.py`: builds publication-oriented derived tables and report assets.
- `trace_figures.py`: derives trace-facing figure inputs and reports.
- `topology_replay.py`: builds replay artifacts over the topology graph.
- `release_manifest.py`: machine-readable release metadata helpers.
- `instances/`: tiny local test instance data packaged with the repo.

### `fms_bench/v2/`

- `__init__.py`: exports the v2 runtime surface.
- `types.py`: v2 runtime dataclasses and strongly typed protocol structures.
- `task_semantics.py`: work package, contract, and commitment semantics.
- `event_interpreter.py`: translates DES events into benchmark-visible decision epochs.
- `area_runtime.py`: runtime logic for Area agents.
- `cell_runtime.py`: runtime logic for Cell agents.
- `runtime_ledger.py`: settlement, completion, reward, and accounting ledger.
- `tree_runtime_env.py`: interpreter-driven v2 tree runtime on top of the shared DES world.
- `intercell_tree_runtime_env.py`: Category B A3C9-capable inter-cell v2 runtime.
- `authority_modes.py`: frozen authority-mode definitions such as `centralized`, `hierarchical`, `mediated_heterarchical_cnp`, and `holonic_hybrid`.
- `controller_api.py`: legal controller boundary and adapter helpers.
- `controller_families.py`: shared controller-family wiring.
- `controller_payloads.py`: structured payloads exchanged by controller-facing runtime code.
- `external_controller_adapter.py`: adapter for external-session or out-of-process controllers.
- `external_tree_drivers.py`: stronger external controller-family implementations.
- `framework_envs.py`: framework-specific environment metadata and install expectations.
- `learned_tree_driver.py`: learned controller path.
- `rl_tree_driver.py`: RL-specific driver path.
- `category_b_controller_families.py`: Category B controller-family helpers and selectors.
- `framework_drivers/`: framework-specific drivers for LangGraph, AgentScope, AG2, CrewAI, Google ADK, and SPADE.

## Notes

- This export is intentionally code-first.
- It is designed to make the A3C9 benchmark slice runnable without carrying internal result history.
- If you publish frozen tables later, add a separate `results/` or `artifacts/` directory rather than mixing them into the code root.
