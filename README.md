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
Benchmark-public-v1/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ requirements.txt
├─ uv.lock
├─ docs/
│  ├─ tree_runtime_category_b_a3c9_suite_matrix_v1.md
│  └─ 2026-04-10-category-b-authority-modes.md
├─ examples/
│  ├─ run_tree_runtime_v2_category_b.py
│  └─ run_tree_runtime_v2_framework_native.py
├─ fms_bench/
│  ├─ config.py
│  ├─ holonic_types.py
│  ├─ simulator.py
│  ├─ tree_intercell_env.py
│  ├─ experiment_runner.py
│  ├─ arrival_profiles.py
│  ├─ due_date_profiles.py
│  ├─ setup_profiles.py
│  ├─ failure_profiles.py
│  ├─ authority_profiles.py
│  ├─ suite_builder.py
│  ├─ tree_builder.py
│  └─ v2/
│     ├─ task_semantics.py
│     ├─ event_interpreter.py
│     ├─ area_runtime.py
│     ├─ cell_runtime.py
│     ├─ runtime_ledger.py
│     ├─ intercell_tree_runtime_env.py
│     ├─ authority_modes.py
│     └─ category_b_controller_families.py
├─ tests/
│  ├─ test_a3c9_runtime_contract.py
│  ├─ test_intercell_authority_modes.py
│  ├─ test_tree_intercell_env.py
│  └─ core runtime/profile tests
└─ external/
   └─ benchmarks/flexible jobshop/instances/
      ├─ Brandimarte1993/
      ├─ KacemHammadiBorne2002/
      ├─ FattahiMehrabadJolai2007/
      └─ HurinkJurischThole1994/
```

## Notes

- This export is intentionally code-first.
- It is designed to make the A3C9 benchmark slice runnable without carrying internal result history.
- If you publish frozen tables later, add a separate `results/` or `artifacts/` directory rather than mixing them into the code root.
