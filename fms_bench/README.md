# FMS-Bench

A data-driven flexible manufacturing system benchmark with an event-driven SMDP structure.

## Features

- Standard instance loading for Taillard JSSP and FJSPLib-compatible FJSP families
- Stochastic sampling utilities for processing variability and failures
- Discrete-event simulator (DES) with event-driven time advancement
- Gymnasium-compatible environment (`IndustrialFMS`) with `reset()` and `step()`
- Holonic plant environment (`HolonicPlantEnv`) with `Plant -> Cell -> Machine` structure
- Energy and carbon tracking with TOU pricing
- Reward decomposition (makespan, tardiness, energy, carbon)
- Evaluation protocol with metrics collection
- Holonic plant evaluation runner with multi-seed JSON/CSV outputs
- Baseline heuristics (FIFO, SPT, LPT, EDD, least-loaded-cell, Contract-Net MAS-like baselines, etc.)

## Package Layout

### Core Modules

- `fms_bench/config.py`: benchmark defaults and constraints
- `fms_bench/models.py`: dataclasses for jobs, operations, machines, factories
- `fms_bench/data_loader.py`: instance parsers (Taillard + generic FJSPLib families)
- `fms_bench/instance_catalog.py`: local instance family catalog and suite recommendations
- `fms_bench/distributions.py`: sampling utilities (Gamma, Poisson, Exponential)
- `fms_bench/simulator.py`: event queue and DES simulation logic
- `fms_bench/env.py`: Gymnasium environment wrapper for policy interaction
- `fms_bench/plant_env.py`: plant-level holonic environment over the shared DES kernel
- `fms_bench/constraints.py`: energy model and carbon tracker
- `fms_bench/suite_builder.py`: multi-cell plant assembly from public benchmark families
- `fms_bench/holonic_types.py`: contract, budget, feedback, typed event/message, and cell config dataclasses

### Extended Modules

- `fms_bench/rewards.py`: reward decomposition (time, makespan, tardiness, energy, carbon)
- `fms_bench/evaluation.py`: flat-environment evaluation protocol and metrics collection
- `fms_bench/holonic_evaluation.py`: plant-level multi-seed evaluation and result serialization
- `fms_bench/baselines.py`: baseline heuristics (Random, FIFO, SPT, LPT, EDD, etc.)
- `fms_bench/mas_baselines.py`: MAS-like Contract-Net plant baselines
- `fms_bench/instances/`: directory for raw benchmark instances

## Quick Start

```python
from fms_bench import IndustrialFMS, HolonicPlantEnv, BenchmarkConfig, PlantCellConfig, SPT_Policy, evaluate_policy

# Create environment
config = BenchmarkConfig()
env = IndustrialFMS(
    instance_path="instances/mt10.txt",
    instance_kind="brandimarte",
    config=config,
    seed=42,
)

# Run with baseline policy
policy = SPT_Policy()
results = evaluate_policy(
    policy=policy,
    instance_path="instances/mt10.txt",
    instance_kind="brandimarte",
    num_episodes=10,
    policy_name="spt",
)

print(results.summary())
```

Holonic plant skeleton:

```python
from pathlib import Path

from fms_bench import BenchmarkConfig, HolonicPlantEnv, PlantCellConfig

cfg = BenchmarkConfig()
cfg.failure.mtbf_hours = 0.0
cfg.failure.mttr_hours = 0.0

env = HolonicPlantEnv(
    cell_configs=[
        PlantCellConfig(
            cell_id=0,
            family_key="kacem",
            instance_kind="kacem",
            instance_path=Path("external/benchmarks/flexible jobshop/instances/KacemHammadiBorne2002/kacem1.txt"),
            wip_cap=1,
        ),
        PlantCellConfig(
            cell_id=1,
            family_key="fattahi",
            instance_kind="fattahi",
            instance_path=Path("external/benchmarks/flexible jobshop/instances/FattahiMehrabadJolai2007/fattahi1.txt"),
            wip_cap=1,
        ),
    ],
    config=cfg,
    seed=42,
)

obs, info = env.reset()
action = info["valid_cells"][0]
obs, reward, terminated, truncated, info = env.step(action)
```

Benchmark-facing plant API example:

```python
import numpy as np

obs, info = env.reset(seed=13)
plant_obs = env.observe("plant")
mask = env.action_mask("plant")
agents = env.agents_to_act()
step = env.step_dict({"plant": int(np.flatnonzero(mask)[0])})
print(agents, step.info["messages"], step.info["typed_events"])
```

See [../docs/benchmark_interface_v1.md](../docs/benchmark_interface_v1.md) for the stable public interface contract.

Holonic evaluation example:

```python
from pathlib import Path

from fms_bench import (
    BenchmarkConfig,
    LeastLoadedCellPolicy,
    PlantCellConfig,
    evaluate_holonic_policy,
)

cfg = BenchmarkConfig()
cfg.failure.mtbf_hours = 0.0
cfg.failure.mttr_hours = 0.0

results = evaluate_holonic_policy(
    policy=LeastLoadedCellPolicy(),
    cell_configs=[
        PlantCellConfig(
            cell_id=0,
            family_key="kacem",
            instance_kind="kacem",
            instance_path=Path("external/benchmarks/flexible jobshop/instances/KacemHammadiBorne2002/kacem1.txt"),
            wip_cap=1,
        ),
        PlantCellConfig(
            cell_id=1,
            family_key="fattahi",
            instance_kind="fattahi",
            instance_path=Path("external/benchmarks/flexible jobshop/instances/FattahiMehrabadJolai2007/fattahi1.txt"),
            wip_cap=1,
        ),
    ],
    config=cfg,
    seeds=[13, 17],
    benchmark_item_id="suite=tiny_smoke__level=L0__cells=kacem+fattahi__split=test",
)

print(results.summary())
```

Contract-Net MAS-like baseline example:

```python
from pathlib import Path

from fms_bench import (
    BenchmarkConfig,
    ContractNetGreedyPolicy,
    PlantCellConfig,
    evaluate_holonic_policy,
)

cfg = BenchmarkConfig()
cfg.failure.mtbf_hours = 0.0
cfg.failure.mttr_hours = 0.0

results = evaluate_holonic_policy(
    policy=ContractNetGreedyPolicy(),
    cell_configs=[
        PlantCellConfig(
            cell_id=0,
            family_key="kacem",
            instance_kind="kacem",
            instance_path=Path("external/benchmarks/flexible jobshop/instances/KacemHammadiBorne2002/kacem1.txt"),
            wip_cap=1,
        ),
        PlantCellConfig(
            cell_id=1,
            family_key="fattahi",
            instance_kind="fattahi",
            instance_path=Path("external/benchmarks/flexible jobshop/instances/FattahiMehrabadJolai2007/fattahi1.txt"),
            wip_cap=1,
        ),
    ],
    config=cfg,
    seeds=[13],
    benchmark_item_id="suite=tiny_smoke__level=L0__cells=kacem+fattahi__split=test",
)

print(results.summary())
```

## Installation

```bash
pip install -r requirements.txt
```

## Benchmark Instances

Place benchmark instances in `fms_bench/instances/` or point the environment at the local benchmark libraries under `external/benchmarks/`:

- **Taillard JSSP**: Standard JSSP format (n_jobs n_machines on first line)
- **FJSPLib families**: Brandimarte, Hurink, Dauzere-Peres-Paulli, Chambers-Barnes, Kacem, Fattahi, Behnke

See [../docs/benchmark_instance_suite.md](../docs/benchmark_instance_suite.md) for the recommended benchmark suite.

## License

MIT
