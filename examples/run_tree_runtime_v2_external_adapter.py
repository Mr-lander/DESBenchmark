from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig, ExternalControllerSession, PlantCellConfig, run_external_controller_loop
from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv


FLEX_ROOT = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"


def make_demo_env() -> TreeRuntimeEnv:
    config = BenchmarkConfig()
    config.horizon_hours = 200
    config.failure.mtbf_hours = 0.0
    config.failure.mttr_hours = 0.0

    common = {
        "family_key": "brandimarte",
        "instance_path": FLEX_ROOT / "Brandimarte1993" / "mk01.txt",
        "instance_kind": "brandimarte",
        "wip_cap": 2,
    }
    cell_configs = [
        PlantCellConfig(
            cell_id=0,
            label="cell-slow-tight",
            processing_time_scale=1.38,
            energy_cap_kwh=220.0,
            carbon_cap_kg=132.0,
            **common,
        ),
        PlantCellConfig(
            cell_id=1,
            label="cell-fast-loose",
            processing_time_scale=0.74,
            energy_cap_kwh=1500.0,
            carbon_cap_kg=900.0,
            **common,
        ),
        PlantCellConfig(
            cell_id=2,
            label="cell-medium",
            processing_time_scale=1.0,
            energy_cap_kwh=780.0,
            carbon_cap_kg=468.0,
            **common,
        ),
    ]
    return TreeRuntimeEnv(
        cell_configs=cell_configs,
        area_to_cells={0: [0, 1], 1: [2]},
        config=config,
        seed=13,
    )


def mask_first_driver(decision: dict[str, object]) -> dict[str, int]:
    actions: dict[str, int] = {}
    for agent_id, agent_view in dict(decision["active_agents"]).items():
        mask = list(agent_view["action_mask"])
        valid = [index for index, enabled in enumerate(mask) if enabled]
        actions[str(agent_id)] = valid[0] if valid else 0
    return actions


def main() -> None:
    env = make_demo_env()
    try:
        session = ExternalControllerSession(env)
        decision = session.reset(seed=13)
        print("reset agents:", decision["agents_to_act"])

        transitions = run_external_controller_loop(session, mask_first_driver, max_steps=3)
        for step_index, transition in enumerate(transitions, start=1):
            next_agents = transition["current_decision"]["agents_to_act"]
            print(
                f"step={step_index} actions={transition['submitted_actions']} "
                f"reward={transition['reward']:.6f} next={next_agents}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
