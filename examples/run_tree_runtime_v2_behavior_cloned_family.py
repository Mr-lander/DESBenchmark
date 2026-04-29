from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import (
    ExternalHierarchicalTreeDriver,
    build_tree_runtime_v2_pilot_suite,
    build_tree_runtime_v2_small_suite,
    collect_behavior_cloning_examples,
    fit_behavior_cloned_tree_driver,
    fit_mlp_behavior_cloned_tree_driver,
)
from fms_bench.experiment_runner import run_experiment_suite

def _case_env_factory(case):
    def factory(seed: int):
        from fms_bench.experiment_runner import make_regime_config

        config = make_regime_config(case.regime_id) if case.regime_id is not None else None
        return case.env_cls(case.cell_configs, config, seed)

    return factory


def main() -> None:
    teacher = ExternalHierarchicalTreeDriver()
    train_cases = build_tree_runtime_v2_pilot_suite(PROJECT_ROOT, seed_split="dev")[:4]
    train_examples = []
    for case in train_cases:
        train_examples.extend(
            collect_behavior_cloning_examples(
                teacher=teacher,
                env_factory=_case_env_factory(case),
                seeds=case.seeds[:2],
                max_decisions=48,
            )
        )

    learned_driver = fit_behavior_cloned_tree_driver(
        train_examples,
        epochs=200,
        lr=0.05,
        controller_name="bc_tree_hierarchical",
    )
    mlp_learned_driver = fit_mlp_behavior_cloned_tree_driver(
        train_examples,
        hidden_dim=12,
        epochs=240,
        lr=0.05,
        seed=11,
        controller_name="mlp_bc_tree_hierarchical",
    )

    eval_cases = build_tree_runtime_v2_small_suite(PROJECT_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        PROJECT_ROOT
        / "dist"
        / "benchmark_runs"
        / f"tree_runtime_v2_small_behavior_cloned_family_{timestamp}"
    )
    summary_path = run_experiment_suite(
        cases=eval_cases,
        policies=[teacher, learned_driver, mlp_learned_driver],
        output_dir=output_dir,
    )
    print(f"trained_examples={len(train_examples)}")
    print(summary_path)


if __name__ == "__main__":
    main()
