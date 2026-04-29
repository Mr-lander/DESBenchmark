from __future__ import annotations

from argparse import ArgumentParser
import copy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import (
    build_tree_runtime_v2_ladder_suite,
    build_tree_runtime_v2_pilot_suite,
    build_tree_runtime_v2_small_suite,
    build_tree_runtime_v2_wide_suite,
    train_policy_gradient_tree_driver,
)
from fms_bench.experiment_runner import run_experiment_suite


SUITE_BUILDERS = {
    "small": build_tree_runtime_v2_small_suite,
    "pilot": build_tree_runtime_v2_pilot_suite,
    "wide": build_tree_runtime_v2_wide_suite,
    "ladder": build_tree_runtime_v2_ladder_suite,
}


def _case_env_factory(case):
    def factory(seed: int):
        from fms_bench.experiment_runner import make_regime_config

        config = make_regime_config(case.regime_id) if case.regime_id is not None else None
        return case.env_cls(case.cell_configs, config, seed)

    return factory


def _cycling_suite_env_factory(cases: list[object]):
    case_factories = [_case_env_factory(case) for case in cases]
    state = {"index": 0}

    def factory(seed: int):
        case_factory = case_factories[state["index"] % len(case_factories)]
        state["index"] += 1
        return case_factory(seed)

    return factory


def _parse_args() -> object:
    parser = ArgumentParser(description="Train and evaluate a policy-gradient tree driver on the public v2 runtime boundary.")
    parser.add_argument("--train-suite", choices=sorted(SUITE_BUILDERS), default="pilot")
    parser.add_argument("--eval-suite", choices=sorted(SUITE_BUILDERS), default="small")
    parser.add_argument("--train-split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--eval-split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--train-case-limit", type=int, default=4)
    parser.add_argument("--eval-case-limit", type=int, default=None)
    parser.add_argument("--train-seed-limit", type=int, default=2)
    parser.add_argument("--eval-seed-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--episodes-per-epoch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-seed", type=int, default=7)
    parser.add_argument("--max-decisions-per-episode", type=int, default=48)
    parser.add_argument("--controller-name", default="policy_gradient_tree_driver")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _limit_cases(cases: list[object], *, case_limit: int | None, seed_limit: int | None) -> list[object]:
    limited = list(cases)
    if case_limit is not None:
        limited = limited[: max(int(case_limit), 0)]
    if seed_limit is None:
        return limited
    result: list[object] = []
    for case in limited:
        limited_seeds = list(case.seeds[: max(int(seed_limit), 0)])
        try:
            result.append(replace(case, seeds=limited_seeds))
            continue
        except TypeError:
            copied = copy.copy(case)
            copied.seeds = limited_seeds
            result.append(copied)
    return result


def main() -> None:
    args = _parse_args()
    train_cases = SUITE_BUILDERS[str(args.train_suite)](PROJECT_ROOT, seed_split=str(args.train_split))
    eval_cases = SUITE_BUILDERS[str(args.eval_suite)](PROJECT_ROOT, seed_split=str(args.eval_split))
    train_cases = _limit_cases(
        train_cases,
        case_limit=args.train_case_limit,
        seed_limit=args.train_seed_limit,
    )
    eval_cases = _limit_cases(
        eval_cases,
        case_limit=args.eval_case_limit,
        seed_limit=args.eval_seed_limit,
    )
    if not train_cases:
        raise SystemExit("No training cases selected.")
    if not eval_cases:
        raise SystemExit("No evaluation cases selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "dist"
            / "benchmark_runs"
            / f"tree_runtime_v2_{args.eval_suite}_policy_gradient_{timestamp}"
        )
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    training_env_factory = _cycling_suite_env_factory(train_cases)
    train_seeds: list[int] = []
    for case in train_cases:
        train_seeds.extend(int(seed) for seed in case.seeds)

    policy = train_policy_gradient_tree_driver(
        env_factory=training_env_factory,
        train_seeds=train_seeds,
        epochs=int(args.epochs),
        episodes_per_epoch=int(args.episodes_per_epoch),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        temperature=float(args.temperature),
        policy_seed=int(args.policy_seed),
        max_decisions_per_episode=int(args.max_decisions_per_episode),
        controller_name=str(args.controller_name),
        save_path=output_dir / "policy_gradient_tree_driver.json",
    )
    (output_dir / "policy_gradient_training_stats.json").write_text(
        json.dumps(policy.training_stats, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    summary_path = run_experiment_suite(
        cases=eval_cases,
        policies=[policy],
        output_dir=output_dir,
    )
    print(f"training_cases={len(train_cases)}")
    print(f"evaluation_cases={len(eval_cases)}")
    print(summary_path)


if __name__ == "__main__":
    main()
