from __future__ import annotations

from argparse import ArgumentParser, REMAINDER
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import (
    SubprocessExternalControllerDriver,
)
from fms_bench.experiment_runner import (
    build_tree_runtime_v2_ladder_suite,
    build_tree_runtime_v2_pilot_suite,
    build_tree_runtime_v2_small_suite,
    build_tree_runtime_v2_wide_suite,
    run_experiment_suite,
)


SUITE_BUILDERS = {
    "small": build_tree_runtime_v2_small_suite,
    "pilot": build_tree_runtime_v2_pilot_suite,
    "wide": build_tree_runtime_v2_wide_suite,
    "ladder": build_tree_runtime_v2_ladder_suite,
}


def _parse_args(argv: list[str] | None = None) -> object:
    parser = ArgumentParser(
        description="Run the benchmark against a custom MAS living in a subprocess.",
    )
    parser.add_argument(
        "--suite",
        default="small",
        choices=sorted(SUITE_BUILDERS),
    )
    parser.add_argument(
        "--controller-name",
        default="external_subprocess_controller",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional debug-only limit on the number of suite cases to run.",
    )
    parser.add_argument(
        "--seed-limit",
        type=int,
        default=None,
        help="Optional debug-only limit on the number of seeds per case.",
    )
    parser.add_argument(
        "--numeric-profile",
        default=None,
        help="Optional named numeric profile overlay, e.g. us_ca_pge_e20_camx_summer_2026q1.",
    )
    parser.add_argument(
        "--due-date-profile",
        default=None,
        help="Optional named due-date profile, e.g. twk_nominal_k3_relaxed_v1.",
    )
    parser.add_argument(
        "--controller-command",
        nargs=REMAINDER,
        required=True,
        help="Subprocess command that reads one decision JSON line and prints one action JSON line.",
    )
    return parser.parse_args(argv)


def _limit_suite(cases: list[object], *, case_limit: int | None, seed_limit: int | None) -> list[object]:
    limited_cases = list(cases)
    if case_limit is not None:
        limited_cases = limited_cases[: max(int(case_limit), 0)]
    if seed_limit is None:
        return limited_cases
    return [
        replace(case, seeds=list(case.seeds[: max(int(seed_limit), 0)]))
        for case in limited_cases
    ]


def _normalize_controller_command(command_parts: list[str]) -> str | list[str]:
    if len(command_parts) == 1:
        return str(command_parts[0])
    return [str(part) for part in command_parts]


def _build_suite_from_args(
    args: object,
    *,
    suite_builders: dict[str, object] | None = None,
) -> list[object]:
    builders = suite_builders or SUITE_BUILDERS
    builder = builders[str(args.suite)]
    return builder(
        PROJECT_ROOT,
        numeric_profile_id=getattr(args, "numeric_profile", None),
        due_date_profile_id=getattr(args, "due_date_profile", None),
    )


def main() -> None:
    args = _parse_args()
    suite = _build_suite_from_args(args)
    suite = _limit_suite(
        suite,
        case_limit=args.case_limit,
        seed_limit=args.seed_limit,
    )
    if not args.controller_command:
        raise ValueError("controller command must not be empty")

    policy = SubprocessExternalControllerDriver(
        _normalize_controller_command(list(args.controller_command)),
        controller_name=str(args.controller_name),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "dist"
            / "benchmark_runs"
            / f"tree_runtime_v2_{args.suite}_{args.controller_name}_{timestamp}"
        )
    summary_path = run_experiment_suite(
        cases=suite,
        policies=[policy],
        output_dir=output_dir,
    )
    print(summary_path)


if __name__ == "__main__":
    main()
