from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import examples.run_tree_runtime_v2_framework_native as framework_native_runner

_CATEGORY_B_KNOWN_SUITES = framework_native_runner.CATEGORY_B_SUITE_IDS


def _parse_args(argv: list[str] | None = None) -> object:
    parser = ArgumentParser(
        description="Run the Category B inter-cell benchmark surface through the native framework runner.",
    )
    parser.add_argument(
        "--framework",
        default="langgraph",
        choices=["agentscope", "langgraph"],
        help="Native framework line to run for Category B.",
    )
    parser.add_argument(
        "--suite",
        default="intercell_wide",
        help=(
            "Category B suite surface to run. Official A3C9 public surfaces are "
            f"`{framework_native_runner.OFFICIAL_CATEGORY_B_A3C9_SUITES[0]}` and "
            f"`{framework_native_runner.OFFICIAL_CATEGORY_B_A3C9_SUITES[1]}`."
        ),
    )
    known_args, forwarded_args = parser.parse_known_args(argv)
    if str(known_args.suite) not in _CATEGORY_B_KNOWN_SUITES:
        parser.error(f"Unsupported Category B suite: {known_args.suite}")
    setattr(known_args, "forwarded_args", list(forwarded_args))
    return known_args


def _build_forwarded_argv(args: object) -> list[str]:
    return [
        "--framework",
        str(args.framework),
        "--suite",
        str(args.suite),
        *list(getattr(args, "forwarded_args", [])),
    ]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    forwarded_argv = _build_forwarded_argv(args)
    framework_native_runner.main(forwarded_argv)


if __name__ == "__main__":
    main()
