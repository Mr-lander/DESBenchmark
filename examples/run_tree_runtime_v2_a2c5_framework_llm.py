from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import AgentScopeLLMTreeDriver, LangGraphLLMTreeDriver
from fms_bench.experiment_runner import build_tree_runtime_v2_ladder_suite, run_experiment_suite


def _parse_args(argv: list[str] | None = None) -> object:
    parser = ArgumentParser(description="Run only the official tree_a2_c5 ladder slice with one LLM-backed framework.")
    parser.add_argument("--framework", choices=["langgraph", "agentscope"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _build_policy(args: object) -> object:
    common = {
        "model_name": str(args.model),
        "base_url": str(args.base_url),
        "api_key": str(args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"),
        "request_timeout": None if float(args.timeout) <= 0.0 else float(args.timeout),
        "strict_framework": True,
    }
    if str(args.framework) == "langgraph":
        return LangGraphLLMTreeDriver(**common)
    return AgentScopeLLMTreeDriver(**common)


def main(argv: list[str] | None = None) -> str:
    args = _parse_args(argv)
    suite = [
        case
        for case in build_tree_runtime_v2_ladder_suite(PROJECT_ROOT)
        if str(case.topology_id) == "tree_a2_c5"
    ]
    if not suite:
        raise SystemExit("No tree_a2_c5 cases found in the ladder suite.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "dist"
            / "benchmark_runs"
            / f"tree_runtime_v2_ladder_a2_c5_{args.framework}_llm_{timestamp}"
        )
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    summary_path = run_experiment_suite(
        cases=suite,
        policies=[_build_policy(args)],
        output_dir=output_dir,
    )
    print(summary_path)
    return str(summary_path)


if __name__ == "__main__":
    raise SystemExit(main())
