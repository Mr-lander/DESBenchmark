from __future__ import annotations

from argparse import ArgumentParser
import copy
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import (
    AG2LLMTreeDriver,
    AG2TreeDriver,
    AgentScopeLLMTreeDriver,
    AgentScopeTreeDriver,
    CrewAILLMTreeDriver,
    CrewAITreeDriver,
    GoogleADKLLMTreeDriver,
    GoogleADKTreeDriver,
    LangGraphLLMTreeDriver,
    LangGraphTreeDriver,
    SPADELLMTreeDriver,
    SPADETreeDriver,
)
from fms_bench.experiment_runner import (
    MAINLINE_CATEGORY_B_AUTHORITY_MODES,
    build_tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep_suite,
    build_tree_runtime_v2_intercell_a3c9_probe_suite,
    build_tree_runtime_v2_intercell_a3c9_dynamics_suite,
    build_tree_runtime_v2_intercell_a3c9_nightmare_suite,
    build_tree_runtime_v2_intercell_a3c9_wide_suite,
    build_tree_runtime_v2_intercell_blocking_probe_suite,
    build_tree_runtime_v2_intercell_brandimarte_probe_suite,
    build_tree_runtime_v2_intercell_dynamics_extension_suite,
    build_tree_runtime_v2_intercell_harder_congestion_candidates_suite,
    build_tree_runtime_v2_intercell_harder_congestion_suite,
    build_tree_runtime_v2_intercell_small_suite,
    build_tree_runtime_v2_intercell_wide_suite,
    build_tree_runtime_v2_ladder_suite,
    build_tree_runtime_v2_pilot_suite,
    build_tree_runtime_v2_small_suite,
    build_tree_runtime_v2_wide_suite,
    run_experiment_suite,
)
from fms_bench.v2.framework_envs import get_framework_env_spec


SUITE_BUILDERS = {
    "small": build_tree_runtime_v2_small_suite,
    "intercell_a3c9_dynamics_budget_sweep": build_tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep_suite,
    "intercell_a3c9_dynamics": build_tree_runtime_v2_intercell_a3c9_dynamics_suite,
    "intercell_a3c9_nightmare": build_tree_runtime_v2_intercell_a3c9_nightmare_suite,
    "intercell_a3c9_probe": build_tree_runtime_v2_intercell_a3c9_probe_suite,
    "intercell_a3c9_wide": build_tree_runtime_v2_intercell_a3c9_wide_suite,
    "intercell_blocking_probe": build_tree_runtime_v2_intercell_blocking_probe_suite,
    "intercell_brandimarte_probe": build_tree_runtime_v2_intercell_brandimarte_probe_suite,
    "intercell_dynamics_extension": build_tree_runtime_v2_intercell_dynamics_extension_suite,
    "intercell_harder_congestion": build_tree_runtime_v2_intercell_harder_congestion_suite,
    "intercell_harder_congestion_candidates": build_tree_runtime_v2_intercell_harder_congestion_candidates_suite,
    "intercell_small": build_tree_runtime_v2_intercell_small_suite,
    "intercell_wide": build_tree_runtime_v2_intercell_wide_suite,
    "pilot": build_tree_runtime_v2_pilot_suite,
    "wide": build_tree_runtime_v2_wide_suite,
    "ladder": build_tree_runtime_v2_ladder_suite,
}
FRAMEWORK_CHOICES = ("ag2", "agentscope", "crewai", "google_adk", "langgraph", "spade")
CATEGORY_B_SUITE_IDS = frozenset(
    suite_id for suite_id in SUITE_BUILDERS if suite_id.startswith("intercell_")
)

_DEPRECATED_SUITE_ALIAS_MAP = {
    "intercell_a3c9_dynamics": "intercell_a3c9_wide",
}
OFFICIAL_CATEGORY_B_A3C9_SUITES = (
    "intercell_a3c9_wide",
    "intercell_a3c9_nightmare",
)
_PUBLIC_SUITE_HELP = (
    "Suite surface to run. Official Category B A3C9 public surfaces are "
    f"`{OFFICIAL_CATEGORY_B_A3C9_SUITES[0]}` and `{OFFICIAL_CATEGORY_B_A3C9_SUITES[1]}`."
)


def _build_driver(framework_key: str, *, strict_framework: bool) -> object:
    if framework_key == "ag2":
        return AG2TreeDriver(strict_framework=strict_framework)
    if framework_key == "agentscope":
        return AgentScopeTreeDriver(strict_framework=strict_framework)
    if framework_key == "crewai":
        return CrewAITreeDriver(strict_framework=strict_framework)
    if framework_key == "google_adk":
        return GoogleADKTreeDriver(strict_framework=strict_framework)
    if framework_key == "langgraph":
        return LangGraphTreeDriver(strict_framework=strict_framework)
    if framework_key == "spade":
        return SPADETreeDriver(strict_framework=strict_framework)
    raise ValueError(f"Unsupported framework key: {framework_key}")


def _build_llm_driver(
    framework_key: str,
    *,
    strict_framework: bool,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_timeout_seconds: float | None,
) -> object:
    if framework_key == "agentscope":
        return AgentScopeLLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "ag2":
        return AG2LLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "crewai":
        return CrewAILLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "google_adk":
        return GoogleADKLLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "langgraph":
        return LangGraphLLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "spade":
        return SPADELLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            strict_framework=strict_framework,
        )
    raise ValueError(
        f"Unsupported LLM-backed framework key: {framework_key}"
    )


def _parse_args(argv: list[str] | None = None) -> object:
    parser = ArgumentParser(
        description="Run one framework-backed tree-runtime benchmark line inside its own uv environment.",
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=sorted(FRAMEWORK_CHOICES),
    )
    parser.add_argument(
        "--suite",
        default="small",
        help=_PUBLIC_SUITE_HELP,
    )
    parser.add_argument(
        "--strict-framework",
        action="store_true",
        help="Fail immediately if the selected framework package is not installed in the current environment.",
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
        "--llm-backend",
        choices=["none", "openai_compat"],
        default="none",
        help="Optional real LLM backend for supported framework drivers.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model id for --llm-backend runs.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Base URL for the OpenAI-compatible backend, e.g. http://127.0.0.1:30000/v1",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for the OpenAI-compatible backend. Defaults to OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds for live LLM backend calls. Use <= 0 to disable.",
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
        "--intercell-max-in-flight",
        type=int,
        default=1,
        help="Bound on concurrently admitted inter-cell route jobs for intercell_small.",
    )
    parser.add_argument(
        "--intercell-inbound-buffer-cap",
        type=int,
        default=1,
        help="Inbound buffer cap override for intercell_small cell configs.",
    )
    parser.add_argument(
        "--authority-mode",
        action="append",
        choices=sorted(MAINLINE_CATEGORY_B_AUTHORITY_MODES),
        default=None,
        help=(
            "Optional authority-mode filter for authority-expanded Category B A3C9 suites. "
            "Repeat to pass multiple modes."
        ),
    )
    args = parser.parse_args(argv)
    suite_id = str(args.suite)
    if suite_id not in SUITE_BUILDERS and suite_id not in _DEPRECATED_SUITE_ALIAS_MAP:
        parser.error(f"Unsupported suite: {suite_id}")
    return args


def _build_suite_from_args(
    args: object,
    *,
    suite_builders: dict[str, object] | None = None,
) -> list[object]:
    builders = suite_builders or SUITE_BUILDERS
    requested_suite = str(args.suite)
    normalized_suite = _DEPRECATED_SUITE_ALIAS_MAP.get(requested_suite, requested_suite)
    if normalized_suite != requested_suite:
        print(
            f"DEPRECATED: {requested_suite} is no longer an official Category B public surface; "
            f"use {normalized_suite}",
            file=sys.stderr,
        )
    builder = builders[normalized_suite]
    builder_kwargs = {
        "numeric_profile_id": getattr(args, "numeric_profile", None),
        "due_date_profile_id": getattr(args, "due_date_profile", None),
    }
    if normalized_suite == "intercell_small":
        builder_kwargs["max_in_flight_jobs"] = int(getattr(args, "intercell_max_in_flight", 1))
        builder_kwargs["inbound_buffer_cap"] = int(getattr(args, "intercell_inbound_buffer_cap", 1))
    if normalized_suite in {"intercell_a3c9_wide", "intercell_a3c9_nightmare"} and getattr(args, "authority_mode", None):
        builder_kwargs["authority_modes"] = tuple(str(mode) for mode in args.authority_mode)
    return builder(
        PROJECT_ROOT,
        **builder_kwargs,
    )


def _limit_suite(cases: list[object], *, case_limit: int | None, seed_limit: int | None) -> list[object]:
    limited_cases = list(cases)
    if case_limit is not None:
        limited_cases = limited_cases[: max(int(case_limit), 0)]
    if seed_limit is None:
        return limited_cases
    result: list[object] = []
    for case in limited_cases:
        limited_seeds = list(case.seeds[: max(int(seed_limit), 0)])
        try:
            result.append(replace(case, seeds=limited_seeds))
            continue
        except TypeError:
            copied = copy.copy(case)
            copied.seeds = limited_seeds
            result.append(copied)
    return result


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = get_framework_env_spec(args.framework)
    suite = _build_suite_from_args(args)
    suite = _limit_suite(
        suite,
        case_limit=args.case_limit,
        seed_limit=args.seed_limit,
    )
    if str(args.llm_backend) == "openai_compat":
        llm_model = str(args.llm_model or "").strip()
        llm_base_url = str(args.llm_base_url or "").strip()
        if not llm_model or not llm_base_url:
            raise SystemExit("--llm-model and --llm-base-url are required when --llm-backend=openai_compat")
        llm_api_key = str(args.llm_api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY")
        policy = _build_llm_driver(
            args.framework,
            strict_framework=bool(args.strict_framework),
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_timeout_seconds=(
                None if float(args.llm_timeout) <= 0.0 else float(args.llm_timeout)
            ),
        )
    else:
        policy = _build_driver(args.framework, strict_framework=bool(args.strict_framework))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "dist"
            / "benchmark_runs"
            / f"tree_runtime_v2_{args.suite}_{spec.framework_key}_{timestamp}"
        )
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise SystemExit(f"Output directory already exists: {output_dir}")
    summary_path = run_experiment_suite(
        cases=suite,
        policies=[policy],
        output_dir=output_dir,
    )
    print(summary_path)


if __name__ == "__main__":
    main()
