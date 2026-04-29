from __future__ import annotations

from argparse import ArgumentParser, Namespace
import copy
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request

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
    build_tree_runtime_v2_intercell_a5c12_harder_candidates_suite,
    build_tree_runtime_v2_intercell_a5c12_harder_suite,
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
    "intercell_a5c12_harder_candidates": build_tree_runtime_v2_intercell_a5c12_harder_candidates_suite,
    "intercell_a5c12_harder": build_tree_runtime_v2_intercell_a5c12_harder_suite,
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
REAL_LLM_PREFLIGHT_TIMEOUT_SECONDS = 10.0
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
    llm_request_retries: int = 2,
    llm_retry_backoff_seconds: float = 0.5,
    llm_retry_backoff_max_seconds: float = 60.0,
) -> object:
    if framework_key == "agentscope":
        return AgentScopeLLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            request_retries=llm_request_retries,
            retry_backoff_seconds=llm_retry_backoff_seconds,
            retry_backoff_max_seconds=llm_retry_backoff_max_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "ag2":
        return AG2LLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            request_retries=llm_request_retries,
            retry_backoff_seconds=llm_retry_backoff_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "crewai":
        return CrewAILLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            request_retries=llm_request_retries,
            retry_backoff_seconds=llm_retry_backoff_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "google_adk":
        return GoogleADKLLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            request_retries=llm_request_retries,
            retry_backoff_seconds=llm_retry_backoff_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "langgraph":
        return LangGraphLLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            request_retries=llm_request_retries,
            retry_backoff_seconds=llm_retry_backoff_seconds,
            retry_backoff_max_seconds=llm_retry_backoff_max_seconds,
            strict_framework=strict_framework,
        )
    if framework_key == "spade":
        return SPADELLMTreeDriver(
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            request_timeout=llm_timeout_seconds,
            request_retries=llm_request_retries,
            retry_backoff_seconds=llm_retry_backoff_seconds,
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
        "--llm-retries",
        type=int,
        default=2,
        help="Per-decision live LLM retry count after the initial request.",
    )
    parser.add_argument(
        "--llm-retry-backoff-seconds",
        type=float,
        default=0.5,
        help="Initial retry backoff in seconds. AgentScope/LangGraph use exponential backoff.",
    )
    parser.add_argument(
        "--llm-retry-backoff-max-seconds",
        type=float,
        default=60.0,
        help="Maximum per-retry backoff in seconds for AgentScope/LangGraph live LLM calls.",
    )
    parser.add_argument(
        "--require-zero-fallback",
        action="store_true",
        help=(
            "Fail the run if any framework LLM decision falls back to the contract path. "
            "Retry is still allowed; only persisted fallback decisions are forbidden."
        ),
    )
    parser.add_argument(
        "--strict-real-llm",
        action="store_true",
        help=(
            "Paper-grade strict mode: require a real LLM backend, require zero fallback, "
            "and validate the final summary as real-llm output."
        ),
    )
    parser.add_argument(
        "--real-llm-response-backed",
        action="store_true",
        help=(
            "Paper-grade response-backed mode: require a real LLM backend, allow "
            "invalid-response fallback, but reject empty/exception decision failures "
            "and no-legal-action runs in the final artifacts."
        ),
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
    if normalized_suite in {
        "intercell_a3c9_wide",
        "intercell_a3c9_nightmare",
        "intercell_a5c12_harder",
        "intercell_a5c12_harder_candidates",
    } and getattr(args, "authority_mode", None):
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


def _validate_strict_real_llm_summary(
    *,
    summary_csv: Path,
    framework: str,
    suite: str,
    uv_project_environment: str,
) -> None:
    from scripts.validate_framework_run import validate

    validate(
        Namespace(
            summary_csv=Path(summary_csv),
            expected_framework=str(framework),
            expected_suite=str(suite),
            expected_output_dir=Path(summary_csv).parent,
            expected_uv_project_environment=str(uv_project_environment),
            expected_controller_mode="external_session",
            expected_llm_backend="openai_compat",
            require_real_llm=True,
            require_real_llm_response_backed=False,
        )
    )


def _validate_response_backed_real_llm_summary(
    *,
    summary_csv: Path,
    framework: str,
    suite: str,
    uv_project_environment: str,
) -> None:
    from scripts.validate_framework_run import validate

    validate(
        Namespace(
            summary_csv=Path(summary_csv),
            expected_framework=str(framework),
            expected_suite=str(suite),
            expected_output_dir=Path(summary_csv).parent,
            expected_uv_project_environment=str(uv_project_environment),
            expected_controller_mode="external_session",
            expected_llm_backend="openai_compat",
            require_real_llm=False,
            require_real_llm_response_backed=True,
        )
    )


def _preflight_openai_compat_endpoint(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float | None,
) -> None:
    normalized_base_url = str(base_url).rstrip("/")
    models_url = f"{normalized_base_url}/models"
    effective_timeout = (
        REAL_LLM_PREFLIGHT_TIMEOUT_SECONDS
        if timeout_seconds is None or float(timeout_seconds) <= 0.0
        else float(timeout_seconds)
    )
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib_request.Request(models_url, headers=headers)
    try:
        with urllib_request.urlopen(request, timeout=effective_timeout) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            raw_body = response.read()
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(
            "real-llm endpoint preflight failed: "
            f"{models_url} -> {type(exc).__name__}: {exc}"
        ) from exc

    if status_code != 200:
        raise SystemExit(
            "real-llm endpoint preflight failed: "
            f"{models_url} returned status {status_code}"
        )

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            "real-llm endpoint preflight failed: "
            f"{models_url} returned non-JSON payload"
        ) from exc

    models = payload.get("models")
    if not isinstance(models, list) or not models:
        # Accept standard OpenAI-compatible /models payloads that expose
        # the model list via the top-level "data" field.
        models = payload.get("data")
    if not isinstance(models, list) or not models:
        raise SystemExit(
            "real-llm endpoint preflight failed: "
            f"{models_url} did not expose a non-empty models list"
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = get_framework_env_spec(args.framework)
    normalized_suite = _DEPRECATED_SUITE_ALIAS_MAP.get(str(args.suite), str(args.suite))
    strict_real_llm = bool(getattr(args, "strict_real_llm", False))
    response_backed_real_llm = bool(getattr(args, "real_llm_response_backed", False))
    if strict_real_llm and response_backed_real_llm:
        raise SystemExit("--strict-real-llm and --real-llm-response-backed are mutually exclusive")
    effective_strict_framework = (
        bool(getattr(args, "strict_framework", False))
        or strict_real_llm
        or response_backed_real_llm
    )
    require_zero_fallback = bool(getattr(args, "require_zero_fallback", False)) or strict_real_llm
    allowed_framework_failure_kinds: tuple[str, ...] | None = None
    if strict_real_llm:
        allowed_framework_failure_kinds = ()
    elif response_backed_real_llm:
        allowed_framework_failure_kinds = ("invalid_response",)
    if (strict_real_llm or response_backed_real_llm) and str(args.llm_backend) != "openai_compat":
        raise SystemExit(
            "real-llm validation modes require --llm-backend=openai_compat"
        )
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
        if strict_real_llm or response_backed_real_llm:
            _preflight_openai_compat_endpoint(
                base_url=llm_base_url,
                api_key=llm_api_key,
                timeout_seconds=(
                    None if float(args.llm_timeout) <= 0.0 else float(args.llm_timeout)
                ),
            )
        policy = _build_llm_driver(
            args.framework,
            strict_framework=effective_strict_framework,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_timeout_seconds=(
                None if float(args.llm_timeout) <= 0.0 else float(args.llm_timeout)
            ),
            llm_request_retries=int(getattr(args, "llm_retries", 2)),
            llm_retry_backoff_seconds=float(
                getattr(args, "llm_retry_backoff_seconds", 0.5)
            ),
            llm_retry_backoff_max_seconds=float(
                getattr(args, "llm_retry_backoff_max_seconds", 60.0)
            ),
        )
    else:
        policy = _build_driver(args.framework, strict_framework=effective_strict_framework)
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
        require_zero_fallback=require_zero_fallback,
        allowed_framework_failure_kinds=allowed_framework_failure_kinds,
    )
    if strict_real_llm:
        _validate_strict_real_llm_summary(
            summary_csv=summary_path,
            framework=spec.framework_name,
            suite=normalized_suite,
            uv_project_environment=spec.uv_project_environment,
        )
    if response_backed_real_llm:
        _validate_response_backed_real_llm_summary(
            summary_csv=summary_path,
            framework=spec.framework_name,
            suite=normalized_suite,
            uv_project_environment=spec.uv_project_environment,
        )
    print(summary_path)


if __name__ == "__main__":
    main()
