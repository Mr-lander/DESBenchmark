from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_tree_runtime_v2_framework_native import (
    _DEPRECATED_SUITE_ALIAS_MAP,
    _build_driver,
    _build_llm_driver,
    _build_suite_from_args,
    _preflight_openai_compat_endpoint,
    _validate_response_backed_real_llm_summary,
    _validate_strict_real_llm_summary,
)
from fms_bench.experiment_runner import (
    MAINLINE_CATEGORY_B_AUTHORITY_MODES,
    recover_experiment_suite_summary,
    run_experiment_suite,
)
from fms_bench.v2.framework_envs import get_framework_env_spec


def _parse_args(argv: list[str] | None = None) -> object:
    parser = ArgumentParser(
        description=(
            "Run one or more explicit case shards for a framework-backed tree-runtime suite "
            "into a shared output root, then rebuild a partial suite summary from completed cases."
        )
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=("ag2", "agentscope", "crewai", "google_adk", "langgraph", "spade"),
    )
    parser.add_argument("--suite", default="intercell_a5c12_harder")
    parser.add_argument(
        "--case-id",
        action="append",
        required=True,
        help="Repeat to select one or more exact case_id values from the suite.",
    )
    parser.add_argument(
        "--authority-mode",
        action="append",
        default=None,
        help="Optional suite authority filter before case-id selection.",
    )
    parser.add_argument(
        "--paradigm-module",
        type=Path,
        default=None,
        help="Optional Python file that registers an extension paradigm before case selection.",
    )
    parser.add_argument("--strict-framework", action="store_true")
    parser.add_argument(
        "--llm-backend",
        choices=("none", "openai_compat"),
        default="none",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout", type=float, default=30.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-retry-backoff-seconds", type=float, default=0.5)
    parser.add_argument("--llm-retry-backoff-max-seconds", type=float, default=60.0)
    parser.add_argument("--require-zero-fallback", action="store_true")
    parser.add_argument("--strict-real-llm", action="store_true")
    parser.add_argument("--real-llm-response-backed", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Shared output root for completed case shards.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any selected case whose summary already exists under output-dir.",
    )
    args = parser.parse_args(argv)
    if args.paradigm_module is not None:
        from examples.run_tree_runtime_v2_framework_native import _load_paradigm_module

        _load_paradigm_module(Path(args.paradigm_module))
    if args.strict_real_llm and args.real_llm_response_backed:
        raise SystemExit(
            "--strict-real-llm and --real-llm-response-backed are mutually exclusive"
        )
    return args


def _select_cases(args: object) -> tuple[list[object], list[str]]:
    suite = list(_build_suite_from_args(args))
    requested_case_ids = [str(case_id) for case_id in getattr(args, "case_id", ())]
    suite_by_case_id = {case.case_id: case for case in suite}
    unknown_case_ids = [case_id for case_id in requested_case_ids if case_id not in suite_by_case_id]
    if unknown_case_ids:
        raise SystemExit("unknown case_id(s): " + ", ".join(unknown_case_ids))
    selected_cases = [suite_by_case_id[case_id] for case_id in requested_case_ids]
    return selected_cases, requested_case_ids


def _build_policy(args: object, *, strict_framework: bool) -> object:
    if str(args.llm_backend) != "openai_compat":
        return _build_driver(str(args.framework), strict_framework=strict_framework)

    llm_model = str(args.llm_model or "").strip()
    llm_base_url = str(args.llm_base_url or "").strip()
    if not llm_model or not llm_base_url:
        raise SystemExit("--llm-model and --llm-base-url are required when --llm-backend=openai_compat")
    llm_api_key = str(args.llm_api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    if bool(args.strict_real_llm) or bool(args.real_llm_response_backed):
        _preflight_openai_compat_endpoint(
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout_seconds=(None if float(args.llm_timeout) <= 0.0 else float(args.llm_timeout)),
        )
    return _build_llm_driver(
        str(args.framework),
        strict_framework=strict_framework,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_timeout_seconds=(None if float(args.llm_timeout) <= 0.0 else float(args.llm_timeout)),
        llm_request_retries=int(getattr(args, "llm_retries", 2)),
        llm_retry_backoff_seconds=float(getattr(args, "llm_retry_backoff_seconds", 0.5)),
        llm_retry_backoff_max_seconds=float(
            getattr(args, "llm_retry_backoff_max_seconds", 60.0)
        ),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = get_framework_env_spec(str(args.framework))
    normalized_suite = _DEPRECATED_SUITE_ALIAS_MAP.get(str(args.suite), str(args.suite))
    effective_strict_framework = (
        bool(getattr(args, "strict_framework", False))
        or bool(getattr(args, "strict_real_llm", False))
        or bool(getattr(args, "real_llm_response_backed", False))
    )
    selected_cases, requested_case_ids = _select_cases(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runnable_cases: list[object] = []
    skipped_case_ids: list[str] = []
    for case in selected_cases:
        case_dir = output_dir / case.case_id
        summary_csv = case_dir / "summary.csv"
        if summary_csv.exists():
            if bool(args.skip_existing):
                skipped_case_ids.append(case.case_id)
                continue
            raise SystemExit(f"case already completed under output-dir: {case.case_id}")
        if case_dir.exists() and any(case_dir.iterdir()):
            raise SystemExit(
                f"case directory already exists with partial contents; resolve manually first: {case_dir}"
            )
        runnable_cases.append(case)

    policy = _build_policy(args, strict_framework=effective_strict_framework)
    require_zero_fallback = bool(getattr(args, "require_zero_fallback", False)) or bool(
        getattr(args, "strict_real_llm", False)
    )
    allowed_framework_failure_kinds: tuple[str, ...] | None = None
    if bool(getattr(args, "strict_real_llm", False)):
        allowed_framework_failure_kinds = ()
    elif bool(getattr(args, "real_llm_response_backed", False)):
        allowed_framework_failure_kinds = ("invalid_response",)

    if runnable_cases:
        run_experiment_suite(
            cases=runnable_cases,
            policies=[policy],
            output_dir=output_dir,
            require_zero_fallback=require_zero_fallback,
            allowed_framework_failure_kinds=allowed_framework_failure_kinds,
        )

    recovered_summary = recover_experiment_suite_summary(output_dir=output_dir)

    if bool(args.strict_real_llm):
        _validate_strict_real_llm_summary(
            summary_csv=recovered_summary,
            framework=spec.framework_name,
            suite=normalized_suite,
            uv_project_environment=spec.uv_project_environment,
        )
    if bool(args.real_llm_response_backed):
        _validate_response_backed_real_llm_summary(
            summary_csv=recovered_summary,
            framework=spec.framework_name,
            suite=normalized_suite,
            uv_project_environment=spec.uv_project_environment,
        )

    print(f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"requested_cases={len(requested_case_ids)}")
    print(f"ran_cases={len(runnable_cases)}")
    print(f"skipped_cases={len(skipped_case_ids)}")
    for case_id in skipped_case_ids:
        print(f"skipped_case_id={case_id}")
    print(recovered_summary)


if __name__ == "__main__":
    main()
