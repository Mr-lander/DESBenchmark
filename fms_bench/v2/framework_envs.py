from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Mapping


def _normalize_framework_key(value: str) -> str:
    normalized = []
    for char in str(value).strip().lower():
        if char.isalnum():
            normalized.append(char)
        else:
            normalized.append("_")
    return "_".join(part for part in "".join(normalized).split("_") if part)


@dataclass(frozen=True)
class FrameworkEnvSpec:
    framework_key: str
    framework_name: str
    python_package: str
    optional_dependency: str
    uv_project_environment: str
    current_native_status: str
    example_script: str = "examples/run_tree_runtime_v2_framework_native.py"

    def sync_environment(self) -> dict[str, str]:
        return {
            "UV_PROJECT_ENVIRONMENT": self.uv_project_environment,
        }

    def recommended_uv_sync_args(
        self,
        *,
        include_test: bool = False,
    ) -> list[str]:
        args = [
            "uv",
            "sync",
            "--extra",
            self.optional_dependency,
        ]
        if include_test:
            args.extend(["--extra", "test"])
        return args

    def recommended_uv_run_args(
        self,
        *,
        suite: str = "small",
        strict_framework: bool = True,
        output_dir: str | Path | None = None,
    ) -> list[str]:
        args = [
            "uv",
            "run",
            "--extra",
            self.optional_dependency,
            "python",
            self.example_script,
            "--framework",
            self.framework_key,
            "--suite",
            str(suite),
        ]
        if strict_framework:
            args.append("--strict-framework")
        if output_dir is not None:
            args.extend(["--output-dir", str(output_dir)])
        return args

    def recommended_uv_sync_command(
        self,
        *,
        include_test: bool = False,
        shell: str = "powershell",
    ) -> str:
        args = " ".join(shlex.quote(part) for part in self.recommended_uv_sync_args(include_test=include_test))
        env_value = self.uv_project_environment.replace("'", "''")
        if shell == "powershell":
            return f"$env:UV_PROJECT_ENVIRONMENT='{env_value}'; {args}"
        return f"UV_PROJECT_ENVIRONMENT={shlex.quote(self.uv_project_environment)} {args}"

    def recommended_uv_run_command(
        self,
        *,
        suite: str = "small",
        strict_framework: bool = True,
        output_dir: str | Path | None = None,
        shell: str = "powershell",
    ) -> str:
        args = " ".join(
            shlex.quote(part)
            for part in self.recommended_uv_run_args(
                suite=suite,
                strict_framework=strict_framework,
                output_dir=output_dir,
            )
        )
        env_value = self.uv_project_environment.replace("'", "''")
        if shell == "powershell":
            return f"$env:UV_PROJECT_ENVIRONMENT='{env_value}'; {args}"
        return f"UV_PROJECT_ENVIRONMENT={shlex.quote(self.uv_project_environment)} {args}"


FRAMEWORK_ENV_SPECS: dict[str, FrameworkEnvSpec] = {
    "ag2": FrameworkEnvSpec(
        framework_key="ag2",
        framework_name="AG2",
        python_package="ag2",
        optional_dependency="framework_ag2",
        uv_project_environment=".uv_envs/ag2",
        current_native_status="native_framework",
    ),
    "agentscope": FrameworkEnvSpec(
        framework_key="agentscope",
        framework_name="AgentScope",
        python_package="agentscope",
        optional_dependency="framework_agentscope",
        uv_project_environment=".uv_envs/agentscope",
        current_native_status="native_framework",
    ),
    "crewai": FrameworkEnvSpec(
        framework_key="crewai",
        framework_name="CrewAI",
        python_package="crewai",
        optional_dependency="framework_crewai",
        uv_project_environment=".uv_envs/crewai",
        current_native_status="native_framework",
    ),
    "google_adk": FrameworkEnvSpec(
        framework_key="google_adk",
        framework_name="Google ADK",
        python_package="google.adk",
        optional_dependency="framework_google_adk",
        uv_project_environment=".uv_envs/google_adk",
        current_native_status="native_framework",
    ),
    "langgraph": FrameworkEnvSpec(
        framework_key="langgraph",
        framework_name="LangGraph",
        python_package="langgraph",
        optional_dependency="framework_langgraph",
        uv_project_environment=".uv_envs/langgraph",
        current_native_status="native_framework",
    ),
    "spade": FrameworkEnvSpec(
        framework_key="spade",
        framework_name="SPADE",
        python_package="spade",
        optional_dependency="framework_spade",
        uv_project_environment=".uv_envs/spade",
        current_native_status="native_framework",
    ),
}


def get_framework_env_spec(framework_key: str) -> FrameworkEnvSpec:
    normalized = _normalize_framework_key(framework_key)
    if normalized not in FRAMEWORK_ENV_SPECS:
        raise KeyError(f"Unknown framework environment profile: {framework_key}")
    return FRAMEWORK_ENV_SPECS[normalized]


def find_framework_env_spec(
    *,
    framework_name: str | None = None,
    python_package: str | None = None,
) -> FrameworkEnvSpec | None:
    normalized_name = _normalize_framework_key(framework_name or "")
    normalized_package = _normalize_framework_key(python_package or "")
    for framework_key, spec in FRAMEWORK_ENV_SPECS.items():
        if normalized_name and normalized_name == framework_key:
            return spec
        if normalized_package and normalized_package == framework_key:
            return spec
        if framework_name is not None and str(framework_name) == spec.framework_name:
            return spec
        if python_package is not None and str(python_package) == spec.python_package:
            return spec
    return None


def framework_env_payload(
    *,
    framework_name: str | None,
    python_package: str | None,
) -> Mapping[str, object]:
    spec = find_framework_env_spec(
        framework_name=framework_name,
        python_package=python_package,
    )
    if spec is None:
        return {}
    return {
        "framework_key": spec.framework_key,
        "optional_dependency": spec.optional_dependency,
        "uv_project_environment": spec.uv_project_environment,
        "expected_backend_mode": spec.current_native_status,
    }
