from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import (
    AG2TreeDriver,
    AgentScopeTreeDriver,
    CrewAITreeDriver,
    ExternalAdaptiveHolonTreeDriver,
    ExternalBlackboardTreeDriver,
    ExternalContractNetTreeDriver,
    ExternalFlatTreeDriver,
    ExternalHierarchicalTreeDriver,
    ExternalSupervisorReactiveTreeDriver,
    GoogleADKTreeDriver,
    LangGraphTreeDriver,
    SPADETreeDriver,
)
from fms_bench.experiment_runner import build_tree_runtime_v2_wide_suite, run_experiment_suite


def main() -> None:
    suite = build_tree_runtime_v2_wide_suite(PROJECT_ROOT)
    policies = [
        ExternalFlatTreeDriver(),
        ExternalHierarchicalTreeDriver(),
        ExternalContractNetTreeDriver(),
        ExternalAdaptiveHolonTreeDriver(),
        ExternalBlackboardTreeDriver(),
        ExternalSupervisorReactiveTreeDriver(),
        AgentScopeTreeDriver(),
        LangGraphTreeDriver(),
        AG2TreeDriver(),
        GoogleADKTreeDriver(),
        CrewAITreeDriver(),
        SPADETreeDriver(),
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        PROJECT_ROOT
        / "dist"
        / "benchmark_runs"
        / f"tree_runtime_v2_wide_external_family_{timestamp}"
    )
    summary_path = run_experiment_suite(cases=suite, policies=policies, output_dir=output_dir)
    print(summary_path)


if __name__ == "__main__":
    main()
