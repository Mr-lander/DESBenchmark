from pathlib import Path
from datetime import datetime
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_bench import (
    ContractNetBudgetedPolicy,
    ContractNetGreedyPolicy,
    FlatMaskFirstPolicy,
    LeastLoadedCellPolicy,
)
from fms_bench.experiment_runner import build_core_v1_small_suite, run_experiment_suite


if __name__ == '__main__':
    repo_root = REPO_ROOT
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = repo_root / 'dist' / 'benchmark_runs' / f'core_v1_small_{stamp}'
    policies = [
        FlatMaskFirstPolicy(),
        LeastLoadedCellPolicy(),
        ContractNetGreedyPolicy(),
        ContractNetBudgetedPolicy(),
    ]
    summary_path = run_experiment_suite(
        cases=build_core_v1_small_suite(repo_root),
        policies=policies,
        output_dir=output_dir,
    )
    print(summary_path)
