from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import FlatTreePolicy, HierarchicalTreePolicy, TreeContractNetPolicy
from fms_bench.experiment_runner import build_tree_benchmark_suite, run_experiment_suite


def main() -> None:
    suite = build_tree_benchmark_suite(PROJECT_ROOT)
    policies = [
        FlatTreePolicy(),
        HierarchicalTreePolicy(),
        TreeContractNetPolicy(),
    ]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = PROJECT_ROOT / 'dist' / 'benchmark_runs' / f'tree_benchmark_v1_small_{timestamp}'
    summary_path = run_experiment_suite(cases=suite, policies=policies, output_dir=output_dir)
    print(summary_path)


if __name__ == '__main__':
    main()
