from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import (
    BenchmarkConfig,
    FlatTreePolicy,
    HierarchicalTreePolicy,
    PlantCellConfig,
    TreeContractNetPolicy,
    TreeHolonicEnv,
    build_tree_topology,
    evaluate_holonic_policy,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cell_configs(repo_root: Path) -> list[PlantCellConfig]:
    root = repo_root / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    return [
        # Public benchmark families used here: Brandimarte (1993), Kacem et al. (2002), Fattahi et al. (2007).
        PlantCellConfig(
            cell_id=0,
            family_key='kacem',
            instance_path=root / 'KacemHammadiBorne2002' / 'kacem1.txt',
            instance_kind='kacem',
            label='cell-kacem',
            wip_cap=1,
        ),
        PlantCellConfig(
            cell_id=1,
            family_key='fattahi',
            instance_path=root / 'FattahiMehrabadJolai2007' / 'fattahi1.txt',
            instance_kind='fattahi',
            label='cell-fattahi',
            wip_cap=1,
        ),
        PlantCellConfig(
            cell_id=2,
            family_key='brandimarte',
            instance_path=root / 'Brandimarte1993' / 'mk01.txt',
            instance_kind='brandimarte',
            label='cell-brandimarte',
            wip_cap=1,
        ),
    ]


def _config() -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.horizon_hours = 200
    config.failure.mtbf_hours = 0.0
    config.failure.mttr_hours = 0.0
    return config


def _env_factory(area_to_cells: dict[int, list[int]]):
    def factory(cell_configs: list[PlantCellConfig], config: BenchmarkConfig, seed: int) -> TreeHolonicEnv:
        return TreeHolonicEnv(
            cell_configs,
            area_to_cells=area_to_cells,
            config=config,
            seed=seed,
        )

    return factory


def _controller_family(policy_name: str) -> str:
    if policy_name.startswith('tree_flat'):
        return 'flat'
    if policy_name.startswith('tree_hierarchical'):
        return 'hierarchical'
    if policy_name.startswith('tree_contract_net'):
        return 'mas_style'
    return 'unknown'


def main() -> None:
    repo_root = _repo_root()
    cell_configs = _cell_configs(repo_root)
    topology = build_tree_topology(cell_configs, area_count=2)
    env_cls = _env_factory(topology.area_to_cells)
    policies = [
        FlatTreePolicy(),
        HierarchicalTreePolicy(),
        TreeContractNetPolicy(),
    ]
    seeds = [13]
    benchmark_item_id = (
        f'suite=tree_v1_smoke__regime=R0__topology={topology.topology_id}'
        '__families=kacem+fattahi+brandimarte__split=test'
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = repo_root / 'dist' / 'benchmark_runs' / f'tree_v1_smoke_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for policy in policies:
        results = evaluate_holonic_policy(
            policy=policy,
            cell_configs=cell_configs,
            config=_config(),
            seeds=seeds,
            benchmark_item_id=benchmark_item_id,
            env_cls=env_cls,
        )
        base = output_dir / policy.name
        results.save_json(base.with_suffix('.json'))
        results.save_episode_csv(base.with_suffix('.csv'))
        summary = results.summary()
        summary['topology_id'] = topology.topology_id
        summary['regime_id'] = 'R0'
        summary['controller_family'] = _controller_family(policy.name)
        rows.append(summary)

    summary_path = output_dir / 'summary.csv'
    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(summary_path)
    for row in rows:
        print(
            row['policy_name'],
            f"makespan_mean={row['makespan_mean']:.3f}",
            f"throughput_mean={row['throughput_mean']:.3f}",
            f"success_rate={row['success_rate']:.3f}",
        )


if __name__ == '__main__':
    main()
