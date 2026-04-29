"""Stable tree-holonic topology builders for public benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .holonic_types import AreaSpec, PlantCellConfig


@dataclass(frozen=True)
class TreeTopology:
    """Resolved Plant / Area / Cell topology for a benchmark item."""

    topology_id: str
    areas: Dict[int, AreaSpec]
    cell_to_area: Dict[int, int]

    @property
    def area_to_cells(self) -> Dict[int, List[int]]:
        return {area_id: list(area.cell_ids) for area_id, area in self.areas.items()}


def build_tree_topology(
    cell_configs: Iterable[PlantCellConfig],
    area_count: int,
    area_partitions: Iterable[Iterable[int]] | None = None,
) -> TreeTopology:
    configs = list(cell_configs)
    if not configs:
        raise ValueError("build_tree_topology requires at least one cell config")
    if area_count <= 0:
        raise ValueError("area_count must be positive")
    if area_count > len(configs):
        raise ValueError("area_count cannot exceed the number of cells")

    ordered_cell_ids = sorted(config.cell_id for config in configs)
    areas: Dict[int, AreaSpec] = {}
    cell_to_area: Dict[int, int] = {}
    if area_partitions is not None:
        normalized_partitions = [
            [int(cell_id) for cell_id in area_cell_ids]
            for area_cell_ids in area_partitions
        ]
        if len(normalized_partitions) != area_count:
            raise ValueError("area_partitions must match area_count")
        flattened = [cell_id for area_cell_ids in normalized_partitions for cell_id in area_cell_ids]
        if sorted(flattened) != ordered_cell_ids:
            raise ValueError("area_partitions must contain each cell id exactly once")
        partitioned_cells = normalized_partitions
    else:
        chunk_sizes = _balanced_chunk_sizes(len(ordered_cell_ids), area_count)
        cursor = 0
        partitioned_cells = []
        for chunk_size in chunk_sizes:
            partitioned_cells.append(ordered_cell_ids[cursor : cursor + chunk_size])
            cursor += chunk_size

    for area_id, area_cell_ids in enumerate(partitioned_cells):
        areas[area_id] = AreaSpec(
            area_id=area_id,
            label=f"area_{area_id}",
            cell_ids=area_cell_ids,
        )
        for cell_id in area_cell_ids:
            cell_to_area[cell_id] = area_id

    topology_id = f"tree_a{area_count}_c{len(ordered_cell_ids)}"
    return TreeTopology(
        topology_id=topology_id,
        areas=areas,
        cell_to_area=cell_to_area,
    )


def _balanced_chunk_sizes(total_items: int, group_count: int) -> List[int]:
    base_size = total_items // group_count
    remainder = total_items % group_count
    return [base_size + (1 if index < remainder else 0) for index in range(group_count)]
