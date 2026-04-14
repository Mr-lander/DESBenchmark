"""Publication-facing table and figure artifact builders."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Sequence
import csv
import json


_KEY_METRICS = (
    "makespan_mean",
    "tardiness_mean",
    "energy_mean",
    "carbon_mean",
    "violation_magnitude_mean",
)

_CORE_METRICS = (
    "makespan_mean",
    "tardiness_mean",
    "energy_mean",
    "carbon_mean",
)

_CONSTRAINT_METRICS = (
    "energy_overshoot_magnitude_total_mean",
    "carbon_overshoot_magnitude_total_mean",
    "energy_overshoot_duration_mean",
    "carbon_overshoot_duration_mean",
    "energy_overshoot_frequency_mean",
    "carbon_overshoot_frequency_mean",
)


@dataclass(frozen=True)
class PublicationArtifactPaths:
    summary_csv: Path
    paper_table_csv: Path
    figure_summary_json: Path
    paper_report_html: Path


@dataclass(frozen=True)
class FocusedCasebookArtifactPaths:
    summary_csv: Path
    casebook_table_csv: Path
    casebook_summary_json: Path
    casebook_report_html: Path


@dataclass(frozen=True)
class ControllerFrontierArtifactPaths:
    merged_summary_csv: Path
    paper_table_csv: Path
    figure_summary_json: Path
    paper_report_html: Path


@dataclass(frozen=True)
class MergedPublicationArtifactPaths:
    merged_summary_csv: Path
    paper_table_csv: Path
    figure_summary_json: Path
    paper_report_html: Path


def build_publication_artifacts(
    *,
    summary_csv: str | Path,
    output_dir: str | Path,
) -> PublicationArtifactPaths:
    summary_path = Path(summary_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_summary_rows(summary_path)
    paper_rows = [_normalize_row(row) for row in rows]
    suite_tier = _stable_single_value(paper_rows, "suite_tier")
    controller_modes = sorted({row["controller_interface_mode"] for row in paper_rows})
    metric_separation_counts = _metric_separation_counts(paper_rows)
    constraint_metric_separation_counts = _constraint_metric_separation_counts(paper_rows)
    spotlight_cases = _spotlight_cases(paper_rows)

    paper_table_csv = out_dir / "paper_table.csv"
    figure_summary_json = out_dir / "figure_summary.json"
    paper_report_html = out_dir / "paper_report.html"

    _write_csv(paper_table_csv, paper_rows)
    figure_payload = {
        "suite_tier": suite_tier,
        "num_cases": len({row["case_id"] for row in paper_rows}),
        "num_rows": len(paper_rows),
        "controller_interface_modes": controller_modes,
        "metric_separation_counts": metric_separation_counts,
        "constraint_metric_separation_counts": constraint_metric_separation_counts,
        "spotlight_cases": spotlight_cases,
    }
    figure_summary_json.write_text(json.dumps(figure_payload, indent=2), encoding="utf-8")
    paper_report_html.write_text(
        _render_html_report(
            suite_tier=suite_tier,
            num_rows=len(paper_rows),
            num_cases=figure_payload["num_cases"],
            controller_modes=controller_modes,
            metric_separation_counts=metric_separation_counts,
            constraint_metric_separation_counts=constraint_metric_separation_counts,
            spotlight_cases=spotlight_cases,
        ),
        encoding="utf-8",
    )

    return PublicationArtifactPaths(
        summary_csv=summary_path,
        paper_table_csv=paper_table_csv,
        figure_summary_json=figure_summary_json,
        paper_report_html=paper_report_html,
    )


def build_casebook_artifacts(
    *,
    summary_csv: str | Path,
    output_dir: str | Path,
    case_ids: Sequence[str],
    headlines: dict[str, str] | None = None,
    rationales: dict[str, str] | None = None,
    highlight_pairs: dict[str, tuple[str, str]] | None = None,
) -> FocusedCasebookArtifactPaths:
    if not case_ids:
        raise ValueError("case_ids must not be empty")

    summary_path = Path(summary_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_normalize_row(row) for row in _load_summary_rows(summary_path)]
    filtered_rows = _select_case_rows(rows, case_ids)
    suite_tier = _stable_single_value(filtered_rows, "suite_tier")

    case_summaries = [
        _build_case_summary(
            case_id=case_id,
            case_rows=[row for row in filtered_rows if row["case_id"] == case_id],
            headline=(headlines or {}).get(case_id),
            rationale=(rationales or {}).get(case_id),
            highlight_pair=(highlight_pairs or {}).get(case_id),
        )
        for case_id in case_ids
    ]

    casebook_table_csv = out_dir / "casebook_table.csv"
    casebook_summary_json = out_dir / "casebook_summary.json"
    casebook_report_html = out_dir / "casebook_report.html"

    _write_csv(casebook_table_csv, filtered_rows)
    summary_payload = {
        "suite_tier": suite_tier,
        "num_cases": len(case_summaries),
        "num_rows": len(filtered_rows),
        "selected_case_ids": list(case_ids),
        "cases": case_summaries,
    }
    casebook_summary_json.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    casebook_report_html.write_text(
        _render_casebook_report(
            suite_tier=suite_tier,
            case_summaries=case_summaries,
        ),
        encoding="utf-8",
    )

    return FocusedCasebookArtifactPaths(
        summary_csv=summary_path,
        casebook_table_csv=casebook_table_csv,
        casebook_summary_json=casebook_summary_json,
        casebook_report_html=casebook_report_html,
    )


def build_controller_frontier_artifacts(
    *,
    baseline_summary_csv: str | Path,
    framework_summary_csv: str | Path,
    output_dir: str | Path,
    merged_summary_csv: str | Path | None = None,
    replace_policies: Sequence[str] | None = None,
) -> ControllerFrontierArtifactPaths:
    baseline_path = Path(baseline_summary_csv)
    framework_path = Path(framework_summary_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_summary_rows(baseline_path)
    framework_rows = _load_summary_rows(framework_path)
    policies_to_replace = list(
        replace_policies
        or sorted(
            {
                row["policy_name"]
                for row in framework_rows
                if row["policy_name"].startswith("external_framework_")
            }
        )
    )
    if not policies_to_replace:
        raise ValueError("no framework policies selected for controller-frontier replacement")

    merged_rows = _merge_controller_frontier_rows(
        baseline_rows=baseline_rows,
        framework_rows=framework_rows,
        replace_policies=policies_to_replace,
    )

    merged_summary_path = (
        Path(merged_summary_csv) if merged_summary_csv is not None else out_dir / "controller_frontier_summary.csv"
    )
    merged_summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(merged_summary_path, merged_rows)

    publication = build_publication_artifacts(
        summary_csv=merged_summary_path,
        output_dir=out_dir,
    )
    return ControllerFrontierArtifactPaths(
        merged_summary_csv=merged_summary_path,
        paper_table_csv=publication.paper_table_csv,
        figure_summary_json=publication.figure_summary_json,
        paper_report_html=publication.paper_report_html,
    )


def build_merged_publication_artifacts(
    *,
    summary_csv_paths: Sequence[str | Path],
    output_dir: str | Path,
    merged_summary_csv: str | Path | None = None,
) -> MergedPublicationArtifactPaths:
    if not summary_csv_paths:
        raise ValueError("summary_csv_paths must not be empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str]] = []
    for summary_csv in summary_csv_paths:
        merged_rows.extend(_load_summary_rows(Path(summary_csv)))
    _validate_unique_case_policy_rows(merged_rows)

    merged_summary_path = (
        Path(merged_summary_csv)
        if merged_summary_csv is not None
        else out_dir / "merged_summary.csv"
    )
    _write_csv(merged_summary_path, merged_rows)

    publication = build_publication_artifacts(
        summary_csv=merged_summary_path,
        output_dir=out_dir,
    )
    return MergedPublicationArtifactPaths(
        merged_summary_csv=merged_summary_path,
        paper_table_csv=publication.paper_table_csv,
        figure_summary_json=publication.figure_summary_json,
        paper_report_html=publication.paper_report_html,
    )


def build_topology_publication_artifacts(
    *,
    summary_csv: str | Path,
    output_root: str | Path,
) -> dict[str, PublicationArtifactPaths]:
    rows = [_normalize_row(row) for row in _load_summary_rows(Path(summary_csv))]
    if not rows:
        raise ValueError("summary_csv must contain at least one row")

    by_topology: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        topology_id = str(row.get("topology_id", "")).strip()
        if not topology_id:
            raise ValueError("topology_id must be present on every summary row")
        by_topology.setdefault(topology_id, []).append(row)

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    topology_artifacts: dict[str, PublicationArtifactPaths] = {}
    for topology_id, topology_rows in sorted(by_topology.items()):
        topology_dir = output_root_path / topology_id
        topology_dir.mkdir(parents=True, exist_ok=True)
        topology_summary_csv = topology_dir / "summary.csv"
        _write_csv(topology_summary_csv, topology_rows)
        topology_artifacts[topology_id] = build_publication_artifacts(
            summary_csv=topology_summary_csv,
            output_dir=topology_dir,
        )
    return topology_artifacts


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _validate_unique_case_policy_rows(rows: Sequence[dict[str, str]]) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    for row in rows:
        key = (str(row.get("case_id", "")), str(row.get("policy_name", "")))
        if key in seen_pairs:
            duplicates.append(f"{key[0]} / {key[1]}")
            continue
        seen_pairs.add(key)
    if duplicates:
        raise ValueError("duplicate case/policy rows: " + ", ".join(sorted(duplicates)))


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    normalized.setdefault("controller_interface_mode", "in_process")
    return normalized


def _merge_controller_frontier_rows(
    *,
    baseline_rows: list[dict[str, str]],
    framework_rows: list[dict[str, str]],
    replace_policies: Sequence[str],
) -> list[dict[str, str]]:
    replace_set = set(replace_policies)
    replacement_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in framework_rows:
        policy_name = row["policy_name"]
        if policy_name not in replace_set:
            continue
        key = (row["case_id"], policy_name)
        if key in replacement_map:
            raise ValueError(
                f"duplicate replacement rows for controller frontier: {key[0]} / {key[1]}"
            )
        replacement_map[key] = dict(row)

    merged_rows: list[dict[str, str]] = []
    baseline_case_ids: list[str] = []
    seen_case_ids: set[str] = set()
    baseline_cases_by_policy: dict[str, set[str]] = {policy_name: set() for policy_name in replace_set}
    missing_pairs: list[str] = []
    used_pairs: set[tuple[str, str]] = set()
    for row in baseline_rows:
        case_id = row["case_id"]
        if case_id not in seen_case_ids:
            baseline_case_ids.append(case_id)
            seen_case_ids.add(case_id)
        policy_name = row["policy_name"]
        if policy_name not in replace_set:
            merged_rows.append(dict(row))
            continue
        baseline_cases_by_policy[policy_name].add(case_id)
        key = (case_id, policy_name)
        replacement_row = replacement_map.get(key)
        if replacement_row is None:
            missing_pairs.append(f"{key[0]} / {key[1]}")
            continue
        merged_rows.append(dict(replacement_row))
        used_pairs.add(key)

    for policy_name in sorted(replace_set):
        if baseline_cases_by_policy[policy_name]:
            continue
        for case_id in baseline_case_ids:
            key = (case_id, policy_name)
            replacement_row = replacement_map.get(key)
            if replacement_row is None:
                missing_pairs.append(f"{key[0]} / {key[1]}")
                continue
            merged_rows.append(dict(replacement_row))
            used_pairs.add(key)

    if missing_pairs:
        raise ValueError(
            "missing replacement rows for controller frontier: "
            + ", ".join(sorted(missing_pairs))
        )

    extra_pairs = sorted(replacement_map.keys() - used_pairs)
    if extra_pairs:
        extra_text = ", ".join(f"{case_id} / {policy_name}" for case_id, policy_name in extra_pairs)
        raise ValueError(f"unexpected replacement rows for controller frontier: {extra_text}")

    return merged_rows


def _select_case_rows(rows: list[dict[str, str]], case_ids: Sequence[str]) -> list[dict[str, str]]:
    requested = list(case_ids)
    by_case = {case_id: [] for case_id in requested}
    for row in rows:
        case_id = row["case_id"]
        if case_id in by_case:
            by_case[case_id].append(row)

    missing = [case_id for case_id, case_rows in by_case.items() if not case_rows]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"unknown case_ids: {missing_text}")

    ordered_rows: list[dict[str, str]] = []
    for case_id in requested:
        ordered_rows.extend(sorted(by_case[case_id], key=lambda row: row["policy_name"]))
    return ordered_rows


def _stable_single_value(rows: Iterable[dict[str, str]], key: str) -> str:
    values = {row[key] for row in rows}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _metric_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "0.0")
    if value in (None, ""):
        return 0.0
    return float(value)


def _metric_separation_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

    counts: dict[str, int] = {}
    for metric in _KEY_METRICS:
        counts[metric] = sum(
            1
            for case_rows in by_case.values()
            if len({case_row[metric] for case_row in case_rows}) > 1
        )
    return counts


def _metric_spread_value(case_rows: list[dict[str, str]], metric: str) -> float:
    if metric in _CONSTRAINT_METRICS:
        values = [_metric_float(row, metric) for row in case_rows]
    else:
        values = [float(row[metric]) for row in case_rows]
    return round(max(values) - min(values), 6)


def _constraint_metric_separation_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

    counts: dict[str, int] = {}
    for metric in _CONSTRAINT_METRICS:
        counts[metric] = sum(
            1
            for case_rows in by_case.values()
            if len({_metric_float(case_row, metric) for case_row in case_rows}) > 1
        )
    return counts


def _constraint_metric_spread(case_rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        metric: _metric_spread_value(case_rows, metric)
        for metric in _CONSTRAINT_METRICS
    }


def _use_structured_constraint_priority(case_rows: list[dict[str, str]]) -> bool:
    return any(
        spread > 0.0
        for spread in _constraint_metric_spread(case_rows).values()
    )


def _constraint_priority_mode(case_rows: list[dict[str, str]]) -> str:
    if _use_structured_constraint_priority(case_rows):
        return "structured"
    return "legacy_violation"


def _behavior_signature_metrics(case_rows: list[dict[str, str]]) -> tuple[str, ...]:
    if _use_structured_constraint_priority(case_rows):
        return _KEY_METRICS + _CONSTRAINT_METRICS
    return _KEY_METRICS


def _comparison_metric_order(case_rows: list[dict[str, str]]) -> tuple[str, ...]:
    if _use_structured_constraint_priority(case_rows):
        return _KEY_METRICS + _CONSTRAINT_METRICS
    return _KEY_METRICS


def _spotlight_priority(case_rows: list[dict[str, str]]) -> tuple[float, ...]:
    constraint_spread = _constraint_metric_spread(case_rows)
    structured = _use_structured_constraint_priority(case_rows)
    return (
        1.0 if structured else 0.0,
        constraint_spread["energy_overshoot_magnitude_total_mean"] if structured else 0.0,
        constraint_spread["carbon_overshoot_magnitude_total_mean"] if structured else 0.0,
        constraint_spread["energy_overshoot_duration_mean"] if structured else 0.0,
        constraint_spread["carbon_overshoot_duration_mean"] if structured else 0.0,
        constraint_spread["energy_overshoot_frequency_mean"] if structured else 0.0,
        constraint_spread["carbon_overshoot_frequency_mean"] if structured else 0.0,
        _metric_spread_value(case_rows, "violation_magnitude_mean") if not structured else 0.0,
        _metric_spread_value(case_rows, "energy_mean"),
        _metric_spread_value(case_rows, "makespan_mean"),
    )


def _comparison_priority(
    row: dict[str, str],
    reference_row: dict[str, str],
    case_rows: list[dict[str, str]],
) -> tuple[float, ... | str]:
    structured = _use_structured_constraint_priority(case_rows)
    if structured:
        return (
            abs(
                _metric_float(row, "energy_overshoot_magnitude_total_mean")
                - _metric_float(reference_row, "energy_overshoot_magnitude_total_mean")
            ),
            abs(
                _metric_float(row, "carbon_overshoot_magnitude_total_mean")
                - _metric_float(reference_row, "carbon_overshoot_magnitude_total_mean")
            ),
            abs(
                _metric_float(row, "energy_overshoot_duration_mean")
                - _metric_float(reference_row, "energy_overshoot_duration_mean")
            ),
            abs(
                _metric_float(row, "carbon_overshoot_duration_mean")
                - _metric_float(reference_row, "carbon_overshoot_duration_mean")
            ),
            abs(
                _metric_float(row, "energy_overshoot_frequency_mean")
                - _metric_float(reference_row, "energy_overshoot_frequency_mean")
            ),
            abs(
                _metric_float(row, "carbon_overshoot_frequency_mean")
                - _metric_float(reference_row, "carbon_overshoot_frequency_mean")
            ),
            abs(float(row["energy_mean"]) - float(reference_row["energy_mean"])),
            abs(float(row["makespan_mean"]) - float(reference_row["makespan_mean"])),
            str(row["policy_name"]),
        )
    return (
        abs(float(row["violation_magnitude_mean"]) - float(reference_row["violation_magnitude_mean"])),
        abs(float(row["energy_mean"]) - float(reference_row["energy_mean"])),
        abs(float(row["makespan_mean"]) - float(reference_row["makespan_mean"])),
        str(row["policy_name"]),
    )


def _spotlight_cases(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

    ranked: list[tuple[tuple[float, float, float], dict[str, object]]] = []
    for case_id, case_rows in by_case.items():
        violation_values = [float(row["violation_magnitude_mean"]) for row in case_rows]
        makespan_values = [float(row["makespan_mean"]) for row in case_rows]
        energy_values = [float(row["energy_mean"]) for row in case_rows]
        constraint_spread = _constraint_metric_spread(case_rows)
        record = {
            "case_id": case_id,
            "regime_id": case_rows[0]["regime_id"],
            "topology_id": case_rows[0]["topology_id"],
            "constraint_priority_mode": _constraint_priority_mode(case_rows),
            "spread": {
                "violation_magnitude_mean": max(violation_values) - min(violation_values),
                "makespan_mean": max(makespan_values) - min(makespan_values),
                "energy_mean": max(energy_values) - min(energy_values),
            },
            "constraint_spread": constraint_spread,
            "rows": [
                {
                    "policy_name": row["policy_name"],
                    "makespan_mean": float(row["makespan_mean"]),
                    "tardiness_mean": float(row["tardiness_mean"]),
                    "energy_mean": float(row["energy_mean"]),
                    "carbon_mean": float(row["carbon_mean"]),
                    "violation_magnitude_mean": float(row["violation_magnitude_mean"]),
                    **{
                        metric: _metric_float(row, metric)
                        for metric in _CONSTRAINT_METRICS
                    },
                    "controller_interface_mode": row["controller_interface_mode"],
                }
                for row in case_rows
            ],
        }
        ranked.append(
            (
                _spotlight_priority(case_rows),
                record,
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in ranked[:3]]


def _build_case_summary(
    *,
    case_id: str,
    case_rows: list[dict[str, str]],
    headline: str | None,
    rationale: str | None,
    highlight_pair: tuple[str, str] | None,
) -> dict[str, object]:
    ordered_rows = sorted(case_rows, key=lambda row: row["policy_name"])
    return {
        "case_id": case_id,
        "headline": headline or _default_case_headline(case_id),
        "rationale": rationale or "",
        "regime_id": ordered_rows[0]["regime_id"],
        "topology_id": ordered_rows[0]["topology_id"],
        "controller_interface_modes": sorted(
            {row["controller_interface_mode"] for row in ordered_rows}
        ),
        "constraint_priority_mode": _constraint_priority_mode(ordered_rows),
        "behavior_group_count": len(_behavior_groups(ordered_rows)),
        "behavior_groups": _behavior_groups(ordered_rows),
        "metric_spread": _metric_spread(ordered_rows),
        "constraint_metric_spread": _constraint_metric_spread(ordered_rows),
        "best_policies_by_metric": _best_policies_by_metric(ordered_rows),
        "best_policies_by_constraint_metric": _best_policies_by_constraint_metric(ordered_rows),
        "highlight_comparison": _highlight_comparison(ordered_rows, highlight_pair),
        "policy_rows": [
            {
                "policy_name": row["policy_name"],
                "makespan_mean": float(row["makespan_mean"]),
                "tardiness_mean": float(row["tardiness_mean"]),
                "energy_mean": float(row["energy_mean"]),
                "carbon_mean": float(row["carbon_mean"]),
                "violation_magnitude_mean": float(row["violation_magnitude_mean"]),
                **{
                    metric: _metric_float(row, metric)
                    for metric in _CONSTRAINT_METRICS
                },
            }
            for row in ordered_rows
        ],
    }


def _default_case_headline(case_id: str) -> str:
    return case_id.split("__", 1)[-1].replace("__", " / ")


def _metric_spread(case_rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        metric: _metric_spread_value(case_rows, metric)
        for metric in _comparison_metric_order(case_rows)
    }


def _best_policies_by_metric(case_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    best: dict[str, list[str]] = {}
    for metric in _comparison_metric_order(case_rows):
        values = [_metric_float(row, metric) for row in case_rows]
        best_value = min(values)
        best[metric] = sorted(
            row["policy_name"]
            for row in case_rows
            if abs(_metric_float(row, metric) - best_value) < 1e-9
        )
    return best


def _best_policies_by_constraint_metric(case_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    best: dict[str, list[str]] = {}
    for metric in _CONSTRAINT_METRICS:
        values = [_metric_float(row, metric) for row in case_rows]
        best_value = min(values)
        best[metric] = sorted(
            row["policy_name"]
            for row in case_rows
            if abs(_metric_float(row, metric) - best_value) < 1e-9
        )
    return best


def _behavior_groups(case_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, ...], list[dict[str, str]]] = {}
    signature_metrics = _behavior_signature_metrics(case_rows)
    for row in case_rows:
        signature = tuple(
            round(
                _metric_float(row, metric) if metric in _CONSTRAINT_METRICS else float(row[metric]),
                6,
            )
            for metric in signature_metrics
        )
        grouped.setdefault(signature, []).append(row)

    groups = [
        {
            "policies": sorted(item["policy_name"] for item in rows),
            "signature": {
                metric: round(
                    _metric_float(rows[0], metric) if metric in _CONSTRAINT_METRICS else float(rows[0][metric]),
                    6,
                )
                for metric in signature_metrics
            },
        }
        for rows in grouped.values()
    ]
    groups.sort(key=lambda item: (-len(item["policies"]), item["policies"][0]))
    return groups


def _highlight_comparison(
    case_rows: list[dict[str, str]],
    highlight_pair: tuple[str, str] | None,
) -> dict[str, object]:
    by_policy = {row["policy_name"]: row for row in case_rows}
    comparison_metric_order = _comparison_metric_order(case_rows)
    if highlight_pair is None:
        reference_name = next(
            (
                row["policy_name"]
                for row in case_rows
                if "hierarchical" in row["policy_name"]
            ),
            sorted(by_policy)[0],
        )
        reference_row = by_policy[reference_name]
        comparison_row = max(
            (row for row in case_rows if row["policy_name"] != reference_name),
            key=lambda row: _comparison_priority(row, reference_row, case_rows),
        )
    else:
        reference_name, comparison_name = highlight_pair
        missing = [
            policy_name
            for policy_name in (reference_name, comparison_name)
            if policy_name not in by_policy
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"unknown highlight pair policy_name: {missing_text}")
        reference_row = by_policy[reference_name]
        comparison_row = by_policy[comparison_name]

    return {
        "reference_policy": reference_row["policy_name"],
        "comparison_policy": comparison_row["policy_name"],
        "constraint_priority_mode": _constraint_priority_mode(case_rows),
        "comparison_metric_order": list(comparison_metric_order),
        "delta": {
            metric: round(
                _metric_float(comparison_row, metric) - _metric_float(reference_row, metric),
                6,
            )
            if metric in _CONSTRAINT_METRICS
            else round(float(comparison_row[metric]) - float(reference_row[metric]), 6)
            for metric in comparison_metric_order
        },
        "reference_metrics": {
            metric: round(_metric_float(reference_row, metric), 6)
            if metric in _CONSTRAINT_METRICS
            else round(float(reference_row[metric]), 6)
            for metric in comparison_metric_order
        },
        "comparison_metrics": {
            metric: round(_metric_float(comparison_row, metric), 6)
            if metric in _CONSTRAINT_METRICS
            else round(float(comparison_row[metric]), 6)
            for metric in comparison_metric_order
        },
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_html_report(
    *,
    suite_tier: str,
    num_rows: int,
    num_cases: int,
    controller_modes: list[str],
    metric_separation_counts: dict[str, int],
    constraint_metric_separation_counts: dict[str, int],
    spotlight_cases: list[dict[str, object]],
) -> str:
    metric_cards = "\n".join(
        f"""
        <article class="card">
          <div class="kicker">{metric}</div>
          <strong>{count}</strong>
        </article>
        """
        for metric, count in metric_separation_counts.items()
    )
    constraint_cards = "\n".join(
        f"""
        <article class="card">
          <div class="kicker">{metric}</div>
          <strong>{count}</strong>
        </article>
        """
        for metric, count in constraint_metric_separation_counts.items()
    )
    spotlight_html = "\n".join(
        f"""
        <article class="case">
          <h3>{item['case_id']}</h3>
          <p>regime={item['regime_id']} topology={item['topology_id']}</p>
          <p>violation spread={item['spread']['violation_magnitude_mean']:.3f}</p>
          <p>energy overshoot spread={item['constraint_spread']['energy_overshoot_magnitude_total_mean']:.3f}</p>
          <p>makespan spread={item['spread']['makespan_mean']:.3f}</p>
        </article>
        """
        for item in spotlight_cases
    )
    controller_mode_text = ", ".join(controller_modes)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tree Runtime Paper Artifacts</title>
  <style>
    :root {{
      --bg: #f3f1ea;
      --panel: rgba(255, 255, 255, 0.9);
      --ink: #1f2a31;
      --muted: #5c6b73;
      --line: rgba(31, 42, 49, 0.12);
      --accent: #8d4f2b;
      --cool: #265f7d;
      --shadow: 0 20px 48px rgba(31, 42, 49, 0.12);
      --sans: "Aptos", "Segoe UI Variable Text", sans-serif;
      --mono: "Cascadia Code", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(38, 95, 125, 0.12), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, #faf8f2 100%);
    }}
    .page {{
      width: min(1200px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 24px;
    }}
    .section {{ margin-top: 20px; }}
    h1, h2, h3 {{ margin: 0; line-height: 1.08; }}
    h1 {{ font-size: clamp(36px, 4vw, 58px); }}
    h2 {{ font-size: 24px; }}
    h3 {{ font-size: 18px; }}
    p {{ color: var(--muted); line-height: 1.7; }}
    code {{ font-family: var(--mono); color: var(--cool); }}
    .stats, .metrics, .cases {{
      display: grid;
      gap: 16px;
      margin-top: 16px;
    }}
    .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .metrics {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
    .cases {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .card, .case {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .kicker {{
      color: var(--accent);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    strong {{ font-size: 30px; display: block; margin-top: 8px; }}
    @media (max-width: 1100px) {{
      .stats, .metrics, .cases {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <p class="kicker">Tree Runtime Paper Artifacts</p>
      <h1>{suite_tier}</h1>
      <p>This page is generated from one benchmark <code>suite_summary.csv</code>. It turns the locked benchmark output into a reproducible paper-facing summary rather than a hand-written note.</p>
      <div class="stats">
        <article class="card"><div class="kicker">Rows</div><strong>{num_rows}</strong></article>
        <article class="card"><div class="kicker">Cases</div><strong>{num_cases}</strong></article>
        <article class="card"><div class="kicker">Controller Modes</div><strong>{controller_mode_text}</strong></article>
      </div>
    </section>
    <section class="section">
      <h2>Metric Separation</h2>
      <p>Counts below report how many benchmark cases separate controllers on at least one value for each public metric.</p>
      <div class="metrics">
        {metric_cards}
      </div>
    </section>
    <section class="section">
      <h2>Constraint Separation</h2>
      <p>Structured overshoot metrics stay separate so energy and carbon are not collapsed into one mixed scalar in paper-facing artifacts.</p>
      <div class="metrics">
        {constraint_cards}
      </div>
    </section>
    <section class="section">
      <h2>Spotlight Cases</h2>
      <p>These cases are ranked by structured constraint spread when it separates controllers; otherwise the builder falls back to the legacy mixed violation spread, then energy spread, then makespan spread.</p>
      <div class="cases">
        {spotlight_html}
      </div>
    </section>
  </main>
</body>
</html>
"""


def _render_casebook_report(
    *,
    suite_tier: str,
    case_summaries: list[dict[str, object]],
) -> str:
    case_sections = "\n".join(_render_case_section(case_summary) for case_summary in case_summaries)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tree Runtime Focused Casebook</title>
  <style>
    :root {{
      --bg: #efe8dd;
      --panel: rgba(255, 251, 245, 0.94);
      --panel-strong: #fffdf9;
      --ink: #1f2022;
      --muted: #5d5e62;
      --line: rgba(31, 32, 34, 0.12);
      --accent: #b54d1d;
      --accent-cool: #185f73;
      --success: #1f6d53;
      --shadow: 0 24px 60px rgba(31, 32, 34, 0.12);
      --sans: "Aptos Display", "Segoe UI Variable Display", sans-serif;
      --mono: "Cascadia Code", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top right, rgba(24, 95, 115, 0.14), transparent 26%),
        radial-gradient(circle at top left, rgba(181, 77, 29, 0.14), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, #faf7f1 100%);
    }}
    .page {{
      width: min(1280px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .hero, .case {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 24px;
    }}
    .case {{ margin-top: 22px; }}
    .kicker {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 12px;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.08; }}
    h1 {{ font-size: clamp(36px, 5vw, 60px); }}
    h2 {{ font-size: clamp(24px, 3vw, 34px); }}
    h3 {{ font-size: 18px; }}
    p {{
      color: var(--muted);
      line-height: 1.7;
      margin: 10px 0 0;
    }}
    code {{
      font-family: var(--mono);
      color: var(--accent-cool);
      font-size: 0.95em;
    }}
    .hero-grid, .facts, .metrics, .groups {{
      display: grid;
      gap: 16px;
      margin-top: 18px;
    }}
    .hero-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .facts {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .metrics {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
    .groups {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .card, .group, .comparison, .policy-table {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .card strong {{
      display: block;
      font-size: 30px;
      margin-top: 8px;
    }}
    .grid-two {{
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 14px;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .delta-up {{ color: var(--accent); }}
    .delta-down {{ color: var(--success); }}
    .group ul {{
      margin: 10px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    @media (max-width: 1100px) {{
      .hero-grid, .facts, .metrics, .groups, .grid-two {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <p class="kicker">Tree Runtime Focused Casebook</p>
      <h1>{escape(suite_tier)}</h1>
      <p>This focused bundle turns a locked <code>suite_summary.csv</code> into a paper-facing casebook for the benchmark cases that actually separate controller families.</p>
      <div class="hero-grid">
        <article class="card"><div class="kicker">Headline Cases</div><strong>{len(case_summaries)}</strong></article>
        <article class="card"><div class="kicker">Shared Contract</div><strong>one legal summary table</strong></article>
        <article class="card"><div class="kicker">Reading Rule</div><strong>tradeoffs, not single winner</strong></article>
      </div>
    </section>
    {case_sections}
  </main>
</body>
</html>
"""


def _render_case_section(case_summary: dict[str, object]) -> str:
    highlight = case_summary["highlight_comparison"]
    metric_cards = "\n".join(
        f"""
        <article class="card">
          <div class="kicker">{escape(metric)}</div>
          <strong>{case_summary['metric_spread'][metric]:.3f}</strong>
        </article>
        """
        for metric in _KEY_METRICS
    )
    group_cards = "\n".join(
        f"""
        <article class="group">
          <div class="kicker">Behavior Group</div>
          <h3>{", ".join(escape(policy) for policy in group['policies'])}</h3>
          <p>signature: <code>{escape(_signature_text(group['signature']))}</code></p>
        </article>
        """
        for group in case_summary["behavior_groups"]
    )
    comparison_rows = "\n".join(
        _render_comparison_row(metric, highlight)
        for metric in highlight["comparison_metric_order"]
    )
    policy_rows = "\n".join(
        f"""
        <tr>
          <td><code>{escape(row['policy_name'])}</code></td>
          <td>{row['makespan_mean']:.3f}</td>
          <td>{row['tardiness_mean']:.3f}</td>
          <td>{row['energy_mean']:.3f}</td>
          <td>{row['violation_magnitude_mean']:.3f}</td>
        </tr>
        """
        for row in case_summary["policy_rows"]
    )
    rationale_html = ""
    if case_summary["rationale"]:
        rationale_html = f"<p>{escape(case_summary['rationale'])}</p>"
    return f"""
    <section class="case">
      <p class="kicker">Headline Case</p>
      <h2>{escape(case_summary['headline'])}</h2>
      <p><code>{escape(case_summary['case_id'])}</code></p>
      {rationale_html}
      <div class="facts">
        <article class="card"><div class="kicker">Regime</div><strong>{escape(case_summary['regime_id'])}</strong></article>
        <article class="card"><div class="kicker">Topology</div><strong>{escape(case_summary['topology_id'])}</strong></article>
        <article class="card"><div class="kicker">Modes</div><strong>{escape(", ".join(case_summary['controller_interface_modes']))}</strong></article>
        <article class="card"><div class="kicker">Behavior Groups</div><strong>{case_summary['behavior_group_count']}</strong></article>
      </div>
      <div class="metrics">
        {metric_cards}
      </div>
      <div class="grid-two">
        <article class="comparison">
          <div class="kicker">Highlighted Comparison</div>
          <h3><code>{escape(highlight['reference_policy'])}</code> vs <code>{escape(highlight['comparison_policy'])}</code></h3>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Reference</th>
                <th>Comparison</th>
                <th>Delta</th>
              </tr>
            </thead>
            <tbody>
              {comparison_rows}
            </tbody>
          </table>
        </article>
        <article class="policy-table">
          <div class="kicker">Policy Rows</div>
          <table>
            <thead>
              <tr>
                <th>Policy</th>
                <th>Makespan</th>
                <th>Tardiness</th>
                <th>Energy</th>
                <th>Violation</th>
              </tr>
            </thead>
            <tbody>
              {policy_rows}
            </tbody>
          </table>
        </article>
      </div>
      <div class="groups">
        {group_cards}
      </div>
    </section>
    """


def _render_comparison_row(metric: str, highlight: dict[str, object]) -> str:
    delta = float(highlight["delta"][metric])
    delta_class = "delta-up" if delta > 0 else "delta-down" if delta < 0 else ""
    delta_prefix = "+" if delta > 0 else ""
    return f"""
    <tr>
      <td><code>{escape(metric)}</code></td>
      <td>{highlight['reference_metrics'][metric]:.3f}</td>
      <td>{highlight['comparison_metrics'][metric]:.3f}</td>
      <td class="{delta_class}">{delta_prefix}{delta:.3f}</td>
    </tr>
    """


def _signature_text(signature: dict[str, float]) -> str:
    return ", ".join(f"{metric}={value:.3f}" for metric, value in signature.items())
