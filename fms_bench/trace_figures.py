"""Publication-facing figure data builders derived from trace artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import json


@dataclass(frozen=True)
class TraceFigureArtifactPaths:
    artifacts_csv: Path
    activation_profile_csv: Path
    protocol_profile_csv: Path
    settlement_profile_csv: Path
    world_event_profile_csv: Path
    trace_figure_summary_json: Path
    trace_figure_report_html: Path


def build_trace_publication_figures(
    *,
    artifacts_csv: str | Path,
    output_dir: str | Path,
) -> TraceFigureArtifactPaths:
    artifacts_path = Path(artifacts_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [_normalize_manifest_row(row) for row in _load_csv_rows(artifacts_path)]
    base_dir = artifacts_path.parent
    case_id = _stable_single_value(manifest_rows, "case_id")
    policies = sorted({row["policy_name"] for row in manifest_rows})
    controller_modes = sorted({row["controller_interface_mode"] for row in manifest_rows})

    activation_profile, activation_totals, dominant_activation_agents = _build_activation_profile(
        manifest_rows,
        base_dir,
    )
    protocol_profile, protocol_totals = _build_protocol_profile(manifest_rows, base_dir)
    settlement_profile, completed_contracts_by_policy, settlement_status_counts = _build_settlement_profile(
        manifest_rows,
        base_dir,
    )
    world_event_profile, world_event_totals, world_event_kind_counts = _build_world_event_profile(
        manifest_rows,
        base_dir,
    )

    activation_profile_csv = out_dir / "activation_profile.csv"
    protocol_profile_csv = out_dir / "protocol_profile.csv"
    settlement_profile_csv = out_dir / "settlement_profile.csv"
    world_event_profile_csv = out_dir / "world_event_profile.csv"
    trace_figure_summary_json = out_dir / "trace_figure_summary.json"
    trace_figure_report_html = out_dir / "trace_figure_report.html"

    _write_csv(activation_profile_csv, activation_profile)
    _write_csv(protocol_profile_csv, protocol_profile)
    _write_csv(settlement_profile_csv, settlement_profile)
    _write_csv(world_event_profile_csv, world_event_profile)

    top_protocol_edges = [
        {
            "policy_name": row["policy_name"],
            "sender": row["sender"],
            "recipient": row["recipient"],
            "kind": row["kind"],
            "count": int(row["count"]),
        }
        for row in sorted(
            protocol_profile,
            key=lambda row: (
                -int(row["count"]),
                row["policy_name"],
                row["sender"],
                row["recipient"],
                row["kind"],
            ),
        )[:5]
    ]

    summary_payload = {
        "case_id": case_id,
        "num_policies": len(policies),
        "policies": policies,
        "controller_interface_modes": controller_modes,
        "activation_total_records_by_policy": activation_totals,
        "dominant_activation_agent_by_policy": dominant_activation_agents,
        "protocol_message_count_by_policy": protocol_totals,
        "top_protocol_edges": top_protocol_edges,
        "settlement_status_counts_by_policy": settlement_status_counts,
        "completed_contracts_by_policy": completed_contracts_by_policy,
        "world_event_count_by_policy": world_event_totals,
        "world_event_kind_counts_by_policy": world_event_kind_counts,
    }
    trace_figure_summary_json.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    trace_figure_report_html.write_text(
        _render_html_report(
            case_id=case_id,
            policies=policies,
            controller_modes=controller_modes,
            activation_totals=activation_totals,
            protocol_totals=protocol_totals,
            completed_contracts_by_policy=completed_contracts_by_policy,
            world_event_totals=world_event_totals,
            top_protocol_edges=top_protocol_edges,
        ),
        encoding="utf-8",
    )

    return TraceFigureArtifactPaths(
        artifacts_csv=artifacts_path,
        activation_profile_csv=activation_profile_csv,
        protocol_profile_csv=protocol_profile_csv,
        settlement_profile_csv=settlement_profile_csv,
        world_event_profile_csv=world_event_profile_csv,
        trace_figure_summary_json=trace_figure_summary_json,
        trace_figure_report_html=trace_figure_report_html,
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_manifest_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    if not normalized.get("controller_interface_mode"):
        normalized["controller_interface_mode"] = "in_process"
    return normalized


def _stable_single_value(rows: Iterable[dict[str, str]], key: str) -> str:
    values = {row[key] for row in rows}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _resolve_trace_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_trace_records(base_dir: Path, row: dict[str, str], key: str) -> list[dict[str, object]]:
    payload = json.loads(_resolve_trace_path(base_dir, row[key]).read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for episode in payload.get("episodes", []):
        records.extend(episode.get("trace", []))
    return records


def _build_activation_profile(
    manifest_rows: list[dict[str, str]],
    base_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, dict[str, object]]]:
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    totals: Counter[str] = Counter()
    agent_counts_by_policy: dict[str, Counter[str]] = defaultdict(Counter)

    for row in manifest_rows:
        policy_name = row["policy_name"]
        controller_mode = row["controller_interface_mode"]
        records = _load_trace_records(base_dir, row, "activation_trace_json")
        totals[policy_name] += len(records)
        for record in records:
            agent_id = str(record["agent_id"])
            activation_cause = str(record["activation_cause"])
            functional_loop = str(record["functional_loop"])
            counts[(policy_name, controller_mode, agent_id, activation_cause, functional_loop)] += 1
            agent_counts_by_policy[policy_name][agent_id] += 1

    profile_rows = [
        {
            "policy_name": policy_name,
            "controller_interface_mode": controller_mode,
            "agent_id": agent_id,
            "activation_cause": activation_cause,
            "functional_loop": functional_loop,
            "count": count,
        }
        for (policy_name, controller_mode, agent_id, activation_cause, functional_loop), count in sorted(counts.items())
    ]
    dominant_agents = {
        policy_name: {
            "agent_id": sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0],
            "count": sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][1],
        }
        for policy_name, counter in sorted(agent_counts_by_policy.items())
        if counter
    }
    return profile_rows, dict(sorted(totals.items())), dominant_agents


def _build_protocol_profile(
    manifest_rows: list[dict[str, str]],
    base_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    totals: Counter[str] = Counter()

    for row in manifest_rows:
        policy_name = row["policy_name"]
        controller_mode = row["controller_interface_mode"]
        records = _load_trace_records(base_dir, row, "protocol_trace_json")
        totals[policy_name] += len(records)
        for record in records:
            counts[
                (
                    policy_name,
                    controller_mode,
                    str(record["sender"]),
                    str(record["recipient"]),
                    str(record["kind"]),
                    str(record.get("scope", "")),
                )
            ] += 1

    profile_rows = [
        {
            "policy_name": policy_name,
            "controller_interface_mode": controller_mode,
            "sender": sender,
            "recipient": recipient,
            "kind": kind,
            "scope": scope,
            "count": count,
        }
        for (policy_name, controller_mode, sender, recipient, kind, scope), count in sorted(counts.items())
    ]
    return profile_rows, dict(sorted(totals.items()))


def _build_settlement_profile(
    manifest_rows: list[dict[str, str]],
    base_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, dict[str, int]]]:
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    contracts_by_policy: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    controller_modes: dict[str, str] = {}

    for row in manifest_rows:
        policy_name = row["policy_name"]
        controller_modes[policy_name] = row["controller_interface_mode"]
        records = _load_trace_records(base_dir, row, "settlement_trace_json")
        for record in records:
            status = str(record["status"])
            contract_id = str(record["contract_id"])
            status_counts[policy_name][status] += 1
            contracts_by_policy[policy_name][contract_id].add(status)

    completed_contracts_by_policy: dict[str, int] = {}
    nested_status_counts: dict[str, dict[str, int]] = {}
    profile_rows: list[dict[str, object]] = []
    for policy_name in sorted(status_counts):
        contracts = contracts_by_policy[policy_name]
        completed_contracts = sum(1 for statuses in contracts.values() if "completed" in statuses)
        partial_contracts = sum(1 for statuses in contracts.values() if "partial" in statuses)
        completed_contracts_by_policy[policy_name] = completed_contracts
        nested_status_counts[policy_name] = dict(sorted(status_counts[policy_name].items()))
        for status, count in sorted(status_counts[policy_name].items()):
            profile_rows.append(
                {
                    "policy_name": policy_name,
                    "controller_interface_mode": controller_modes[policy_name],
                    "status": status,
                    "count": count,
                    "unique_contracts": len(contracts),
                    "completed_contracts": completed_contracts,
                    "partial_contracts": partial_contracts,
                }
            )

    return profile_rows, completed_contracts_by_policy, nested_status_counts


def _build_world_event_profile(
    manifest_rows: list[dict[str, str]],
    base_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, dict[str, int]]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    totals: Counter[str] = Counter()
    kind_counts_by_policy: dict[str, Counter[str]] = defaultdict(Counter)

    for row in manifest_rows:
        policy_name = row["policy_name"]
        controller_mode = row["controller_interface_mode"]
        records = _load_trace_records(base_dir, row, "world_event_trace_json")
        totals[policy_name] += len(records)
        for record in records:
            kind = str(record["kind"])
            scope = str(record["scope"])
            counts[(policy_name, controller_mode, kind, scope)] += 1
            kind_counts_by_policy[policy_name][kind] += 1

    profile_rows = [
        {
            "policy_name": policy_name,
            "controller_interface_mode": controller_mode,
            "kind": kind,
            "scope": scope,
            "count": count,
        }
        for (policy_name, controller_mode, kind, scope), count in sorted(counts.items())
    ]
    nested_kind_counts = {
        policy_name: dict(sorted(counter.items()))
        for policy_name, counter in sorted(kind_counts_by_policy.items())
    }
    return profile_rows, dict(sorted(totals.items())), nested_kind_counts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_html_report(
    *,
    case_id: str,
    policies: list[str],
    controller_modes: list[str],
    activation_totals: dict[str, int],
    protocol_totals: dict[str, int],
    completed_contracts_by_policy: dict[str, int],
    world_event_totals: dict[str, int],
    top_protocol_edges: list[dict[str, object]],
) -> str:
    policy_cards = "\n".join(
        f"""
        <article class="card">
          <div class="kicker">{policy_name}</div>
          <strong>{activation_totals.get(policy_name, 0)}</strong>
          <p>activations</p>
          <p>{protocol_totals.get(policy_name, 0)} protocol messages</p>
          <p>{completed_contracts_by_policy.get(policy_name, 0)} completed contracts</p>
          <p>{world_event_totals.get(policy_name, 0)} world events</p>
        </article>
        """
        for policy_name in policies
    )
    edge_cards = "\n".join(
        f"""
        <article class="edge">
          <h3>{item['policy_name']}</h3>
          <p><code>{item['sender']}</code> -> <code>{item['recipient']}</code></p>
          <p>{item['kind']}</p>
          <strong>{item['count']}</strong>
        </article>
        """
        for item in top_protocol_edges
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trace Figure Artifacts</title>
  <style>
    :root {{
      --bg: #eef3ef;
      --panel: rgba(255, 255, 255, 0.92);
      --ink: #182329;
      --muted: #5b6a6e;
      --line: rgba(24, 35, 41, 0.12);
      --accent: #9a542b;
      --cool: #29667b;
      --shadow: 0 20px 48px rgba(24, 35, 41, 0.12);
      --sans: "Aptos", "Segoe UI Variable Text", sans-serif;
      --mono: "Cascadia Code", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(41, 102, 123, 0.12), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, #fafcf8 100%);
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
    h1 {{ font-size: clamp(34px, 4vw, 56px); }}
    h2 {{ font-size: 24px; }}
    h3 {{ font-size: 18px; }}
    p {{ color: var(--muted); line-height: 1.7; }}
    code {{ font-family: var(--mono); color: var(--cool); }}
    .stats, .policies, .edges {{
      display: grid;
      gap: 16px;
      margin-top: 16px;
    }}
    .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .policies {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .edges {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .card, .edge {{
      background: rgba(255, 255, 255, 0.94);
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
      .stats, .policies, .edges {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <p class="kicker">Trace Figure Artifacts</p>
      <h1>{case_id}</h1>
      <p>This page is generated from one case-level <code>artifacts.csv</code>. It turns runtime traces into reproducible figure inputs for activation, protocol, settlement, and world-event views.</p>
      <div class="stats">
        <article class="card"><div class="kicker">Policies</div><strong>{len(policies)}</strong></article>
        <article class="card"><div class="kicker">Controller Modes</div><strong>{", ".join(controller_modes)}</strong></article>
        <article class="card"><div class="kicker">Top Edge Count</div><strong>{top_protocol_edges[0]['count'] if top_protocol_edges else 0}</strong></article>
      </div>
    </section>
    <section class="section">
      <h2>Policy Totals</h2>
      <p>Each card condenses one policy into counts that can later become activation, throughput, or protocol-volume figures without re-reading Python internals.</p>
      <div class="policies">
        {policy_cards}
      </div>
    </section>
    <section class="section">
      <h2>Top Protocol Edges</h2>
      <p>These aggregated edges are ranked by message count and are intended as figure seeds for message-flow or contract-propagation diagrams.</p>
      <div class="edges">
        {edge_cards}
      </div>
    </section>
  </main>
</body>
</html>
"""
