# Paradigms

In the paper, a coordination paradigm is an instantiation of protocol `Gamma`.
In DESBench, a paradigm controls authority distribution, legal action semantics,
and recovery paths while the suite/profile, simulator, metrics, implementation
carrier, controller/model, and seed split stay fixed.

## Official Paradigms

| Paradigm | Routing authority | Cell contract actions | Area local reroute | Plant reroute required | Mediated bid round |
| --- | --- | --- | --- | --- | --- |
| `centralized` | `centralized` | `accept_only` | false | false | false |
| `hierarchical` | `hierarchical` | `accept_only` | false | true | false |
| `heterarchical_cnp` | `mediated_cnp` | `accept_reject` | false | false | true |
| `holonic_hybrid` | `holonic_hybrid` | `accept_reject` | true | false | false |

Interpretation:

- `centralized`: plant-level centralized routing; cells only accept assigned contracts.
- `hierarchical`: layered plant/area routing; plant reroute is the required escalation path.
- `heterarchical_cnp`: mediated contract-net style routing; cells can accept or reject and a mediated bid round is enabled.
- `holonic_hybrid`: hybrid holonic routing; cells can accept or reject and area-local reroute is enabled.

The legacy alias `mediated_heterarchical_cnp` remains accepted for backward compatibility, but public docs and released result tables should use `heterarchical_cnp`.

## Add A New Paradigm

Use a standalone extension file when your new MAS architecture can be expressed
with the existing authority switches. Start from
`examples/custom_paradigm.py`:

```python
from fms_bench.paradigms import AuthorityMode, register_authority_mode

register_authority_mode(
    AuthorityMode(
        mode_id="my_hybrid",
        routing_authority_mode="holonic_hybrid",
        cell_contract_actions="accept_reject",
        area_local_reroute_enabled=True,
        plant_reroute_required=False,
        mediated_bid_round_enabled=True,
    )
)
```

Run it with the public runner:

```bash
uv run python examples/run_suite.py \
  --paradigm-module examples/custom_paradigm.py \
  --framework langgraph \
  --suite intercell_a5c12_harder \
  --authority-mode my_hybrid \
  --case-limit 1 \
  --seed-limit 1
```

Use the wider runtime path only when the paradigm needs new activation,
message, or action semantics beyond the existing switches. In that case, add
tests for legal action masks, activation order, reroute/escalation behavior,
and exported audit fields before reporting results.

## Valid Comparison Rule

When reporting a new paradigm, keep these axes fixed inside the comparison
block:

- `suite_id` and `profile_id`
- seed split
- implementation carrier
- controller/model backend
- simulator and metric definitions

Do not change topology, route grammar, profile knobs, or metric definitions
while claiming a pure paradigm comparison. Those changes define a new benchmark
condition, not just a new coordination paradigm.
