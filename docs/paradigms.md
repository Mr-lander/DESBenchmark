# Paradigms And Authority Modes

Paradigms are defined in `fms_bench/v2/authority_modes.py`. They are benchmark-level authority configurations, not framework-specific features.

| Paradigm | Routing Authority | Cell Contract Actions | Area Local Reroute | Plant Reroute Required | Mediated Bid Round |
| --- | --- | --- | --- | --- | --- |
| `centralized` | `centralized` | `accept_only` | false | false | false |
| `hierarchical` | `hierarchical` | `accept_only` | false | true | false |
| `heterarchical_cnp` | `mediated_cnp` | `accept_reject` | false | false | true |
| `holonic_hybrid` | `holonic_hybrid` | `accept_reject` | true | false | false |

## Interpretation

- `centralized`: plant-level centralized routing; cells only accept assigned contracts.
- `hierarchical`: layered plant/area routing; plant reroute is the required escalation path.
- `heterarchical_cnp`: mediated contract-net style routing; cells can accept or reject and a mediated bid round is enabled.
- `holonic_hybrid`: hybrid holonic routing; cells can accept or reject and area-local reroute is enabled.

These descriptions are direct summaries of the fields in `AuthorityMode`. The legacy alias `mediated_heterarchical_cnp` remains accepted for backward compatibility, but public docs and released result tables should use `heterarchical_cnp`.

## Implementation Boundary

In the paper, a paradigm is the coordination protocol `Gamma`. In this repository, the release-facing `Gamma` variants are encoded as `AuthorityMode` records. The authority mode is passed into the tree runtime and changes the protocol-conditioned interface: who can act, which actions are legal, whether rejection is exposed, and whether reroute or mediated bid behavior is enabled.

The framework adapter does not define the paradigm. It receives the payload produced by the runtime under the selected authority mode and returns one legal action. This is why the formal A5C12 matrix is reported as:

```text
model x framework implementation x paradigm x benchmark profile x seed
```

## Adding A Paradigm

If a new paradigm is just a new combination of the existing authority switches, add it to `_AUTHORITY_MODE_REGISTRY` in `fms_bench/v2/authority_modes.py`, then decide whether it belongs in `MAINLINE_CATEGORY_B_AUTHORITY_MODES` in `fms_bench/experiment_runner.py`.

If it needs new activation, message, or action semantics, the registry change is not enough. Review and test the runtime consumers:

- `fms_bench/v2/tree_runtime_env.py`
- `fms_bench/v2/area_runtime.py`
- `fms_bench/v2/cell_runtime.py`
- `fms_bench/v2/controller_payloads.py`

New paradigm results should be labeled as extension experiments unless the official paper matrix is deliberately revised.

See [extension_guide.md](extension_guide.md) for the full checklist.
