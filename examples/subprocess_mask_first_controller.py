from __future__ import annotations

import json
import sys


def _choose_first_valid_action(agent_view: dict[str, object]) -> int:
    mask = list(agent_view.get("action_mask", []))
    valid = [index for index, enabled in enumerate(mask) if enabled]
    return valid[0] if valid else 0


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        decision = json.loads(line)
        actions = {
            str(agent_id): _choose_first_valid_action(dict(agent_view))
            for agent_id, agent_view in dict(decision.get("active_agents", {})).items()
        }
        sys.stdout.write(json.dumps(actions) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
