from collections import defaultdict
from pathlib import Path


def get_rollout_result(rollout_dir: Path) -> float:
    # Compute success rate for each row
    n_success = defaultdict(lambda: 0)
    n_tot = defaultdict(lambda: 0)
    for rollout in rollout_dir.glob("row_*.json"):
        _, row, res, _ = rollout.name.split("_")
        row = int(row)
        n_tot[row] += 1
        if res == "success":
            n_success[row] += 1
    return n_success, n_tot