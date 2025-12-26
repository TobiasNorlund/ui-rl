from collections import defaultdict
from ruamel.yaml import YAML
import math
import re
import sys


def main(config: str, rollouts: list[str], temperature: float, min_fail_rate: float):
    yaml = YAML()
    yaml.preserve_quotes = True

    num_success = defaultdict(lambda: 0)
    num_fail = defaultdict(lambda: 0)

    for rollout_path in rollouts:
        row_match = re.search(r"row_\d+", rollout_path)
        if not row_match:
            raise ValueError(f"Couldn't get row from '{rollout_path}'", file=sys.stderr)
        row_group = row_match.group()

        if "success" in rollout_path:
            num_success[row_group] += 1
        elif "fail" in rollout_path:
            num_fail[row_group] += 1
        else:
            raise ValueError(f"Couldn't parse label from '{rollout_path}'")

    # Compute sampling weights/probabilities = softmax(fail_rate[group_1], fail_rate[group_2], ...)
    all_groups = set(num_success.keys()) | set(num_fail.keys())
    fail_rate = {}
    sampling_weight = {}
    for group in all_groups:
        fr = num_fail[group] / (num_success[group] + num_fail[group])
        fail_rate[group] = max(fr, min_fail_rate)
    Z = sum(math.exp(fr / temperature) for fr in fail_rate.values())
    for group in all_groups:
        sampling_weight[group] = math.exp(fail_rate[group] / temperature) / Z

    with open(config, 'r') as f:
        data = yaml.load(f)
    for entry in data["train_rollouts"]:
        group = entry.get('name')
        entry["sampling_weight"] = sampling_weight[group]
    yaml.dump(data, sys.stdout)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("rollouts", nargs="+")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-fail-rate", type=float, default=0.1)
    args = parser.parse_args()

    main(**vars(args))
