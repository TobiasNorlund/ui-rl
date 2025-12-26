from collections import defaultdict
import re
import sys
from ruamel.yaml import YAML


def update_yaml_rollouts(train_config_path: str, new_rollout_paths: list[str]):
    # Initialize ruamel.yaml to preserve formatting
    yaml = YAML()
    yaml.preserve_quotes = True
    
    # 1. Group new_rollout_paths
    new_rollouts = defaultdict(list)
    for rollout_path in new_rollout_paths:
        # Get filename without extension (e.g., row_000)
        row_match = re.search(r"row_\d+", rollout_path)
        if not row_match:
            print(f"WARNING: Couldn't get row from '{rollout_path}'", file=sys.stderr)
            continue
        row_group = row_match.group()       
        new_rollouts[row_group].append(rollout_path)

    # 2. Load the YAML data
    with open(train_config_path, 'r') as f:
        data = yaml.load(f)

    # 3. Update the data structure
    if 'train_rollouts' in data:
        for entry in data['train_rollouts']:
            group_name = entry.get('name')
            if group_name in new_rollouts:
                # Ensure the 'rollouts' key exists
                if 'rollouts' not in entry:
                    entry['rollouts'] = []
                
                # Append the new paths
                for new_rollout in new_rollouts[group_name]:
                    if new_rollout not in entry["rollouts"]:
                        entry['rollouts'].append(new_rollout)
            
    else:
        print(f"ERROR: No 'train_rollouts' in {train_config_path}", file=sys.stderr)
        exit(1)

    yaml.dump(data, sys.stdout)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config")
    parser.add_argument("new_rollouts", nargs="+")
    args = parser.parse_args()

    update_yaml_rollouts(args.train_config, args.new_rollouts)

    print("YAML updated successfully.", file=sys.stderr)