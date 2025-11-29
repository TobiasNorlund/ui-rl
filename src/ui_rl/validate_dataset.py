from typing import List
from train import load_dataset
from tqdm import tqdm


def main(rollouts: List[str]):
    ds = load_dataset(rollouts)
    for i in tqdm(range(len(ds))):
        try:
            _ = ds[i]
        except AssertionError as e:
            print(i, str(e))


if __name__ == "__main__":
    successful = [path.strip() for path in open("rollouts/20251116_161543/successful.txt")]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", nargs="+", default=successful)
    args = parser.parse_args()
    main(**vars(args))