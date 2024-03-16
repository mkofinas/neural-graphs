import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch

from experiments.utils import common_parser, set_logger


def generate_splits(
    data_path,
    save_path,
    name="fmnist_splits.json",
):
    script_dir = Path(__file__).parent
    inr_path = script_dir / Path(data_path)
    with open(inr_path, "r") as f:
        data = json.load(f)

    splits = ["train", "val", "test"]
    data_split = defaultdict(lambda: defaultdict(list))
    for split in splits:
        print(f"Processing {split} split")

        data_split[split]["path"] = [
            (Path(data_path).parent / Path(*Path(di).parts[-2:])).as_posix()
            for di in data[split]
        ]

        data_split[split]["label"] = [
            torch.load(p, map_location=lambda storage, loc: storage)["label"]
            for p in data_split[split]["path"]
        ]

        print(f"Finished processing {split} split")

    save_path = script_dir / Path(save_path) / name
    with open(save_path, "w") as file:
        json.dump(data_split, file)


if __name__ == "__main__":
    parser = ArgumentParser(
        "INR Classification - Fashion MNIST - preprocess data", parents=[common_parser]
    )
    parser.add_argument(
        "--name", type=str, default="fmnist_splits.json", help="json file name"
    )
    parser.set_defaults(
        save_path=".",
        data_path="fmnist_inrs/splits.json",
    )
    args = parser.parse_args()

    set_logger()

    generate_splits(
        args.data_path,
        args.save_path,
        name=args.name,
    )
