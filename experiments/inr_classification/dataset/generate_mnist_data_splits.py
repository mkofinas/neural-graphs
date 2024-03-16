import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from experiments.utils import common_parser, set_logger


def generate_splits(
    data_path, save_path, name="mnist_splits.json", val_size=5000, random_state=None
):
    script_dir = Path(__file__).parent
    inr_path = script_dir / Path(data_path)
    data_split = defaultdict(lambda: defaultdict(list))
    for p in list(inr_path.glob("mnist_png_*/**/*.pth")):
        s = "train" if "train" in p.as_posix() else "test"
        data_split[s]["path"].append(p.relative_to(script_dir).as_posix())
        data_split[s]["label"].append(p.parent.parent.stem.split("_")[-2])

    # val split
    train_indices, val_indices = train_test_split(
        range(len(data_split["train"]["path"])),
        test_size=val_size,
        random_state=random_state,
    )
    data_split["val"]["path"] = [data_split["train"]["path"][v] for v in val_indices]
    data_split["val"]["label"] = [data_split["train"]["label"][v] for v in val_indices]

    data_split["train"]["path"] = [
        data_split["train"]["path"][v] for v in train_indices
    ]
    data_split["train"]["label"] = [
        data_split["train"]["label"][v] for v in train_indices
    ]

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, "
        f"test size: {len(data_split['test']['path'])}"
    )

    save_path = script_dir / Path(save_path) / name
    with open(save_path, "w") as file:
        json.dump(data_split, file)


if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate data splits", parents=[common_parser])
    parser.add_argument(
        "--name", type=str, default="mnist_splits.json", help="json file name"
    )
    parser.add_argument(
        "--val-size", type=int, default=5000, help="number of validation examples"
    )
    parser.add_argument(
        "--random-state", type=int, default=None, help="random state for split"
    )
    parser.set_defaults(
        save_path=".",
        data_path="mnist-inrs",
    )
    args = parser.parse_args()

    set_logger()

    generate_splits(
        data_path=args.data_path,
        save_path=args.save_path,
        name=args.name,
        val_size=args.val_size,
        random_state=args.random_state,
    )
