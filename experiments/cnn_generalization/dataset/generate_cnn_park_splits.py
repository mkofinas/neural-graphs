import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from itertools import groupby

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from experiments.utils import common_parser, set_logger


def generate_splits(
    data_path,
    save_path,
    name="cnn_park_splits.json",
    val_size=10000,
    test_size=10000,
    seed=42,
):
    script_dir = Path(__file__).parent
    data_path = script_dir / Path(data_path)
    # We have to sort the files to make sure that the order between checkpoints
    # and progresses is the same. We will randomize later.
    checkpoints = sorted(data_path.glob("cifar10_zooV2/*/*/checkpoint.pt"))
    checkpoint_parents = sorted(list(set([c.parent.parent for c in checkpoints])))
    progresses = {
        ckpt.as_posix(): torch.load(ckpt, map_location="cpu")["last_result"]["test/acc"]
        for ckpt in checkpoints
    }

    checkpoint_steps = {
        ckpt.as_posix(): torch.load(ckpt, map_location="cpu")["iteration"]
        for ckpt in checkpoints
    }
    print(
        len(checkpoint_steps),
        len(progresses),
        len(checkpoint_parents),
        len(checkpoints),
    )

    trainval_indices, test_indices = train_test_split(
        range(len(checkpoint_parents)), test_size=test_size, random_state=seed
    )
    train_indices, val_indices = train_test_split(
        trainval_indices, test_size=val_size, random_state=seed
    )
    grouped_checkpoints = [
        list(g) for _, g in groupby(checkpoints, lambda x: x.parent.parent)
    ]

    data_split = defaultdict(lambda: defaultdict(list))
    data_split["train"]["path"] = sum(
        [
            [
                ckpt.relative_to(script_dir).as_posix()
                for ckpt in grouped_checkpoints[idx]
            ]
            for idx in train_indices
        ],
        [],
    )
    data_split["train"]["score"] = sum(
        [
            [progresses[str(ckpt)] for ckpt in grouped_checkpoints[idx]]
            for idx in train_indices
        ],
        [],
    )
    data_split["train"]["step"] = sum(
        [
            [checkpoint_steps[str(ckpt)] for ckpt in grouped_checkpoints[idx]]
            for idx in train_indices
        ],
        [],
    )
    permutation = np.random.permutation(len(data_split["train"]["path"]))
    data_split["train"]["path"] = [
        data_split["train"]["path"][idx] for idx in permutation
    ]
    data_split["train"]["score"] = [
        data_split["train"]["score"][idx] for idx in permutation
    ]
    data_split["train"]["step"] = [
        data_split["train"]["step"][idx] for idx in permutation
    ]

    data_split["val"]["path"] = sum(
        [
            [
                ckpt.relative_to(script_dir).as_posix()
                for ckpt in grouped_checkpoints[idx]
            ]
            for idx in val_indices
        ],
        [],
    )
    data_split["val"]["score"] = sum(
        [
            [progresses[str(ckpt)] for ckpt in grouped_checkpoints[idx]]
            for idx in val_indices
        ],
        [],
    )
    data_split["val"]["step"] = sum(
        [
            [checkpoint_steps[str(ckpt)] for ckpt in grouped_checkpoints[idx]]
            for idx in val_indices
        ],
        [],
    )
    permutation = np.random.permutation(len(data_split["val"]["path"]))
    data_split["val"]["path"] = [data_split["val"]["path"][idx] for idx in permutation]
    data_split["val"]["score"] = [
        data_split["val"]["score"][idx] for idx in permutation
    ]
    data_split["val"]["step"] = [data_split["val"]["step"][idx] for idx in permutation]

    data_split["test"]["path"] = sum(
        [
            [
                ckpt.relative_to(script_dir).as_posix()
                for ckpt in grouped_checkpoints[idx]
            ]
            for idx in test_indices
        ],
        [],
    )
    data_split["test"]["score"] = sum(
        [
            [progresses[str(ckpt)] for ckpt in grouped_checkpoints[idx]]
            for idx in test_indices
        ],
        [],
    )
    data_split["test"]["step"] = sum(
        [
            [checkpoint_steps[str(ckpt)] for ckpt in grouped_checkpoints[idx]]
            for idx in test_indices
        ],
        [],
    )
    permutation = np.random.permutation(len(data_split["test"]["path"]))
    data_split["test"]["path"] = [
        data_split["test"]["path"][idx] for idx in permutation
    ]
    data_split["test"]["score"] = [
        data_split["test"]["score"][idx] for idx in permutation
    ]
    data_split["test"]["step"] = [
        data_split["test"]["step"][idx] for idx in permutation
    ]

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, "
        f"test size: {len(data_split['test']['path'])}"
        f"train score size: {len(data_split['train']['score'])}, "
        f"val score size: {len(data_split['val']['score'])}, "
        f"test score size: {len(data_split['test']['score'])}"
        f"train step size: {len(data_split['train']['step'])}, "
        f"val step size: {len(data_split['val']['step'])}, "
        f"test step size: {len(data_split['test']['step'])}"
    )

    save_path = script_dir / Path(save_path) / name
    with open(save_path, "w") as file:
        json.dump(data_split, file)


if __name__ == "__main__":
    parser = ArgumentParser(
        "CNN Generalization - generate data splits", parents=[common_parser]
    )
    parser.add_argument(
        "--name", type=str, default="cnn_park_splits.json", help="json file name"
    )
    parser.add_argument(
        "--val-size", type=int, default=25, help="number of validation examples"
    )
    parser.add_argument(
        "--test-size", type=int, default=50, help="number of test examples"
    )
    parser.set_defaults(
        save_path=".",
        data_path=".",
    )
    args = parser.parse_args()

    set_logger()

    generate_splits(
        args.data_path,
        args.save_path,
        name=args.name,
        val_size=args.val_size,
        test_size=args.test_size,
    )
