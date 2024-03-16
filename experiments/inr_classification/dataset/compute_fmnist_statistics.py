from argparse import ArgumentParser

from experiments.inr_classification.dataset.compute_mnist_statistics import (
    compute_stats,
)
from experiments.utils import common_parser

if __name__ == "__main__":
    parser = ArgumentParser(
        "Fashion MNIST - generate statistics", parents=[common_parser]
    )
    parser.add_argument(
        "--splits-path", type=str, default="fmnist_splits.json", help="json file name"
    )
    parser.add_argument(
        "--statistics-path",
        type=str,
        default="fmnist_statistics.pth",
        help="Pytorch statistics file name",
    )
    parser.set_defaults(
        save_path=".",
        data_path=".",
    )
    args = parser.parse_args()

    compute_stats(
        data_path=args.data_path,
        save_path=args.save_path,
        splits_path=args.splits_path,
        statistics_path=args.statistics_path,
    )
