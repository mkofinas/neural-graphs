from argparse import ArgumentParser
from pathlib import Path

import torch

from experiments.data import INRDataset
from experiments.utils import common_parser


def compute_stats(
    data_path: str, save_path: str, splits_path: str, statistics_path: str
):
    script_dir = Path(__file__).parent
    data_path = script_dir / Path(data_path)

    train_set = INRDataset(
        dataset_dir=data_path,
        splits_path=splits_path,
        split="train",
        statistics_path=None,
        normalize=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=len(train_set), shuffle=False, num_workers=4
    )

    train_data = next(iter(train_loader))

    train_weights_mean = [w.mean().item() for w in train_data.weights]
    train_weights_std = [w.std().item() for w in train_data.weights]
    train_biases_mean = [w.mean().item() for w in train_data.biases]
    train_biases_std = [w.std().item() for w in train_data.biases]

    print(f"weights_mean: {train_weights_mean}")
    print(f"weights_std: {train_weights_std}")
    print(f"biases_mean: {train_biases_mean}")
    print(f"biases_std: {train_biases_std}")

    dws_weights_mean = [w.mean(0) for w in train_data.weights]
    dws_weights_std = [w.std(0) for w in train_data.weights]
    dws_biases_mean = [w.mean(0) for w in train_data.biases]
    dws_biases_std = [w.std(0) for w in train_data.biases]

    statistics = {
        "weights": {"mean": dws_weights_mean, "std": dws_weights_std},
        "biases": {"mean": dws_biases_mean, "std": dws_biases_std},
    }

    out_path = script_dir / Path(save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    torch.save(statistics, out_path / statistics_path)


if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate statistics", parents=[common_parser])
    parser.add_argument(
        "--splits-path", type=str, default="mnist_splits.json", help="json file name"
    )
    parser.add_argument(
        "--statistics-path",
        type=str,
        default="mnist_statistics.pth",
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
