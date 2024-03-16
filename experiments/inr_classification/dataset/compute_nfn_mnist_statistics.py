from argparse import ArgumentParser
from pathlib import Path

import torch

from experiments.data_nfn import SirenDataset
from experiments.utils import common_parser


def compute_stats(data_path: str, save_path: str):
    script_dir = Path(__file__).parent
    data_path = script_dir / Path(data_path)
    dset = SirenDataset(data_path, "randinit_smaller")

    train_set = torch.utils.data.Subset(dset, range(45_000))

    all_weights = [d[0][0] for d in train_set]
    all_biases = [d[0][1] for d in train_set]

    weights_mean = []
    weights_std = []
    biases_mean = []
    biases_std = []
    for i in range(len(all_weights[0])):
        weights_mean.append(torch.stack([w[i] for w in all_weights]).mean().item())
        weights_std.append(torch.stack([w[i] for w in all_weights]).std().item())
        biases_mean.append(torch.stack([b[i] for b in all_biases]).mean().item())
        biases_std.append(torch.stack([b[i] for b in all_biases]).std().item())
    print(weights_mean)
    print(weights_std)
    print(biases_mean)
    print(biases_std)

    dws_weights_mean = []
    dws_weights_std = []
    dws_biases_mean = []
    dws_biases_std = []
    for i in range(len(all_weights[0])):
        dws_weights_mean.append(
            torch.stack([w[i] for w in all_weights]).mean(0).squeeze(0).unsqueeze(-1)
        )
        dws_weights_std.append(
            torch.stack([w[i] for w in all_weights]).std(0).squeeze(0).unsqueeze(-1)
        )
        dws_biases_mean.append(
            torch.stack([b[i] for b in all_biases]).mean(0).squeeze(0).unsqueeze(-1)
        )
        dws_biases_std.append(
            torch.stack([b[i] for b in all_biases]).std(0).squeeze(0).unsqueeze(-1)
        )

    statistics = {
        "weights": {"mean": dws_weights_mean, "std": dws_weights_std},
        "biases": {"mean": dws_biases_mean, "std": dws_biases_std},
    }

    out_path = script_dir / Path(save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    torch.save(statistics, out_path / "nfn_mnist_statistics.pth")


if __name__ == "__main__":
    parser = ArgumentParser("NFN MNIST - generate statistics", parents=[common_parser])
    parser.set_defaults(
        data_path="nfn-mnist-inrs",
        save_path=".",
    )
    args = parser.parse_args()

    compute_stats(data_path=args.data_path, save_path=args.save_path)
