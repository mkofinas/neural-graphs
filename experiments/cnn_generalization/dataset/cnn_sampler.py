from dataclasses import dataclass
import random
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CNNConfig:
    n_layers: int
    n_classes: int
    channels: list[int]
    kernel_size: list[int]
    stride: list[int]
    padding: list[int]
    residual: list[int]
    activation: list[str]


DEFAULT_CONFIG_OPTIONS = {
    "n_layers": [2, 3, 4, 5],
    "n_classes": 10,
    "in_channels": 3,
    "channels": [4, 8, 16, 32],
    "kernel_size": [3, 5, 7],
    "stride": [1],
    "activation": ["relu", "gelu", "tanh", "sigmoid", "leaky_relu"],
}


ACTIVATION_FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "none": lambda x: x,
}


class CNN(nn.Module):
    def __init__(self, cfg: CNNConfig) -> None:
        super().__init__()
        self._assert_cfg(cfg)
        self.cfg = cfg

        self.layers = nn.ModuleList()
        for i in range(len(cfg.channels) - 1):
            in_channels = cfg.channels[i]
            out_channels = cfg.channels[i + 1]
            kernel_size = cfg.kernel_size[i]
            stride = cfg.stride[i]
            padding = cfg.padding[i]
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
            self.layers.append(conv)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(cfg.channels[-1], cfg.n_classes)

    def _assert_cfg(self, cfg: CNNConfig) -> None:
        assert (
            len(cfg.channels) - 1
            == len(cfg.kernel_size)
            == len(cfg.stride)
            == len(cfg.residual)
            == len(cfg.activation)
            == cfg.n_layers
        )
        assert len(cfg.channels) >= 2
        assert cfg.residual[0] == -1
        assert cfg.n_layers - 1 not in cfg.residual
        for i, r in enumerate(cfg.residual):
            assert r < i - 1 or r < 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = dict()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.cfg.residual[i] > -1:
                # shared channels between residual and current layer
                ch_res = self.cfg.channels[self.cfg.residual[i] + 1]
                ch_out = self.cfg.channels[i + 1]
                ch = min(ch_res, ch_out)
                x = x.clone()
                x[:, :ch] += residuals[self.cfg.residual[i]][:, :ch]
            x = ACTIVATION_FN[self.cfg.activation[i]](x)
            # ------ x here is the node in the computation graph ------
            if i in self.cfg.residual:
                # save the residual for later use
                residuals[i] = x

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def sample_cnn_config(options: Union[dict, None] = None) -> CNNConfig:
    if options is None:
        options = DEFAULT_CONFIG_OPTIONS

    n_layers = random.choice(options["n_layers"])
    n_classes = options["n_classes"]
    channels = [options["in_channels"]] + [
        random.choice(options["channels"]) for _ in range(n_layers)
    ]
    kernel_size = [random.choice(options["kernel_size"]) for _ in range(n_layers)]
    stride = [random.choice(options["stride"]) for _ in range(n_layers)]
    # padding based on (odd) kernel size
    padding = [kernel_size[i] // 2 for i in range(n_layers)]
    # residuals can come from any previous layer, but not the preceding one
    residual = [-1] + [random.choice([-1, *range(i)]) for i in range(n_layers - 1)]
    activation = [random.choice(options["activation"]) for _ in range(n_layers)]
    return CNNConfig(
        n_layers=n_layers,
        n_classes=n_classes,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        residual=residual,
        activation=activation,
    )
