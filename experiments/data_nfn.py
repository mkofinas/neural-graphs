import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from nn import inr


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.num_layers = len(self.net)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output, coords


class SirenPerLayer(Siren):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        outputs = [coords]
        for layer in self.net:
            outputs.append(layer(outputs[-1]))
        return outputs


class BatchSiren(nn.Module):
    def __init__(
        self,
        in_features=2,
        hidden_features=32,
        hidden_layers=1,
        out_features=1,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        img_shape=None,
        input_init=None,
    ):
        super().__init__()

        inr_module = Siren(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        fmodel, params = inr.make_functional(inr_module)

        vparams, vshapes = inr.params_to_tensor(params)
        self.sirens = torch.vmap(inr.wrap_func(fmodel, vshapes))

        inputs = input_init if input_init is not None else get_mgrid(img_shape[0], 2)
        self.inputs = nn.Parameter(inputs, requires_grad=False)

        self.reshape_weights = Rearrange("b i h0 1 -> b (h0 i)")
        self.reshape_biases = Rearrange("b h0 1 -> b h0")

    def forward(self, weights, biases):
        params_flat = torch.cat(
            [
                torch.cat(
                    [self.reshape_weights(w), self.reshape_biases(b)],
                    dim=-1,
                )
                for w, b in zip(weights, biases)
            ],
            dim=-1,
        )

        out = self.sirens(params_flat, self.inputs.expand(params_flat.shape[0], -1, -1))
        return out[0]


def state_dict_to_tensors(state_dict):
    """Converts a state dict into two lists of equal length:
    1. list of weight tensors
    2. list of biases, or None if no bias
    Assumes the state_dict key order is [0.weight, 0.bias, 1.weight, 1.bias, ...]
    """
    weights, biases = [], []
    keys = list(state_dict.keys())
    i = 0
    while i < len(keys):
        weights.append(state_dict[keys[i]][None])
        i += 1
        assert keys[i].endswith("bias")
        biases.append(state_dict[keys[i]][None])
        i += 1
    return weights, biases


class SirenDataset(Dataset):
    def __init__(self, data_path, prefix="randinit_test"):
        idx_pattern = r"net(\d+)\.pth"
        label_pattern = r"_(\d)s"
        self.idx_to_path = {}
        self.idx_to_label = {}
        for siren_path in glob.glob(os.path.join(data_path, f"{prefix}_*/*.pth")):
            idx = int(re.search(idx_pattern, siren_path).group(1))
            self.idx_to_path[idx] = siren_path
            label = int(re.search(label_pattern, siren_path).group(1))
            self.idx_to_label[idx] = label
        assert sorted(list(self.idx_to_path.keys())) == list(
            range(len(self.idx_to_path))
        )

    def __getitem__(self, idx):
        sd = torch.load(self.idx_to_path[idx])
        weights, biases = state_dict_to_tensors(sd)
        return (weights, biases), self.idx_to_label[idx]

    def __len__(self):
        return len(self.idx_to_path)


DEF_TFM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
    ]
)


class SirenAndOriginalDataset(Dataset):
    def __init__(self, siren_path, siren_prefix, data_path, data_tfm=DEF_TFM):
        self.siren_dset = SirenDataset(siren_path, prefix=siren_prefix)
        if "mnist" in siren_path:
            self.data_type = "mnist"
            print("Loading MNIST")
            MNIST_train = MNIST(
                data_path, transform=data_tfm, train=True, download=True
            )
            MNIST_test = MNIST(
                data_path, transform=data_tfm, train=False, download=True
            )
            self.dset = Subset(
                ConcatDataset([MNIST_train, MNIST_test]), range(len(self.siren_dset))
            )
            self.input_dset = Subset(
                ConcatDataset(
                    [
                        MNIST(data_path, transform=DEF_TFM, train=True),
                        MNIST(data_path, transform=DEF_TFM, train=False),
                    ]
                ),
                range(len(self.siren_dset)),
            )
        elif "fashion" in siren_path:
            self.data_type = "fashion"
            print("Loading FashionMNIST")
            fMNIST_train = FashionMNIST(
                data_path, transform=data_tfm, train=True, download=True
            )
            fMNIST_test = FashionMNIST(
                data_path, transform=data_tfm, train=False, download=True
            )
            self.dset = ConcatDataset([fMNIST_train, fMNIST_test])
            self.input_dset = ConcatDataset(
                [
                    FashionMNIST(data_path, transform=DEF_TFM, train=True),
                    FashionMNIST(data_path, transform=DEF_TFM, train=False),
                ]
            )
        else:
            self.data_type = "cifar"
            print("Loading CIFAR10")
            CIFAR_train = CIFAR10(
                data_path, transform=data_tfm, train=True, download=True
            )
            CIFAR_test = CIFAR10(
                data_path, transform=data_tfm, train=False, download=True
            )
            self.dset = ConcatDataset([CIFAR_train, CIFAR_test])
            self.input_dset = ConcatDataset(
                [
                    CIFAR10(data_path, transform=DEF_TFM, train=True),
                    CIFAR10(data_path, transform=DEF_TFM, train=False),
                ]
            )
        assert len(self.siren_dset) == len(
            self.dset
        ), f"{len(self.siren_dset)} != {len(self.dset)}"

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        params, siren_label = self.siren_dset[idx]
        img, data_label = self.dset[idx]
        input_img, _ = self.input_dset[idx]
        assert siren_label == data_label
        # return params, img, data_label
        return params, img, input_img
