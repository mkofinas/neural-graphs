import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from rff.layers import GaussianEncoding, PositionalEncoding
from torch import nn


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def params_to_tensor(params):
    return torch.cat([p.flatten() for p in params]), [p.shape for p in params]


def tensor_to_params(tensor, shapes):
    params = []
    start = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()
        param = tensor[start : start + size].reshape(shape)
        params.append(param)
        start += size
    return tuple(params)


def wrap_func(func, shapes):
    def wrapped_func(params, *args, **kwargs):
        params = tensor_to_params(params, shapes)
        return func(params, *args, **kwargs)

    return wrapped_func


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight: torch.Tensor, bias: torch.Tensor, c: float, w0: float):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            # bias.uniform_(-w_std, w_std)
            bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class INR(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        n_layers: int = 3,
        hidden_features: int = 32,
        out_features: int = 1,
        pe_features: Optional[int] = None,
        fix_pe=True,
    ):
        super().__init__()

        if pe_features is not None:
            if fix_pe:
                self.layers = [PositionalEncoding(sigma=10, m=pe_features)]
                encoded_dim = in_features * pe_features * 2
            else:
                self.layers = [
                    GaussianEncoding(
                        sigma=10, input_size=in_features, encoded_size=pe_features
                    )
                ]
                encoded_dim = pe_features * 2
            self.layers.append(Siren(dim_in=encoded_dim, dim_out=hidden_features))
        else:
            self.layers = [Siren(dim_in=in_features, dim_out=hidden_features)]
        for i in range(n_layers - 2):
            self.layers.append(Siren(hidden_features, hidden_features))
        self.layers.append(nn.Linear(hidden_features, out_features))
        self.seq = nn.Sequential(*self.layers)
        self.num_layers = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + 0.5


class INRPerLayer(INR):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = [x]
        for layer in self.seq:
            nodes.append(layer(nodes[-1]))
        nodes[-1] = nodes[-1] + 0.5
        return nodes


def make_functional(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values
