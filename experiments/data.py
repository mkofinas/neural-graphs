import json
import random
from pathlib import Path
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import transforms

from experiments.utils import make_coordinates
from nn import inr
from nn.inr import INR


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class ImageBatch(NamedTuple):
    image: torch.Tensor
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(*[t.to(device) for t in self])

    def __len__(self):
        return len(self.image)


class INRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        splits_path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="statistics.pth",
        translation_scale=0.25,
        rotation_degree=45,
        noise_scale=1e-1,
        drop_rate=1e-2,
        resize_scale=0.2,
        pos_scale=0.0,
        quantile_dropout=0.0,
        class_mapping=None,
    ):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        self.root = self.splits_path.parent
        with self.splits_path.open("r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [
            Path(dataset_dir) / Path(p) for p in self.dataset["path"]
        ]

        self.augmentation = augmentation
        self.permutation = permutation
        self.normalize = normalize
        if self.normalize:
            statistics_path = (
                (Path(dataset_dir) / Path(statistics_path)).expanduser().resolve()
            )
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.translation_scale = translation_scale
        self.rotation_degree = rotation_degree
        self.noise_scale = noise_scale
        self.drop_rate = drop_rate
        self.resize_scale = resize_scale
        self.pos_scale = pos_scale
        self.quantile_dropout = quantile_dropout

        if class_mapping is not None:
            self.class_mapping = class_mapping
            self.dataset["label"] = [
                self.class_mapping[l] for l in self.dataset["label"]
            ]

    def __len__(self):
        return len(self.dataset["label"])

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    @staticmethod
    def rotation_mat(degree=30.0):
        angle = torch.empty(1).uniform_(-degree, degree)
        angle_rad = angle * (torch.pi / 180)
        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )
        return rotation_matrix

    def _augment(self, weights, biases):
        """Augmentations for MLP (and some INR specific ones)

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)
        # translation
        translation = torch.empty(weights[0].shape[0]).uniform_(
            -self.translation_scale, self.translation_scale
        )
        order = random.sample(range(1, len(weights)), 1)[0]
        bias_res = translation
        i = 0
        for i in range(order):
            bias_res = bias_res @ weights[i]

        new_biases[i] += bias_res

        # rotation
        if new_weights[0].shape[0] == 2:
            rot_mat = self.rotation_mat(self.rotation_degree)
            new_weights[0] = rot_mat @ new_weights[0]

        # noise
        new_weights = [w + w.std() * self.noise_scale for w in new_weights]
        new_biases = [
            b + b.std() * self.noise_scale if b.shape[0] > 1 else b for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        # scale
        # todo: can also apply to deeper layers
        rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.resize_scale
        new_weights[0] = new_weights[0] * rand_scale

        # positive scale
        if self.pos_scale > 0:
            for i in range(len(new_weights) - 1):
                # todo: we do a lot of duplicated stuff here
                out_dim = new_biases[i].shape[0]
                scale = torch.from_numpy(
                    np.random.uniform(
                        1 - self.pos_scale, 1 + self.pos_scale, out_dim
                    ).astype(np.float32)
                )
                inv_scale = 1.0 / scale
                new_weights[i] = new_weights[i] * scale
                new_biases[i] = new_biases[i] * scale
                new_weights[i + 1] = (new_weights[i + 1].T * inv_scale).T

        if self.quantile_dropout > 0:
            do_q = torch.empty(1).uniform_(0, self.quantile_dropout)
            q = torch.quantile(
                torch.cat([v.flatten().abs() for v in new_weights + new_biases]), q=do_q
            )
            new_weights = [torch.where(w.abs() < q, 0, w) for w in new_weights]
            new_biases = [torch.where(w.abs() < q, 0, w) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _permute(weights, biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        return new_weights, new_biases

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        label = int(self.dataset["label"][item])

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        # Add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        return Batch(weights=weights, biases=biases, label=label)


class INRDummyDataset(torch.utils.data.Dataset):
    def __init__(self, layer_layout):
        self.layer_layout = layer_layout

    def __len__(self):
        return 1

    def __getitem__(self, item):
        # Generate dummy weights using layer_layout
        weights = tuple(
            [
                torch.randn(self.layer_layout[i], self.layer_layout[i + 1])
                for i in range(len(self.layer_layout) - 1)
            ]
        )
        biases = tuple(
            [
                torch.randn(self.layer_layout[i])
                for i in range(1, len(self.layer_layout))
            ]
        )
        label = 0
        # Add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        return Batch(weights=weights, biases=biases, label=label)


class INRAndImageDataset(INRDataset):
    def __init__(self, img_ds, img_offset, style_function, dataset_name, **kwargs):
        super().__init__(**kwargs)
        self.img_offset = img_offset
        self.img_ds = img_ds
        self.dataset_name = dataset_name

        self.img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                style_function,
                transforms.ToTensor(),
                # transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ]
        )
        self.input_img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, item):
        batch = super().__getitem__(item)
        if self.dataset_name == "mnist":
            img_id = int(self.dataset["path"][item].parts[-3].split("_")[-1])
        else:
            img_id = int(
                self.dataset["path"][item].parts[-1].split("_")[-1].split(".")[0]
            )
        img, _ = self.img_ds[img_id]
        transformed_img = self.img_transform(img)
        input_img = self.input_img_transform(img)
        return batch, transformed_img, input_img


class BatchSiren(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_layers=3,
        hidden_features=32,
        img_shape=None,
        input_init=None,
    ):
        super().__init__()
        inr_module = INR(
            in_features=in_features,
            n_layers=n_layers,
            hidden_features=hidden_features,
            out_features=out_features,
        )
        fmodel, params = inr.make_functional(inr_module)

        vparams, vshapes = inr.params_to_tensor(params)
        self.sirens = torch.vmap(inr.wrap_func(fmodel, vshapes))

        inputs = (
            input_init if input_init is not None else make_coordinates(img_shape, 1)
        )
        self.inputs = nn.Parameter(inputs, requires_grad=False)

        self.reshape_weights = Rearrange("b i o 1 -> b (o i)")
        self.reshape_biases = Rearrange("b o 1 -> b o")

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
        return out
