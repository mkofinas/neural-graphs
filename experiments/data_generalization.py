import json
import math
import os
import pickle
import random
from pathlib import Path
from typing import NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters
from torchvision.models.vision_transformer import _vision_transformer

from experiments.cnn_generalization.utils import cnn_to_tg_data, pad_and_flatten_kernel
from experiments.transformer_generalization.utils import vit_to_tg_data


class CNNBatch(NamedTuple):
    weights: Tuple
    biases: Tuple
    y: Union[torch.Tensor, float]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            y=self.y.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class CNNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        splits_path,
        split="train",
        normalize=False,
        augmentation=False,
        statistics_path="dataset/statistics.pth",
        noise_scale=1e-1,
        drop_rate=1e-2,
        max_kernel_size=(3, 3),
        linear_as_conv=False,
        flattening_method="repeat_nodes",
        max_num_hidden_layers=3,
    ):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        with self.splits_path.open("r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [
            (Path(dataset_dir) / Path(p)).as_posix() for p in self.dataset["path"]
        ]

        self.augmentation = augmentation
        self.normalize = normalize
        if self.normalize:
            statistics_path = (
                (Path(dataset_dir) / Path(statistics_path)).expanduser().resolve()
            )
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.noise_scale = noise_scale
        self.drop_rate = drop_rate

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

    def __len__(self):
        return len(self.dataset["score"])

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple(
            (w - m.flatten()[None, None, ...]) / s.flatten()[None, None, ...]
            for w, m, s in zip(weights, wm, ws)
        )
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    def _augment(self, weights, biases):
        """Augmentations for MLP (and some INR specific ones)

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)
        # noise
        new_weights = [
            w + w.std() * self.noise_scale * torch.randn_like(w) for w in new_weights
        ]
        new_biases = [
            (
                b + b.std() * self.noise_scale * torch.randn_like(b)
                if b.shape[0] > 1
                else b
            )
            for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _transform_weights_biases(w, max_kernel_size, linear_as_conv=False):
        """
        Convolutional weights are 4D, and they are stored in the following
        order: [out_channels, in_channels, height, width]
        Linear weights are 2D, and they are stored in the following order:
        [out_features, in_features]

        1. We transpose the in_channels and out_channels dimensions in
        convolutions, and the in_features and out_features dimensions in linear
        layers
        2. We have a maximum HxW value, and pad the convolutional kernel with
        0s if necessary
        3. We flatten the height and width dimensions of the convolutional
        weights
        4. We unsqueeze the last dimension of weights and biases
        """
        if w.ndim == 1:
            w = w.unsqueeze(-1)
            return w

        w = w.transpose(0, 1)

        # TODO: Simplify the logic here
        if linear_as_conv:
            if w.ndim == 2:
                w = w.unsqueeze(-1).unsqueeze(-1)
            w = pad_and_flatten_kernel(w, max_kernel_size)
        else:
            w = (
                pad_and_flatten_kernel(w, max_kernel_size)
                if w.ndim == 4
                else w.unsqueeze(-1)
            )

        return w

    @staticmethod
    def _cnn_to_mlp_repeat_nodes(weights, biases, conv_mask):
        final_conv_layer = max([i for i, w in enumerate(conv_mask) if w])
        final_feature_map_size = (
            weights[final_conv_layer + 1].shape[0] // weights[final_conv_layer].shape[1]
        )
        weights[final_conv_layer] = weights[final_conv_layer].repeat(
            1, final_feature_map_size, 1
        )
        biases[final_conv_layer] = biases[final_conv_layer].repeat(
            final_feature_map_size, 1
        )
        return weights, biases, final_feature_map_size

    @staticmethod
    def _cnn_to_mlp_extra_layer(weights, biases, conv_mask, max_kernel_size):
        final_conv_layer = max([i for i, w in enumerate(conv_mask) if w])
        final_feature_map_size = (
            weights[final_conv_layer + 1].shape[0] // weights[final_conv_layer].shape[1]
        )
        dtype = weights[final_conv_layer].dtype
        # NOTE: We assume that the final feature map is square
        spatial_resolution = int(math.sqrt(final_feature_map_size))
        new_weights = (
            torch.eye(weights[final_conv_layer + 1].shape[0])
            .unflatten(0, (weights[final_conv_layer].shape[1], final_feature_map_size))
            .transpose(1, 2)
            .unflatten(-1, (spatial_resolution, spatial_resolution))
        )
        new_weights = pad_and_flatten_kernel(new_weights, max_kernel_size)

        new_biases = torch.zeros(
            (weights[final_conv_layer + 1].shape[0], 1),
            dtype=dtype,
        )
        weights = (
            weights[: final_conv_layer + 1]
            + [new_weights]
            + weights[final_conv_layer + 1 :]
        )
        biases = (
            biases[: final_conv_layer + 1]
            + [new_biases]
            + biases[final_conv_layer + 1 :]
        )
        return weights, biases, final_feature_map_size

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        # Create a mask to denote which layers are convolutional and which are linear
        conv_mask = [
            1 if w.ndim == 4 else 0 for k, w in state_dict.items() if "weight" in k
        ]

        layer_layout = [list(state_dict.values())[0].shape[1]] + [
            v.shape[0] for k, v in state_dict.items() if "bias" in k
        ]

        weights = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "weight" in k
        ]
        biases = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "bias" in k
        ]
        score = float(self.dataset["score"][item])

        # NOTE: We assume that the architecture includes linear layers and
        # convolutional layers
        if self.flattening_method == "repeat_nodes":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_repeat_nodes(
                weights, biases, conv_mask
            )
        elif self.flattening_method == "extra_layer":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_extra_layer(
                weights, biases, conv_mask, self.max_kernel_size
            )
        elif self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
        )
        return data


class NFNZooDataset(CNNDataset):
    def __init__(
        self,
        data_path,
        split,
        idcs_file=None,
        augmentation=False,
        normalize=False,
        statistics_path="dataset/zoo_cifar_nfn_statistics.pth",
        noise_scale=1e-1,
        drop_rate=1e-2,
        max_kernel_size=(3, 3),
        linear_as_conv=False,
        flattening_method="repeat_nodes",
        max_num_hidden_layers=3,
        data_format="graph",
    ):
        data = np.load(os.path.join(data_path, "weights.npy"))
        # Hardcoded shuffle order for consistent test set.
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        data = data[shuffled_idcs]
        metrics = pd.read_csv(
            os.path.join(data_path, "metrics.csv.gz"), compression="gzip"
        )
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(os.path.join(data_path, "layout.csv"))
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[split]
        data = data[idcs]
        self.metrics = metrics.iloc[idcs]

        # iterate over rows of layout
        # for each row, get the corresponding weights from data
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"] : row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

        self.augmentation = augmentation
        self.normalize = normalize
        if self.normalize:
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.noise_scale = noise_scale
        self.drop_rate = drop_rate

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

        if data_format not in ("graph", "nfn"):
            raise ValueError(f"data_format {data_format} not recognized.")
        self.data_format = data_format

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def get_original(self, idx):
        # insert a channel dim
        weights = tuple(w[idx] for w in self.weights)
        biases = tuple(b[idx] for b in self.biases)
        score = self.metrics.iloc[idx].test_accuracy.item()
        return (weights, biases), score

    def __getitem__(self, idx):
        weights = [torch.from_numpy(w[idx]) for w in self.weights]
        biases = [torch.from_numpy(b[idx]) for b in self.biases]
        score = self.metrics.iloc[idx].test_accuracy.item()

        if self.data_format == "nfn":
            return CNNBatch(weights=weights, biases=biases, y=score)

        # Create a mask to denote which layers are convolutional and which are linear
        conv_mask = [1 if w.ndim == 4 else 0 for w in weights]

        layer_layout = [weights[0].shape[1]] + [v.shape[0] for v in biases]

        weights = [
            self._transform_weights_biases(
                w, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for w in weights
        ]
        biases = [
            self._transform_weights_biases(
                b, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for b in biases
        ]

        # NOTE: We assume that the architecture includes linear layers and
        # convolutional layers
        if self.flattening_method == "repeat_nodes":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_repeat_nodes(
                weights, biases, conv_mask
            )
        elif self.flattening_method == "extra_layer":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_extra_layer(
                weights, biases, conv_mask, self.max_kernel_size
            )
        elif self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
        )
        return data


class CNNParkCIFAR10(CNNDataset):
    def __init__(
        self,
        dataset_dir,
        splits_path,
        split="train",
        normalize=False,
        augmentation=False,
        statistics_path=None,
        noise_scale=1e-1,
        drop_rate=1e-2,
        max_kernel_size=(7, 7),
        linear_as_conv=False,
        flattening_method="repeat_nodes",
        max_num_hidden_layers=5,
        data_format="graph",
    ):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        with self.splits_path.open("r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [
            (Path(dataset_dir) / Path(p)).as_posix() for p in self.dataset["path"]
        ]

        # max_step = max(self.dataset["step"])
        # self.dataset["path"] = [
        #     p for p, s in zip(self.dataset["path"], self.dataset["step"])
        #     if s == max_step
        # ]
        # self.dataset["score"] = [
        #     score for score, step in zip(self.dataset["score"], self.dataset["step"])
        #     if step == max_step
        # ]
        # self.dataset["step"] = [
        #     step for step in self.dataset["step"] if step == max_step
        # ]

        self.augmentation = augmentation
        self.normalize = normalize
        if self.normalize:
            statistics_path = (
                (Path(dataset_dir) / Path(statistics_path)).expanduser().resolve()
            )
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.noise_scale = noise_scale
        self.drop_rate = drop_rate

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

        if data_format not in ("graph", "nfn", "stat"):
            raise ValueError(f"data_format {data_format} not recognized.")
        self.data_format = data_format

    @staticmethod
    def compute_stats(tensor: torch.Tensor) -> torch.Tensor:
        """Computes the statistics of the given tensor."""
        mean = tensor.mean()  # (B, C)
        var = tensor.var()  # (B, C)
        q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to(tensor.device)
        quantiles = torch.quantile(tensor, q)  # (5, B, C)
        return torch.stack([mean, var, *quantiles], dim=-1)  # (B, C, 7)

    def __getitem__(self, item):
        path = Path(self.dataset["path"][item])
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint["model"]

        # Create a mask to denote which layers are convolutional and which are linear
        conv_mask = [
            1 if w.ndim == 4 else 0 for k, w in state_dict.items() if "weight" in k
        ]

        layer_layout = [list(state_dict.values())[0].shape[1]] + [
            v.shape[0] for k, v in state_dict.items() if "bias" in k
        ]
        # initial_weight_shapes = [
        #     w.shape[-2:] for k, w in state_dict.items()
        #     if "weight" in k and w.ndim == 4
        # ]
        if self.data_format == "stat":
            # Early Exit for Baseline Statistics model, cannot incorporate
            # residuals or activations
            max_size = 7 * (self.max_num_hidden_layers + 1)
            weights = [v.flatten() for k, v in state_dict.items() if "weight" in k]
            weight_stats = torch.cat([self.compute_stats(w) for w in weights], dim=0)
            weight_stats = F.pad(weight_stats, (0, max_size - weight_stats.shape[0]))
            biases = [v for k, v in state_dict.items() if "bias" in k]
            bias_stats = torch.cat([self.compute_stats(b) for b in biases], dim=0)
            bias_stats = F.pad(bias_stats, (0, max_size - bias_stats.shape[0]))
            score = float(self.dataset["score"][item])
            return CNNBatch(weights=weight_stats, biases=bias_stats, y=score)

        weights = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "weight" in k
        ]
        biases = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "bias" in k
        ]
        score = float(self.dataset["score"][item])

        residual_connections = checkpoint["config"]["residual"]
        activations = checkpoint["config"]["activation"]

        # NOTE: We assume that the architecture includes linear layers and
        # convolutional layers
        if self.flattening_method == "repeat_nodes":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_repeat_nodes(
                weights, biases, conv_mask
            )
        elif self.flattening_method == "extra_layer":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_extra_layer(
                weights, biases, conv_mask, self.max_kernel_size
            )
        elif self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
            residual_connections=residual_connections,
            activations=activations,
        )
        return data


class TransformerParkCIFAR10(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        splits_path,
        metadata_file,
        split="train",
        normalize=False,
        augmentation=False,
        statistics_path=None,
        noise_scale=1e-1,
        drop_rate=1e-2,
        max_kernel_size=(2, 2),
        flattening_method=None,
        max_num_hidden_layers=30,
        data_format="graph",
    ):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        with self.splits_path.open("r") as f:
            self.indices = json.load(f)[self.split]

        metadata_file = metadata_file
        data_file = metadata_file.replace("_meta_data.pkl", ".dat")
        with open(Path(dataset_dir) / metadata_file, "rb") as fp:
            self.meta_data = pickle.load(fp)
        self.meta_data = [self.meta_data[idx] for idx in self.indices]
        # Keys: loss_train, acc_train, loss_val, acc_val, loss_test, acc_test, cfg, params

        # Read checkpoints
        # num of models, num of epochs, max num of params (zero padding is used for smaller models)
        mmap_sz = (100, 50, 21994)
        self.dataset = torch.from_numpy(
            np.asarray(
                np.memmap(
                    Path(dataset_dir) / "transformer_park" / data_file,
                    dtype="float32",
                    mode="r",
                    shape=mmap_sz,
                )
            )
        )
        self.dataset = self.dataset[self.indices]
        num_epochs = self.dataset.shape[1]
        # Flatten checkpoints
        self.dataset = self.dataset.flatten(0, 1)
        # Repeat for metadata, keep corresponding losses and accuracies, and
        # copy the config and params
        self.meta_data = [
            {
                k: (v if k in ("cfg", "params") else v[epoch_index])
                for k, v in md.items()
            }
            for md in self.meta_data
            for epoch_index in range(num_epochs)
        ]

        self.augmentation = augmentation
        self.normalize = normalize
        if self.normalize:
            statistics_path = (
                (Path(dataset_dir) / Path(statistics_path)).expanduser().resolve()
            )
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.noise_scale = noise_scale
        self.drop_rate = drop_rate

        self.max_kernel_size = max_kernel_size
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

        if data_format not in ("graph", "nfn", "stat"):
            raise ValueError(f"data_format {data_format} not recognized.")
        self.data_format = data_format

    @staticmethod
    def compute_stats(tensor: torch.Tensor) -> torch.Tensor:
        """Computes the statistics of the given tensor."""
        mean = tensor.mean()  # (B, C)
        var = tensor.var()  # (B, C)
        q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to(tensor.device)
        quantiles = torch.quantile(tensor, q)  # (5, B, C)
        return torch.stack([mean, var, *quantiles], dim=-1)  # (B, C, 7)

    def __len__(self):
        return self.dataset.shape[0]

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple(
            (w - m.flatten()[None, None, ...]) / s.flatten()[None, None, ...]
            for w, m, s in zip(weights, wm, ws)
        )
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    def _augment(self, weights, biases):
        """Augmentations for MLP (and some INR specific ones)

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)
        # noise
        new_weights = [
            w + w.std() * self.noise_scale * torch.randn_like(w) for w in new_weights
        ]
        new_biases = [
            (
                b + b.std() * self.noise_scale * torch.randn_like(b)
                if b.shape[0] > 1
                else b
            )
            for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _transform_weights_biases(name, w, max_kernel_size):
        """
        Convolutional weights are 4D, and they are stored in the following
        order: [out_channels, in_channels, height, width]
        Linear weights are 2D, and they are stored in the following order:
        [out_features, in_features]

        1. We transpose the in_channels and out_channels dimensions in
        convolutions, and the in_features and out_features dimensions in linear
        layers
        2. We have a maximum HxW value, and pad the convolutional kernel with
        0s if necessary
        3. We flatten the height and width dimensions of the convolutional
        weights
        4. We unsqueeze the last dimension of weights and biases
        """
        # TODO: Convert LayerNorm weight to diagonal matrix
        # TODO: Convert in_proj_weight to 3-dimensional features
        if "ln" in name and "weight" in name:
            w = w.diag()

        if w.ndim == 1:
            if "in_proj" in name:
                w = w.unflatten(0, (3, w.shape[0] // 3)).transpose(-1, -2)
            else:
                w = w.unsqueeze(-1)
            return w

        w = w.transpose(0, 1)

        w = (
            pad_and_flatten_kernel(w, max_kernel_size)
            if w.ndim == 4
            else (
                w.unflatten(1, (3, w.shape[1] // 3)).transpose(-1, -2)
                if "in_proj" in name
                else w.unsqueeze(-1)
            )
        )

        return w

    def __getitem__(self, item):
        # NOTE: We ignore `class_token` and `pos_embedding` for now
        checkpoint = self.dataset[item]
        cfg = self.meta_data[item]["cfg"]
        net = _vision_transformer(weights=None, progress=False, **cfg).eval()
        vector_to_parameters(checkpoint, net.parameters())
        state_dict = net.state_dict()

        layer_layout = [list(state_dict.values())[1].shape[1]] + [
            (v.shape[0] // 3 if "in_proj_bias" in k else v.shape[0])
            for k, v in state_dict.items()
            if "bias" in k
        ]
        if self.data_format == "stat":
            # Early Exit for Baseline Statistics model
            max_size = 7 * (self.max_num_hidden_layers + 1)

            weights = [v.flatten() for k, v in state_dict.items() if "weight" in k]
            weight_stats = torch.cat([self.compute_stats(w) for w in weights], dim=0)
            weight_stats = F.pad(weight_stats, (0, max_size - weight_stats.shape[0]))

            biases = [v for k, v in state_dict.items() if "bias" in k]
            bias_stats = torch.cat([self.compute_stats(b) for b in biases], dim=0)
            bias_stats = F.pad(bias_stats, (0, max_size - bias_stats.shape[0]))

            score = float(self.meta_data[item]["acc_test"])
            return CNNBatch(weights=weight_stats, biases=bias_stats, y=score)

        weights = [
            self._transform_weights_biases(
                k,
                v,
                self.max_kernel_size,
            )
            for k, v in state_dict.items()
            if "weight" in k
        ]
        biases = [
            self._transform_weights_biases(
                k,
                v,
                self.max_kernel_size,
            )
            for k, v in state_dict.items()
            if "bias" in k
        ]
        score = float(self.meta_data[item]["acc_test"])

        final_feature_map_size = 1

        weights = tuple(weights)
        biases = tuple(biases)

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        node_dims = torch.LongTensor(
            [1] * 3 + sum([[b.shape[-1]] * b.shape[-2] for b in biases], [])
        )
        edge_dims = torch.LongTensor(
            sum([[w.shape[-1]] * w.shape[1] * w.shape[0] for w in weights], [])
        )

        data = vit_to_tg_data(
            weights,
            biases,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
            edge_dims=edge_dims,
            node_dims=node_dims,
        )
        return data
