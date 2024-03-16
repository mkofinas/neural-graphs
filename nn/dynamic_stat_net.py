import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicStatFeaturizer(nn.Module):
    def __init__(
        self,
        max_kernel_size,
        max_num_hidden_layers,
        max_kernel_height,
        max_kernel_width,
    ):
        super().__init__()
        self.max_kernel_size = max_kernel_size
        self.max_kernel_dimensions = (max_kernel_height, max_kernel_width)
        self.max_num_hidden_layers = max_num_hidden_layers
        self.max_size = 2 * 7 * (max_num_hidden_layers + 1)

    def forward(self, batch) -> torch.Tensor:
        out = []
        for i in range(len(batch)):
            elem = []
            biases = batch[i].x.split(batch[i].layer_layout)[1:]
            prods = [
                batch[i].layer_layout[j] * batch[i].layer_layout[j + 1]
                for j in range(len(batch[i].layer_layout) - 1)
            ]
            weights = batch[i].edge_attr.split(prods)
            for j, (weight, bias) in enumerate(zip(weights, biases)):
                if j < len(batch[i].initial_weight_shapes):
                    weight = weight.unflatten(-1, self.max_kernel_dimensions)
                    # TODO: Rewrite in a more efficient way
                    start0 = (
                        self.max_kernel_dimensions[0]
                        - batch[i].initial_weight_shapes[j][0]
                    ) // 2
                    end0 = start0 + batch[i].initial_weight_shapes[j][0]
                    start1 = (
                        self.max_kernel_dimensions[1]
                        - batch[i].initial_weight_shapes[j][1]
                    ) // 2
                    end1 = start1 + batch[i].initial_weight_shapes[j][1]
                    weight = weight[:, start0:end0, start1:end1]
                elem.append(self.compute_stats(weight))
                elem.append(self.compute_stats(bias))
            elem = torch.cat(elem, dim=-1)
            elem = F.pad(elem, (0, self.max_size - elem.shape[0]))
            out.append(elem)

        return torch.stack(out, dim=0)

    def compute_stats(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the statistics of the given tensor."""
        mean = tensor.mean()  # (B, C)
        var = tensor.var()  # (B, C)
        q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to(tensor.device)
        quantiles = torch.quantile(tensor, q)  # (5, B, C)
        return torch.stack([mean, var, *quantiles], dim=-1)  # (B, C, 7)


class DynamicStatNet(nn.Module):
    """Outputs a scalar."""

    def __init__(
        self,
        h_size,
        dropout=0.0,
        max_kernel_size=49,
        max_num_hidden_layers=5,
        max_kernel_height=7,
        max_kernel_width=7,
    ):
        super().__init__()
        num_features = 2 * 7 * (max_num_hidden_layers + 1)
        self.hypernetwork = nn.Sequential(
            # DynamicStatFeaturizer(max_kernel_size, max_num_hidden_layers, max_kernel_height, max_kernel_width),
            nn.Linear(num_features, h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_size, 1),
        )

    def forward(self, batch):
        stacked_weights = torch.stack(batch[0], dim=0)
        stacked_biases = torch.stack(batch[1], dim=0)
        stacked_params = torch.cat([stacked_weights, stacked_biases], dim=-1)
        return self.hypernetwork(stacked_params)
