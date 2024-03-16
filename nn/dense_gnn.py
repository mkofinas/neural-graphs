import hydra
import torch
import torch.nn as nn
import torch_geometric

from nn.gnn import nn_to_edge_index, to_pyg_batch


def graph_to_wb(
    edge_features,
    node_features,
    weights,
    biases,
    normalize=False,
    weights_mean=None,
    weights_std=None,
    biases_mean=None,
    biases_std=None,
):
    new_weights = []
    new_biases = []

    start = 0
    for i, w in enumerate(weights):
        size = torch.prod(torch.tensor(w.shape[1:]))
        w_mean = weights_mean[i] if normalize and weights_mean is not None else 0
        w_std = weights_std[i] if normalize and weights_std is not None else 1
        new_weights.append(
            edge_features[:, start : start + size].view(w.shape) * w_std + w_mean
        )
        start += size

    start = 0
    for i, b in enumerate(biases):
        size = torch.prod(torch.tensor(b.shape[1:]))
        b_mean = biases_mean[i] if normalize and biases_mean is not None else 0
        b_std = biases_std[i] if normalize and biases_std is not None else 1
        new_biases.append(
            node_features[:, start : start + size].view(b.shape) * b_std + b_mean
        )
        start += size

    return new_weights, new_biases


class GNNParams(nn.Module):
    def __init__(
        self,
        d_hid,
        d_out,
        graph_constructor,
        gnn_backbone,
        layer_layout,
        rev_edge_features,
        stats=None,
        normalize=False,
        compile=False,
        jit=False,
        out_scale=0.01,
    ):
        super().__init__()
        self.nodes_per_layer = layer_layout
        self.layer_idx = torch.cumsum(torch.tensor([0] + layer_layout), dim=0)

        edge_index = nn_to_edge_index(self.nodes_per_layer, "cpu", dtype=torch.long)
        if rev_edge_features:
            edge_index = torch.cat([edge_index, edge_index.flip(dims=(0,))], dim=-1)
        self.register_buffer(
            "edge_index",
            edge_index,
            persistent=False,
        )

        self.construct_graph = hydra.utils.instantiate(
            graph_constructor,
            d_node=d_hid,
            d_edge=d_hid,
            layer_layout=layer_layout,
            rev_edge_features=rev_edge_features,
            stats=stats,
        )

        self.proj_edge = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )
        self.proj_node = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )

        gnn_kwargs = dict()
        if gnn_backbone.get("deg", False) is None:
            extended_layout = [0] + layer_layout
            deg = torch.zeros(max(extended_layout) + 1, dtype=torch.long)
            for li in range(len(extended_layout) - 1):
                deg[extended_layout[li]] += extended_layout[li + 1]

            gnn_kwargs["deg"] = deg
        self.gnn = hydra.utils.instantiate(gnn_backbone, **gnn_kwargs)
        if jit:
            self.gnn = torch.jit.script(self.gnn)
        if compile:
            self.gnn = torch_geometric.compile(self.gnn)

        self.weight_scale = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(out_scale))
                for _ in range(len(layer_layout) - 1)
            ]
        )
        self.bias_scale = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(out_scale))
                for _ in range(len(layer_layout) - 1)
            ]
        )
        self.stats = stats
        self.normalize = normalize

    def forward(self, inputs):
        node_features, edge_features, _ = self.construct_graph(inputs)

        batch = to_pyg_batch(node_features, edge_features, self.edge_index)
        node_out, edge_out = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
        )
        edge_features = edge_out.reshape(edge_features.shape[0], -1, edge_out.shape[-1])
        node_features = node_out.reshape(node_features.shape[0], -1, node_out.shape[-1])
        edge_features = self.proj_edge(edge_features)
        node_features = self.proj_node(node_features)

        weights, biases = graph_to_wb(
            edge_features=edge_features,
            node_features=node_features,
            weights=inputs[0],
            biases=inputs[1],
            normalize=self.normalize,
            **self.stats,
        )

        weights = [w * s for w, s in zip(weights, self.weight_scale)]
        biases = [b * s for b, s in zip(biases, self.bias_scale)]

        return weights, biases
