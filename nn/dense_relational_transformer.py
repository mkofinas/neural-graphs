import hydra
import torch
import torch.nn as nn

from nn.relational_transformer import RTLayer


def graphs_to_batch(
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

    row_offset = 0
    col_offset = weights[0].shape[1]  # no edge to input nodes
    for i, w in enumerate(weights):
        _, num_in, num_out, _ = w.shape
        w_mean = weights_mean[i] if normalize and weights_mean is not None else 0
        w_std = weights_std[i] if normalize and weights_std is not None else 1
        new_weights.append(
            edge_features[
                :, row_offset : row_offset + num_in, col_offset : col_offset + num_out
            ]
            * w_std
            + w_mean
        )
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[1]  # no bias in input nodes
    for i, b in enumerate(biases):
        _, num_out, _ = b.shape
        b_mean = biases_mean[i] if normalize and biases_mean is not None else 0
        b_std = biases_std[i] if normalize and biases_std is not None else 1
        new_biases.append(
            node_features[:, row_offset : row_offset + num_out] * b_std + b_mean
        )
        row_offset += num_out

    return new_weights, new_biases


class RelationalTransformerParams(nn.Module):
    def __init__(
        self,
        d_node,
        d_edge,
        d_attn_hid,
        d_node_hid,
        d_edge_hid,
        d_out_hid,
        d_out,
        n_layers,
        n_heads,
        layer_layout,
        graph_constructor,
        dropout=0.0,
        node_update_type="rt",
        disable_edge_updates=False,
        rev_edge_features=False,
        modulate_v=True,
        use_ln=True,
        tfixit_init=False,
        stats=None,
        normalize=False,
        out_scale=0.01,
    ):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
        self.construct_graph = hydra.utils.instantiate(
            graph_constructor,
            d_node=d_node,
            d_edge=d_edge,
            layer_layout=layer_layout,
            rev_edge_features=rev_edge_features,
            stats=stats,
        )

        self.layers = nn.ModuleList(
            [
                torch.jit.script(
                    RTLayer(
                        d_node,
                        d_edge,
                        d_attn_hid,
                        d_node_hid,
                        d_edge_hid,
                        n_heads,
                        dropout,
                        node_update_type=node_update_type,
                        disable_edge_updates=disable_edge_updates,
                        modulate_v=modulate_v,
                        use_ln=use_ln,
                        tfixit_init=tfixit_init,
                        n_layers=n_layers,
                    )
                )
                for _ in range(n_layers)
            ]
        )

        self.proj_edge = nn.Sequential(
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, d_out),
        )
        self.proj_node = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_out),
        )

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
        node_features, edge_features, mask = self.construct_graph(inputs)

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, mask)

        node_features = self.proj_node(node_features)
        edge_features = self.proj_edge(edge_features)

        weights, biases = graphs_to_batch(
            edge_features,
            node_features,
            *inputs,
            normalize=self.normalize,
            **self.stats,
        )
        weights = [w * s for w, s in zip(weights, self.weight_scale)]
        biases = [b * s for b, s in zip(biases, self.bias_scale)]
        return weights, biases
