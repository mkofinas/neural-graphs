import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def pad_and_flatten_kernel(kernel, max_kernel_size):
    full_padding = (
        max_kernel_size[0] - kernel.shape[2],
        max_kernel_size[1] - kernel.shape[3],
    )
    padding = (
        full_padding[0] // 2,
        full_padding[0] - full_padding[0] // 2,
        full_padding[1] // 2,
        full_padding[1] - full_padding[1] // 2,
    )
    return F.pad(kernel, padding).flatten(2, 3)


def cnn_to_graph(
    weights,
    biases,
    weights_mean=None,
    weights_std=None,
    biases_mean=None,
    biases_std=None,
):
    weights_mean = weights_mean if weights_mean is not None else [0.0] * len(weights)
    weights_std = weights_std if weights_std is not None else [1.0] * len(weights)
    biases_mean = biases_mean if biases_mean is not None else [0.0] * len(biases)
    biases_std = biases_std if biases_std is not None else [1.0] * len(biases)

    # The graph will have as many nodes as the total number of channels in the
    # CNN, plus the number of output dimensions for each linear layer
    device = weights[0].device
    num_input_nodes = weights[0].shape[0]
    num_nodes = num_input_nodes + sum(b.shape[0] for b in biases)

    edge_features = torch.zeros(
        num_nodes, num_nodes, weights[0].shape[-1], device=device
    )
    edge_feature_masks = torch.zeros(
        num_nodes, num_nodes, device=device, dtype=torch.bool
    )
    adjacency_matrix = torch.zeros(
        num_nodes, num_nodes, device=device, dtype=torch.bool
    )

    row_offset = 0
    col_offset = num_input_nodes  # no edge to input nodes
    for i, w in enumerate(weights):
        num_in, num_out = w.shape[:2]
        edge_features[
            row_offset : row_offset + num_in,
            col_offset : col_offset + num_out,
            : w.shape[-1],
        ] = (w - weights_mean[i]) / weights_std[i]
        edge_feature_masks[
            row_offset : row_offset + num_in, col_offset : col_offset + num_out
        ] = (w.shape[-1] == 1)
        adjacency_matrix[
            row_offset : row_offset + num_in, col_offset : col_offset + num_out
        ] = True
        row_offset += num_in
        col_offset += num_out

    node_features = torch.cat(
        [
            torch.zeros((num_input_nodes, 1), device=device, dtype=biases[0].dtype),
            *[(b - biases_mean[i]) / biases_std[i] for i, b in enumerate(biases)],
        ]
    )

    return node_features, edge_features, edge_feature_masks, adjacency_matrix


def cnn_to_tg_data(
    weights,
    biases,
    conv_mask,
    weights_mean=None,
    weights_std=None,
    biases_mean=None,
    biases_std=None,
    **kwargs,
):
    node_features, edge_features, edge_feature_masks, adjacency_matrix = cnn_to_graph(
        weights, biases, weights_mean, weights_std, biases_mean, biases_std
    )
    edge_index = adjacency_matrix.nonzero().t()

    num_input_nodes = weights[0].shape[0]
    cnn_sizes = [w.shape[1] for i, w in enumerate(weights) if conv_mask[i]]
    num_cnn_nodes = num_input_nodes + sum(cnn_sizes)
    send_nodes = num_input_nodes + sum(cnn_sizes[:-1])
    spatial_embed_mask = torch.zeros_like(node_features[:, 0], dtype=torch.bool)
    spatial_embed_mask[send_nodes:num_cnn_nodes] = True
    node_types = torch.cat(
        [
            torch.zeros(num_cnn_nodes, dtype=torch.long),
            torch.ones(node_features.shape[0] - num_cnn_nodes, dtype=torch.long),
        ]
    )
    if "residual_connections" in kwargs and "layer_layout" in kwargs:
        residual_edge_index = get_residuals_graph(
            kwargs["residual_connections"],
            kwargs["layer_layout"],
        )
        edge_index = torch.cat([edge_index, residual_edge_index], dim=1)
        # TODO: Do this in a more general way, now it works for square kernels
        center_pixel_index = edge_features.shape[-1] // 2
        edge_features[
            residual_edge_index[0], residual_edge_index[1], center_pixel_index
        ] = 1.0

    data = Data(
        x=node_features,
        edge_attr=edge_features[edge_index[0], edge_index[1]],
        edge_index=edge_index,
        mlp_edge_masks=edge_feature_masks[edge_index[0], edge_index[1]],
        spatial_embed_mask=spatial_embed_mask,
        node_types=node_types,
        conv_mask=conv_mask,
        **kwargs,
    )

    return data


def get_residuals_graph(residual_connections, layer_layout):
    residual_layer_index = torch.LongTensor(
        [(e, i) for i, e in enumerate(residual_connections) if e >= 0]
    )
    if residual_layer_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    residual_layer_index = residual_layer_index.T
    layout = torch.tensor(layer_layout)
    hidden_layout = layout[1:-1]
    min_residuals = hidden_layout[residual_layer_index].min(0, keepdim=True).values
    starting_indices = torch.cumsum(layout, dim=0)[residual_layer_index]

    residual_edge_index = torch.cat(
        [
            torch.stack(
                [
                    torch.arange(
                        starting_indices[0, i],
                        starting_indices[0, i] + min_residuals[0, i],
                        dtype=torch.long,
                    ),
                    torch.arange(
                        starting_indices[1, i],
                        starting_indices[1, i] + min_residuals[0, i],
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            for i in range(starting_indices.shape[1])
        ],
        dim=1,
    )
    return residual_edge_index
