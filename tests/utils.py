import torch
import torch.nn.functional as F
import torch_geometric


def permute_weights_biases(weights, biases, permutations):
    perm_weights = tuple(
        (
            weights[i][:, :, permutations[i], :]
            if i == 0
            else (
                weights[i][:, permutations[i - 1], :, :][:, :, permutations[i], :]
                if i < len(weights) - 1
                else weights[i][:, permutations[i - 1], :, :]
            )
        )
        for i in range(len(weights))
    )
    perm_biases = tuple(
        biases[i][:, permutations[i], :] if i < len(biases) - 1 else biases[i]
        for i in range(len(biases))
    )
    return perm_weights, perm_biases


def wb_to_batch(weights, biases, layer_layout, pad):
    batch_size = weights[0].shape[0]
    x = torch.cat(
        [
            torch.zeros(
                (biases[0].shape[0], layer_layout[0], 1),
                dtype=biases[0].dtype,
                device=biases[0].device,
            ),
            *biases,
        ],
        dim=1,
    )
    cumsum_layout = [0] + torch.tensor(layer_layout).cumsum(dim=0).tolist()
    edge_index = torch.cat(
        [
            torch.cartesian_prod(
                torch.arange(cumsum_layout[i], cumsum_layout[i + 1]),
                torch.arange(cumsum_layout[i + 1], cumsum_layout[i + 2]),
            ).T
            for i in range(len(cumsum_layout) - 2)
        ],
        dim=1,
    )
    edge_attr = torch.cat(
        [
            weights[0].flatten(1, 2),
            weights[1].flatten(1, 2),
            F.pad(weights[-1], pad=pad).flatten(1, 2),
        ],
        dim=1,
    )
    batch = torch_geometric.data.Batch.from_data_list(
        [
            torch_geometric.data.Data(
                x=x[i],
                edge_index=edge_index,
                edge_attr=edge_attr[i],
                layer_layout=layer_layout,
                conv_mask=[1 if w.shape[-1] > 1 else 0 for w in weights],
                fmap_size=1,
            )
            for i in range(batch_size)
        ]
    )
    return batch
