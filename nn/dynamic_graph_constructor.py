import torch
import torch.nn as nn
import torch.nn.functional as F
from rff.layers import GaussianEncoding
from torch_geometric.utils import to_dense_adj, to_dense_batch

from nn.activation_embedding import ActivationEmbedding


def sparse_to_dense(
    batch, weights_mean=None, weights_std=None, biases_mean=None, biases_std=None
):
    dense_x, dense_node_mask = to_dense_batch(batch.x, batch.batch)
    adj = to_dense_adj(batch.edge_index, batch.batch)
    dense_edge_attr = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
    dense_node_types = (
        to_dense_batch(batch.node_types, batch.batch)[0]
        if hasattr(batch, "node_types")
        else None
    )
    # `dense_edge_feature_masks` is needed in case of linear_as_conv=False
    dense_edge_feature_masks = (
        to_dense_adj(batch.edge_index, batch.batch, batch.mlp_edge_masks)
        if hasattr(batch, "mlp_edge_masks")
        else None
    )

    if weights_mean is not None and weights_std is not None:
        edge_layout = [
            [
                batch[i].layer_layout[j] * batch[i].layer_layout[j + 1]
                for j in range(len(batch[i].layer_layout) - 1)
            ]
            for i in range(len(batch))
        ]
        edge_layer_indices = [
            torch.arange(len(el)).repeat_interleave(torch.tensor(el))
            for el in edge_layout
        ]
        dense_edge_layer_indices = to_dense_adj(
            batch.edge_index,
            batch.batch,
            torch.cat(edge_layer_indices).to(batch.edge_index.device),
        )
        weights_mean = torch.Tensor([0.0] + weights_mean).to(dense_edge_attr.device)
        weights_std = torch.Tensor([1.0] + weights_std).to(dense_edge_attr.device)
        weights_mean = weights_mean[dense_edge_layer_indices].unsqueeze(-1)
        weights_std = weights_std[dense_edge_layer_indices].unsqueeze(-1)
        dense_edge_attr = (dense_edge_attr - weights_mean) / weights_std

    if biases_mean is not None and biases_std is not None:
        layer_indices = [
            torch.arange(len(batch[i].layer_layout)).repeat_interleave(
                torch.tensor(batch[i].layer_layout)
            )
            for i in range(len(batch))
        ]
        max_num_nodes = max([len(li) for li in layer_indices])
        dense_layer_indices = torch.stack(
            [F.pad(li, (0, max_num_nodes - len(li))) for li in layer_indices],
            dim=0,
        )
        biases_mean = torch.Tensor([0.0] + biases_mean).to(dense_x.device)
        biases_std = torch.Tensor([1.0] + biases_std).to(dense_x.device)
        biases_mean = biases_mean[dense_layer_indices].unsqueeze(-1)
        biases_std = biases_std[dense_layer_indices].unsqueeze(-1)
        dense_x = (dense_x - biases_mean) / biases_std

    return (
        dense_x,
        adj,
        dense_edge_attr,
        dense_node_types,
        dense_node_mask,
        dense_edge_feature_masks,
    )


class GraphConstructor(nn.Module):
    def __init__(
        self,
        d_in,
        d_edge_in,
        d_node,
        d_edge,
        d_out,
        max_num_hidden_layers,
        rev_edge_features=False,
        zero_out_bias=False,
        zero_out_weights=False,
        inp_factor=1,
        input_layers=1,
        sin_emb=False,
        sin_emb_dim=128,
        use_pos_embed=True,
        num_probe_features=0,
        inr_model=None,
        stats=None,
        input_channels=3,
        linear_as_conv=True,
        flattening_method="repeat_nodes",
        max_spatial_resolution=64,
        num_classes=10,
        use_act_embed=True,
    ):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.zero_out_bias = zero_out_bias
        self.zero_out_weights = zero_out_weights
        self.use_pos_embed = use_pos_embed
        self.stats = stats if stats is not None else {}

        self.num_inputs = input_channels
        self.d_out = d_out
        self.num_classes = num_classes
        self._d_edge = d_edge
        self.max_num_hidden_layers = max_num_hidden_layers
        self.flattening_method = flattening_method
        self.linear_as_conv = linear_as_conv
        self.use_act_embed = use_act_embed

        self.pos_embed = nn.Parameter(
            torch.randn(max_num_hidden_layers + input_channels + num_classes, d_node)
        )
        if flattening_method == "repeat_nodes":
            self.spatial_embed = nn.Parameter(
                torch.randn(max_spatial_resolution, d_node)
            )

        proj_weight = []
        proj_bias = []
        if sin_emb:
            proj_weight.append(
                GaussianEncoding(
                    sigma=inp_factor,
                    input_size=d_edge_in + (2 * d_edge_in if rev_edge_features else 0),
                    encoded_size=sin_emb_dim,
                )
            )
            proj_weight.append(nn.Linear(2 * sin_emb_dim, d_edge))
            proj_bias.append(
                GaussianEncoding(
                    sigma=inp_factor,
                    input_size=d_in,
                    encoded_size=sin_emb_dim,
                )
            )
            proj_bias.append(nn.Linear(2 * sin_emb_dim, d_node))
        else:
            proj_weight.append(
                nn.Linear(
                    d_edge_in + (2 * d_edge_in if rev_edge_features else 0), d_edge
                )
            )
            proj_bias.append(nn.Linear(d_in, d_node))

        for i in range(input_layers - 1):
            proj_weight.append(nn.SiLU())
            proj_weight.append(nn.Linear(d_edge, d_edge))
            proj_bias.append(nn.SiLU())
            proj_bias.append(nn.Linear(d_node, d_node))

        self.proj_weight = nn.Sequential(*proj_weight)
        self.proj_bias = nn.Sequential(*proj_bias)

        self.proj_node_in = nn.Linear(d_node, d_node)
        self.proj_edge_in = nn.Linear(d_edge, d_edge)

        if not linear_as_conv:
            # Use different projections for convolutional layers and linear layers
            proj_mlp_weight = []
            if sin_emb:
                proj_mlp_weight.append(
                    GaussianEncoding(
                        sigma=inp_factor,
                        input_size=d_in + (2 * d_in if rev_edge_features else 0),
                        encoded_size=sin_emb_dim,
                    )
                )
                proj_mlp_weight.append(nn.Linear(2 * sin_emb_dim, d_edge))
            else:
                proj_mlp_weight.append(
                    nn.Linear(d_in + (2 * d_in if rev_edge_features else 0), d_edge)
                )

            for i in range(input_layers - 1):
                proj_mlp_weight.append(nn.SiLU())
                proj_mlp_weight.append(nn.Linear(d_edge, d_edge))

            self.proj_mlp_weight = nn.Sequential(*proj_mlp_weight)
            self.proj_mlp_edge_in = nn.Linear(d_edge, d_edge)

        if use_act_embed:
            self.act_emb = ActivationEmbedding(d_node)

        if num_probe_features > 0:
            self.gpf = None
            # self.gpf = GraphProbeFeatures(
            #     d_in=layer_layout[0],
            #     num_inputs=num_probe_features,
            #     inr_model=inr_model,
            #     input_init=None,
            #     proj_dim=d_node,
            # )
        else:
            self.gpf = None

    def get_pos_embed_layout(self, layer_layouts):
        pos_embed_layout = [
            [1] * layout[0] + layout[1:-1] + [1] * layout[-1]
            for layout in layer_layouts
        ]
        layer_indices = [
            list(range(len(pel) - self.num_classes))
            + list(
                range(
                    self.num_inputs + self.max_num_hidden_layers,
                    self.num_inputs + self.max_num_hidden_layers + self.num_classes,
                )
            )
            for pel in pos_embed_layout
        ]

        max_pe = max([sum(pel) for pel in pos_embed_layout])
        pos_embed = torch.stack(
            [
                F.pad(
                    torch.cat(
                        [self.pos_embed[i].expand(n, -1) for i, n in zip(li, pel)],
                        dim=0,
                    ),
                    (0, 0, 0, max_pe - sum(pel)),
                )
                for li, pel in zip(layer_indices, pos_embed_layout)
            ],
            dim=0,
        )
        return pos_embed

    def get_spatial_embed(self, batch):
        final_conv_layer = [
            max([i for i, w in enumerate(batch[i].conv_mask) if w])
            for i in range(len(batch))
        ]

        spatial_embed = torch.cat(
            [
                self.spatial_embed[: batch.fmap_size[i]].repeat_interleave(
                    batch[i].layer_layout[final_conv_layer[i] + 1], dim=0
                )
                for i in range(len(batch))
            ],
            dim=0,
        )
        full_spatial_embed = torch.zeros(
            (batch.x.shape[0], spatial_embed.shape[-1]),
            dtype=spatial_embed.dtype,
            device=spatial_embed.device,
        )
        full_spatial_embed[batch.spatial_embed_mask] = spatial_embed
        dense_spatial_embed = to_dense_batch(full_spatial_embed, batch.batch)[0]
        return dense_spatial_embed

    def update_layer_layout(self, batch):
        if self.flattening_method is None:
            return

        final_conv_layer = [
            max([i for i, w in enumerate(batch[i].conv_mask) if w])
            for i in range(len(batch))
        ]

        if self.flattening_method == "repeat_nodes":
            for i in range(len(batch)):
                batch[i].layer_layout[final_conv_layer[i] + 1] *= batch[
                    i
                ].fmap_size.item()
        elif self.flattening_method == "extra_layer":
            for i in range(len(batch)):
                batch[i].layer_layout.insert(
                    final_conv_layer[i] + 2,
                    batch[i].fmap_size.item()
                    * batch[i].layer_layout[final_conv_layer[i] + 1],
                )
        else:
            raise NotImplementedError

    def get_act_embed(self, batch):
        activation_embeddings = torch.cat(
            [
                self.act_emb(
                    batch[i].activations, batch[i].layer_layout, batch.x.device
                )
                for i in range(len(batch))
            ],
            dim=0,
        )
        activation_embeddings = to_dense_batch(activation_embeddings, batch.batch)[0]
        return activation_embeddings

    def forward(self, batch):
        if self.flattening_method == "repeat_nodes":
            dense_spatial_embed = self.get_spatial_embed(batch)
        elif self.flattening_method == "extra_layer":
            # TODO: add edges
            pass
        elif self.flattening_method is None:
            pass
        else:
            raise NotImplementedError

        self.update_layer_layout(batch)

        if hasattr(batch, "activations") and self.use_act_embed:
            activation_embeddings = self.get_act_embed(batch)

        (
            node_features,
            _,
            edge_features,
            _,
            node_mask,
            edge_feature_masks,
        ) = sparse_to_dense(batch, **self.stats)
        mask = edge_features.sum(dim=-1, keepdim=True) != 0

        linear_indices = (
            edge_features.shape[-1] * torch.arange(3) if self.rev_edge_features else [0]
        )
        if self.rev_edge_features:
            rev_edge_features = edge_features.transpose(-2, -3)
            edge_features = torch.cat(
                [edge_features, rev_edge_features, edge_features + rev_edge_features],
                dim=-1,
            )
            mask = mask | mask.transpose(-2, -3)

        node_features = self.proj_bias(node_features)
        if self.linear_as_conv:
            edge_features = self.proj_weight(edge_features)
        else:
            conv_edge_features = self.proj_weight(edge_features[~edge_feature_masks])
            mlp_edge_features = self.proj_mlp_weight(
                edge_features[edge_feature_masks][..., linear_indices]
            )
            edge_features = torch.zeros(
                (*edge_features.shape[:-1], self._d_edge),
                dtype=edge_features.dtype,
                device=edge_features.device,
            )
            edge_features[~edge_feature_masks] = conv_edge_features
            edge_features[edge_feature_masks] = mlp_edge_features

        if self.zero_out_weights:
            edge_features = torch.zeros_like(edge_features)
        if self.zero_out_bias:
            # only zero out bias, not gpf
            node_features = torch.zeros_like(node_features)

        if self.gpf is not None:
            probe_features = self.gpf(*batch)
            node_features = node_features + probe_features

        node_features = self.proj_node_in(node_features)
        edge_features = self.proj_edge_in(edge_features)

        if self.flattening_method == "repeat_nodes":
            node_features = node_features + dense_spatial_embed

        if self.use_pos_embed:
            pos_embed = self.get_pos_embed_layout(batch.layer_layout)
            node_features = node_features + pos_embed

        # Add activation embeddings
        if hasattr(batch, "activations") and self.use_act_embed:
            node_features = node_features + activation_embeddings

        return node_features, edge_features, mask, node_mask
