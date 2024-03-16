import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.pooling import HeterogeneousAggregator
from nn.relational_transformer import RTLayer


class DynamicRelationalTransformer(nn.Module):
    def __init__(
        self,
        d_in,
        d_node,
        d_edge,
        d_attn_hid,
        d_node_hid,
        d_edge_hid,
        d_out_hid,
        d_out,
        n_layers,
        n_heads,
        graph_constructor,
        dropout=0.0,
        node_update_type="rt",
        disable_edge_updates=False,
        use_cls_token=True,
        pooling_method="cat",
        pooling_layer_idx="last",
        rev_edge_features=False,
        modulate_v=True,
        use_ln=True,
        tfixit_init=False,
        input_channels=3,
        num_classes=10,
        layer_layout=None,
    ):
        super().__init__()
        assert use_cls_token == (pooling_method == "cls_token")
        self.pooling_method = pooling_method
        self.pooling_layer_idx = pooling_layer_idx

        self.rev_edge_features = rev_edge_features
        self.out_features = d_out
        self.num_classes = num_classes
        self.construct_graph = hydra.utils.instantiate(
            graph_constructor,
            d_in=d_in,
            d_node=d_node,
            d_edge=d_edge,
            d_out=d_out,
            rev_edge_features=rev_edge_features,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(d_node))

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
                        float(dropout),
                        node_update_type=node_update_type,
                        disable_edge_updates=(
                            (disable_edge_updates or (i == n_layers - 1))
                            and pooling_method != "mean_edge"
                            and pooling_layer_idx != "all"
                        ),
                        modulate_v=modulate_v,
                        use_ln=use_ln,
                        tfixit_init=tfixit_init,
                        n_layers=n_layers,
                    )
                )
                for i in range(n_layers)
            ]
        )
        num_graph_features = d_node
        if pooling_method == "cat" and pooling_layer_idx == "last":
            num_graph_features = num_classes * d_node
        elif pooling_method == "cat" and pooling_layer_idx == "all":
            # NOTE: Only allowed with datasets of fixed architectures
            num_graph_features = sum(layer_layout) * d_node
        elif pooling_method in ("mean_edge", "max_edge"):
            num_graph_features = d_edge

        if pooling_method in (
            "mean",
            "max",
            "cat",
            "attentional_aggregation",
            "set_transformer",
            "graph_multiset_transformer",
        ):
            self.pool = HeterogeneousAggregator(
                d_node,
                d_out_hid,
                d_node,
                pooling_method,
                pooling_layer_idx,
                input_channels,
                num_classes,
            )

        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_out_hid),
            nn.ReLU(),
            nn.Linear(d_out_hid, d_out_hid),
            nn.ReLU(),
            nn.Linear(d_out_hid, d_out),
        )

    def forward(self, batch):
        node_features, edge_features, mask, node_mask = self.construct_graph(batch)

        if self.use_cls_token:
            node_features = torch.cat(
                [
                    # repeat(self.cls_token, "d -> b 1 d", b=node_features.size(0)),
                    self.cls_token.unsqueeze(0).expand(node_features.size(0), 1, -1),
                    node_features,
                ],
                dim=1,
            )
            edge_features = F.pad(edge_features, (0, 0, 1, 0, 1, 0), value=0)

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, mask)

        if self.pooling_method == "cls_token":
            graph_features = node_features[:, 0]
        elif self.pooling_method == "mean_edge" and self.pooling_layer_idx == "all":
            graph_features = edge_features.mean(dim=(1, 2))
        elif self.pooling_method == "max_edge" and self.pooling_layer_idx == "all":
            graph_features = edge_features.flatten(1, 2).max(dim=1).values
        elif self.pooling_method == "mean_edge" and self.pooling_layer_idx == "last":
            valid_layer_indices = (
                torch.arange(node_mask.shape[1], device=node_mask.device)[None, :]
                * node_mask
            )
            last_layer_indices = valid_layer_indices.topk(
                k=self.num_classes, dim=1
            ).values.fliplr()
            batch_range = torch.arange(node_mask.shape[0], device=node_mask.device)[
                :, None
            ]
            graph_features = edge_features[batch_range, last_layer_indices, :].mean(
                dim=(1, 2)
            )
        else:
            # FIXME: Node features are not masked, some contain garbage
            graph_features = self.pool(
                node_features, batch.layer_layout, node_mask=node_mask
            )

        return self.proj_out(graph_features)
