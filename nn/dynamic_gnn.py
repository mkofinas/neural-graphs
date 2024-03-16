import hydra
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.utils import to_dense_batch

from nn.pooling import HeterogeneousAggregator


def to_pyg_batch(node_features, edge_features, edge_index, node_mask):
    data_list = [
        torch_geometric.data.Data(
            x=node_features[i][node_mask[i]],
            edge_index=edge_index[i],
            edge_attr=edge_features[i, edge_index[i][0], edge_index[i][1]],
        )
        for i in range(node_features.shape[0])
    ]
    return torch_geometric.data.Batch.from_data_list(data_list)


class GNNForGeneralization(nn.Module):
    def __init__(
        self,
        d_hid,
        d_out,
        graph_constructor,
        gnn_backbone,
        rev_edge_features,
        pooling_method,
        pooling_layer_idx,
        compile=False,
        jit=False,
        input_channels=3,
        num_classes=10,
        layer_layout=None,
    ):
        super().__init__()
        self.pooling_method = pooling_method
        self.pooling_layer_idx = pooling_layer_idx
        self.out_features = d_out
        self.num_classes = num_classes
        self.rev_edge_features = rev_edge_features

        self.construct_graph = hydra.utils.instantiate(
            graph_constructor,
            d_node=d_hid,
            d_edge=d_hid,
            d_out=d_out,
            rev_edge_features=rev_edge_features,
            input_channels=input_channels,
            num_classes=num_classes,
        )

        num_graph_features = d_hid
        if pooling_method == "cat" and pooling_layer_idx == "last":
            num_graph_features = num_classes * d_hid
        elif pooling_method == "cat" and pooling_layer_idx == "all":
            # NOTE: Only allowed with datasets of fixed architectures
            num_graph_features = sum(layer_layout) * d_hid

        self.pool = HeterogeneousAggregator(
            d_hid,
            d_hid,
            d_hid,
            pooling_method,
            pooling_layer_idx,
            input_channels,
            num_classes,
        )

        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )

        gnn_kwargs = dict()
        gnn_kwargs["deg"] = torch.tensor(gnn_backbone["deg"], dtype=torch.long)

        self.gnn = hydra.utils.instantiate(gnn_backbone, **gnn_kwargs)
        if jit:
            self.gnn = torch.jit.script(self.gnn)
        if compile:
            self.gnn = torch_geometric.compile(self.gnn)

    def forward(self, batch):
        # self.register_buffer("edge_index", batch.edge_index, persistent=False)
        node_features, edge_features, _, node_mask = self.construct_graph(batch)

        if self.rev_edge_features:
            edge_index = [
                torch.cat(
                    [batch[i].edge_index, batch[i].edge_index.flip(dims=(0,))], dim=-1
                )
                for i in range(len(batch))
            ]
        else:
            edge_index = [batch[i].edge_index for i in range(len(batch))]

        new_batch = to_pyg_batch(node_features, edge_features, edge_index, node_mask)
        out_node, out_edge = self.gnn(
            x=new_batch.x,
            edge_index=new_batch.edge_index,
            edge_attr=new_batch.edge_attr,
        )
        node_features = to_dense_batch(out_node, new_batch.batch)[0]

        graph_features = self.pool(
            node_features, batch.layer_layout, node_mask=node_mask
        )

        return self.proj_out(graph_features)
