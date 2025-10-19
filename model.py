from torch_geometric.nn import NNConv, global_max_pool
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeNet(nn.Module):
    """
    A simple GNN model that incorporates edge attributes via a learned transformation (NNConv).
    """
    def __init__(self, in_channels, edge_in_channels, global_in_channels,
                 hidden_channels=64, num_layers=4):
        super(EdgeNet, self).__init__()

        # We'll build a small feedforward that processes edge attributes
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels * hidden_channels)
        )

        # GNN Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(NNConv(in_channels, hidden_channels, self.edge_nn))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            # Need a new instance of edge_nn for each layer or a universal approach
            edge_nn = nn.Sequential(
                nn.Linear(edge_in_channels, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_channels * hidden_channels)
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, edge_nn))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final linear layers after pooling
        # We incorporate global features by concatenating them after the pooling step
        self.post_pool = nn.Sequential(
            nn.Linear(hidden_channels + global_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output 1 for regression
        )

    def forward(self, x, edge_index, edge_attr, batch, u):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_in_channels]
        # u: [num_graphs, global_in_channels] but used after pooling

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        # global pooling
        x = global_max_pool(x, batch)  # [num_graphs, hidden_channels]

        # concat with global features
        # u has shape [num_graphs, global_in_channels], so matching dimension
        x = torch.cat([x, u], dim=-1)

        # final FFN
        out = self.post_pool(x)
        return out.view(-1)

