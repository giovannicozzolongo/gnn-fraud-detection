"""Static GNN models: GCN, GraphSAGE."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.head = GCNConv(hidden_dim, 1)
        self.dropout = dropout

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        return self.head(x, edge_index).squeeze(-1)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.head = SAGEConv(hidden_dim, 1)
        self.dropout = dropout

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        return self.head(x, edge_index).squeeze(-1)


def get_model(model_type, in_dim, **kwargs):
    models = {"gcn": GCN, "graphsage": GraphSAGE}
    if model_type not in models:
        raise ValueError(f"unknown model: {model_type}")
    return models[model_type](in_dim, **kwargs)
