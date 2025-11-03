"""Static GNN models: GCN, GraphSAGE, GAT, and JK variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3
    ):
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
    def __init__(
        self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3
    ):
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


class GraphSAGEJK(nn.Module):
    """GraphSAGE with JumpingKnowledge: concatenates all intermediate layer outputs.

    On a sparse graph, different layers capture different scales of the
    neighborhood. Concatenating preserves both local and extended signals.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        # JK: classifier takes concatenated outputs from all layers
        self.classifier = nn.Linear(hidden_dim * num_layers, 1)
        self.dropout = dropout
        self.num_layers = num_layers

    def encode(self, x, edge_index):
        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        # concatenate all layer representations
        return torch.cat(layer_outputs, dim=-1)

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.classifier(h).squeeze(-1)


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.head = GATConv(hidden_dim * heads, 1, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, return_attention: bool = False):
        x = self.encode(x, edge_index)
        if return_attention:
            x, attn = self.head(x, edge_index, return_attention_weights=True)
            return x.squeeze(-1), attn
        return self.head(x, edge_index).squeeze(-1)


def get_model(model_type: str, in_dim: int, **kwargs) -> nn.Module:
    models = {
        "gcn": GCN,
        "graphsage": GraphSAGE,
        "graphsage_jk": GraphSAGEJK,
        "gat": GAT,
    }
    if model_type not in models:
        raise ValueError(f"unknown model: {model_type}. choose from {list(models.keys())}")
    return models[model_type](in_dim, **kwargs)
