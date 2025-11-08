"""Temporal GNN: time-aware graph neural network for evolving transaction graphs.

Adds learnable temporal embeddings to node features so the GNN can
distinguish when a transaction happened, not just where it sits in the graph.
Through message passing, the model learns temporal patterns like
"recent transactions linked to old fraud clusters are suspicious."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TimeEncoder(nn.Module):
    """Learnable embedding for discrete timesteps."""

    def __init__(self, num_timesteps: int, time_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_timesteps + 1, time_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.embed(timesteps)


class TemporalSAGE(nn.Module):
    """GraphSAGE with temporal encoding concatenated to input features.

    Simple but effective: the time embedding lets the model learn
    that fraud patterns shift across the 49 timesteps.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_timesteps: int = 49,
        time_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.time_enc = TimeEncoder(num_timesteps, time_dim)
        aug_dim = in_dim + time_dim

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(aug_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.head = SAGEConv(hidden_dim, 1)
        self.dropout = dropout

    def _augment(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_enc(timesteps)
        return torch.cat([x, t_emb], dim=-1)

    def encode(self, x, edge_index, timesteps):
        """Return hidden embeddings before classification head."""
        x = self._augment(x, timesteps)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, timesteps):
        h = self.encode(x, edge_index, timesteps)
        return self.head(h, edge_index).squeeze(-1)


# TODO: try GAT variant with temporal encoding
def get_temporal_model(
    in_dim: int,
    hidden_dim: int = 128,
    num_timesteps: int = 49,
    time_dim: int = 16,
    **kwargs,
) -> TemporalSAGE:
    return TemporalSAGE(
        in_dim, hidden_dim, num_timesteps=num_timesteps, time_dim=time_dim, **kwargs
    )
