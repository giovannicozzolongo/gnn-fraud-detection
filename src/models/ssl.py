"""Self-supervised pre-training via masked feature reconstruction.

Only ~2% of Elliptic transactions have fraud labels. This module
pre-trains the GNN encoder on ALL nodes (including unlabeled ones)
by masking random features and training the network to reconstruct them.
The pre-trained encoder then initializes the downstream classifier.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

logger = logging.getLogger(__name__)


class GraphMAE(nn.Module):
    """Masked autoencoder for graphs.

    Masks a fraction of node features, encodes the corrupted graph,
    and reconstructs the original features at masked positions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        in_dim: int,
        hidden_dim: int,
        mask_rate: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_rate = mask_rate

        # learnable token that replaces masked features
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        nn.init.xavier_normal_(self.mask_token)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x, edge_index, timesteps=None, batch_size=None):
        # mask seed nodes only (first batch_size nodes in NeighborLoader batch)
        n_seeds = batch_size if batch_size is not None else x.shape[0]
        mask = torch.rand(n_seeds, device=x.device) < self.mask_rate

        x_masked = x.clone()
        x_masked[:n_seeds][mask] = self.mask_token

        if timesteps is not None:
            h = self.encoder.encode(x_masked, edge_index, timesteps)
        else:
            h = self.encoder.encode(x_masked, edge_index)

        # reconstruct only masked seed nodes
        x_recon = self.decoder(h[:n_seeds][mask])
        loss = F.mse_loss(x_recon, x[:n_seeds][mask])
        return loss


def pretrain_mae(
    mae: GraphMAE,
    data,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 2048,
    num_neighbors: list[int] | None = None,
    has_timesteps: bool = False,
) -> None:
    """Mini-batch GraphMAE pre-training with NeighborLoader.

    Uses ALL nodes (not just labeled) since reconstruction
    doesn't need fraud labels.
    """
    if num_neighbors is None:
        num_neighbors = [25, 10]

    # sample from all nodes, not just labeled
    all_nodes = torch.arange(data.num_nodes)
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=all_nodes,
        shuffle=True,
    )

    mae = mae.to(device)
    mae.train()
    optimizer = torch.optim.Adam(mae.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            kwargs = {"timesteps": batch.timestep} if has_timesteps else {}
            loss = mae(
                batch.x,
                batch.edge_index,
                batch_size=batch.batch_size,
                **kwargs,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"pretrain ep {epoch:3d}: recon_loss={avg_loss:.4f}")

    logger.info("pre-training done")
