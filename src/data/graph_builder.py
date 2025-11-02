"""Build PyG Data object from Elliptic Bitcoin CSV files."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def load_elliptic(raw_dir: str = "data/raw") -> Data:
    """Load Elliptic Bitcoin dataset and build a PyG graph.

    CSV layout:
      - features: txId, timestep (1-49), 165 numeric features
      - edgelist: txId1, txId2
      - classes: txId, class (1=licit, 2=illicit, "unknown")

    Unknown-label nodes stay in the graph (they still propagate messages)
    but are excluded from train/eval via labeled_mask.
    """
    raw = Path(raw_dir)

    # -- features --
    feat_df = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    tx_ids = feat_df[0].values
    timesteps = feat_df[1].values.astype(int)
    features = feat_df.iloc[:, 2:].values.astype(np.float32)

    n_nodes = len(tx_ids)
    feat_dim = features.shape[1]
    tx_to_idx = {int(tx): i for i, tx in enumerate(tx_ids)}

    logger.info(
        f"loaded {n_nodes} transactions ({feat_dim} features, "
        f"timesteps {timesteps.min()}-{timesteps.max()})"
    )

    # -- edges --
    edge_df = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    col_src, col_dst = edge_df.columns[0], edge_df.columns[1]
    src = edge_df[col_src].map(tx_to_idx)
    dst = edge_df[col_dst].map(tx_to_idx)

    valid = src.notna() & dst.notna()
    if (~valid).any():
        logger.warning(f"dropped {(~valid).sum()} edges with unknown node ids")
    edge_index = np.stack([src[valid].values.astype(int), dst[valid].values.astype(int)])

    logger.info(f"loaded {edge_index.shape[1]} edges")

    # -- labels --
    class_df = pd.read_csv(raw / "elliptic_txs_classes.csv")
    cls_col = class_df.columns[1]
    class_map = dict(zip(class_df.iloc[:, 0], class_df[cls_col].astype(str).str.strip()))

    labels = np.full(n_nodes, -1, dtype=np.int64)
    for i, tx in enumerate(tx_ids):
        cls = class_map.get(int(tx), "unknown")
        if cls == "1":
            labels[i] = 1  # illicit
        elif cls == "2":
            labels[i] = 0  # licit

    n_licit = (labels == 0).sum()
    n_illicit = (labels == 1).sum()
    n_unknown = (labels == -1).sum()
    imb = n_illicit / max(n_licit + n_illicit, 1)
    logger.info(
        f"labels: licit={n_licit}, illicit={n_illicit}, unknown={n_unknown} ({imb:.1%} illicit)"
    )

    # -- build Data --
    data = Data(
        x=torch.tensor(features),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
    )
    data.timestep = torch.tensor(timesteps, dtype=torch.long)
    data.labeled_mask = torch.tensor(labels >= 0)

    return data


def normalize_features(data: Data, fit_mask: torch.Tensor) -> Data:
    """Z-score normalize features. Fit on fit_mask only to avoid leakage."""
    scaler = StandardScaler()
    x = data.x.numpy()
    scaler.fit(x[fit_mask.numpy()])
    data.x = torch.tensor(scaler.transform(x), dtype=torch.float32)
    return data


def get_snapshot_indices(data: Data) -> dict[int, np.ndarray]:
    """Group node indices by timestep."""
    ts = data.timestep.numpy()
    return {int(t): np.where(ts == t)[0] for t in np.unique(ts)}
