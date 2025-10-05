"""Temporal train/val/test splits for Elliptic Bitcoin dataset.

Random splits would leak future transactions into training -- temporal
splits mirror real deployment where you train on past and predict future.
"""

import logging

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def temporal_split(
    data: Data,
    train_range: tuple[int, int] = (1, 34),
    val_range: tuple[int, int] = (35, 42),
    test_range: tuple[int, int] = (43, 49),
) -> Data:
    """Split by timestep. Only labeled nodes get masks; unknown labels excluded."""
    ts = data.timestep
    labeled = data.labeled_mask

    train_mask = (ts >= train_range[0]) & (ts <= train_range[1]) & labeled
    val_mask = (ts >= val_range[0]) & (ts <= val_range[1]) & labeled
    test_mask = (ts >= test_range[0]) & (ts <= test_range[1]) & labeled

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    logger.info(
        f"temporal split: train={train_mask.sum()} (t{train_range[0]}-{train_range[1]}), "
        f"val={val_mask.sum()} (t{val_range[0]}-{val_range[1]}), "
        f"test={test_mask.sum()} (t{test_range[0]}-{test_range[1]})"
    )

    # sanity checks
    assert (train_mask & val_mask).sum() == 0, "train/val overlap"
    assert (train_mask & test_mask).sum() == 0, "train/test overlap"
    return data


def get_class_weights(data: Data) -> torch.Tensor:
    """Inverse frequency weights from training labels."""
    train_y = data.y[data.train_mask]
    n = len(train_y)
    n_classes = train_y.max().item() + 1

    weights = []
    for c in range(n_classes):
        n_c = (train_y == c).sum().item()
        weights.append(n / (n_classes * max(n_c, 1)))

    w = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"class weights: {w.tolist()}")
    return w
