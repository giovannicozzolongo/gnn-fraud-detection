"""Tests for the Elliptic data pipeline."""

import pytest
import torch

from src.data.graph_builder import load_elliptic, normalize_features
from src.data.splits import get_class_weights, temporal_split


@pytest.fixture(scope="module")
def data():
    return load_elliptic()


@pytest.fixture(scope="module")
def split_data(data):
    return temporal_split(data)


def test_node_count(data):
    # elliptic has 203769 transactions
    assert data.num_nodes == 203769


def test_feature_dim(data):
    # 165 features (167 cols minus txId and timestep)
    assert data.x.shape[1] == 165


def test_labels_valid(data):
    unique = set(torch.unique(data.y).tolist())
    assert unique == {-1, 0, 1}


def test_timestep_range(data):
    assert data.timestep.min() == 1
    assert data.timestep.max() == 49


def test_temporal_split_no_overlap(split_data):
    tr = split_data.train_mask
    va = split_data.val_mask
    te = split_data.test_mask
    assert (tr & va).sum() == 0
    assert (tr & te).sum() == 0


def test_temporal_split_no_leakage(split_data):
    """Train nodes must come from earlier timesteps than test nodes."""
    train_ts = split_data.timestep[split_data.train_mask]
    test_ts = split_data.timestep[split_data.test_mask]
    assert train_ts.max() < test_ts.min()


def test_class_weights(split_data):
    w = get_class_weights(split_data)
    assert w.shape == (2,)
    # illicit weight should be higher (rare class)
    assert w[1] > w[0]


def test_normalize_preserves_shape(split_data):
    orig_shape = split_data.x.shape
    normed = normalize_features(split_data, split_data.train_mask)
    assert normed.x.shape == orig_shape
