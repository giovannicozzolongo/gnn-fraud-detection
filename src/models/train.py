"""Training pipeline for all model variants on Elliptic Bitcoin."""

import argparse
import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

from src.data.graph_builder import load_elliptic, normalize_features
from src.data.splits import temporal_split
from src.evaluation.metrics import compute_metrics, log_metrics
from src.models.baselines import train_random_forest, train_xgboost
from src.models.gnn import get_model
from src.models.losses import FocalLoss
from src.models.ssl import GraphMAE, pretrain_mae
from src.models.temporal import get_temporal_model
from src.utils.config import get_device, load_config, set_seed

logger = logging.getLogger(__name__)


# ---- low-level train/eval ----


def train_epoch(model, loader, optimizer, criterion, device, temporal: bool = False) -> float:
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if temporal:
            out = model(batch.x, batch.edge_index, batch.timestep)
        else:
            out = model(batch.x, batch.edge_index)

        # loss only on seed nodes with known labels
        out = out[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        labeled = y >= 0

        if labeled.sum() == 0:
            continue

        loss = criterion(out[labeled], y[labeled].float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, temporal: bool = False):
    model.eval()
    all_proba, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        if temporal:
            out = model(batch.x, batch.edge_index, batch.timestep)
        else:
            out = model(batch.x, batch.edge_index)

        out = out[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        labeled = y >= 0

        if labeled.sum() == 0:
            continue

        proba = torch.sigmoid(out[labeled]).cpu().numpy()
        all_proba.append(proba)
        all_labels.append(y[labeled].cpu().numpy())

    all_proba = np.concatenate(all_proba)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_proba)
    return auc, all_proba, all_labels


# ---- GNN training ----


def _make_loaders(data, cfg):
    num_neighbors = cfg["training"]["num_neighbors"]
    bs = cfg["training"]["batch_size"]

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=bs,
        input_nodes=data.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=bs,
        input_nodes=data.val_mask,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=bs,
        input_nodes=data.test_mask,
    )
    return train_loader, val_loader, test_loader


def train_gnn(
    model_type: str = "graphsage",
    config_path: str = "configs/experiment_config.yaml",
    temporal: bool = False,
    ssl_pretrain: bool = False,
) -> dict:
    """Full training pipeline for a GNN variant.

    Args:
        model_type: gcn, graphsage, or gat
        temporal: if True, use time-encoded GNN
        ssl_pretrain: if True, pre-train encoder with GraphMAE first
    """
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])
    device = get_device()

    tag = model_type
    if temporal:
        tag += "+temporal"
    if ssl_pretrain:
        tag += "+ssl"
    logger.info(f"training {tag} on {device}")

    # data
    data = load_elliptic(cfg["data"]["raw_dir"])
    split_cfg = cfg["data"]["temporal_split"]
    data = temporal_split(
        data,
        train_range=tuple(split_cfg["train"]),
        val_range=tuple(split_cfg["val"]),
        test_range=tuple(split_cfg["test"]),
    )
    data = normalize_features(data, data.train_mask)

    # model
    gnn_cfg = cfg["model"]["gnn"]
    in_dim = data.x.shape[1]

    if temporal:
        t_cfg = cfg["model"]["temporal"]
        model = get_temporal_model(
            in_dim,
            hidden_dim=gnn_cfg["hidden_dim"],
            num_timesteps=cfg["data"]["num_timesteps"],
            time_dim=t_cfg["time_dim"],
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
        )
    else:
        model_kwargs = {
            "hidden_dim": gnn_cfg["hidden_dim"],
            "num_layers": gnn_cfg["num_layers"],
            "dropout": gnn_cfg["dropout"],
        }
        if model_type == "gat":
            model_kwargs["heads"] = gnn_cfg.get("heads", 4)
        model = get_model(model_type, in_dim, **model_kwargs)

    # optional ssl pre-training
    if ssl_pretrain:
        ssl_cfg = cfg["model"]["ssl"]
        mae = GraphMAE(model, in_dim, gnn_cfg["hidden_dim"], mask_rate=ssl_cfg["mask_rate"])
        pretrain_mae(
            mae,
            data,
            device,
            epochs=ssl_cfg["pretrain_epochs"],
            lr=ssl_cfg["pretrain_lr"],
            batch_size=cfg["training"]["batch_size"],
            num_neighbors=cfg["training"]["num_neighbors"],
            has_timesteps=temporal,
        )
        # encoder weights are now pre-trained in-place

    model = model.to(device)
    logger.info(f"params: {sum(p.numel() for p in model.parameters()):,}")

    # loss -- focal loss handles extreme imbalance better than weighted BCE
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    train_loader, val_loader, test_loader = _make_loaders(data, cfg)

    # training loop
    best_val_auc = 0
    patience_counter = 0
    patience = cfg["training"]["patience"]
    best_state = None

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, temporal=temporal)
        val_auc, _, _ = evaluate(model, val_loader, device, temporal=temporal)
        scheduler.step(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(f"ep {epoch:3d}: loss={loss:.4f} auc={val_auc:.4f} lr={lr_now:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model = model.to(device)
    _, test_proba, test_labels = evaluate(model, test_loader, device, temporal=temporal)
    test_metrics = compute_metrics(test_labels, test_proba)
    log_metrics(test_metrics, f"{tag} test")

    return {"model": model, "best_val_auc": best_val_auc, "test_metrics": test_metrics}


def train_baselines(config_path: str = "configs/experiment_config.yaml") -> dict:
    """Train XGBoost and RF on node features (no graph structure)."""
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    data = load_elliptic(cfg["data"]["raw_dir"])
    split_cfg = cfg["data"]["temporal_split"]
    data = temporal_split(
        data,
        train_range=tuple(split_cfg["train"]),
        val_range=tuple(split_cfg["val"]),
        test_range=tuple(split_cfg["test"]),
    )
    data = normalize_features(data, data.train_mask)

    x = data.x.numpy()
    y = data.y.numpy()

    x_train, y_train = x[data.train_mask], y[data.train_mask]
    x_val, y_val = x[data.val_mask], y[data.val_mask]
    x_test, y_test = x[data.test_mask], y[data.test_mask]

    results = {}

    # xgboost
    xgb_params = dict(cfg["model"]["xgboost"])
    xgb_clf = train_xgboost(x_train, y_train, x_val, y_val, xgb_params)
    xgb_proba = xgb_clf.predict_proba(x_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_proba)
    log_metrics(xgb_metrics, "xgboost test")
    results["xgboost"] = xgb_metrics

    # random forest
    rf_params = dict(cfg["model"]["random_forest"])
    rf_clf = train_random_forest(x_train, y_train, x_val, y_val, rf_params)
    rf_proba = rf_clf.predict_proba(x_test)[:, 1]
    rf_metrics = compute_metrics(y_test, rf_proba)
    log_metrics(rf_metrics, "rf test")
    results["rf"] = rf_metrics

    return results


# ---- main ----


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--model", default=None, help="gcn/graphsage/gat (default: all)")
    parser.add_argument("--temporal", action="store_true", help="use time-encoded GNN")
    parser.add_argument("--ssl", action="store_true", help="pre-train with GraphMAE")
    parser.add_argument("--baselines", action="store_true", help="train XGBoost + RF only")
    args = parser.parse_args()

    if args.baselines:
        results = train_baselines(args.config)
    else:
        cfg = load_config(args.config)
        models_to_train = [args.model] if args.model else [cfg["model"]["gnn"]["type"]]
        results = {}
        for m in models_to_train:
            logger.info("=" * 50)
            res = train_gnn(
                model_type=m,
                config_path=args.config,
                temporal=args.temporal,
                ssl_pretrain=args.ssl,
            )
            tag = m
            if args.temporal:
                tag += "+temporal"
            if args.ssl:
                tag += "+ssl"
            results[tag] = res["test_metrics"]

    logger.info("=" * 50)
    logger.info("final results:")
    for name, metrics in results.items():
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        logger.info(f"  {name:30s}: {', '.join(parts)}")
