"""Ablation study: isolate the contribution of each component.

Progressive ablation:
  1. XGBoost (features only, no graph)
  2. GraphSAGE 3L (graph structure)
  3. GraphSAGE JK 3L (JumpingKnowledge)
  4. GraphSAGE JK 3L + focal loss (best config)
"""

import logging

import torch

from src.data.graph_builder import load_elliptic, normalize_features
from src.data.splits import get_class_weights, temporal_split
from src.evaluation.metrics import compute_metrics, log_metrics
from src.models.baselines import train_random_forest, train_xgboost
from src.models.gnn import get_model
from src.models.losses import FocalLoss
from src.models.train import _make_loaders, evaluate, train_epoch
from src.utils.config import get_device, load_config, set_seed

logger = logging.getLogger(__name__)


def _train_and_eval(model, data, cfg, device, criterion, tag="") -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    train_loader, val_loader, test_loader = _make_loaders(data, cfg)

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, _, _ = evaluate(model, val_loader, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                break

    model.load_state_dict(best_state)
    model = model.to(device)
    _, test_proba, test_labels = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_labels, test_proba)
    log_metrics(metrics, tag)
    return metrics


def run_ablation(config_path: str = "configs/experiment_config.yaml") -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])
    device = get_device()

    data = load_elliptic(cfg["data"]["raw_dir"])
    split_cfg = cfg["data"]["temporal_split"]
    data = temporal_split(
        data,
        train_range=tuple(split_cfg["train"]),
        val_range=tuple(split_cfg["val"]),
        test_range=tuple(split_cfg["test"]),
    )
    data = normalize_features(data, data.train_mask)

    gnn_cfg = cfg["model"]["gnn"]
    in_dim = data.x.shape[1]
    results = {}

    weights = get_class_weights(data)
    pos_weight = torch.tensor([weights[1] / weights[0]], device=device)

    # weighted BCE for baselines and simple GNN
    import torch.nn as nn

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal = FocalLoss(alpha=0.75, gamma=2.0)

    # 1. xgboost (features only)
    logger.info("=" * 40 + " xgboost")
    x, y = data.x.numpy(), data.y.numpy()
    xgb = train_xgboost(
        x[data.train_mask],
        y[data.train_mask],
        x[data.val_mask],
        y[data.val_mask],
        dict(cfg["model"]["xgboost"]),
    )
    xgb_proba = xgb.predict_proba(x[data.test_mask])[:, 1]
    results["xgboost"] = compute_metrics(y[data.test_mask], xgb_proba)
    log_metrics(results["xgboost"], "xgboost")

    # 2. random forest
    logger.info("=" * 40 + " random forest")
    rf = train_random_forest(
        x[data.train_mask],
        y[data.train_mask],
        x[data.val_mask],
        y[data.val_mask],
        dict(cfg["model"]["random_forest"]),
    )
    rf_proba = rf.predict_proba(x[data.test_mask])[:, 1]
    results["random_forest"] = compute_metrics(y[data.test_mask], rf_proba)
    log_metrics(results["random_forest"], "random_forest")

    # 3. GraphSAGE 3L
    logger.info("=" * 40 + " graphsage 3L")
    model = get_model(
        "graphsage",
        in_dim,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
    ).to(device)
    results["graphsage_3L"] = _train_and_eval(model, data, cfg, device, bce, "graphsage_3L")

    # 4. GraphSAGE JK 3L (+ JumpingKnowledge)
    logger.info("=" * 40 + " graphsage JK 3L")
    model = get_model(
        "graphsage_jk",
        in_dim,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
    ).to(device)
    results["graphsage_jk"] = _train_and_eval(model, data, cfg, device, bce, "graphsage_jk")

    # 5. GraphSAGE JK 3L + focal loss (best config)
    logger.info("=" * 40 + " graphsage JK 3L + focal")
    model = get_model(
        "graphsage_jk",
        in_dim,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
    ).to(device)
    results["graphsage_jk_focal"] = _train_and_eval(
        model, data, cfg, device, focal, "graphsage_jk_focal"
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_ablation()

    logger.info("=" * 50)
    logger.info("ablation summary:")
    for name, metrics in results.items():
        logger.info(
            f"  {name:25s}: ROC-AUC={metrics['roc_auc']:.4f} "
            f"PR-AUC={metrics['pr_auc']:.4f} F1={metrics['f1']:.4f}"
        )
