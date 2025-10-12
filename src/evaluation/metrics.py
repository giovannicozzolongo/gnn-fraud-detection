"""Evaluation metrics for binary classification."""

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: ground truth binary labels
        y_proba: predicted probabilities for positive class
        threshold: classification threshold for F1
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
    }

    return metrics


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    tag = f"{prefix} " if prefix else ""
    parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    logger.info(f"{tag}{', '.join(parts)}")


# TODO: add per-class precision/recall
def get_confusion_matrix(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    y_pred = (y_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)
