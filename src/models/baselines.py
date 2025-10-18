"""XGBoost and Random Forest baselines (features only, no graph)."""

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """Train XGBoost with early stopping on validation AUC."""
    if params is None:
        params = {}

    spw = params.pop("scale_pos_weight", "auto")
    if spw == "auto":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        max_depth=params.get("max_depth", 6),
        n_estimators=params.get("n_estimators", 500),
        learning_rate=params.get("learning_rate", 0.05),
        scale_pos_weight=spw,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_proba = clf.predict_proba(x_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    logger.info(f"xgboost: val ROC-AUC={val_auc:.4f} (best iter {clf.best_iteration})")
    return clf


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train RF baseline."""
    if params is None:
        params = {}

    clf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 500),
        class_weight=params.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    val_proba = clf.predict_proba(x_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    logger.info(f"random forest: val ROC-AUC={val_auc:.4f}")
    return clf
