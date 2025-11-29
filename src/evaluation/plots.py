"""Figures for the README: ablation bars, temporal analysis, model comparison."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("muted")


def plot_ablation(results: dict, save_path: str = "figures/ablation.png") -> None:
    """Progressive ablation: each bar adds one component."""
    display = {
        "xgboost": "XGBoost\n(features only)",
        "static_gnn": "Static GNN\n(+ graph)",
        "temporal_gnn": "Temporal GNN\n(+ time)",
        "temporal_gnn+ssl": "Temporal GNN + SSL\n(+ pre-training)",
    }
    names = [display.get(k, k) for k in results]
    aucs = [results[k]["roc_auc"] for k in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, aucs, color=COLORS[: len(names)])

    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=11,
        )

    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.5, 1.02)
    ax.set_title("Ablation: What Does Each Component Add?")
    ax.invert_yaxis()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {save_path}")


def plot_model_comparison(results: dict, save_path: str = "figures/model_comparison.png") -> None:
    """Bar chart: ROC-AUC, PR-AUC, F1 per model."""
    models = list(results.keys())
    metrics = ["roc_auc", "pr_auc", "f1"]
    labels = ["ROC-AUC", "PR-AUC", "F1"]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [results[m][metric] for m in models]
        ax.bar(x + i * width, values, width, label=label, color=COLORS[i])

    ax.set_ylabel("Score")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison on Elliptic Bitcoin")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {save_path}")


def plot_temporal_performance(
    per_timestep: dict[int, float], save_path: str = "figures/temporal_auc.png"
) -> None:
    """ROC-AUC per timestep in the test set -- shows if model degrades on newer data."""
    timesteps = sorted(per_timestep.keys())
    aucs = [per_timestep[t] for t in timesteps]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, aucs, "o-", color=COLORS[0], linewidth=2, markersize=5)
    ax.axhline(np.mean(aucs), color="gray", linestyle="--", alpha=0.5, label="mean")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Detection Performance Over Time (Test Set)")
    ax.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = json.loads(Path("logs/results_ablation.json").read_text())
    plot_ablation(results)
