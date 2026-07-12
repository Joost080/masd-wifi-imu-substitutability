import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path = None,
    title: str = "Confusion Matrix",
):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(log_csv: Path, save_path: Path = None):
    import pandas as pd
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["val_acc"])
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_f1(
    f1_scores: np.ndarray,
    labels: list = None,
    save_path: Path = None,
    title: str = "Per-class F1",
):
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(f1_scores))
    ax.bar(x, f1_scores)
    ax.set_xticks(x)
    ax.set_xticklabels(labels or [str(i) for i in x], rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
