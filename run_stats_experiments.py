"""
Train and evaluate the two new stats-features variants:

  B. stats_imu_only       — MLP on 54 hand-crafted IMU features
  C. stats_imu_raw_stats  — DeepConvLSTM ⊕ stats encoder (late fusion)

The raw-only control is the existing `audit_dropout_imu` checkpoint — no
retraining needed for variant A.

Side-experiment, NOT in proposal. See experiment_log.md for context.

Usage (from the research/ directory):
    python run_stats_experiments.py
    python run_stats_experiments.py --skip-train   # eval only
"""

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.loaders import get_dataloader
from src.models.stats_mlp import StatsMLP, DeepConvLSTMStats
from src.training.trainer import Trainer
from src.utils.metrics import (
    weighted_accuracy,
    macro_f1,
    per_class_f1,
    get_confusion_matrix,
)

CONFIGS = [
    Path("configs/imu_stats_only.yaml"),
    Path("configs/imu_raw_stats.yaml"),
]

MODEL_REGISTRY = {
    "stats_mlp": StatsMLP,
    "deepconvlstm_stats": DeepConvLSTMStats,
}


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg_model: dict) -> torch.nn.Module:
    kwargs = dict(cfg_model)
    model_type = kwargs.pop("type")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'.")
    return MODEL_REGISTRY[model_type](**kwargs)


def run_training(cfg: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / cfg["experiment"] / timestamp

    d = cfg["data"]
    train_loader, val_loader = get_dataloader(
        mode=d["modality"],
        split="train",
        batch_size=d["batch_size"],
        val_split=d["val_split"],
        num_workers=d["num_workers"],
        seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg["model"])
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    t = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        exp_dir=exp_dir,
        early_stop_patience=t["early_stop_patience"],
    )
    trainer.train(num_epochs=t["epochs"])
    return exp_dir


def latest_run_dir(experiment: str) -> Path:
    root = Path("experiments") / experiment
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "best_model.pt").exists())
    if not runs:
        raise FileNotFoundError(f"No completed runs found under {root}")
    return runs[-1]


def run_evaluation(cfg: dict, run_dir: Path) -> None:
    d = cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = get_dataloader(
        mode=d["modality"],
        split="test",
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg["model"]).to(device)
    ckpt = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            *inputs, y = batch
            inputs = [t.to(device) for t in inputs]
            y = y.to(device)
            logits = model(*inputs)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    num_classes = cfg["model"]["num_classes"]
    wacc = weighted_accuracy(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred)
    pcf1 = per_class_f1(y_true, y_pred, num_classes=num_classes)
    cm = get_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    print(f"  Test accuracy : {wacc:.4f}")
    print(f"  Test macro-F1 : {mf1:.4f}")
    print(f"  Test samples  : {len(y_true)}")

    (run_dir / "test_metrics.json").write_text(
        json.dumps({"weighted_accuracy": float(wacc), "macro_f1": float(mf1),
                    "n_samples": int(len(y_true)), "checkpoint": str(ckpt)}, indent=2)
    )
    with open(run_dir / "test_per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "f1"])
        for i, v in enumerate(pcf1):
            w.writerow([i, float(v)])
    np.save(run_dir / "test_confusion_matrix.npy", cm)

    canonical_dir = Path("experiments") / cfg["experiment"]
    shutil.copy(run_dir / "test_confusion_matrix.npy",
                canonical_dir / "test_confusion_matrix.npy")
    shutil.copy(run_dir / "test_per_class_f1.csv",
                canonical_dir / "test_per_class_f1.csv")
    shutil.copy(run_dir / "test_metrics.json",
                canonical_dir / "test_metrics.json")

    print(f"  Saved to {run_dir}")
    print(f"  Canonical copy at {canonical_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; evaluate the latest existing run for each config."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for config_path in CONFIGS:
        cfg = load_cfg(config_path)
        name = cfg["experiment"]
        print(f"{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")

        if args.skip_train:
            run_dir = latest_run_dir(name)
            print(f"  Skipping training — using {run_dir}")
        else:
            print("  Training...")
            run_dir = run_training(cfg)

        print("  Evaluating...")
        run_evaluation(cfg, run_dir)
        print()

    print("All done.")
    print()
    print("Comparison: audit_dropout_imu (raw-only) is the control. Look up its")
    print("test_metrics.json in experiments/audit_dropout_imu/ for the baseline.")


if __name__ == "__main__":
    main()
