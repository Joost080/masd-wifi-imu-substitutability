"""
Evaluate a trained model on the held-out MASD test split.

Usage:
    python scripts/analysis/evaluate.py configs/imu_baseline.yaml
    python scripts/analysis/evaluate.py configs/imu_baseline.yaml experiments/rq2_imu/20260504_142010

If the run path is omitted, the most recent timestamp dir under the experiment
folder is used.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.loaders import get_dataloader
from src.utils.metrics import (
    weighted_accuracy,
    macro_f1,
    per_class_f1,
    get_confusion_matrix,
)
from scripts.train.run_experiment import build_model


def latest_run_dir(experiment: str) -> Path:
    root = Path("experiments") / experiment
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")
    return runs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_dir = args.run_dir or latest_run_dir(cfg["experiment"])
    ckpt = run_dir / "best_model.pt"
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d = cfg["data"]
    test_loader = get_dataloader(
        mode=d["modality"],
        split="test",
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x1, x2, y = [t.to(device) for t in batch]
                logits = model(x1, x2)
            else:
                x, y = [t.to(device) for t in batch]
                logits = model(x)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    wacc = weighted_accuracy(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred)
    pcf1 = per_class_f1(y_true, y_pred, num_classes=cfg["model"]["num_classes"])
    cm = get_confusion_matrix(y_true, y_pred, num_classes=cfg["model"]["num_classes"])

    print(f"\nTest weighted accuracy: {wacc:.4f}")
    print(f"Test macro F1:          {mf1:.4f}")
    print(f"Test samples:           {len(y_true)}")

    out = {
        "weighted_accuracy": float(wacc),
        "macro_f1": float(mf1),
        "n_samples": int(len(y_true)),
        "checkpoint": str(ckpt),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(out, indent=2))

    with open(run_dir / "test_per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "f1"])
        for i, v in enumerate(pcf1):
            w.writerow([i, float(v)])

    np.save(run_dir / "test_confusion_matrix.npy", cm)
    print(f"\nSaved test_metrics.json, test_per_class_f1.csv, test_confusion_matrix.npy to {run_dir}")


if __name__ == "__main__":
    main()
