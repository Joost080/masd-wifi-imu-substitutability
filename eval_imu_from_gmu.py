"""
Co-training causal test (Track 1A).

The noise-ablation in ablation_zero_wifi.py shows that the trained GMU is
functionally invariant to WiFi at inference (gate fully closed). What it
does NOT show is whether the IMU pathway inside GMU is the same encoder
that a standalone IMU model would have learned, or whether co-training
with WiFi changed it. This script tests that directly.

Method (no retraining):
  1. Load a trained GMU checkpoint.
  2. Copy its IMU-side weights (imu_encoder + lstm + dropout + head) into a
     plain DeepConvLSTM. The architectures are weight-compatible: the
     GMU's per-modality encoder is built by LateFusionModel._build_conv_encoder,
     which produces the same nn.Sequential structure as DeepConvLSTM.conv_blocks;
     the GMU lstm/head/dropout are identical to DeepConvLSTM's.
  3. Evaluate the assembled IMU-only model on the IMU test split.

Compared with the IMU-solo audit baseline (`audit_dropout_imu`):
  - direct-from-GMU acc ≈ IMU-solo acc → IMU pathway is independently as good
    as the standalone IMU model; WiFi encoder + gate are dead weight at infer.
  - direct-from-GMU acc < IMU-solo acc → WiFi co-training actively harmed the
    IMU pathway (clean negative result; strengthens "GMU is no better than IMU"
    multi-seed finding).
  - direct-from-GMU acc > IMU-solo acc → co-training shaped the IMU encoder
    better; revives the +3.6pp single-seed claim at the encoder level.

Multi-seed: runs over seed_{0..N-1} of the GMU experiment and writes
aggregate summary + per-class F1 in the same layout as run_multiseed.py so
analyze_stats_perclass.py and other downstream tools can read it.

Usage:
    python eval_imu_from_gmu.py configs/audit/gmu_fusion.yaml --num-seeds 5
    python eval_imu_from_gmu.py configs/audit/easy_gmu_fusion.yaml --num-seeds 5

Outputs (under experiments/<gmu_exp>/imu_from_gmu/):
    seed_<n>/test_metrics.json, test_per_class_f1.csv, test_confusion_matrix.npy
    cotraining_summary.json, cotraining_summary.csv, cotraining_per_class_f1.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.loaders import get_dataloader
from src.models.deepconvlstm import DeepConvLSTM
from src.utils.metrics import (
    weighted_accuracy,
    macro_f1,
    per_class_f1,
    get_confusion_matrix,
)


def imu_from_gmu_state_dict(gmu_state: dict) -> dict:
    """Map GMU state_dict keys onto a DeepConvLSTM state_dict.

    GMU has: wifi_encoder.*, imu_encoder.*, gmu.*, lstm.*, dropout.*, head.*
    DeepConvLSTM has: conv_blocks.*, lstm.*, dropout.*, head.*

    The imu_encoder and conv_blocks are both built as nn.Sequential of 4
    inner Sequentials (Conv1d-ReLU-MaxPool1d-Dropout), with identical
    in_channels=9 / conv_channels=64. Their parameter shapes match 1:1.
    """
    mapping = {}
    for k, v in gmu_state.items():
        if k.startswith("imu_encoder."):
            mapping[k.replace("imu_encoder.", "conv_blocks.", 1)] = v
        elif (k.startswith("lstm.")
              or k.startswith("head.")
              or k.startswith("dropout.")):
            mapping[k] = v
    return mapping


def build_imu_model_from_gmu_cfg(gmu_cfg: dict) -> DeepConvLSTM:
    m = gmu_cfg["model"]
    return DeepConvLSTM(
        in_channels=9,
        num_classes=m["num_classes"],
        conv_channels=m.get("conv_channels", 64),
        conv_kernel=m.get("conv_kernel", 5),
        lstm_hidden=m.get("lstm_hidden", 128),
        conv_dropout=m.get("conv_dropout", 0.5),
        lstm_dropout=m.get("lstm_dropout", 0.5),
        head_dropout=m.get("head_dropout", 0.5),
    )


def eval_one_seed(gmu_exp: str, seed: int, gmu_cfg: dict, device: str) -> dict:
    ckpt = Path("experiments") / gmu_exp / f"seed_{seed}" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Train the GMU multi-seed first."
        )
    gmu_state = torch.load(ckpt, map_location=device)
    imu_state = imu_from_gmu_state_dict(gmu_state)

    model = build_imu_model_from_gmu_cfg(gmu_cfg).to(device)
    missing, unexpected = model.load_state_dict(imu_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch on seed {seed}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    d = gmu_cfg["data"]
    test_loader = get_dataloader(
        mode="imu",
        split="test",
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = [t.to(device) for t in batch]
            logits = model(x)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return {
        "seed": seed,
        "weighted_accuracy": float(weighted_accuracy(y_true, y_pred)),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "_y_true": y_true,
        "_y_pred": y_pred,
    }


def save_per_seed(seed_dir: Path, result: dict, n_classes: int):
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if not k.startswith("_")}
    (seed_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    pcf1 = per_class_f1(result["_y_true"], result["_y_pred"], num_classes=n_classes)
    with open(seed_dir / "test_per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "f1"])
        for i, v in enumerate(pcf1):
            w.writerow([i, float(v)])

    cm = get_confusion_matrix(result["_y_true"], result["_y_pred"], num_classes=n_classes)
    np.save(seed_dir / "test_confusion_matrix.npy", cm)


def aggregate(out_root: Path, results: list, n_classes: int) -> dict:
    accs = np.array([r["weighted_accuracy"] for r in results])
    f1s = np.array([r["macro_f1"] for r in results])
    ddof = 1 if len(results) > 1 else 0
    summary = {
        "num_seeds": len(results),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=ddof)),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std(ddof=ddof)),
        "per_seed": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
    }
    (out_root / "cotraining_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_root / "cotraining_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "weighted_accuracy", "macro_f1", "n_samples"])
        for r in results:
            w.writerow([r["seed"], r["weighted_accuracy"], r["macro_f1"], r["n_samples"]])

    # Aggregate per-class F1 across seeds (mean ± std)
    pcf1_per_seed = []
    for r in results:
        pcf1 = per_class_f1(r["_y_true"], r["_y_pred"], num_classes=n_classes)
        pcf1_per_seed.append(pcf1)
    arr = np.stack(pcf1_per_seed)
    with open(out_root / "cotraining_per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "f1_mean", "f1_std", "n_seeds"])
        for c in range(arr.shape[1]):
            w.writerow([
                c,
                float(arr[:, c].mean()),
                float(arr[:, c].std(ddof=ddof)),
                arr.shape[0],
            ])
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Causal test of the co-training mechanism. Extracts the IMU "
            "pathway weights from a trained GMU checkpoint, plugs them into a "
            "DeepConvLSTM, and evaluates on IMU test data. No retraining.\n\n"
            "Compare the resulting acc against the IMU-only baseline at the "
            "same dropout setting (e.g., audit_dropout_imu multiseed_summary)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config", type=Path,
        help="GMU YAML config used for the multi-seed run "
             "(e.g. configs/audit/gmu_fusion.yaml).",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument(
        "--out-name", type=str, default="imu_from_gmu",
        help="Sub-directory name under experiments/<gmu_exp>/.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        gmu_cfg = yaml.safe_load(f)
    gmu_exp = gmu_cfg["experiment"]
    n_classes = gmu_cfg["model"]["num_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"GMU experiment: {gmu_exp}")
    print(f"Num seeds:      {args.num_seeds}")
    print(f"Num classes:    {n_classes}")
    print(f"Device:         {device}")
    print()

    out_root = Path("experiments") / gmu_exp / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in range(args.num_seeds):
        r = eval_one_seed(gmu_exp, seed, gmu_cfg, device)
        print(f"  seed {seed}: acc={r['weighted_accuracy']:.4f}  f1={r['macro_f1']:.4f}")
        save_per_seed(out_root / f"seed_{seed}", r, n_classes)
        results.append(r)

    summary = aggregate(out_root, results, n_classes)
    print()
    print(f"Aggregate ({summary['num_seeds']} seeds):")
    print(f"  acc {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(f"  f1  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print()
    print(f"Outputs under {out_root}/")
    print()
    print("Compare against IMU-only at the same dropout convention. For Hard:")
    print("  cat experiments/audit_dropout_imu/multiseed_summary.json")
    print("For Easy:")
    print("  cat experiments/audit_dropout_easy_imu/multiseed_summary.json")
    print()
    print("Welch t-test between the two summary CSVs gives the causal answer:")
    print("  H0: GMU IMU pathway = IMU solo (no co-training effect on encoder).")


if __name__ == "__main__":
    main()
