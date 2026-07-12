"""
Analyze learned GMU gate values for RQ4.

Produces:
  <run_dir>/gate_stats_per_class.csv  - per-class mean/std/entropy of gate
  <run_dir>/gate_trajectories.npy     - (27, T) mean gate per class per timestep
  results/rq4_gate_bar.pdf/png        - sorted bar chart of mean gate, all 27 classes
  results/rq4_gate_trajectories.pdf/png - gate over time for selected activities

Usage:
    python scripts/analysis/analyze_gates.py configs/gmu_fusion.yaml
    python scripts/analysis/analyze_gates.py configs/gmu_fusion.yaml experiments/rq3_gmu_fusion/20260505_123456
    python scripts/analysis/analyze_gates.py configs/gmu_fusion.yaml \\
        --wifi-f1 experiments/rq1_wifi/test_per_class_f1.csv \\
        --imu-f1 experiments/rq2_imu/test_per_class_f1.csv \\
        --activities 3 7 12 18 24

Gate convention: g=1 means WiFi fully used, g=0 means IMU fully used.

--activities: 5 class indices (0-26) to show in the trajectory figure.
  If omitted, auto-selects: 2 highest gate (WiFi-preferring),
  2 lowest gate (IMU-preferring), 1 closest to 0.5 (ambiguous).
  If --wifi-f1 and --imu-f1 are also given, the last slot becomes
  the class with the largest |WiFi_F1 - IMU_F1| gap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scripts.train.run_experiment import build_model
from src.data.loaders import get_dataloader

# MASD 27-class label mapping (Li et al. 2025, Table 1 — Easy→Medium→Hard ordering).
LABEL_MAP = {
    0:  "standing",
    1:  "walking",
    2:  "jumping",
    3:  "sitting",
    4:  "lying",
    5:  "wave right hand",
    6:  "drink water",
    7:  "torso-twisting",
    8:  "kick right foot",
    9:  "right hand up",
    10: "draw clockwise",
    11: "turn left",
    12: "turn right",
    13: "wave left hand",
    14: "throw",
    15: "kick left foot",
    16: "golf swing",
    17: "basketball shooting",
    18: "boxing",
    19: "squatting",
    20: "push",
    21: "pull",
    22: "bending (stand)",
    23: "bending (sit)",
    24: "leg stretch",
    25: "left hand up",
    26: "draw counterclockwise",
}


def latest_run_dir(experiment: str) -> Path:
    root = Path("experiments") / experiment
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")
    return runs[-1]


def load_f1_csv(path: Path) -> np.ndarray:
    f1 = np.zeros(27)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            f1[int(row["class"])] = float(row["f1"])
    return f1


def gate_entropy(gates: np.ndarray) -> float:
    """Mean binary entropy of per-sample scalar gate values."""
    g = np.clip(gates, 1e-7, 1 - 1e-7)
    return float(-np.mean(g * np.log(g) + (1 - g) * np.log(1 - g)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--wifi-f1", type=Path, default=None,
                        help="Per-class F1 CSV from WiFi baseline (rq1)")
    parser.add_argument("--imu-f1", type=Path, default=None,
                        help="Per-class F1 CSV from IMU baseline (rq2)")
    parser.add_argument("--activities", type=int, nargs="+", default=None,
                        help="Class indices to show in the trajectory figure (5 recommended)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix appended to figure filenames (e.g. '_27_perchannel'). "
                             "Default empty = overwrites the canonical rq4_gate_bar.pdf/png.")
    args = parser.parse_args()
    suffix = args.out_suffix

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_dir = args.run_dir or latest_run_dir(cfg["experiment"])
    ckpt = run_dir / "best_model.pt"
    print(f"Run dir:    {run_dir}")
    print(f"Checkpoint: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")

    d = cfg["data"]
    test_loader = get_dataloader(
        mode=d["modality"], split="test",
        batch_size=d["batch_size"], num_workers=d["num_workers"], seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_labels, all_gates = [], []
    with torch.no_grad():
        for wifi, imu, y in test_loader:
            wifi, imu = wifi.to(device), imu.to(device)
            _, g = model(wifi, imu, return_gate=True)  # g: (B, T, 1) or (B, T, C)
            # For per-channel gates, average across the channel dim so the rest
            # of the analysis (per-class mean/std/entropy, trajectories) is
            # apples-to-apples with the scalar variant.
            if g.dim() == 3 and g.shape[-1] > 1:
                g = g.mean(dim=-1, keepdim=True)
            all_labels.append(y.numpy())
            all_gates.append(g.squeeze(-1).cpu().numpy())  # (B, T)

    all_labels = np.concatenate(all_labels)        # (N,)
    all_gates = np.concatenate(all_gates, axis=0)  # (N, T)
    num_classes = cfg["model"]["num_classes"]
    T = all_gates.shape[1]
    print(f"Collected gates for {len(all_labels)} samples, T={T} timesteps")

    # Per-class statistics
    rows = []
    trajectories = np.zeros((num_classes, T))
    mean_gate_per_class = np.zeros(num_classes)

    for c in range(num_classes):
        mask = all_labels == c
        n = int(mask.sum())
        if n == 0:
            rows.append((c, LABEL_MAP[c], 0, 0.0, 0.0, 0.0))
            continue
        g_c = all_gates[mask]                   # (n, T)
        g_scalar = g_c.mean(axis=1)             # (n,) — per-sample mean over time
        trajectories[c] = g_c.mean(axis=0)      # (T,) — mean trajectory
        mean_gate_per_class[c] = float(g_scalar.mean())
        rows.append((c, LABEL_MAP[c], n,
                     float(g_scalar.mean()), float(g_scalar.std()),
                     gate_entropy(g_scalar)))

    with open(run_dir / "gate_stats_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "label", "n", "mean_gate", "std_gate", "entropy"])
        for row in rows:
            w.writerow(row)
    np.save(run_dir / "gate_trajectories.npy", trajectories)
    print(f"Saved gate_stats_per_class.csv and gate_trajectories.npy to {run_dir}")

    # Print summary table to stdout
    print("\nClass | Label      | n    | mean_gate | std  | entropy")
    print("-" * 60)
    for r in sorted(rows, key=lambda x: x[3], reverse=True):
        print(f"  {r[0]:2d}  | {r[1]:10s} | {r[2]:4d} | {r[3]:.4f}    | {r[4]:.4f} | {r[5]:.4f}")

    # Select 5 activities for trajectory figure
    if args.activities:
        selected = list(args.activities)
    else:
        order = np.argsort(mean_gate_per_class)
        # 2 most WiFi-preferring (highest gate), 2 most IMU-preferring (lowest gate)
        selected = list(order[-2:][::-1]) + list(order[:2])
        # 5th: disagreement class if F1 data provided, else most uncertain (nearest 0.5)
        if args.wifi_f1 and args.imu_f1:
            wifi_f1 = load_f1_csv(args.wifi_f1)
            imu_f1 = load_f1_csv(args.imu_f1)
            gap = np.abs(wifi_f1 - imu_f1)
            fifth = int(np.argmax(gap))
            print(f"\nDisagreement class: {fifth} ({LABEL_MAP[fifth]}) — "
                  f"WiFi F1={wifi_f1[fifth]:.3f}, IMU F1={imu_f1[fifth]:.3f}, gap={gap[fifth]:.3f}")
        else:
            fifth = int(np.argmin(np.abs(mean_gate_per_class - 0.5)))
        if fifth not in selected:
            selected.append(fifth)
        else:
            # Pick next-most-uncertain not already selected
            uncertain_order = np.argsort(np.abs(mean_gate_per_class - 0.5))
            for c in uncertain_order:
                if int(c) not in selected:
                    selected.append(int(c))
                    break

    print(f"\nTrajectory figure classes: {selected}")
    print("  " + ", ".join(f"{c} ({LABEL_MAP[c]})" for c in selected))

    Path("results").mkdir(exist_ok=True)

    # Figure 1: sorted horizontal bar chart
    order = np.argsort(mean_gate_per_class)
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = ["#e07b54" if mean_gate_per_class[i] > 0.5 else "#5b8db8" for i in order]
    y_pos = np.arange(num_classes)
    std_vals = np.array([r[4] for r in rows])
    ax.barh(y_pos, mean_gate_per_class[order], xerr=std_vals[order],
            color=colors, height=0.7, capsize=3, ecolor="gray", alpha=0.85)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABEL_MAP[i] for i in order], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean gate value")
    ax.set_title("Per-class modality preference (GMU gate)\n"
                 "blue = IMU preferred  |  orange = WiFi preferred")
    fig.tight_layout()
    bar_stem = f"results/rq4_gate_bar{suffix}"
    fig.savefig(f"{bar_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{bar_stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {bar_stem}.pdf/png")

    # Figure 2: gate trajectories for selected activities
    n_sel = len(selected)
    t_axis = np.linspace(0, 5, T)
    fig, axes = plt.subplots(1, n_sel, figsize=(3 * n_sel, 3), sharey=True)
    if n_sel == 1:
        axes = [axes]
    for ax, cls in zip(axes, selected):
        ax.plot(t_axis, trajectories[cls], color="steelblue", linewidth=1.5)
        ax.fill_between(t_axis, trajectories[cls], 0.5,
                        where=trajectories[cls] > 0.5,
                        alpha=0.15, color="#e07b54", label="WiFi")
        ax.fill_between(t_axis, 0.5, trajectories[cls],
                        where=trajectories[cls] < 0.5,
                        alpha=0.15, color="#5b8db8", label="IMU")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title(f"{LABEL_MAP[cls]}\n(class {cls})", fontsize=8)
    axes[0].set_ylabel("Gate (0=IMU, 1=WiFi)", fontsize=8)
    fig.suptitle("GMU gate trajectories over 5-second window", fontsize=10)
    fig.tight_layout()
    traj_stem = f"results/rq4_gate_trajectories{suffix}"
    fig.savefig(f"{traj_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{traj_stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {traj_stem}.pdf/png")


if __name__ == "__main__":
    main()
