"""
Plot per-epoch gate statistics across seeds of a fusion experiment (Track 2B).

Reads experiments/<exp>/seed_<n>/gate_trajectory.csv for each seed and
overlays them as a single mean ± std curve over epochs, plus a smaller panel
for sigmoid entropy. Optionally overlays multiple experiments (e.g. baseline
GMU vs moddrop GMU) on the same axes for direct comparison.

CSV schema (written by Trainer when training.track_gate=true):
    epoch, gate_mean, gate_std, gate_entropy, gate_min, gate_max, n_samples

Usage:
    python analyze_gate_trajectory.py audit_dropout_gmu_trajectory \\
        audit_dropout_gmu_moddrop --num-seeds 5

The first positional argument is the "reference" experiment and is drawn in
blue; subsequent experiments are drawn in red / green / orange. All
trajectories truncate at the shortest run (some seeds may have early-stopped
sooner than others).

Outputs (under results/):
    gate_trajectory_<refexp>_vs_<otherexp>.pdf
    gate_trajectory_<refexp>_vs_<otherexp>_entropy.pdf
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_COLORS = ["steelblue", "indianred", "darkseagreen", "darkorange"]


def load_seed_trajectories(exp: str, num_seeds: int) -> dict:
    """Return arrays of shape (S, E) for mean, std, entropy. E is min #epochs."""
    rows_per_seed = []
    for s in range(num_seeds):
        path = Path("experiments") / exp / f"seed_{s}" / "gate_trajectory.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No gate_trajectory.csv at {path}. Did you train with "
                f"training.track_gate=true?"
            )
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows_per_seed.append(rows)

    min_len = min(len(rs) for rs in rows_per_seed)
    if min_len == 0:
        raise ValueError(f"All gate_trajectory.csv for '{exp}' are empty")
    epochs = np.array([int(rows_per_seed[0][i]["epoch"]) for i in range(min_len)])
    mean = np.zeros((num_seeds, min_len))
    ent = np.zeros((num_seeds, min_len))
    gmin = np.zeros((num_seeds, min_len))
    gmax = np.zeros((num_seeds, min_len))
    for s, rs in enumerate(rows_per_seed):
        for i in range(min_len):
            mean[s, i] = float(rs[i]["gate_mean"])
            ent[s, i] = float(rs[i]["gate_entropy"])
            gmin[s, i] = float(rs[i]["gate_min"])
            gmax[s, i] = float(rs[i]["gate_max"])
    return {
        "epochs": epochs,
        "mean":   mean,    # (S, E)
        "entropy": ent,
        "min":    gmin,
        "max":    gmax,
    }


def plot_overlay(experiments: list, num_seeds: int,
                 out_dir: Path, suffix: str):
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, exp in enumerate(experiments):
        d = load_seed_trajectories(exp, num_seeds)
        mu = d["mean"].mean(axis=0)
        sd = d["mean"].std(axis=0, ddof=1 if num_seeds > 1 else 0)
        color = _COLORS[i % len(_COLORS)]
        ax.plot(d["epochs"], mu, color=color, label=exp, linewidth=1.5)
        ax.fill_between(d["epochs"], mu - sd, mu + sd, color=color, alpha=0.18)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6,
               label="sigmoid centre (init)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean gate value (0 = IMU, 1 = WiFi)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Per-epoch gate trajectory (mean ± std across {num_seeds} seeds)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_pdf = out_dir / f"gate_trajectory{suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_dir / f"gate_trajectory{suffix}.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_pdf}")

    # Entropy panel
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, exp in enumerate(experiments):
        d = load_seed_trajectories(exp, num_seeds)
        mu = d["entropy"].mean(axis=0)
        sd = d["entropy"].std(axis=0, ddof=1 if num_seeds > 1 else 0)
        color = _COLORS[i % len(_COLORS)]
        ax.plot(d["epochs"], mu, color=color, label=exp, linewidth=1.5)
        ax.fill_between(d["epochs"], mu - sd, mu + sd, color=color, alpha=0.18)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sigmoid entropy of gate (0 = collapsed)")
    ax.set_ylim(-0.02, np.log(2.0) + 0.05)  # max entropy = log(2)
    ax.axhline(np.log(2.0), color="gray", linestyle="--", linewidth=0.6,
               label="max entropy (uniform)")
    ax.set_title(f"Per-epoch gate entropy (mean ± std across {num_seeds} seeds)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_pdf = out_dir / f"gate_trajectory_entropy{suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_dir / f"gate_trajectory_entropy{suffix}.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-epoch gate trajectories from one or more multi-seed "
            "fusion experiments. The first argument is the reference; "
            "subsequent ones are overlaid for comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiments", nargs="+",
                        help="Experiment names (e.g. audit_dropout_gmu_trajectory).")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--suffix", type=str, default=None,
                        help="Filename suffix (default: derived from experiment names).")
    args = parser.parse_args()

    if args.suffix is None:
        if len(args.experiments) == 1:
            args.suffix = f"_{args.experiments[0]}"
        else:
            args.suffix = "_" + "_vs_".join(args.experiments)

    plot_overlay(args.experiments, args.num_seeds, args.out_dir, args.suffix)

    # Numeric summary at the end of training (last epoch) for each experiment
    print()
    print("=== Final-epoch gate stats (last logged epoch per seed) ===")
    for exp in args.experiments:
        d = load_seed_trajectories(exp, args.num_seeds)
        mu_final = d["mean"][:, -1].mean()
        sd_final = d["mean"][:, -1].std(ddof=1 if args.num_seeds > 1 else 0)
        ent_final = d["entropy"][:, -1].mean()
        print(
            f"  {exp:<40s} mean={mu_final:.4f} ± {sd_final:.4f}  "
            f"entropy={ent_final:.4f}"
        )


if __name__ == "__main__":
    main()
