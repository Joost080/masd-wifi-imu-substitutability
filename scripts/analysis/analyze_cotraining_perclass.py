"""
Per-class Track 1A analysis: where does the IMU encoder degrade when
extracted from a trained GMU vs trained standalone?

For each class c, we report:
    F1_solo(c)        = mean F1 across seeds for audit_dropout_imu
    F1_from_gmu(c)    = mean F1 across seeds for IMU-from-GMU eval
    delta(c)          = F1_solo(c) - F1_from_gmu(c)

Plus an aggregate paired Wilcoxon signed-rank test on the per-seed macro F1
(5 paired observations: standalone IMU vs IMU-from-GMU on the same seed).

Usage:
    python scripts/analysis/analyze_cotraining_perclass.py \
        --solo-exp audit_dropout_imu \
        --gmu-exp  audit_dropout_gmu \
        --num-seeds 5 \
        --suffix hard

The same script handles Easy:
    python scripts/analysis/analyze_cotraining_perclass.py \
        --solo-exp audit_dropout_easy_imu \
        --gmu-exp  audit_dropout_easy_gmu \
        --num-seeds 5 \
        --suffix easy

Outputs (under results/):
    cotraining_perclass_delta_<suffix>.csv
    cotraining_perclass_delta_<suffix>.pdf
    cotraining_perclass_stats_<suffix>.json
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats


def load_perclass_f1(exp_root: Path, num_seeds: int) -> tuple:
    """Returns (classes, f1_matrix) where f1_matrix is (S, C)."""
    rows_per_seed = []
    classes = None
    for s in range(num_seeds):
        path = exp_root / f"seed_{s}" / "test_per_class_f1.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        cls = [int(r["class"]) for r in rows]
        f1s = [float(r["f1"]) for r in rows]
        if classes is None:
            classes = cls
        elif classes != cls:
            raise ValueError(
                f"Class order mismatch at {path}: {cls} vs {classes}"
            )
        rows_per_seed.append(f1s)
    return classes, np.array(rows_per_seed)


def load_seed_macro_f1(exp_root: Path, num_seeds: int) -> np.ndarray:
    """Returns array of shape (S,) of macro_f1 per seed from test_metrics.json."""
    out = []
    for s in range(num_seeds):
        path = exp_root / f"seed_{s}" / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with open(path) as f:
            m = json.load(f)
        out.append(float(m["macro_f1"]))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-exp", required=True,
                        help="Experiment name for standalone IMU (e.g. audit_dropout_imu).")
    parser.add_argument("--gmu-exp", required=True,
                        help="GMU experiment name (e.g. audit_dropout_gmu). "
                             "Reads imu_from_gmu/seed_<n>/ inside it.")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--suffix", required=True,
                        help="Filename suffix, e.g. 'hard' or 'easy'.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--exp-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    solo_root = args.exp_root / args.solo_exp
    gmu_imu_root = args.exp_root / args.gmu_exp / "imu_from_gmu"

    classes, f1_solo = load_perclass_f1(solo_root, args.num_seeds)
    classes_g, f1_gmu = load_perclass_f1(gmu_imu_root, args.num_seeds)
    if classes != classes_g:
        raise ValueError("Class index ordering differs between solo and GMU-IMU CSVs")

    mean_solo = f1_solo.mean(axis=0)
    std_solo  = f1_solo.std(axis=0, ddof=1)
    mean_gmu  = f1_gmu.mean(axis=0)
    std_gmu   = f1_gmu.std(axis=0, ddof=1)
    delta     = mean_solo - mean_gmu

    # Paired macro-F1 test
    solo_macro = load_seed_macro_f1(solo_root, args.num_seeds)
    gmu_macro  = load_seed_macro_f1(gmu_imu_root, args.num_seeds)
    paired_diff = solo_macro - gmu_macro
    try:
        wilc = scipy_stats.wilcoxon(solo_macro, gmu_macro, alternative="greater")
        wilc_stat = float(wilc.statistic)
        wilc_p = float(wilc.pvalue)
    except ValueError as e:
        wilc_stat, wilc_p = float("nan"), float("nan")
        print(f"Wilcoxon could not be computed: {e}")
    t = scipy_stats.ttest_rel(solo_macro, gmu_macro)
    t_stat = float(t.statistic)
    t_p = float(t.pvalue)

    out_csv = args.out_dir / f"cotraining_perclass_delta_{args.suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "f1_solo_mean", "f1_solo_std",
                    "f1_imu_from_gmu_mean", "f1_imu_from_gmu_std", "delta"])
        order = np.argsort(-delta)  # largest drop first
        for i in order:
            w.writerow([classes[i],
                        f"{mean_solo[i]:.4f}", f"{std_solo[i]:.4f}",
                        f"{mean_gmu[i]:.4f}",  f"{std_gmu[i]:.4f}",
                        f"{delta[i]:.4f}"])
    print(f"Saved {out_csv}")

    stats_out = {
        "suffix": args.suffix,
        "solo_exp": args.solo_exp,
        "gmu_exp": args.gmu_exp,
        "num_seeds": args.num_seeds,
        "macro_f1_solo_mean": float(solo_macro.mean()),
        "macro_f1_solo_std":  float(solo_macro.std(ddof=1)),
        "macro_f1_imu_from_gmu_mean": float(gmu_macro.mean()),
        "macro_f1_imu_from_gmu_std":  float(gmu_macro.std(ddof=1)),
        "mean_paired_drop": float(paired_diff.mean()),
        "wilcoxon_statistic": wilc_stat,
        "wilcoxon_p_one_sided": wilc_p,
        "paired_t_statistic": t_stat,
        "paired_t_p_two_sided": t_p,
        "class_with_largest_drop": int(classes[int(np.argmax(delta))]),
        "largest_drop_value": float(delta.max()),
        "class_with_smallest_drop": int(classes[int(np.argmin(delta))]),
        "smallest_drop_value": float(delta.min()),
        "num_classes_with_drop_gt_0.10": int((delta > 0.10).sum()),
        "num_classes_improved_by_gmu":  int((delta < 0).sum()),
    }
    out_json = args.out_dir / f"cotraining_perclass_stats_{args.suffix}.json"
    with open(out_json, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Saved {out_json}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = np.argsort(-delta)
    x = np.arange(len(classes))
    bar_colors = ["#c0392b" if d > 0 else "#27ae60" for d in delta[order]]
    ax.bar(x, delta[order], color=bar_colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(classes[i]) for i in order], rotation=0, fontsize=7)
    ax.set_xlabel("Class index (sorted by drop)")
    ax.set_ylabel("F1 drop: solo − IMU-from-GMU")
    ax.set_title(
        f"Per-class F1 drop from co-training "
        f"({args.suffix}; macro F1 solo={solo_macro.mean():.3f}"
        f" → from-GMU={gmu_macro.mean():.3f}, "
        f"Wilcoxon p={wilc_p:.3g})"
    )
    fig.tight_layout()
    out_pdf = args.out_dir / f"cotraining_perclass_delta_{args.suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(args.out_dir / f"cotraining_perclass_delta_{args.suffix}.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_pdf}")

    print()
    print("=== Summary ===")
    print(f"Macro F1 solo:         {solo_macro.mean():.4f} ± {solo_macro.std(ddof=1):.4f}")
    print(f"Macro F1 IMU-from-GMU: {gmu_macro.mean():.4f} ± {gmu_macro.std(ddof=1):.4f}")
    print(f"Mean paired drop:      {paired_diff.mean():.4f}")
    print(f"Paired t-test:         t={t_stat:.3f}, p={t_p:.4g}")
    print(f"Wilcoxon (greater):    W={wilc_stat:.3f}, p={wilc_p:.4g}")
    print(f"Classes with drop > 0.10: {(delta > 0.10).sum()} / {len(classes)}")
    print(f"Classes improved by GMU:  {(delta < 0).sum()} / {len(classes)}")


if __name__ == "__main__":
    main()
