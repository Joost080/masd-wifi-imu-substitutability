"""
Per-class F1 comparison across the three stats-features variants:

  A. raw-only       -- audit_dropout_imu  (canonical 27-class baseline at p_LSTM=0.5)
  B. stats-only     -- stats_imu_only
  C. raw + stats    -- stats_imu_raw_stats

Reads each experiment's per-class F1 CSV. By default reads the canonical single-
seed CSV (test_per_class_f1.csv); with --multiseed it reads
multiseed_per_class_f1.csv (mean ± std across seeds).

Output: results/stats_perclass_comparison.csv with one row per class containing
A_f1, B_f1, C_f1, and pairwise deltas. Also prints a sorted table to stdout
highlighting the classes where the variants disagree most.

Usage:
    python scripts/analysis/analyze_stats_perclass.py                # single-seed (canonical CSVs)
    python scripts/analysis/analyze_stats_perclass.py --multiseed    # multi-seed (after run_multiseed.py)
    python scripts/analysis/analyze_stats_perclass.py --a-exp audit_dropout_imu  # override A
"""

import argparse
import csv
from pathlib import Path

# MASD 27-class label map. Identical to analyze_gates.py.
LABEL_MAP = {
    0: "standing", 1: "walking", 2: "jumping", 3: "sitting", 4: "lying",
    5: "wave right hand", 6: "drink water", 7: "torso-twisting",
    8: "kick right foot", 9: "right hand up", 10: "draw clockwise",
    11: "turn left", 12: "turn right", 13: "wave left hand", 14: "throw",
    15: "kick left foot", 16: "golf swing", 17: "basketball shooting",
    18: "boxing", 19: "squatting", 20: "push", 21: "pull",
    22: "bending (stand)", 23: "bending (sit)", 24: "leg stretch",
    25: "left hand up", 26: "draw counterclockwise",
}


def _resolve_single_seed_csv(experiment: str) -> Path:
    """Find a test_per_class_f1.csv for an experiment.

    Search order: experiment-level canonical -> latest timestamped subdir ->
    error. Some early runs (e.g. audit_dropout_imu) saved best_model.pt only
    and never produced the canonical CSV at the experiment level.
    """
    base = Path("experiments") / experiment
    candidate = base / "test_per_class_f1.csv"
    if candidate.exists():
        return candidate
    if base.exists():
        subdirs = sorted(p for p in base.iterdir()
                         if p.is_dir() and (p / "test_per_class_f1.csv").exists())
        if subdirs:
            return subdirs[-1] / "test_per_class_f1.csv"
    raise FileNotFoundError(
        f"No test_per_class_f1.csv under experiments/{experiment}/. "
        f"Either the canonical file or a timestamped run dir must contain it."
    )


def load_perclass(experiment: str, multiseed: bool) -> dict[int, dict]:
    """Return {class_id: {'mean': f1, 'std': f1_or_None}} for one experiment."""
    if multiseed:
        path = Path("experiments") / experiment / "multiseed_per_class_f1.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run run_multiseed.py for '{experiment}' first."
            )
        out = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                out[int(row["class"])] = {
                    "mean": float(row["f1_mean"]),
                    "std":  float(row["f1_std"]),
                }
        return out

    path = _resolve_single_seed_csv(experiment)
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row["class"])] = {"mean": float(row["f1"]), "std": None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-exp", default="rq2_imu_corrected",
                        help="Experiment name for variant A (raw-only). "
                             "Default is rq2_imu_corrected because that's where the "
                             "canonical IMU per-class CSV exists locally.")
    parser.add_argument("--b-exp", default="stats_imu_only",
                        help="Experiment name for variant B (stats-only).")
    parser.add_argument("--c-exp", default="stats_imu_raw_stats",
                        help="Experiment name for variant C (raw + stats). "
                             "Ignored when --no-c is passed.")
    parser.add_argument("--no-c", action="store_true",
                        help="Skip variant C (raw+stats fusion). Use for the WiFi "
                             "case where only A (raw) and B (stats) exist yet.")
    parser.add_argument("--multiseed", action="store_true",
                        help="Read multiseed_per_class_f1.csv instead of single-seed CSV.")
    parser.add_argument("--out", default="results/stats_perclass_comparison.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    print(f"Mode: {'multi-seed' if args.multiseed else 'single-seed'}"
          + (" (A vs B only)" if args.no_c else ""))
    a = load_perclass(args.a_exp, args.multiseed)
    b = load_perclass(args.b_exp, args.multiseed)
    c = None if args.no_c else load_perclass(args.c_exp, args.multiseed)

    classes = sorted(set(a) & set(b)) if c is None else sorted(set(a) & set(b) & set(c))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        if args.multiseed:
            if c is None:
                w.writerow(["class", "label",
                            "A_f1_mean", "A_f1_std",
                            "B_f1_mean", "B_f1_std",
                            "B_minus_A"])
                for cls in classes:
                    w.writerow([
                        cls, LABEL_MAP.get(cls, ""),
                        a[cls]["mean"], a[cls]["std"],
                        b[cls]["mean"], b[cls]["std"],
                        b[cls]["mean"] - a[cls]["mean"],
                    ])
            else:
                w.writerow(["class", "label",
                            "A_f1_mean", "A_f1_std",
                            "B_f1_mean", "B_f1_std",
                            "C_f1_mean", "C_f1_std",
                            "B_minus_A", "C_minus_A", "B_minus_C"])
                for cls in classes:
                    w.writerow([
                        cls, LABEL_MAP.get(cls, ""),
                        a[cls]["mean"], a[cls]["std"],
                        b[cls]["mean"], b[cls]["std"],
                        c[cls]["mean"], c[cls]["std"],
                        b[cls]["mean"] - a[cls]["mean"],
                        c[cls]["mean"] - a[cls]["mean"],
                        b[cls]["mean"] - c[cls]["mean"],
                    ])
        else:
            if c is None:
                w.writerow(["class", "label", "A_f1", "B_f1", "B_minus_A"])
                for cls in classes:
                    w.writerow([
                        cls, LABEL_MAP.get(cls, ""),
                        a[cls]["mean"], b[cls]["mean"],
                        b[cls]["mean"] - a[cls]["mean"],
                    ])
            else:
                w.writerow(["class", "label", "A_f1", "B_f1", "C_f1",
                            "B_minus_A", "C_minus_A", "B_minus_C"])
                for cls in classes:
                    w.writerow([
                        cls, LABEL_MAP.get(cls, ""),
                        a[cls]["mean"], b[cls]["mean"], c[cls]["mean"],
                        b[cls]["mean"] - a[cls]["mean"],
                        c[cls]["mean"] - a[cls]["mean"],
                        b[cls]["mean"] - c[cls]["mean"],
                    ])
    print(f"Saved -> {args.out}")

    # Stdout table sort key: |B-C| if C exists (highlight C anomalies);
    # |B-A| otherwise (highlight where stats vs raw disagree most)
    print(f"\n{'='*80}")
    if c is None:
        print(f"Per-class F1 -- sorted by |B - A| (where stats and raw disagree most)")
    else:
        print(f"Per-class F1 -- sorted by |B - C| (variants where stats-only and "
              f"fusion disagree most)")
    print(f"{'='*80}")
    if args.multiseed:
        if c is None:
            header = f"{'cls':>3} {'label':<22} {'A f1':>8} {'B f1':>8} {'B-A':>7}"
        else:
            header = f"{'cls':>3} {'label':<22} {'A f1':>8} {'B f1':>8} {'C f1':>8} {'B-A':>7} {'C-A':>7} {'B-C':>7}"
    else:
        if c is None:
            header = f"{'cls':>3} {'label':<22} {'A f1':>7} {'B f1':>7} {'B-A':>7}"
        else:
            header = f"{'cls':>3} {'label':<22} {'A f1':>7} {'B f1':>7} {'C f1':>7} {'B-A':>7} {'C-A':>7} {'B-C':>7}"
    print(header)
    print("-" * len(header))
    if c is None:
        ranked = sorted(classes, key=lambda c_: abs(b[c_]["mean"] - a[c_]["mean"]), reverse=True)
    else:
        ranked = sorted(classes, key=lambda c_: abs(b[c_]["mean"] - c[c_]["mean"]), reverse=True)
    for cls in ranked:
        label = LABEL_MAP.get(cls, "")[:22]
        af, bf = a[cls]["mean"], b[cls]["mean"]
        if c is None:
            print(f"{cls:>3} {label:<22} {af:>7.3f} {bf:>7.3f} {bf-af:>+7.3f}")
        else:
            cf = c[cls]["mean"]
            print(f"{cls:>3} {label:<22} {af:>7.3f} {bf:>7.3f} {cf:>7.3f} "
                  f"{bf-af:>+7.3f} {cf-af:>+7.3f} {bf-cf:>+7.3f}")

    # Aggregate verdicts
    n = len(classes)
    b_beats_a = sum(1 for cls in classes if b[cls]["mean"] > a[cls]["mean"])
    if c is not None:
        c_beats_a = sum(1 for cls in classes if c[cls]["mean"] > a[cls]["mean"])
        b_beats_c = sum(1 for cls in classes if b[cls]["mean"] > c[cls]["mean"])
    print(f"\n{'='*80}")
    print("Aggregate per-class outcomes")
    print(f"{'='*80}")
    print(f"  B beats A on {b_beats_a}/{n} classes ({100*b_beats_a/n:.0f}%)")
    if c is not None:
        print(f"  C beats A on {c_beats_a}/{n} classes ({100*c_beats_a/n:.0f}%)")
        print(f"  B beats C on {b_beats_c}/{n} classes ({100*b_beats_c/n:.0f}%)  "
              f"<- if >50%, the C<B anomaly is broad-based, not driven by a few classes")


if __name__ == "__main__":
    main()
