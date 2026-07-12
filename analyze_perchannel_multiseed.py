"""
Multi-seed per-channel gate diagnostic (Track 1B).

The single-seed per-channel diagnostic (diagnostic_perchannel.py) found 47/64
structured channels (30 variance-routers, 17 competence-routers) on the
audit-dropout per-channel GMU. That single-seed count is the paper's headline
diagnostic number. This script re-runs the per-channel analysis on every seed
of a multi-seed per-channel GMU experiment and reports the channel-category
counts as mean ± std across seeds, so the claim is publication-rigorous.

Method (per seed, identical to diagnostic_perchannel.py):
  1. Load best_model.pt from experiments/<gmu_exp>/seed_<n>/.
  2. Forward the test set with return_gate=True, collect per-class per-channel
     mean gate value across (samples, time).
  3. For each of the 64 channels, compute Pearson r against:
        - WiFi-only per-class F1 (cor_wifi)
        - IMU-only per-class F1 (cor_imu)
        - IMU-sigma per class (cor_sigma)
  4. Classify each channel:
        variance-router         : cor_sigma < -THR
        competence-router       : cor_wifi  > +THR
        anti-variance-router    : cor_sigma > +THR
        anti-competence-router  : cor_wifi  < -THR
        neutral                 : |cor_sigma| <= THR and |cor_wifi| <= THR

Aggregation:
  - mean ± std of category counts across seeds
  - mean ± std of (cor_sigma range, cor_wifi range)
  - per-channel mean correlations (so a "stable" channel category over seeds
    can be identified for the heatmap caption)

Inputs are the SAME wifi-f1 / imu-f1 baselines as diagnostic_perchannel.py
(per-class F1 from the single-modality audit runs).

Usage (from research/):
    python analyze_perchannel_multiseed.py \
        configs/audit/gmu_perchannel.yaml \
        --wifi-f1 experiments/audit_dropout_wifi/seed_0/test_per_class_f1.csv \
        --imu-f1  experiments/audit_dropout_imu/seed_0/test_per_class_f1.csv \
        --num-seeds 5

Outputs (under results/):
    perchannel_multiseed_summary.json
    perchannel_multiseed_table.csv       (per-channel mean ± std correlations)
    perchannel_multiseed_category_counts.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.loaders import get_dataloader
from src.models.fusion import GMULateFusionModel


def load_per_class_f1(path: Path, n_classes: int) -> np.ndarray:
    f1 = np.full(n_classes, np.nan)
    with open(path) as f:
        for row in csv.DictReader(f):
            idx = int(row["class"])
            if idx < n_classes:
                f1[idx] = float(row["f1"])
    return f1


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    xs, ys = x[mask], y[mask]
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def perchannel_for_seed(
    ckpt: Path,
    gmu_cfg: dict,
    wifi_f1: np.ndarray,
    imu_f1: np.ndarray,
    device: str,
) -> dict:
    """Run the per-channel diagnostic for one checkpoint. Returns arrays.

    Mirrors the diagnostic_perchannel.py logic; kept here to avoid imports
    that script-level rather than module-level.
    """
    model = GMULateFusionModel(**gmu_cfg["model"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    n = gmu_cfg["model"]["num_classes"]
    gate_dim = gmu_cfg["model"].get("gate_dim", 1)
    if gate_dim == "channel":
        gate_dim = gmu_cfg["model"].get("conv_channels", 64)
    if gate_dim <= 1:
        raise ValueError(f"per-channel diagnostic requires gate_dim>1; got {gate_dim}")

    d = gmu_cfg["data"]
    test_loader = get_dataloader(
        mode=d["modality"], split="test",
        batch_size=d["batch_size"], num_workers=d["num_workers"], seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    gate_sum = np.zeros((n, gate_dim))
    gate_cnt = np.zeros(n, dtype=int)
    sigma_sum = np.zeros(n)
    sigma_cnt = np.zeros(n, dtype=int)

    with torch.no_grad():
        for wifi, imu, y in test_loader:
            wifi_d, imu_d = wifi.to(device), imu.to(device)
            _, g = model(wifi_d, imu_d, return_gate=True)
            g_per_sample = g.cpu().numpy().mean(axis=1)  # (B, gate_dim)
            sigma_per_sample = imu.numpy().std(axis=1).mean(axis=1)  # (B,)
            for cls, g_row, sig in zip(y.numpy(), g_per_sample, sigma_per_sample):
                c = int(cls)
                gate_sum[c] += g_row
                gate_cnt[c] += 1
                sigma_sum[c] += float(sig)
                sigma_cnt[c] += 1

    counts = np.maximum(gate_cnt, 1)[:, None]
    gate_per_class_per_channel = gate_sum / counts
    imu_sigma = np.where(sigma_cnt > 0, sigma_sum / np.maximum(sigma_cnt, 1), np.nan)

    cor_wifi = np.zeros(gate_dim)
    cor_imu = np.zeros(gate_dim)
    cor_sigma = np.zeros(gate_dim)
    for c in range(gate_dim):
        gc = gate_per_class_per_channel[:, c]
        cor_wifi[c] = pearson(gc, wifi_f1)
        cor_imu[c] = pearson(gc, imu_f1)
        cor_sigma[c] = pearson(gc, imu_sigma)

    return {
        "gate_per_class_per_channel": gate_per_class_per_channel,
        "cor_wifi": cor_wifi,
        "cor_imu": cor_imu,
        "cor_sigma": cor_sigma,
    }


def categorize(cor_sigma: np.ndarray, cor_wifi: np.ndarray, thr: float) -> dict:
    return {
        "variance_router":   int(np.sum(cor_sigma < -thr)),
        "competence_router": int(np.sum(cor_wifi  >  thr)),
        "anti_variance":     int(np.sum(cor_sigma >  thr)),
        "anti_competence":   int(np.sum(cor_wifi  < -thr)),
        "neutral":           int(np.sum((np.abs(cor_sigma) <= thr)
                                        & (np.abs(cor_wifi) <= thr))),
        "structured":        int(np.sum((cor_sigma < -thr) | (cor_wifi > thr)
                                        | (cor_sigma > thr) | (cor_wifi < -thr))),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed per-channel gate diagnostic. Aggregates the single-seed "
            "diagnostic_perchannel.py classifications across N seeds and reports "
            "mean ± std category counts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path,
                        help="GMU per-channel config (e.g. configs/audit/gmu_perchannel.yaml).")
    parser.add_argument("--wifi-f1", type=Path, required=True,
                        help="per-class F1 CSV from the WiFi-only audit run.")
    parser.add_argument("--imu-f1",  type=Path, required=True,
                        help="per-class F1 CSV from the IMU-only audit run.")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="|cor| threshold for 'shows pattern' classification. "
                             "Default 0.30 matches the single-seed diagnostic.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-suffix", type=str, default=None,
                        help="Suffix on output filenames. Defaults to "
                             "'_<experiment_name>' so different runs don't "
                             "overwrite each other.")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    with open(args.config) as f:
        gmu_cfg = yaml.safe_load(f)
    gmu_exp = gmu_cfg["experiment"]
    n = gmu_cfg["model"]["num_classes"]
    if args.out_suffix is None:
        args.out_suffix = f"_{gmu_exp}"

    wifi_f1 = load_per_class_f1(args.wifi_f1, n)
    imu_f1 = load_per_class_f1(args.imu_f1, n)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GMU experiment: {gmu_exp}")
    print(f"Device:         {device}")
    print(f"Threshold:      |r| > {args.threshold}")
    print()

    per_seed = []
    for seed in range(args.num_seeds):
        ckpt = Path("experiments") / gmu_exp / f"seed_{seed}" / "best_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        d = perchannel_for_seed(ckpt, gmu_cfg, wifi_f1, imu_f1, device)
        cats = categorize(d["cor_sigma"], d["cor_wifi"], args.threshold)
        print(
            f"  seed {seed}: structured={cats['structured']:>2}  "
            f"var={cats['variance_router']:>2}  "
            f"comp={cats['competence_router']:>2}  "
            f"neut={cats['neutral']:>2}  "
            f"cor_sigma in [{d['cor_sigma'].min():+.2f},{d['cor_sigma'].max():+.2f}]  "
            f"cor_wifi in [{d['cor_wifi'].min():+.2f},{d['cor_wifi'].max():+.2f}]"
        )
        per_seed.append({"seed": seed, **d, **cats})

    # --- aggregate ---
    gate_dim = per_seed[0]["cor_sigma"].shape[0]
    cs_arr = np.stack([s["cor_sigma"] for s in per_seed])   # (S, C)
    cw_arr = np.stack([s["cor_wifi"]  for s in per_seed])
    ci_arr = np.stack([s["cor_imu"]   for s in per_seed])
    ddof = 1 if args.num_seeds > 1 else 0

    cat_keys = ["variance_router", "competence_router", "anti_variance",
                "anti_competence", "neutral", "structured"]
    cat_mean = {k: float(np.mean([s[k] for s in per_seed])) for k in cat_keys}
    cat_std  = {k: float(np.std([s[k] for s in per_seed], ddof=ddof)) for k in cat_keys}

    range_sigma_per_seed = np.stack([
        [s["cor_sigma"].min(), s["cor_sigma"].max()] for s in per_seed
    ])
    range_wifi_per_seed = np.stack([
        [s["cor_wifi"].min(), s["cor_wifi"].max()] for s in per_seed
    ])
    summary = {
        "experiment": gmu_exp,
        "num_seeds": args.num_seeds,
        "gate_dim": int(gate_dim),
        "threshold": args.threshold,
        "category_counts": {
            k: {"mean": cat_mean[k], "std": cat_std[k]} for k in cat_keys
        },
        "cor_sigma_range_mean": {
            "min": float(range_sigma_per_seed[:, 0].mean()),
            "max": float(range_sigma_per_seed[:, 1].mean()),
        },
        "cor_wifi_range_mean": {
            "min": float(range_wifi_per_seed[:, 0].mean()),
            "max": float(range_wifi_per_seed[:, 1].mean()),
        },
        "per_seed_categories": [
            {"seed": s["seed"], **{k: s[k] for k in cat_keys}} for s in per_seed
        ],
    }

    out_json = args.out_dir / f"perchannel_multiseed_summary{args.out_suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Per-channel table (mean ± std across seeds for each channel)
    out_table = args.out_dir / f"perchannel_multiseed_table{args.out_suffix}.csv"
    with open(out_table, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel",
                    "cor_sigma_mean", "cor_sigma_std",
                    "cor_wifi_mean",  "cor_wifi_std",
                    "cor_imu_mean",   "cor_imu_std"])
        for c in range(gate_dim):
            w.writerow([
                c,
                f"{cs_arr[:, c].mean():+.4f}",
                f"{cs_arr[:, c].std(ddof=ddof):.4f}",
                f"{cw_arr[:, c].mean():+.4f}",
                f"{cw_arr[:, c].std(ddof=ddof):.4f}",
                f"{ci_arr[:, c].mean():+.4f}",
                f"{ci_arr[:, c].std(ddof=ddof):.4f}",
            ])

    out_cats = args.out_dir / f"perchannel_multiseed_category_counts{args.out_suffix}.csv"
    with open(out_cats, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "mean", "std", "per_seed"])
        for k in cat_keys:
            per = ",".join(str(s[k]) for s in per_seed)
            w.writerow([k, f"{cat_mean[k]:.2f}", f"{cat_std[k]:.2f}", per])

    print()
    print(f"=== Aggregate ({args.num_seeds} seeds, |r| > {args.threshold}) ===")
    for k in cat_keys:
        print(f"  {k:<20s}  {cat_mean[k]:5.2f} ± {cat_std[k]:.2f}")
    print()
    print(f"  cor_sigma range (mean): [{summary['cor_sigma_range_mean']['min']:+.3f}, "
          f"{summary['cor_sigma_range_mean']['max']:+.3f}]")
    print(f"  cor_wifi  range (mean): [{summary['cor_wifi_range_mean']['min']:+.3f}, "
          f"{summary['cor_wifi_range_mean']['max']:+.3f}]")
    print()
    print(f"Wrote:")
    print(f"  {out_json}")
    print(f"  {out_table}")
    print(f"  {out_cats}")


if __name__ == "__main__":
    main()
