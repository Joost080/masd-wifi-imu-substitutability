"""
Per-channel-aware gate diagnostic.

The audit found that diagnostic_posture.py averages the 64 per-channel gate
values into a single scalar per sample/timestep before computing per-class
statistics. That averaging may hide channel-specific structure: channel 1
could variance-route on postures while channel 50 competence-routes on
locomotion, and the average would look uniform.

This script does NOT average across channels. For each of the 64 channels
separately, it computes:
  - per-class mean gate value (across samples and time)
  - Pearson correlation with WiFi-F1, IMU-F1, and IMU-sigma over the 27 classes

Outputs:
  - results/perchannel_gate_table<suffix>.csv
        rows = (channel, cor_wifi_f1, cor_imu_f1, cor_imu_sigma)
  - results/perchannel_gate_heatmap<suffix>.pdf
        64 × num_classes heatmap of mean gate; channels sorted by
        cor(gate_c, IMU-sigma); classes sorted by IMU-sigma
  - results/perchannel_gate_scatter<suffix>.pdf
        scatter of per-channel correlations:
            x = cor(gate_c, WiFi-F1)   (positive = competence-routing channel)
            y = cor(gate_c, IMU-sigma) (negative = variance-routing channel)

Usage:
    python scripts/analysis/diagnostic_perchannel.py configs/gmu_fusion_perchannel.yaml \
        experiments/rq5_gmu_perchannel/<TIMESTAMP> \
        --wifi-f1 experiments/rq1_wifi/20260504_223532/test_per_class_f1_rq1.csv \
        --imu-f1  experiments/rq2_imu/20260504_225400/test_per_class_f1_rq2.csv

Requires a model trained with gate_dim > 1 (per-channel gating).
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

LABEL_MAP = {
    0:  "standing",      1:  "walking",       2:  "jumping",       3:  "sitting",
    4:  "lying",         5:  "wave right",    6:  "drink",         7:  "torso-twist",
    8:  "kick right",    9:  "right hand up", 10: "draw cw",       11: "turn left",
    12: "turn right",    13: "wave left",     14: "throw",         15: "kick left",
    16: "golf swing",    17: "basketball",    18: "boxing",        19: "squatting",
    20: "push",          21: "pull",          22: "bend (stand)",  23: "bend (sit)",
    24: "leg stretch",   25: "left hand up",  26: "draw ccw",
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path,
                        help="YAML config of the per-channel GMU run")
    parser.add_argument("run_dir", type=Path,
                        help="Run directory containing best_model.pt")
    parser.add_argument("--wifi-f1", type=Path, required=True)
    parser.add_argument("--imu-f1",  type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-suffix", type=str, default="")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n = cfg["model"]["num_classes"]
    gate_dim = cfg["model"].get("gate_dim", 1)
    if gate_dim == 1:
        raise SystemExit("This diagnostic requires gate_dim > 1 (per-channel gating). "
                         f"Config has gate_dim={gate_dim}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Run dir:    {args.run_dir}")
    print(f"Device:     {device}")
    print(f"gate_dim:   {gate_dim}")
    print(f"num_classes: {n}")

    d = cfg["data"]
    test_loader = get_dataloader(
        mode=d["modality"], split="test",
        batch_size=d["batch_size"], num_workers=d["num_workers"], seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.run_dir / "best_model.pt",
                                      map_location=device))
    model.eval()

    # --- collect per-class, per-channel mean gate + per-class IMU-sigma ---
    gate_sum  = np.zeros((n, gate_dim))   # sum over samples and time
    gate_cnt  = np.zeros(n, dtype=int)    # samples per class (×T factor below)
    sigma_sum = np.zeros(n)
    sigma_cnt = np.zeros(n, dtype=int)

    with torch.no_grad():
        for wifi, imu, y in test_loader:
            wifi_d, imu_d = wifi.to(device), imu.to(device)
            _, g = model(wifi_d, imu_d, return_gate=True)   # (B, T, gate_dim)
            g_np = g.cpu().numpy()                          # (B, T, gate_dim)
            B, T, _ = g_np.shape
            # mean over time per sample → (B, gate_dim)
            g_per_sample = g_np.mean(axis=1)
            # IMU temporal activity per sample: mean across channels of std across time
            imu_np = imu.numpy()
            sigma_per_sample = imu_np.std(axis=1).mean(axis=1)  # (B,)

            for cls, g_row, sig in zip(y.numpy(), g_per_sample, sigma_per_sample):
                c = int(cls)
                gate_sum[c]  += g_row
                gate_cnt[c]  += 1
                sigma_sum[c] += float(sig)
                sigma_cnt[c] += 1

    counts = np.maximum(gate_cnt, 1)[:, None]   # (n, 1) for broadcasting
    gate_per_class_per_channel = gate_sum / counts   # (n, gate_dim)
    imu_sigma = np.where(sigma_cnt > 0,
                         sigma_sum / np.maximum(sigma_cnt, 1), np.nan)
    print(f"Collected gates for {gate_cnt.sum()} samples across {n} classes")

    wifi_f1 = load_per_class_f1(args.wifi_f1, n)
    imu_f1  = load_per_class_f1(args.imu_f1,  n)

    # --- per-channel correlations ---
    cor_wifi  = np.zeros(gate_dim)
    cor_imu   = np.zeros(gate_dim)
    cor_sigma = np.zeros(gate_dim)
    for c in range(gate_dim):
        gate_c = gate_per_class_per_channel[:, c]
        cor_wifi[c]  = pearson(gate_c, wifi_f1)
        cor_imu[c]   = pearson(gate_c, imu_f1)
        cor_sigma[c] = pearson(gate_c, imu_sigma)

    # --- save per-channel table ---
    table_path = args.out_dir / f"perchannel_gate_table{args.out_suffix}.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "mean_gate_overall",
                    "cor_wifi_f1", "cor_imu_f1", "cor_imu_sigma"])
        for c in range(gate_dim):
            w.writerow([c, f"{gate_per_class_per_channel[:, c].mean():.4f}",
                        f"{cor_wifi[c]:+.4f}", f"{cor_imu[c]:+.4f}",
                        f"{cor_sigma[c]:+.4f}"])
    print(f"Saved {table_path}")

    # --- summary stats ---
    THR = 0.30   # absolute correlation threshold for "shows pattern"
    n_var_router  = int(np.sum(cor_sigma < -THR))    # variance-router channels
    n_comp_router = int(np.sum(cor_wifi  >  THR))    # competence-router channels
    n_anti_var    = int(np.sum(cor_sigma >  THR))    # opposite of variance-router
    n_anti_comp   = int(np.sum(cor_wifi  < -THR))
    n_neutral     = int(np.sum((np.abs(cor_sigma) <= THR)
                               & (np.abs(cor_wifi) <= THR)))

    print("\n=== Per-channel gating pattern summary "
          f"(N={gate_dim} channels, threshold |r| > {THR}) ===")
    print(f"  variance-router channels        (cor_sigma < -{THR}): {n_var_router:3d}")
    print(f"  competence-router channels       (cor_wifi  > +{THR}): {n_comp_router:3d}")
    print(f"  anti-variance-router channels   (cor_sigma > +{THR}): {n_anti_var:3d}")
    print(f"  anti-competence-router channels  (cor_wifi  < -{THR}): {n_anti_comp:3d}")
    print(f"  neutral (no clear pattern)                          : {n_neutral:3d}")
    print(f"\n  cor(gate, IMU-sigma) range : [{cor_sigma.min():+.3f}, {cor_sigma.max():+.3f}]")
    print(f"  cor(gate, WiFi-F1) range   : [{cor_wifi.min():+.3f}, {cor_wifi.max():+.3f}]")

    if n_var_router + n_comp_router + n_anti_var + n_anti_comp >= gate_dim // 4:
        print("\n  Diverse per-channel structure detected — channel-averaged "
              "diagnostic was MASKING this. The per-channel gate IS doing "
              "something class-specific; previous 'uniform gate' claim must "
              "be revised.")
    else:
        print("\n  Most channels are in the neutral band. The per-channel gate "
              "averaged claim is approximately correct — the gate genuinely "
              "lacks per-class structure across channels.")

    # --- heatmap: channels (sorted by cor_sigma) × classes (sorted by IMU-sigma) ---
    chan_order  = np.argsort(cor_sigma)               # negative cor on top
    class_order = np.argsort(imu_sigma)               # static-posture classes left
    M = gate_per_class_per_channel[class_order][:, chan_order].T  # (gate_dim, n)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.30), 8))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                   vmin=0.0, vmax=max(0.5, float(M.max())))
    ax.set_yticks([0, gate_dim // 2, gate_dim - 1])
    ax.set_yticklabels(
        [f"ch {chan_order[0]} (cor σ={cor_sigma[chan_order[0]]:+.2f})",
         f"ch {chan_order[gate_dim // 2]}",
         f"ch {chan_order[-1]} (cor σ={cor_sigma[chan_order[-1]]:+.2f})"],
        fontsize=8,
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels([LABEL_MAP[c] for c in class_order],
                       rotation=70, ha="right", fontsize=7)
    ax.set_xlabel("Class (sorted by IMU-sigma, low → high)")
    ax.set_ylabel("Gate channel (sorted by cor(gate, IMU-sigma))")
    ax.set_title(f"Per-channel gate heatmap — {gate_dim} channels × {n} classes")
    fig.colorbar(im, ax=ax, label="Mean gate value")
    fig.tight_layout()
    out_pdf = args.out_dir / f"perchannel_gate_heatmap{args.out_suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(args.out_dir / f"perchannel_gate_heatmap{args.out_suffix}.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_pdf}")

    # --- scatter of per-channel correlations ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(cor_wifi, cor_sigma, s=30, alpha=0.7,
               edgecolor="black", linewidth=0.4, c="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.axhline(-THR, color="#e07b54", linestyle=":", linewidth=0.6,
               label=f"variance-router threshold (cor_sigma = -{THR})")
    ax.axvline( THR, color="#5b8db8", linestyle=":", linewidth=0.6,
               label=f"competence-router threshold (cor_wifi = +{THR})")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("cor(gate_channel, WiFi-F1)  →  competence-router →")
    ax.set_ylabel("cor(gate_channel, IMU-sigma)  →  ↑ anti-variance | ↓ variance-router")
    ax.set_title(f"Per-channel routing pattern distribution ({gate_dim} channels)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out_pdf = args.out_dir / f"perchannel_gate_scatter{args.out_suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(args.out_dir / f"perchannel_gate_scatter{args.out_suffix}.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
