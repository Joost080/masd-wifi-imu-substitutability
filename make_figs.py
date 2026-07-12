"""
Generate the paper figures from the committed experiment metrics under
experiments/ and results/. Pure visualisation of already-recorded results --
no model runs, no GPU, no dataset download needed.

    python make_figs.py        # writes figures/fig_*.pdf and .png
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_WIFI, C_IMU, C_FUSE = "#c44e52", "#4c72b0", "#55a868"
CAT_COLORS = {
    "static_posture": "#4c72b0", "locomotion": "#c44e52",
    "upper_body_arm": "#dd8452", "bilateral_torso": "#8172b3",
    "sports_limb": "#55a868",
}
CAT_LABEL = {
    "static_posture": "Static posture", "locomotion": "Locomotion",
    "upper_body_arm": "Upper-body / arm", "bilateral_torso": "Bilateral / torso",
    "sports_limb": "Dynamic limb / sports",
}


def acc(name):
    return json.load(open(EXP / name / "multiseed_summary.json"))["acc_mean"]


# ---------------------------------------------------------------------------
# Figure 1 -- the substitutability gap (headline). WiFi vs IMU, both regimes,
# every backbone. Numbers from the corrected-pipeline multi-seed runs (log).
# ---------------------------------------------------------------------------
def fig_headline():
    labels = ["DCL", "ResNet", "2D-CNN", "DCL", "ResNet"]
    hard = [(0.064, C_WIFI), (0.091, C_WIFI), (0.1045, C_WIFI), (0.734, C_IMU), (0.800, C_IMU)]
    easy = [(0.342, C_WIFI), (0.469, C_WIFI), (0.465, C_WIFI), (0.952, C_IMU), (0.940, C_IMU)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, data, title, chance in [
        (axes[0], hard, "Hard (27 classes)", 1/27),
        (axes[1], easy, "Easy (5 classes)", 1/5)]:
        vals = [d[0] for d in data]
        cols = [d[1] for d in data]
        x = np.arange(len(data))
        ax.bar(x, vals, color=cols, width=0.7)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v*100:.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(chance, ls=":", color="0.4", lw=1)
        ax.text(len(data) - 0.5, chance + 0.01, "chance", ha="right", va="bottom",
                fontsize=7, color="0.4")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 1.08); ax.set_title(title)
        ax.axvspan(-0.5, 2.5, color=C_WIFI, alpha=0.06)
        ax.axvspan(2.5, 4.5, color=C_IMU, alpha=0.06)
        ax.text(1.0, 1.04, "WiFi", ha="center", va="top", fontsize=8.5, color=C_WIFI, fontweight="bold")
        ax.text(3.5, 1.04, "IMU", ha="center", va="top", fontsize=8.5, color=C_IMU, fontweight="bold")
    axes[0].set_ylabel("Weighted accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "fig_headline.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- per-class substitutability boundary. WiFi-F1 vs IMU-F1 per class,
# sorted by IMU-F1. None of the 27 WiFi classes clears a usable threshold.
# ---------------------------------------------------------------------------
def fig_boundary():
    df = pd.read_csv(RES / "substitutability_perclass.csv").sort_values("imu_f1")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(3.4, 4.3))
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.plot([r.wifi_f1, r.imu_f1], [yi, yi], color="0.8", lw=1, zorder=1)
    ax.scatter(df.imu_f1, y, color=C_IMU, s=16, zorder=3, label="IMU")
    ax.scatter(df.wifi_f1, y, color=C_WIFI, s=16, zorder=3, label="WiFi")
    ax.axvline(0.20, ls="--", color="0.3", lw=1)
    ax.text(0.205, 1.0, "usable\nthreshold", fontsize=6.5, color="0.3", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(df.name, fontsize=6)
    ax.set_xlabel("Per-class F1 (Hard 27-class)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.7, len(df) - 0.3)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_boundary.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- the capstone. Proper CSI representations and fusion, evaluated
# subject-independently. (a) representation matters but stays at the floor;
# (b) fusion with proper WiFi never beats IMU-alone.
# ---------------------------------------------------------------------------
def fig_capstone():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # (a) representation, subject-held-out (wifirb_*)
    ax = axes[0]
    reps = ["signed", "amp", "doppler"]
    hard = [acc(f"wifirb_{r}_deepconvlstm") for r in reps]
    easy = [acc("wifirb_signed_deepconvlstm_easy"), acc("wifirb_amp_deepconvlstm_easy"), np.nan]
    x = np.arange(3); w = 0.38
    ax.bar(x - w/2, hard, w, color=C_WIFI, label="Hard (27)")
    ax.bar(x + w/2, easy, w, color="#e8a2a2", label="Easy (5)")
    for xi, v in zip(x - w/2, hard):
        ax.text(xi, v + 0.015, f"{v*100:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w/2, easy):
        if not np.isnan(v):
            ax.text(xi, v + 0.015, f"{v*100:.0f}", ha="center", va="bottom", fontsize=7)
    _imu_easy_si = acc("aligned_imu_easy")  # subject-independent IMU-Easy (matches this panel's split)
    ax.axhline(_imu_easy_si, ls="-", color=C_IMU, lw=1); ax.text(2.4, _imu_easy_si, "IMU (Easy)", color=C_IMU, fontsize=6.5, va="bottom", ha="right")
    ax.axhline(0.55, ls=":", color="0.4", lw=1); ax.text(2.4, 0.55, "SKELAR WiFi (Easy)", color="0.4", fontsize=6.5, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["signed\n(ours, old)", "amplitude", "Doppler"], fontsize=7.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Weighted accuracy")
    ax.set_title("(a) CSI representation\n(subject-independent)", fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    # (b) fusion with proper WiFi (aligned_*), Hard
    ax = axes[1]
    bars = [
        ("IMU\nalone", acc("aligned_imu"), C_IMU),
        ("GMU\n+amp", acc("aligned_gmu_amp"), C_FUSE),
        ("GMU\n+Dopp.", acc("aligned_gmu_doppler"), C_FUSE),
        ("late\n+amp", acc("aligned_late_amp"), C_FUSE),
        ("late\n+Dopp.", acc("aligned_late_doppler"), C_FUSE),
    ]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.7)
    for xi, b in zip(x, bars):
        ax.text(xi, b[1] + 0.012, f"{b[1]*100:.0f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(acc("aligned_imu"), ls="--", color=C_IMU, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
    ax.set_ylim(0, 0.85)
    ax.set_title("(b) Fusion with proper WiFi\n(Hard, subject-independent)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_capstone.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 (appendix) -- feature sufficiency by activity type. Per-class F1
# delta, stats-MLP minus best-tuned DeepConvLSTM (conv-dropout 0.1), coloured
# by activity category. Negative = temporal model needed.
# ---------------------------------------------------------------------------
def fig_featuresuff():
    cat = pd.read_csv(RES / "substitutability_perclass.csv")[["class", "name", "category"]]
    stats = pd.read_csv(EXP / "stats_imu_only" / "multiseed_per_class_f1.csv")[["class", "f1_mean"]].rename(columns={"f1_mean": "stats"})
    deep = pd.read_csv(EXP / "rq2_imu_corrected" / "multiseed_per_class_f1.csv")[["class", "f1_mean"]].rename(columns={"f1_mean": "deep"})
    df = cat.merge(stats, on="class").merge(deep, on="class")
    df["delta"] = df["stats"] - df["deep"]
    df = df.sort_values("delta")
    y = np.arange(len(df))
    colors = [CAT_COLORS[c] for c in df.category]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh(y, df.delta, color=colors)
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(df.name, fontsize=7)
    ax.set_xlabel("F1(stats MLP, 27K) - F1(best DeepConvLSTM, 299K)")
    ax.set_ylim(-0.7, len(df) - 0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[k]) for k in CAT_LABEL]
    ax.legend(handles, [CAT_LABEL[k] for k in CAT_LABEL], loc="lower right",
              frameon=False, fontsize=7)
    ax.text(0.012, len(df) - 1.4, "summary stats suffice", ha="left", fontsize=7.5, color="0.3")
    ax.text(-0.012, 1.2, "temporal model needed", ha="right", fontsize=7.5, color="0.3")
    fig.tight_layout()
    fig.savefig(OUT / "fig_featuresuff.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 0 (method) -- pipeline schematic of the GMU fusion model, matching the
# code: per-modality CONV encoders -> gate -> SHARED 2-layer LSTM -> head. The
# backbone variants are single-modality baselines (noted below), not branches
# of the fusion model. Pure schematic, no data dependence.
# ---------------------------------------------------------------------------
def fig_method():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    C_W, C_I, C_F, C_L, C_N = "#f4e3e3", "#e4ecf6", "#e6f1e9", "#fdf2e0", "#f0f0f0"
    fig, ax = plt.subplots(figsize=(7.0, 3.05))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.016",
            mutation_aspect=2.4, linewidth=0.8, edgecolor="0.45", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
            lw=0.9, color="0.35", shrinkA=0, shrinkB=0))

    # --- pipeline (upper ~60% of the canvas) ---
    yw, yi, hh = 0.72, 0.41, 0.20
    cw, ci, cg = yw + hh / 2, yi + hh / 2, 0.67   # row centres + main-line centre
    # inputs
    box(0.005, yw, 0.155, hh, "WiFi CSI\n$500\\times224$\n(signed/amp/Dopp.)", C_W)
    box(0.005, yi, 0.155, hh, "IMU\n$150\\times9$\n(acc+gyro+mag)", C_I)
    # per-modality CONV encoders (conv-only in the fusion model)
    box(0.205, yw, 0.17, hh, "WiFi conv encoder\n$4\\times$(Conv-ReLU-\nMaxPool)", C_W)
    box(0.205, yi, 0.17, hh, "IMU conv encoder\n$4\\times$(Conv-ReLU-\nMaxPool)", C_I)
    # gate (converges the two conv-feature streams)
    box(0.42, 0.51, 0.125, 0.32, "GMU gate $g_t$\n(scalar /\nper-channel)\nor concat", C_F)
    # SHARED LSTM -- after the gate
    box(0.585, 0.58, 0.13, 0.18, "shared\n2-layer LSTM\n(128)", C_L)
    # head + output
    box(0.735, 0.59, 0.08, 0.16, "linear\nhead", C_N)
    box(0.835, 0.56, 0.13, 0.22, "27 cls (Hard)\n5 cls (Easy)", C_N)

    arrow(0.16, cw, 0.205, cw)
    arrow(0.16, ci, 0.205, ci)
    arrow(0.375, cw, 0.42, 0.75)       # WiFi conv feats -> gate
    arrow(0.375, ci, 0.42, 0.59)       # IMU conv feats  -> gate
    arrow(0.545, cg, 0.585, cg)        # gate -> shared LSTM
    arrow(0.715, cg, 0.735, cg)        # LSTM -> head
    arrow(0.815, cg, 0.835, cg)        # head -> output
    ax.text(0.1825, 0.665, "$\\downarrow$150\n(fusion)", ha="center", va="center",
            fontsize=5.2, color="0.45")
    ax.text(0.455, 0.85, "per-timestep $h_t\\in\\mathbb{R}^{64}$", ha="center",
            va="bottom", fontsize=5.4, color="0.45")

    # --- notes (lower ~35% of the canvas, evenly spaced, clear bottom margin) ---
    # band A -- single-modality baselines (where the backbone sweep lives)
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.205), 0.90, 0.135, boxstyle="round,pad=0.004,rounding_size=0.01",
        mutation_aspect=5, linewidth=0.6, edgecolor="0.7", facecolor="#fbf7f2"))
    ax.text(0.5, 0.298,
            "Single-modality baseline:  one stream $\\rightarrow$ backbone "
            "$\\rightarrow$ linear head",
            ha="center", va="center", fontsize=6.2, color="0.25")
    ax.text(0.5, 0.238,
            "backbone $=$ DeepConvLSTM (conv$+$LSTM);  WiFi also 1D-ResNet / 2D-CNN,  "
            "IMU also 1D-ResNet / feature-MLP",
            ha="center", va="center", fontsize=6.2, color="0.4")
    # band B -- evaluation protocols
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.06), 0.90, 0.095, boxstyle="round,pad=0.004,rounding_size=0.01",
        mutation_aspect=7, linewidth=0.6, edgecolor="0.7", facecolor="#fafafa"))
    ax.text(0.5, 0.1075,
            "Evaluation:  window-level split   |   "
            "subject-independent (4 of 19 participants held out)",
            ha="center", va="center", fontsize=6.2, color="0.25")
    fig.savefig(OUT / "fig_method.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_method()
    fig_headline(); fig_boundary(); fig_capstone(); fig_featuresuff()
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.pdf"))))
