"""
Compose the Hard vs Easy per-channel heatmaps into a single side-by-side
figure for the presentation. Reads the existing per-channel diagnostic PNGs
in results/ and writes a composite to results/perchannel_compare_hard_vs_easy.{pdf,png}.

Usage:
    python scripts/analysis/make_perchannel_compare.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    results = Path("results")
    hard_path = results / "perchannel_gate_heatmap_audit_perchannel.png"
    easy_path = results / "perchannel_gate_heatmap_audit_easy_perchannel.png"
    for p in (hard_path, easy_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    hard_img = mpimg.imread(hard_path)
    easy_img = mpimg.imread(easy_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].imshow(hard_img)
    axes[0].set_title(
        "Hard (27-class): per-channel gate is unstable\n"
        "structured channels = 33.2 ± 18.8 (CV 57%)",
        fontsize=10,
    )
    axes[0].axis("off")

    axes[1].imshow(easy_img)
    axes[1].set_title(
        "Easy (5-class): per-channel gate is stable\n"
        "structured channels = 55.4 ± 0.9 (CV 1.6%)",
        fontsize=10,
    )
    axes[1].axis("off")

    fig.suptitle(
        "Per-channel GMU routing emerges robustly only when WiFi has informative signal",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_pdf = results / "perchannel_compare_hard_vs_easy.pdf"
    out_png = results / "perchannel_compare_hard_vs_easy.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
