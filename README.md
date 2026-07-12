# Can Ambient WiFi Substitute for a Body-Worn IMU?

Code, CSI-rebuild scripts, and experiment metrics for:

> **Can Ambient WiFi Substitute for a Body-Worn IMU? A Representation-Corrected, Multi-Seed Study
> of Cross-Modal Substitution for Human Activity Recognition at Scale**
> Joost Liebeton, Egemen İşgüder, Jeroen Klein Brinke, Özlem Durmaz İncel (University of Twente).
> Under review at the Reproduce! workshop, UbiComp/ISWC 2026.

The paper asks under what conditions ambient WiFi CSI can substitute for a body-worn IMU on the
MASD benchmark (27 activities, 20 participants). Along the way it corrects the CSI representation
exposed by MASD's released ML-ready files (a degraded *signed* encoding — the real part of the
complex CSI — rather than the documented amplitude), rebuilds proper amplitude and antenna-ratio
Doppler from the raw complex release, recovers the participant identifiers the processed files
omit, and re-evaluates the headline comparisons subject-independently over five seeds.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Models (DeepConvLSTM, 1D-ResNet, 2D-CNN, GMU/early/late fusion, stats MLPs), data pipeline, trainer, metrics |
| `configs/` | YAML configs for every trained configuration (incl. `ablation/`, `audit/`, `easy/` variants) |
| `experiments/` | **Metrics of all runs in the paper** (`multiseed_summary.json`, per-class F1 CSVs, training logs). Model checkpoints are stripped for size |
| `results/` | Aggregated analysis outputs (per-class substitutability tables, gate statistics) |
| `scripts/rebuild/` | `build_wifi_proper.py` rebuilds signed / amplitude / Doppler CSI **from the raw complex MASD release** and recovers subject identifiers (auto-downloads the raw files); `build_aligned.py` builds the aligned proper-WiFi + IMU dataset |
| `scripts/train/` | Training/evaluation runners: `run_experiment.py` + `run_multiseed.py` (window-level protocol, Table 1), `run_wifirb.py` + `run_aligned.py` + shell wrappers (subject-independent re-test, Table 2), stats-MLP and dropout-audit runs |
| `scripts/analysis/` | Fusion diagnostics D1–D3 (`analyze_gates.py`, `ablation_zero_wifi.py`, `eval_imu_from_gmu.py`, `analyze_gate_trajectory.py`) and the per-class boundary / stats-vs-deep / per-channel analyses |
| `scripts/checks/` | Pipeline sanity checks (incl. the pilot overfitting test referenced in Methods) |
| `tests/` | Unit tests for data, models, metrics |
| `make_figs.py` | Regenerates the four paper figures **from the committed metrics — no GPU or dataset needed** |

All commands below are run from the repository root.

## Setup

Python ≥ 3.10. PyTorch is pinned to the CUDA 12.1 build used for the paper (2.5.1):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

(CPU-only or newer CUDA builds of `torch==2.5.1` also work for the smaller models.)

## Data

The MASD dataset (Li et al., 2025) is **not redistributed here**.

1. **Processed ML-ready files** (window-level protocol, Table 1): download from the MASD release
   and place as `data/train_wifi.npy`, `data/train_imu.npy`, `data/train_labels.npy`,
   `data/test_wifi.npy`, `data/test_imu.npy`, `data/test_labels.npy`.
2. **Raw complex CSI** (rebuild + subject-independent protocol, Table 2): fetched automatically —
   `build_wifi_proper.py` downloads the per-participant raw files from UCSD DataPlanet
   (`perma:83.ucsddata/FVZWII`) into a local cache and derives the representations.

## Reproducing the paper

Every headline number is a mean ± std over **5 model-initialisation seeds (0–4)** with the data
split seed fixed at **42**. The subject-independent protocol holds out participants **4, 9, 13,
18** (chosen once to span the participant-index range, fixed across all seeds and configurations;
15 train / 4 test of the 19 participants with usable CSI). Training: Adam (lr 1e-3, batch 32),
≤ 80 epochs, early stopping on validation loss (patience 10); z-score statistics from the 80%
train partition only.

```bash
# Figures only (works immediately from the committed metrics)
python make_figs.py

# Table 1 — window-level multi-seed runs, e.g.:
python scripts/train/run_multiseed.py --num-seeds 5 --configs configs/imu_corrected.yaml configs/gmu_corrected.yaml

# Table 2 — from-raw rebuild, then subject-independent runs:
python scripts/rebuild/build_wifi_proper.py --out-dir data/wifirb --window 500 --stride 250
python scripts/train/run_wifirb.py    # WiFi-alone by representation/backbone
python scripts/rebuild/build_aligned.py   # aligned proper-WiFi + IMU dataset
python scripts/train/run_aligned.py   # fusion re-test (IMU control reproduces 0.6923)

# Fusion diagnostics (D1–D3) and per-class analyses:
python scripts/analysis/analyze_gates.py
python scripts/analysis/ablation_zero_wifi.py
python scripts/analysis/eval_imu_from_gmu.py
python scripts/analysis/analyze_substitutability.py
python scripts/analysis/analyze_stats_perclass.py
```

Each script's docstring documents its options; the shell scripts
(`scripts/train/run_arch_amp.sh`, `scripts/train/run_phase2.sh`) record the exact
sequences used for the paper's GPU runs.

## Citation

Citation entry to follow upon publication in the UbiComp/ISWC 2026 Adjunct Proceedings. Until
then, please cite the paper as "under review at the Reproduce! workshop, UbiComp/ISWC 2026".

## License

MIT — see [LICENSE](LICENSE).
