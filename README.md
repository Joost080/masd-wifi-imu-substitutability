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
| `tests/` | Unit tests for data, models, metrics |
| `build_wifi_proper.py` | Rebuilds signed / amplitude / Doppler CSI **from the raw complex MASD release** and recovers subject identifiers (auto-downloads the raw files) |
| `build_aligned.py` | Builds the aligned proper-WiFi + IMU dataset for the subject-independent fusion re-test |
| `run_experiment.py`, `run_multiseed.py` | Single-config and n-seed training/evaluation (window-level protocol, Table 1) |
| `run_wifirb.py`, `run_aligned.py`, `run_arch_amp.sh`, `run_phase2.sh` | Subject-independent re-test on the rebuild (Table 2) |
| `run_stats_experiments.py`, `run_corrected_experiments.py`, `run_overnight.py` | Feature-MLP (SQ4) and dropout-audit runs |
| `analyze_gates.py`, `ablation_zero_wifi.py`, `eval_imu_from_gmu.py`, `analyze_gate_trajectory.py` | Fusion diagnostics D1–D3 (gate closure, WiFi-invariance counterfactual, modality dropout / encoder extraction) |
| `analyze_substitutability.py`, `analyze_stats_perclass.py`, `analyze_perchannel_multiseed.py`, `analyze_cotraining_perclass.py`, `diagnostic_perchannel.py`, `make_perchannel_compare.py` | Per-class boundary, stats-vs-deep, and per-channel gate analyses |
| `make_figs.py` | Regenerates the four paper figures **from the committed metrics — no GPU or dataset needed** |
| `sanity_overfit.py`, `smoke_test_a0.py`, `smoke_test_c.py` | Pipeline sanity checks (incl. the pilot overfitting test referenced in Methods) |

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
python run_multiseed.py --num-seeds 5 --configs configs/imu_corrected.yaml configs/gmu_corrected.yaml

# Table 2 — from-raw rebuild, then subject-independent runs:
python build_wifi_proper.py --out-dir data/wifirb --window 500 --stride 250
python run_wifirb.py                  # WiFi-alone by representation/backbone
python build_aligned.py               # aligned proper-WiFi + IMU dataset
python run_aligned.py                 # fusion re-test (IMU control reproduces 0.6923)

# Fusion diagnostics (D1–D3) and per-class analyses:
python analyze_gates.py
python ablation_zero_wifi.py
python eval_imu_from_gmu.py
python analyze_substitutability.py
python analyze_stats_perclass.py
```

Each script's docstring documents its options; the shell scripts (`run_arch_amp.sh`,
`run_phase2.sh`) record the exact sequences used for the paper's GPU runs.

## Citation

Citation entry to follow upon publication in the UbiComp/ISWC 2026 Adjunct Proceedings. Until
then, please cite the paper as "under review at the Reproduce! workshop, UbiComp/ISWC 2026".

## License

MIT — see [LICENSE](LICENSE).
