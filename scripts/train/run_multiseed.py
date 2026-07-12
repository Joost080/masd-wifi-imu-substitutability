"""
Multi-seed training and evaluation.

Trains each config N times with different model-init seeds (0 … N-1).
The data split seed stays fixed at the value in the config (seed=42),
so every run sees the exact same train/val/test windows.

After all runs, prints a mean ± std summary table and saves JSON + CSV.

Results layout:
    experiments/<experiment>/seed_<n>/best_model.pt
    experiments/<experiment>/seed_<n>/test_metrics.json
    experiments/<experiment>/multiseed_summary.json
    experiments/<experiment>/multiseed_summary.csv

Usage (from the research/ directory):
    python scripts/train/run_multiseed.py                           # 3 seeds, default configs
    python scripts/train/run_multiseed.py --num-seeds 5
    python scripts/train/run_multiseed.py --configs configs/imu_corrected.yaml configs/gmu_corrected.yaml
    python scripts/train/run_multiseed.py --skip-train              # summarise existing runs only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.loaders import get_dataloader
from src.models.deepconvlstm import DeepConvLSTM
from src.models.fusion import GMULateFusionModel, EarlyFusionModel, LateFusionModel
from src.models.resnet1d import ResNet1D
from src.models.csi_resnet2d import CSIResNet2D
from src.models.stats_mlp import (
    StatsMLP,
    DeepConvLSTMStats,
    StatsStatsMLP,
    GatedStatsFusion,
)
from src.training.trainer import Trainer
from src.utils.metrics import weighted_accuracy, macro_f1, per_class_f1, get_confusion_matrix

DEFAULT_CONFIGS = [
    Path("configs/imu_corrected.yaml"),
    Path("configs/gmu_corrected.yaml"),
]

_TYPED_REGISTRY = {
    # stats-feature MLPs (existing side-experiment family)
    "stats_mlp":           StatsMLP,
    "deepconvlstm_stats":  DeepConvLSTMStats,
    "stats_stats_mlp":     StatsStatsMLP,
    "gated_stats_fusion":  GatedStatsFusion,
    # raw modality CSI-native backbone (Track 1C)
    "resnet1d":            ResNet1D,
    # 2D CNN over the CSI image (§C)
    "csi_resnet2d":        CSIResNet2D,
}


def _build_typed_model(cfg):
    kwargs = dict(cfg["model"])
    model_type = kwargs.pop("type")
    if model_type not in _TYPED_REGISTRY:
        raise ValueError(f"Unknown model.type '{model_type}'")
    return _TYPED_REGISTRY[model_type](**kwargs)


_MODEL_BUILDERS = {
    "imu":           lambda cfg: DeepConvLSTM(**cfg["model"]),
    "wifi":          lambda cfg: DeepConvLSTM(**cfg["model"]),
    "gmu_fusion":    lambda cfg: GMULateFusionModel(**cfg["model"]),
    "gmu_fusion_moddrop": lambda cfg: GMULateFusionModel(**{
        k: v for k, v in cfg["model"].items() if k != "moddrop_p"
    }),
    "early_fusion":  lambda cfg: EarlyFusionModel(**cfg["model"]),
    "late_fusion":   lambda cfg: LateFusionModel(**cfg["model"]),
    "imu_stats":      _build_typed_model,
    "imu_raw_stats":  _build_typed_model,
    "wifi_stats":     _build_typed_model,
    "imu_wifi_stats": _build_typed_model,
    "wifi_resnet":   _build_typed_model,
    # --- §C magnetometer ablation + 2D CNN ---
    "imu6":          lambda cfg: DeepConvLSTM(**cfg["model"]),   # 6-axis DeepConvLSTM
    "imu6_stats":    _build_typed_model,                          # 6-axis stats-MLP
    "imu6_resnet":   _build_typed_model,                          # 6-axis ResNet-1D
    "wifi2d":        _build_typed_model,                          # 2D CNN over CSI image
    "imu_stats_spec": _build_typed_model,                         # time+spectral stats-MLP (Finding-3 test)
}


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int):
    """Seed all RNGs that affect model initialisation and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg: dict) -> torch.nn.Module:
    modality = cfg["data"]["modality"]
    if modality not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown modality '{modality}'")
    return _MODEL_BUILDERS[modality](cfg)


def train_one_seed(cfg: dict, seed: int) -> Path:
    seed_everything(seed)

    exp_dir = Path("experiments") / cfg["experiment"] / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    d = cfg["data"]
    train_loader, val_loader = get_dataloader(
        mode=d["modality"],
        split="train",
        batch_size=d["batch_size"],
        val_split=d["val_split"],
        num_workers=d["num_workers"],
        seed=d["seed"],           # data split seed — fixed, not the model seed
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg)
    print(f"    params: {sum(p.numel() for p in model.parameters()):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    t = cfg["training"]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        exp_dir=exp_dir,
        early_stop_patience=t["early_stop_patience"],
        moddrop_p=float(t.get("moddrop_p", 0.0)),
        track_gate=bool(t.get("track_gate", False)),
    )
    trainer.train(num_epochs=t["epochs"])
    return exp_dir


def evaluate_one_seed(cfg: dict, seed: int) -> dict:
    exp_dir = Path("experiments") / cfg["experiment"] / f"seed_{seed}"
    ckpt = exp_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Run without --skip-train first."
        )

    # Re-seed before model construction so state_dict loads into the right architecture
    seed_everything(seed)

    d = cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = get_dataloader(
        mode=d["modality"],
        split="test",
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x1, x2, y = [t.to(device) for t in batch]
                logits = model(x1, x2)
            else:
                x, y = [t.to(device) for t in batch]
                logits = model(x)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = float(weighted_accuracy(y_true, y_pred))
    f1  = float(macro_f1(y_true, y_pred))
    result = {
        "seed": seed,
        "weighted_accuracy": acc,
        "macro_f1": f1,
        "n_samples": int(len(y_true)),
    }
    (exp_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))

    # Per-class F1 + confusion matrix per seed (same layout as single-seed runners)
    num_classes = cfg["model"]["num_classes"]
    pcf1 = per_class_f1(y_true, y_pred, num_classes=num_classes)
    cm = get_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    with open(exp_dir / "test_per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "f1"])
        for i, v in enumerate(pcf1):
            w.writerow([i, float(v)])
    np.save(exp_dir / "test_confusion_matrix.npy", cm)

    print(f"    seed={seed}  acc={acc:.4f}  f1={f1:.4f}")
    return result


def save_summary(experiment: str, results: list) -> dict:
    accs = [r["weighted_accuracy"] for r in results]
    f1s  = [r["macro_f1"]          for r in results]
    ddof = 1 if len(results) > 1 else 0      # sample std when n>1, else 0
    summary = {
        "experiment": experiment,
        "num_seeds":  len(results),
        "acc_mean":   float(np.mean(accs)),
        "acc_std":    float(np.std(accs, ddof=ddof)),
        "f1_mean":    float(np.mean(f1s)),
        "f1_std":     float(np.std(f1s, ddof=ddof)),
        "per_seed":   results,
    }
    out = Path("experiments") / experiment
    (out / "multiseed_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "multiseed_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "weighted_accuracy", "macro_f1", "n_samples"])
        for r in results:
            w.writerow([r["seed"], r["weighted_accuracy"], r["macro_f1"], r["n_samples"]])

    # Per-class F1 aggregated across seeds.
    # Reads each seed's test_per_class_f1.csv; produces mean ± std per class.
    pcf1_per_seed = []
    for r in results:
        seed_dir = out / f"seed_{r['seed']}"
        csv_path = seed_dir / "test_per_class_f1.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            seed_pcf1 = [float(row[1]) for row in reader]
        pcf1_per_seed.append(seed_pcf1)
    if pcf1_per_seed:
        arr = np.array(pcf1_per_seed)            # (S, K)
        with open(out / "multiseed_per_class_f1.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "f1_mean", "f1_std", "n_seeds"])
            for c in range(arr.shape[1]):
                f1_mean = float(arr[:, c].mean())
                f1_std = float(arr[:, c].std(ddof=ddof))
                w.writerow([c, f1_mean, f1_std, arr.shape[0]])

    print(f"  Summary saved -> experiments/{experiment}/multiseed_summary.json")
    return summary


def print_table(summaries: list):
    print(f"\n{'='*70}")
    print(f"{'MULTI-SEED SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Experiment':<30}  {'N':>3}  {'Acc mean±std':^22}  {'F1 mean±std':^22}")
    print(f"{'-'*30}  {'-'*3}  {'-'*22}  {'-'*22}")
    for s in summaries:
        print(
            f"{s['experiment']:<30}  {s['num_seeds']:>3}  "
            f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}          "
            f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train and/or evaluate multi-seed experiments. Two ways to pick seeds:\n"
            "  --num-seeds N       seeds 0..N-1 (default, also triggers the aggregate "
            "                       multiseed_summary.* + multiseed_per_class_f1.csv writes)\n"
            "  --seeds 0 1 2 ...    run only these specific seeds; aggregate is skipped\n"
            "                       (use a final --num-seeds N --skip-train pass to write it)\n"
            "\n"
            "Run experiments in parallel across GPUs by opening multiple terminals and "
            "pinning each with CUDA_VISIBLE_DEVICES. Two common patterns:\n"
            "  - different experiments, one per GPU (no --seeds flag needed):\n"
            "      CUDA_VISIBLE_DEVICES=0 python scripts/train/run_multiseed.py --configs A.yaml --num-seeds 5\n"
            "      CUDA_VISIBLE_DEVICES=1 python scripts/train/run_multiseed.py --configs B.yaml --num-seeds 5\n"
            "  - same experiment, split seeds across GPUs:\n"
            "      CUDA_VISIBLE_DEVICES=0 python scripts/train/run_multiseed.py --configs A.yaml --seeds 0 1 2\n"
            "      CUDA_VISIBLE_DEVICES=1 python scripts/train/run_multiseed.py --configs A.yaml --seeds 3 4\n"
            "      # after both finish, in any terminal:\n"
            "      python scripts/train/run_multiseed.py --configs A.yaml --num-seeds 5 --skip-train\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs", nargs="+", type=Path, default=DEFAULT_CONFIGS,
        help="YAML config files to run (default: imu_corrected + gmu_corrected).",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3,
        help="Number of seeds to run (seeds 0..N-1). Default: 3. Mutually exclusive with --seeds.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Run only these specific seeds. Use to split work across GPUs. "
             "When set, the aggregate multiseed_summary.* files are NOT written -- "
             "do a final pass with --num-seeds N --skip-train to aggregate.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; evaluate and summarise existing seed runs only.",
    )
    args = parser.parse_args()

    if args.seeds is not None:
        seeds_to_run = list(args.seeds)
        partial = True
    else:
        seeds_to_run = list(range(args.num_seeds))
        partial = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds_str = ", ".join(str(s) for s in seeds_to_run)
    mode = "partial (aggregate skipped)" if partial else "full"
    print(f"Device: {device}  |  Seeds: [{seeds_str}]  |  Mode: {mode}\n")

    summaries = []

    for config_path in args.configs:
        cfg = load_cfg(config_path)
        name = cfg["experiment"]
        print(f"{'='*60}")
        print(f"Experiment : {name}")
        print(f"Config     : {config_path}")
        print(f"{'='*60}")

        results = []
        for seed in seeds_to_run:
            print(f"\n  Seed {seed}:")
            if not args.skip_train:
                print("    Training...")
                train_one_seed(cfg, seed)
            print("    Evaluating...")
            results.append(evaluate_one_seed(cfg, seed))

        if partial:
            print(f"\n  Partial run on seeds [{seeds_str}] -- aggregate not written.")
            print(f"  After all seeds finish, run:")
            print(f"    python scripts/train/run_multiseed.py --configs {config_path} "
                  f"--num-seeds <total> --skip-train")
        else:
            summaries.append(save_summary(name, results))
        print()

    if summaries:
        print_table(summaries)
    print("All done.")


if __name__ == "__main__":
    main()
