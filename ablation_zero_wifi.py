"""
Counterfactual-WiFi ablation on a trained scalar GMU.

The scalar GMU's gate is fully closed at test time (mean=std=0 across all
classes). The paper hypothesises this means the +3.6 pp Hard advantage of
GMU over IMU-only is co-training of the IMU encoder, not inference-time
fusion. The clean test: zero out the WiFi tensor at inference time on the
trained GMU and measure the accuracy change. If the gate is genuinely
closed, accuracy should be approximately preserved.

This script does NOT retrain anything. It loads a trained scalar-GMU
checkpoint and runs three test-set evaluations:

  1. baseline:  WiFi tensor as-is        (sanity: should reproduce paper number)
  2. zero:      WiFi tensor set to 0     (paper prediction: within ±0.5 pp)
  3. noise:     WiFi tensor as N(0, 1)   (sanity check on the prediction)

Usage:
    python ablation_zero_wifi.py configs/audit/gmu_fusion.yaml \\
        experiments/audit_dropout_gmu/<TIMESTAMP>

If the run path is omitted, the most recent timestamp dir under the
experiment folder is used.

Output:
    <run_dir>/ablation_zero_wifi.json
        {
          "baseline":   {"weighted_accuracy": ..., "macro_f1": ..., "n": ...},
          "zero_wifi":  {"weighted_accuracy": ..., "macro_f1": ..., "n": ...},
          "noise_wifi": {"weighted_accuracy": ..., "macro_f1": ..., "n": ...},
          "delta_zero_minus_baseline":  ...,
          "delta_noise_minus_baseline": ...
        }
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.loaders import get_dataloader
from src.utils.metrics import weighted_accuracy, macro_f1
from run_experiment import build_model


def latest_run_dir(experiment: str) -> Path:
    root = Path("experiments") / experiment
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")
    return runs[-1]


def evaluate_pass(model, loader, device, wifi_mode: str, noise_seed: int = 0):
    """One test-set forward pass.

    wifi_mode: 'as_is' | 'zero' | 'noise'
        'zero':  WiFi tensor replaced by zeros of identical shape.
        'noise': WiFi tensor replaced by N(0, 1) of identical shape.
    """
    if wifi_mode not in ("as_is", "zero", "noise"):
        raise ValueError(f"unknown wifi_mode={wifi_mode}")

    rng = torch.Generator(device="cpu").manual_seed(noise_seed)

    y_true, y_pred = [], []
    with torch.no_grad():
        for wifi, imu, y in loader:
            if wifi_mode == "zero":
                wifi = torch.zeros_like(wifi)
            elif wifi_mode == "noise":
                wifi = torch.randn(wifi.shape, generator=rng)
            wifi_d = wifi.to(device)
            imu_d = imu.to(device)
            logits = model(wifi_d, imu_d)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {
        "weighted_accuracy": float(weighted_accuracy(y_true, y_pred)),
        "macro_f1":          float(macro_f1(y_true, y_pred)),
        "n":                 int(len(y_true)),
    }


def fmt(d: dict) -> str:
    return f"acc={d['weighted_accuracy']:.4f}  f1={d['macro_f1']:.4f}  n={d['n']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["model"].get("gate_dim", 1) != 1:
        raise SystemExit(
            "This ablation targets the scalar-gate GMU (gate_dim=1). "
            f"Config has gate_dim={cfg['model'].get('gate_dim')}."
        )

    run_dir = args.run_dir or latest_run_dir(cfg["experiment"])
    ckpt = run_dir / "best_model.pt"
    print(f"Run dir:    {run_dir}")
    print(f"Checkpoint: {ckpt}")
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")

    d = cfg["data"]
    test_loader = get_dataloader(
        mode=d["modality"], split="test",
        batch_size=d["batch_size"], num_workers=d["num_workers"], seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    print("\n=== counterfactual-WiFi ablation ===")
    baseline   = evaluate_pass(model, test_loader, device, "as_is")
    print(f"  baseline (WiFi as-is) : {fmt(baseline)}")
    zero_wifi  = evaluate_pass(model, test_loader, device, "zero")
    print(f"  WiFi = 0              : {fmt(zero_wifi)}")
    noise_wifi = evaluate_pass(model, test_loader, device, "noise",
                                noise_seed=args.noise_seed)
    print(f"  WiFi = N(0,1)         : {fmt(noise_wifi)}")

    d_zero  = zero_wifi["weighted_accuracy"]  - baseline["weighted_accuracy"]
    d_noise = noise_wifi["weighted_accuracy"] - baseline["weighted_accuracy"]
    print(f"\n  delta(zero  - baseline) : {d_zero:+.4f}  ({d_zero*100:+.2f} pp)")
    print(f"  delta(noise - baseline) : {d_noise:+.4f}  ({d_noise*100:+.2f} pp)")

    out = {
        "checkpoint":                    str(ckpt),
        "config":                        str(args.config),
        "num_classes":                   int(cfg["model"]["num_classes"]),
        "class_filter":                  d.get("class_filter"),
        "noise_seed":                    int(args.noise_seed),
        "baseline":                      baseline,
        "zero_wifi":                     zero_wifi,
        "noise_wifi":                    noise_wifi,
        "delta_zero_minus_baseline":     float(d_zero),
        "delta_noise_minus_baseline":    float(d_noise),
    }
    out_path = run_dir / "ablation_zero_wifi.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
