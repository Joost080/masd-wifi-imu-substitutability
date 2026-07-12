"""
Run a single training experiment from a YAML config.

Usage:
    python scripts/train/run_experiment.py configs/wifi_baseline.yaml
    python scripts/train/run_experiment.py configs/late_fusion.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import yaml
import torch
from datetime import datetime
from pathlib import Path

from src.data.loaders import get_dataloader
from src.models.deepconvlstm import DeepConvLSTM
from src.models.fusion import EarlyFusionModel, LateFusionModel, GMULateFusionModel
from src.models.resnet1d import ResNet1D
from src.training.trainer import Trainer


def build_model(cfg: dict):
    modality = cfg["data"]["modality"]
    m = cfg["model"]
    if modality in ("wifi", "imu"):
        return DeepConvLSTM(**m)
    elif modality == "early_fusion":
        return EarlyFusionModel(**m)
    elif modality == "late_fusion":
        return LateFusionModel(**m)
    elif modality in ("gmu_fusion", "gmu_fusion_moddrop"):
        # moddrop strips its own training-only kwarg; run_experiment forwards the
        # full block since GMULateFusionModel ignores unknown kwargs only when
        # caller filters them. Keep the moddrop_p out here.
        kwargs = {k: v for k, v in m.items() if k != "moddrop_p"}
        return GMULateFusionModel(**kwargs)
    elif modality == "wifi_resnet":
        kwargs = dict(m)
        kwargs.pop("type", None)
        return ResNet1D(**kwargs)
    raise ValueError(f"Unknown modality: {modality}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / cfg["experiment"] / timestamp

    d = cfg["data"]
    train_loader, val_loader = get_dataloader(
        mode=d["modality"],
        split="train",
        batch_size=d["batch_size"],
        val_split=d["val_split"],
        num_workers=d["num_workers"],
        seed=d["seed"],
        class_filter=d.get("class_filter"),
    )

    model = build_model(cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    t = cfg["training"]
    if t.get("optimizer", "sgd") == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=t["lr"], momentum=t["momentum"]
        )

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
    print(f"\nDone. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
