"""
Server-side smoke test for the §C experiments (run after `git pull`, before training).

Checks the magnetometer ablation (6-axis acc+gyro), the ResNet-1D-on-IMU input,
and the 2D-CNN-on-CSI model: shapes, channel-slicing consistency, and forwards.

Usage (from research/):  python scripts/checks/smoke_test_c.py
Exit 0 = all passed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import sys
import numpy as np
import torch

from src.data.dataset import IMUDataset, IMUStatsDataset
from src.data.loaders import get_dataloader
from src.models.deepconvlstm import DeepConvLSTM
from src.models.resnet1d import ResNet1D
from src.models.stats_mlp import StatsMLP
from src.models.csi_resnet2d import CSIResNet2D

ok_all = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    ok_all.append(ok)


def main():
    print("=" * 64); print("§C SMOKE TEST (mag ablation + ResNet-IMU + 2D-CNN)"); print("=" * 64)

    # --- 6-axis raw consistency: imu6 == imu[:, :6] ---
    print("\n[C1] 6-axis IMU (acc+gyro) raw")
    imu9 = IMUDataset(split="test")
    imu6 = IMUDataset(split="test", channels=[0, 1, 2, 3, 4, 5])
    x9, _ = imu9[0]; x6, _ = imu6[0]
    check("imu6 sample is (150, 6)", tuple(x6.shape) == (150, 6), str(tuple(x6.shape)))
    check("imu6 == imu[:, :6] (same data + sliced stats)",
          torch.allclose(x6, x9[:, :6], atol=1e-5))

    # --- 6-axis stats: 36 features = first 36 cols of the 54-feature vector ---
    print("\n[C1] 6-axis IMU stats")
    s9 = IMUStatsDataset(split="test")
    s6 = IMUStatsDataset(split="test", channels=[0, 1, 2, 3, 4, 5])
    f9, _ = s9[0]; f6, _ = s6[0]
    check("imu6_stats is 36-D", tuple(f6.shape) == (36,), str(tuple(f6.shape)))
    check("imu6_stats == imu_stats[:, :36]", torch.allclose(f6, f9[:36], atol=1e-5))

    # --- loaders dispatch the new modalities + models forward ---
    print("\n[C1/C2] loaders + model forwards")
    tr, _ = get_dataloader("imu6", split="train", batch_size=4, num_workers=0)
    xb, yb = next(iter(tr))
    check("imu6 loader batch (4,150,6)", tuple(xb.shape) == (4, 150, 6), str(tuple(xb.shape)))
    with torch.no_grad():
        check("DeepConvLSTM(in_channels=6) -> (4,27)",
              tuple(DeepConvLSTM(in_channels=6, num_classes=27)(xb).shape) == (4, 27))
        check("ResNet1D(in_channels=6) -> (4,27)",
              tuple(ResNet1D(in_channels=6, num_classes=27)(xb).shape) == (4, 27))
        trs, _ = get_dataloader("imu6_stats", split="train", batch_size=4, num_workers=0)
        fb, _ = next(iter(trs))
        check("StatsMLP(in_features=36) -> (4,27)",
              tuple(StatsMLP(in_features=36, num_classes=27)(fb).shape) == (4, 27))

    # --- Finding-3 spectral stats (108-D) ---
    print("\n[Fin3] time+spectral stats-MLP")
    trsp, _ = get_dataloader("imu_stats_spec", split="train", batch_size=4, num_workers=0)
    fsp, _ = next(iter(trsp))
    check("imu_stats_spec batch is 108-D", tuple(fsp.shape) == (4, 108), str(tuple(fsp.shape)))
    with torch.no_grad():
        check("StatsMLP(in_features=108) -> (4,27)",
              tuple(StatsMLP(in_features=108, num_classes=27)(fsp).shape) == (4, 27))

    # --- 2D CNN on CSI image (Easy 5-class) ---
    print("\n[C3] 2D CNN on CSI image")
    tr2, _ = get_dataloader("wifi2d", split="train", batch_size=4, num_workers=0,
                            class_filter=[0, 1, 2, 3, 4])
    wb, _ = next(iter(tr2))
    check("wifi2d loader batch (4,500,224)", tuple(wb.shape) == (4, 500, 224), str(tuple(wb.shape)))
    m2d = CSIResNet2D(in_channels=1, num_classes=5, stem_channels=32)
    with torch.no_grad():
        check("CSIResNet2D -> (4,5)", tuple(m2d(wb).shape) == (4, 5))
    print(f"     CSIResNet2D params: {sum(p.numel() for p in m2d.parameters()):,}")

    print("\n" + "=" * 64)
    print(f"RESULT: {sum(ok_all)}/{len(ok_all)} checks passed")
    print("=" * 64)
    sys.exit(0 if all(ok_all) else 1)


if __name__ == "__main__":
    main()
