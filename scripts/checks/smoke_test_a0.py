"""
Server-side smoke test for the §A0 pipeline changes (run after `git pull`).

Verifies, with torch present, the two changes that the supervisor's replan rests on:

  A0.1  Normalization mu/sigma come from the 80% TRAIN partition only (val excluded),
        and that partition is *byte-for-byte* the same train subset that
        loaders.get_dataloader's random_split produces.

  A0.2  Fusion now feeds WiFi downsampled 500 -> 150 (adaptive avg pool) and native
        IMU at 150; the fusion models forward cleanly at the new temporal length.

Usage (from research/):
    python scripts/checks/smoke_test_a0.py            # quick (Easy-filter WiFi gather is small)
    python scripts/checks/smoke_test_a0.py --full     # also primes the 27-class WiFi stats cache (~3 GB read)

Exit code 0 = all checks passed; nonzero = a check failed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import random_split

from src.data.dataset import (
    _train_partition_rows,
    _train_stats,
    _train_feature_stats,
    _downsample_to,
    IMUDataset,
    WiFiDataset,
    EarlyFusionDataset,
    DATA_DIR,
    _VAL_SPLIT,
    _SPLIT_SEED,
)
from src.data.loaders import get_dataloader
from src.models.fusion import EarlyFusionModel, LateFusionModel, GMULateFusionModel

_PASS, _FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail=""):
    tag = _PASS if ok else _FAIL
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    _results.append(ok)
    return ok


def _raw_rows_from_random_split(ds, val_split, seed):
    """Replay the loader's split and map the train subset back to raw file rows."""
    N = len(ds)
    n_val = int(N * val_split)
    n_train = N - n_val
    train_ds, _ = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    pos = np.asarray(train_ds.indices)
    if ds.indices is not None:                      # class-filtered: position -> member row
        rows = np.asarray(ds.indices)[pos]
    else:
        rows = pos
    return np.sort(rows), n_train, n_val


def test_partition_equivalence(class_filter, label):
    print(f"\n[A0.1] Partition == random_split  ({label})")
    ds = IMUDataset(split="train", class_filter=class_filter,
                    val_split=_VAL_SPLIT, split_seed=_SPLIT_SEED)
    expected, n_train, n_val = _raw_rows_from_random_split(ds, _VAL_SPLIT, _SPLIT_SEED)
    got = _train_partition_rows(class_filter, _VAL_SPLIT, _SPLIT_SEED)
    check("train-partition rows match random_split's train subset exactly",
          np.array_equal(got, expected),
          f"n_train={len(got)} (expected {n_train})")

    # Val exclusion: train and val partitions are disjoint and cover all member rows.
    y = np.load(DATA_DIR / "train_labels.npy")
    if class_filter is None:
        member = np.arange(len(y))
    else:
        member = np.flatnonzero(np.isin(y, sorted(set(class_filter))))
    val_rows = np.setdiff1d(member, got)
    check("|train| = N - int(N*0.2)", len(got) == len(member) - int(len(member) * _VAL_SPLIT))
    check("train ∩ val = ∅ and train ∪ val = all member rows",
          len(np.intersect1d(got, val_rows)) == 0 and (len(got) + len(val_rows) == len(member)),
          f"|train|={len(got)} |val|={len(val_rows)} |member|={len(member)}")


def test_stats_exclude_val(class_filter, label, full):
    print(f"\n[A0.1] Stats use train partition only  ({label})")
    # IMU stats are cheap (small file); always test them.
    rows = _train_partition_rows(class_filter, _VAL_SPLIT, _SPLIT_SEED)
    X = np.load(DATA_DIR / "train_imu.npy", mmap_mode="r")
    mean_part, std_part = _train_stats("imu", class_filter, _VAL_SPLIT, _SPLIT_SEED)
    mean_full = np.asarray(X).mean(axis=(0, 1))
    # Partition stats should differ from full-file stats (val excluded), but only slightly.
    diff = float(np.abs(mean_part - mean_full).max())
    check("partition IMU mean differs from full-file mean (val truly excluded)", diff > 0.0,
          f"max|Δmean|={diff:.3e}")

    # Test split must reuse the TRAIN-partition stats, not its own.
    tr = IMUDataset(split="train", class_filter=class_filter)
    te = IMUDataset(split="test", class_filter=class_filter)
    check("test set normalized with train-partition stats",
          np.allclose(tr.mean, te.mean) and np.allclose(tr.std, te.std))

    # Feature stats path (used by stats-MLP family) also restricted to the partition.
    fm, fs = _train_feature_stats("imu", class_filter, _VAL_SPLIT, _SPLIT_SEED)
    check("IMU feature stats computed (finite, positive std)",
          np.isfinite(fm).all() and (fs > 0).all(), f"F={len(fm)}")

    if full and class_filter is None:
        wm, ws = _train_stats("wifi", None, _VAL_SPLIT, _SPLIT_SEED)
        check("27-class WiFi partition stats computed (cache primed)",
              np.isfinite(wm).all() and (ws > 0).all(), f"C={len(wm)}")


def test_downsample():
    print("\n[A0.2] WiFi 500 -> 150 adaptive avg pooling")
    const = np.full((500, 224), 3.14, dtype=np.float32)
    out = _downsample_to(const, 150)
    check("downsample shape (500,224) -> (150,224)", out.shape == (150, 224))
    check("constant window preserved by average pooling", np.allclose(out, 3.14, atol=1e-5))
    ramp = np.linspace(0, 1, 500, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32)
    out_r = _downsample_to(ramp, 150)
    check("ramp mean preserved (≈0.5)", abs(float(out_r.mean()) - 0.5) < 1e-3,
          f"mean={float(out_r.mean()):.4f}")


def test_fusion_forward(class_filter, num_classes, label):
    print(f"\n[A0.2] Fusion datasets + models forward at T=150  ({label})")
    train_loader, _ = get_dataloader(
        "gmu_fusion", split="train", batch_size=4, val_split=_VAL_SPLIT,
        num_workers=0, seed=_SPLIT_SEED, class_filter=class_filter,
    )
    wifi, imu, y = next(iter(train_loader))
    check("WiFi batch is (B,150,224)", tuple(wifi.shape) == (4, 150, 224), str(tuple(wifi.shape)))
    check("IMU batch is (B,150,9) native", tuple(imu.shape) == (4, 150, 9), str(tuple(imu.shape)))

    with torch.no_grad():
        for name, model in [
            ("GMULateFusionModel", GMULateFusionModel(num_classes=num_classes)),
            ("LateFusionModel", LateFusionModel(num_classes=num_classes)),
            ("EarlyFusionModel", EarlyFusionModel(num_classes=num_classes)),
        ]:
            out = model(wifi, imu)
            check(f"{name} forward -> (B,{num_classes})", tuple(out.shape) == (4, num_classes),
                  str(tuple(out.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Also gather 27-class WiFi partition stats (~3 GB read; primes cache).")
    args = ap.parse_args()

    print("=" * 64)
    print("§A0 SMOKE TEST  (normalization partition + fusion realignment)")
    print("=" * 64)

    test_partition_equivalence(None, "27-class")
    test_partition_equivalence([0, 1, 2, 3, 4], "Easy 5-class")
    test_stats_exclude_val(None, "27-class", args.full)
    test_stats_exclude_val([0, 1, 2, 3, 4], "Easy 5-class", args.full)
    test_downsample()
    test_fusion_forward([0, 1, 2, 3, 4], 5, "Easy 5-class")
    if args.full:
        test_fusion_forward(None, 27, "27-class")

    print("\n" + "=" * 64)
    n_ok = sum(_results)
    n_tot = len(_results)
    print(f"RESULT: {n_ok}/{n_tot} checks passed")
    print("=" * 64)
    sys.exit(0 if n_ok == n_tot else 1)


if __name__ == "__main__":
    main()
