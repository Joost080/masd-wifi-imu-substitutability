import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import upsample_imu, compute_imu_stats, normalize_imu, compute_wifi_stats, normalize_wifi


def test_upsample_imu_output_shape():
    X = np.random.randn(4, 150, 9).astype(np.float32)
    out = upsample_imu(X, target_len=500)
    assert out.shape == (4, 500, 9)


def test_upsample_imu_preserves_endpoints():
    X = np.random.randn(2, 150, 9).astype(np.float32)
    out = upsample_imu(X, target_len=500)
    np.testing.assert_allclose(out[:, 0, :], X[:, 0, :], rtol=1e-4)
    np.testing.assert_allclose(out[:, -1, :], X[:, -1, :], rtol=1e-4)


def test_normalize_imu_zero_mean():
    X = np.random.randn(100, 150, 9).astype(np.float32)
    stats = compute_imu_stats(X)
    X_norm = normalize_imu(X, stats)
    np.testing.assert_allclose(X_norm.mean(axis=(0, 1)), np.zeros(9), atol=1e-4)


def test_normalize_wifi_zero_mean():
    X = np.random.randn(50, 500, 224).astype(np.float32)
    stats = compute_wifi_stats(X)
    X_norm = normalize_wifi(X, stats)
    np.testing.assert_allclose(X_norm.mean(axis=(0, 1)), np.zeros(224), atol=1e-4)


def test_imu_stats_train_only_applied_to_test():
    """Stats from train must not be recomputed on test — this checks the pattern works."""
    X_train = np.random.randn(100, 150, 9).astype(np.float32) + 5.0
    X_test = np.random.randn(20, 150, 9).astype(np.float32) + 5.0
    stats = compute_imu_stats(X_train)
    X_test_norm = normalize_imu(X_test, stats)
    # Test set mean after normalization should be close to 0 (same distribution)
    assert X_test_norm.mean() < 1.0
