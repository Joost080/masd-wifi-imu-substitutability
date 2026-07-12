import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.deepconvlstm import DeepConvLSTM
from src.models.fusion import EarlyFusionModel, LateFusionModel

B, T = 4, 500
C_WIFI, C_IMU = 224, 9


def test_deepconvlstm_wifi_output_shape():
    model = DeepConvLSTM(in_channels=C_WIFI)
    out = model(torch.randn(B, T, C_WIFI))
    assert out.shape == (B, 27)


def test_deepconvlstm_imu_output_shape():
    model = DeepConvLSTM(in_channels=C_IMU)
    out = model(torch.randn(B, T, C_IMU))
    assert out.shape == (B, 27)


def test_early_fusion_output_shape():
    model = EarlyFusionModel()
    wifi = torch.randn(B, T, C_WIFI)
    imu = torch.randn(B, T, C_IMU)
    out = model(wifi, imu)
    assert out.shape == (B, 27)


def test_late_fusion_output_shape():
    model = LateFusionModel()
    wifi = torch.randn(B, T, C_WIFI)
    imu = torch.randn(B, T, C_IMU)
    out = model(wifi, imu)
    assert out.shape == (B, 27)


def test_deepconvlstm_no_nan_on_random_input():
    model = DeepConvLSTM(in_channels=C_WIFI)
    out = model(torch.randn(B, T, C_WIFI))
    assert not torch.isnan(out).any()


def test_late_fusion_no_nan():
    model = LateFusionModel()
    out = model(torch.randn(B, T, C_WIFI), torch.randn(B, T, C_IMU))
    assert not torch.isnan(out).any()
