"""
1D ResNet for WiFi CSI (Track 1C).

Motivation: the proposal explicitly lists "Repeating the WiFi-only baseline
with a CSI-native backbone such as ResNet-18 to quantify any gap due to
architecture choice" as future work (Section "Backbone choice for CSI"). The
SKELAR paper itself uses a ResNet backbone on MASD WiFi and reports 55%
weighted accuracy on the Easy 5-class subset, well above our DeepConvLSTM-on-CSI
audit number of 0.328. If a CSI-native backbone closes most of that gap, the
WiFi-related claims in the paper (CSI is intrinsically weak on MASD) need to
soften toward "CSI is weak under DeepConvLSTM." If the gap persists, the
modality-not-backbone claim is hardened.

Architecture: an 18-layer 1D ResNet adapted from He et al. 2016.
  Stem: Conv1d(in_channels -> stem_channels, kernel=7, stride=2, padding=3) + BN + ReLU
        MaxPool1d(kernel=3, stride=2, padding=1)
  4 stages of 2 BasicBlock1d each
    stage_channels = [stem, 2*stem, 4*stem, 4*stem]    (slightly smaller last
                                                        stage than full ResNet-18
                                                        for a 1-3M param budget)
    stage_strides  = [1, 2, 2, 2]
  AdaptiveAvgPool1d(1) -> Flatten -> Dropout(head_dropout) -> Linear(channels, num_classes)

Input convention matches DeepConvLSTM (and matches WiFi/IMU datasets):
  x: (B, T, C) -> permuted to (B, C, T) internally.

Param count at default settings (stem=64, in_channels=224, num_classes=27):
  ~2.8M, large enough to be a fair CSI-native baseline, small enough to stay
  one order of magnitude below the X-Fi reference (>10M).
"""

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    """2-conv residual block. Identity skip when in_channels == out_channels
    and stride == 1; otherwise a 1x1 conv on the skip path."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet1D(nn.Module):
    """1D ResNet-18-style backbone for sequence inputs (B, T, C)."""

    def __init__(
        self,
        in_channels: int = 224,
        num_classes: int = 27,
        stem_channels: int = 64,
        layers_per_stage: tuple = (2, 2, 2, 2),
        last_stage_channels: int | None = None,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        last_c = last_stage_channels if last_stage_channels is not None else 4 * stem_channels
        widths = [stem_channels, 2 * stem_channels, 4 * stem_channels, last_c]
        strides = [1, 2, 2, 2]
        self._in_channels = stem_channels
        stages = []
        for i, (w, s, n) in enumerate(zip(widths, strides, layers_per_stage)):
            stages.append(self._make_stage(w, n, s))
        self.stages = nn.Sequential(*stages)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(widths[-1], num_classes)

        self._init_weights()

    def _make_stage(self, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock1d(self._in_channels, out_channels, stride=stride)]
        self._in_channels = out_channels
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1d(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)          # (B, C, T)
        x = self.stem(x)                # (B, stem, T/4)
        x = self.stages(x)              # (B, last_c, T/32)
        x = self.global_pool(x).squeeze(-1)  # (B, last_c)
        x = self.dropout(x)
        return self.head(x)
