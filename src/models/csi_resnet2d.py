"""
2D ResNet for WiFi CSI (§C / supervisor "if time allows").

The SOTA WiFi-HAR approach treats the CSI amplitude matrix as a
(time x subcarrier) image and applies a 2D CNN; ResNet-18 reaches 98.11% on
UT-HAR this way. This is the first SOTA-architecture WiFi baseline on MASD. It
is expected to stay at/below SKELAR's 55-66% Easy ceiling and not change the
Medium/Hard conclusion, but it closes the architecture gap cleanly.

Input convention matches the other models: x is (B, T, S) = (B, 500, 224); we
add a singleton channel -> (B, 1, T, S) and run a ResNet-18-style 2D CNN.
`stem_channels` defaults small (the Easy 5-class train set is ~1.9k windows, so
a full-width ResNet-18 would overfit); raise it for a larger-capacity baseline.
"""

import torch
import torch.nn as nn


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CSIResNet2D(nn.Module):
    """ResNet-18-style 2D CNN over the CSI (time x subcarrier) amplitude image."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        stem_channels: int = 32,
        layers_per_stage: tuple = (2, 2, 2, 2),
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        widths = [stem_channels, 2 * stem_channels, 4 * stem_channels, 8 * stem_channels]
        strides = [1, 2, 2, 2]
        self._in = stem_channels
        self.stages = nn.Sequential(*[
            self._make_stage(widths[i], layers_per_stage[i], strides[i]) for i in range(4)
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(widths[-1], num_classes)
        self._init_weights()

    def _make_stage(self, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock2d(self._in, out_channels, stride=stride)]
        self._in = out_channels
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock2d(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, T, S) -> (B, 1, T, S)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)
