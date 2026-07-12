"""
Models for the stats-features side-experiment.

StatsMLP          — variant B, MLP on hand-crafted features only.
DeepConvLSTMStats — variant C, raw DeepConvLSTM ⊕ stats encoder, concat-and-head.
"""

import torch
import torch.nn as nn

from .deepconvlstm import DeepConvLSTM
from .fusion import GatedMultimodalUnit


class StatsMLP(nn.Module):
    """Small MLP for the stats-only baseline.

    Input: (B, F) feature vector. Output: (B, num_classes) logits.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 27,
        hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DeepConvLSTMStats(nn.Module):
    """Late fusion of a raw DeepConvLSTM trunk and a stats-MLP encoder.

    The raw trunk reuses DeepConvLSTM up to (but not including) its
    classification head, giving a (B, lstm_hidden) embedding. The stats
    encoder produces a (B, stats_embed) embedding. Concat → classifier.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 27,
        conv_channels: int = 64,
        conv_kernel: int = 5,
        lstm_hidden: int = 128,
        conv_dropout: float = 0.5,
        lstm_dropout: float = 0.5,
        head_dropout: float = 0.5,
        stats_in_features: int = 54,
        stats_hidden: int = 64,
    ):
        super().__init__()
        self.raw_trunk = DeepConvLSTM(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_channels=conv_channels,
            conv_kernel=conv_kernel,
            lstm_hidden=lstm_hidden,
            conv_dropout=conv_dropout,
            lstm_dropout=lstm_dropout,
            head_dropout=head_dropout,
        )
        # Replace the classifier head with identity; we'll keep the pooled
        # embedding and add our own joint head below.
        self.raw_trunk.head = nn.Identity()

        self.stats_encoder = nn.Sequential(
            nn.Linear(stats_in_features, stats_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(stats_hidden, stats_hidden),
            nn.ReLU(),
        )
        # No extra dropout here: the raw trunk already applies head_dropout
        # before its (now identity) head, and stats_encoder has its own dropout.
        self.head = nn.Linear(lstm_hidden + stats_hidden, num_classes)

    def forward(self, raw, stats):
        raw_emb = self.raw_trunk(raw)        # (B, lstm_hidden)
        stats_emb = self.stats_encoder(stats)  # (B, stats_hidden)
        return self.head(torch.cat([raw_emb, stats_emb], dim=-1))


class StatsStatsMLP(nn.Module):
    """Concat-and-MLP fusion of two stats feature vectors.

    Used for the stats-stats side-experiment: IMU stats (54-D) + WiFi CSI stats
    (33-D) -> 87-D concat -> small MLP -> logits. Tests whether weak CSI
    stats add complementary value to strong IMU stats in the simplest
    possible fusion setting.

    Designed for symmetry with StatsMLP (variant B): same backbone shape,
    just a wider input.
    """

    def __init__(
        self,
        imu_features: int = 54,
        wifi_features: int = 33,
        num_classes: int = 27,
        hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        in_features = imu_features + wifi_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, imu_stats, wifi_stats):
        return self.net(torch.cat([imu_stats, wifi_stats], dim=-1))


class GatedStatsFusion(nn.Module):
    """GMU-style gated fusion of two stats feature vectors.

    Projects each input vector to a shared `embed_dim` (no activation, matching
    the existing project convention that omits the Arevalo 2017 tanh -- see
    audit log entry 2026-05-08 minor deviation). The two projections are then
    fed to the existing `GatedMultimodalUnit`, so the gate semantics and the
    parameter accounting match the raw-modality GMU work in `fusion.py`:
      g = sigmoid(W_wifi h_wifi + W_imu h_imu)   (B, gate_dim)
      h = g * h_wifi + (1 - g) * h_imu           (B, embed_dim)
    Convention: gate near 1 -> WiFi-preferring, gate near 0 -> IMU-preferring,
    same as `GMULateFusionModel`.
    """

    def __init__(
        self,
        imu_features: int = 54,
        wifi_features: int = 33,
        embed_dim: int = 64,
        num_classes: int = 27,
        hidden: int = 128,
        dropout: float = 0.5,
        gate_dim=1,
    ):
        super().__init__()
        self.proj_imu = nn.Linear(imu_features, embed_dim)
        self.proj_wifi = nn.Linear(wifi_features, embed_dim)
        if gate_dim == "channel":
            gate_dim = embed_dim
        self.gmu = GatedMultimodalUnit(embed_dim, gate_dim=gate_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, imu_stats, wifi_stats, return_gate: bool = False):
        h_imu = self.proj_imu(imu_stats)
        h_wifi = self.proj_wifi(wifi_stats)
        h, g = self.gmu(h_wifi, h_imu)
        logits = self.head(h)
        if return_gate:
            return logits, g
        return logits
