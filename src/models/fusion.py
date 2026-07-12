import torch
import torch.nn as nn
from .deepconvlstm import DeepConvLSTM


class GatedMultimodalUnit(nn.Module):
    """Gated multimodal unit (Arevalo et al. 2017).

    gate_dim=1: scalar gate per timestep, broadcast over the hidden dim
        g_t = sigmoid(W_wifi * h_wifi + W_imu * h_imu)        (B, T, 1)
        Total gate params: d+1+d = 2d+1
    gate_dim=d: per-channel gate (the original GMU formulation)
        g_t = sigmoid(W_wifi * h_wifi + W_imu * h_imu)        (B, T, d)
        Total gate params: d*d+d+d*d = 2d^2+d
    h_t = g_t * h_wifi + (1 - g_t) * h_imu                    (B, T, d)
    """

    def __init__(self, d: int, gate_dim: int = 1):
        super().__init__()
        if gate_dim not in (1, d):
            raise ValueError(f"gate_dim must be 1 (scalar) or {d} (per-channel), got {gate_dim}")
        self.proj_wifi = nn.Linear(d, gate_dim, bias=True)
        self.proj_imu = nn.Linear(d, gate_dim, bias=False)

    def forward(self, h_wifi, h_imu):
        g = torch.sigmoid(self.proj_wifi(h_wifi) + self.proj_imu(h_imu))  # (B,T,1) or (B,T,d)
        return g * h_wifi + (1 - g) * h_imu, g


class EarlyFusionModel(nn.Module):
    """
    Concatenate WiFi (B, 500, 224) + upsampled IMU (B, 500, 9) → (B, 500, 233).
    Single shared DeepConvLSTM encoder.
    """

    def __init__(self, num_classes: int = 27, **kwargs):
        super().__init__()
        self.encoder = DeepConvLSTM(in_channels=233, num_classes=num_classes, **kwargs)

    def forward(self, wifi, imu):
        # wifi: (B, 500, 224), imu: (B, 500, 9) — imu must already be upsampled
        x = torch.cat([wifi, imu], dim=-1)  # (B, 500, 233)
        return self.encoder(x)


class LateFusionModel(nn.Module):
    """
    Modality-specific conv encoders → concat → shared 2-layer LSTM → head.

    Extends the FedOpenHAR data-type-specific layer concept to heterogeneous
    sensor fusion (WiFi CSI + IMU).
    """

    def __init__(
        self,
        num_classes: int = 27,
        conv_channels: int = 64,
        conv_kernel: int = 5,
        lstm_hidden: int = 128,
        conv_dropout: float = 0.5,
        lstm_dropout: float = 0.5,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.wifi_encoder = self._build_conv_encoder(224, conv_channels, conv_kernel, conv_dropout)
        self.imu_encoder = self._build_conv_encoder(9, conv_channels, conv_kernel, conv_dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels * 2,   # 64 + 64
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(lstm_hidden, num_classes)

    @staticmethod
    def _build_conv_encoder(in_channels, conv_channels, conv_kernel, dropout):
        layers = []
        for i in range(4):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels if i == 0 else conv_channels,
                        conv_channels,
                        conv_kernel,
                        padding=conv_kernel // 2,
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, wifi, imu):
        # wifi: (B, T, 224), imu: (B, T, 9)
        w = self.wifi_encoder(wifi.permute(0, 2, 1)).permute(0, 2, 1)   # (B, T, 64)
        m = self.imu_encoder(imu.permute(0, 2, 1)).permute(0, 2, 1)     # (B, T, 64)
        x, _ = self.lstm(torch.cat([w, m], dim=-1))                     # (B, T, 128)
        return self.head(self.dropout(x[:, -1, :]))


class GMULateFusionModel(nn.Module):
    """Late fusion with Gated Multimodal Unit (Arevalo et al. 2017, GMU).

    Replaces the concat in LateFusionModel with a scalar gate per timestep.
    LSTM input_size = conv_channels (64), not 2*conv_channels like concat.
    Per proposal §3.2 (configuration 5).
    """

    def __init__(
        self,
        num_classes: int = 27,
        conv_channels: int = 64,
        conv_kernel: int = 5,
        lstm_hidden: int = 128,
        conv_dropout: float = 0.5,
        lstm_dropout: float = 0.5,
        head_dropout: float = 0.5,
        gate_dim: int = 1,
    ):
        super().__init__()
        self.wifi_encoder = LateFusionModel._build_conv_encoder(224, conv_channels, conv_kernel, conv_dropout)
        self.imu_encoder = LateFusionModel._build_conv_encoder(9, conv_channels, conv_kernel, conv_dropout)
        if gate_dim == "channel":
            gate_dim = conv_channels
        self.gmu = GatedMultimodalUnit(conv_channels, gate_dim=gate_dim)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, wifi, imu, return_gate: bool = False):
        w = self.wifi_encoder(wifi.permute(0, 2, 1)).permute(0, 2, 1)   # (B, T, 64)
        m = self.imu_encoder(imu.permute(0, 2, 1)).permute(0, 2, 1)     # (B, T, 64)
        h, g = self.gmu(w, m)                                            # (B, T, 64), (B, T, 1)
        out, _ = self.lstm(h)                                            # (B, T, 128)
        logits = self.head(self.dropout(out[:, -1, :]))
        if return_gate:
            return logits, g
        return logits
