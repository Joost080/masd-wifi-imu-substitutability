import torch.nn as nn


class DeepConvLSTM(nn.Module):
    """
    4 × Conv1d(64, k=5, ReLU, Dropout) → 2 × LSTM(128) → Linear(27)

    Per-proposal architecture for WiFi-only and IMU-only baselines.
    in_channels=224 for WiFi, in_channels=9 for IMU.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 27,
        conv_channels: int = 64,
        conv_kernel: int = 5,
        lstm_hidden: int = 128,
        conv_dropout: float = 0.5,
        lstm_dropout: float = 0.5,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            *[
                self._conv_block(
                    in_channels if i == 0 else conv_channels,
                    conv_channels,
                    conv_kernel,
                    conv_dropout,
                )
                for i in range(4)
            ]
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(lstm_hidden, num_classes)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, kernel: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, C) — permute to (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)     # back to (B, T, conv_channels)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])   # last timestep
        return self.head(x)
