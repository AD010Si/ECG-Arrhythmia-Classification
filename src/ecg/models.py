"""Model zoo. All models are randomly initialised and trained from scratch."""
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Fully connected baseline on [signal | RR] features."""

    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _conv_encoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    )


class CNNOnly(nn.Module):
    """1D CNN over the raw 250-sample heartbeat window."""

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.cnn = _conv_encoder()
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.cnn(x))


class FusionCNN(nn.Module):
    """Two-branch network: CNN on the waveform + small MLP on RR-interval features.

    Input is the concatenated vector ``[signal (250) | pre_rr, post_rr, rr_ratio]``;
    the last ``n_rr`` columns are routed to the RR branch.
    """

    def __init__(self, n_classes: int = 4, n_rr: int = 3, head_hidden: int = 64):
        super().__init__()
        self.n_rr = n_rr
        self.cnn = _conv_encoder()
        self.rr = nn.Sequential(nn.Linear(n_rr, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(128 + 64, head_hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(head_hidden, n_classes)
        )

    def forward(self, x):
        x_sig, x_rr = x[:, :-self.n_rr], x[:, -self.n_rr:]
        sig_feat = self.cnn(x_sig.unsqueeze(1))
        return self.head(torch.cat([sig_feat, self.rr(x_rr)], dim=1))


def build_model(name: str, in_dim: int, n_classes: int = 4) -> nn.Module:
    if name == "mlp":
        return SimpleMLP(in_dim, n_classes=n_classes)
    if name == "cnn":
        return CNNOnly(n_classes)
    if name == "fusion":
        return FusionCNN(n_classes)
    raise ValueError(f"unknown model '{name}' (choose mlp | cnn | fusion)")
