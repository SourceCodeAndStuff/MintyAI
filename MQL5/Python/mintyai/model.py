"""MintyAI model: a dilated causal temporal convolutional network.

Deliberately built from ops that MetaTrader 5's bundled ONNX Runtime
handles without drama (Conv, BatchNormalization, Erf, Add, Mul, Gemm,
ReduceMean, Slice, Transpose, Softmax). No recurrent or attention layers:
a marginal accuracy gain is not worth an EA that fails to load a model on
a live account.

Input normalisation and the calibration temperature are baked into the
graph as constants, so there is no side-car JSON to drift out of sync
with the weights.

The graph has two outputs:

  probs [1, 3]  calibrated P(down), P(flat), P(up)
  risk  [1, 2]  mu and log-sigma of the bar's outcome in R units

The risk head exists because position size needs something the classifier
cannot express. Kelly wants edge / variance, and a class distribution over
three barrier outcomes says nothing about the dispersion of the result.
Predicting it, rather than exposing it as a knob, is what lets the EA size
itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn


# sigma lives in R units, where 1.0 is one full barrier. Below ~e^-4 the
# Kelly divide blows up; above ~e^1.5 the model is claiming a spread no
# barrier-bounded trade can produce.
LOG_SIGMA_MIN = -4.0
LOG_SIGMA_MAX = 1.5


class CausalBlock(nn.Module):
    """Dilated causal conv -> norm -> GELU -> pointwise conv, with residual."""

    def __init__(self, channels: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.point = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left-pad only: no timestep may see its own future
        y = Fn.pad(x, (self.pad, 0))
        y = self.conv(y)
        y = self.norm(y)
        y = Fn.gelu(y)
        y = self.point(y)
        y = self.drop(y)
        return x + y


class MintyAINet(nn.Module):
    def __init__(
        self,
        n_features: int,
        window: int,
        channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        self.n_features = n_features
        self.window = window
        self.n_classes = n_classes

        # folded-in preprocessing constants
        self.register_buffer("feat_mean", torch.zeros(1, n_features, 1))
        self.register_buffer("feat_std", torch.ones(1, n_features, 1))
        self.register_buffer("temperature", torch.ones(1))

        self.stem = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.ModuleList(
            [CausalBlock(channels, d, dropout=dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_classes),
        )
        # outcome head: [mu, log_sigma] of r = (exit - close) / barrier,
        # signed for a long. Shares the trunk, so it also acts as an
        # auxiliary task on the representation.
        self.risk_head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 2),
        )

    def set_normalisation(self, mean, std) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1)
        std_t = torch.clamp(std_t, min=1e-6)
        self.feat_mean.copy_(mean_t)
        self.feat_std.copy_(std_t)

    def set_temperature(self, t: float) -> None:
        self.temperature.fill_(max(float(t), 1e-3))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, W, F] of RAW features, exactly as MintyAIFeatures.mqh emits."""
        z = x.transpose(1, 2)  # -> [B, F, W]
        z = (z - self.feat_mean) / self.feat_std
        z = torch.clamp(z, -8.0, 8.0)

        z = self.stem(z)
        for block in self.blocks:
            z = block(z)

        last = z[:, :, -1]
        avg = z.mean(dim=2)
        return torch.cat([last, avg], dim=1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def risk(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 2] = (mu, log_sigma) of the long-side outcome in R units.

        log_sigma is clamped: an unclamped sigma can collapse toward zero on
        a quiet stretch, and the EA divides by its square to size a trade.
        """
        h = self.trunk(x)
        out = self.risk_head(h)
        mu = torch.clamp(out[:, 0:1], -3.0, 3.0)
        log_sigma = torch.clamp(out[:, 1:2], LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return torch.cat([mu, log_sigma], dim=1)

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Both heads off a single trunk pass - used in training."""
        h = self.trunk(x)
        out = self.risk_head(h)
        mu = torch.clamp(out[:, 0:1], -3.0, 3.0)
        log_sigma = torch.clamp(out[:, 1:2], LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return self.head(h), torch.cat([mu, log_sigma], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(probs [B, 3] = down/flat/up, risk [B, 2] = mu/log_sigma)."""
        logits, risk = self.heads(x)
        return torch.softmax(logits / self.temperature, dim=1), risk


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
