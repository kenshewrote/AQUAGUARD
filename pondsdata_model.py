"""
Same dual-head architecture as lstm_forecast.py's DOForecastLSTM (main 6-hour
trajectory head + dedicated 1-hour head, do_slope feature), but with
horizon_steps as a constructor argument instead of a hardcoded module
constant - lstm_forecast.py's version bakes in HORIZON_STEPS=72 (5-min
steps, Oman dataset). Pondsdata's native cadence is 20 minutes, so its
horizon is 18 steps for the same 6 real hours. Kept as a separate class
so the live app's model code is never touched by this comparison work.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PondsDOForecastLSTM(nn.Module):
    def __init__(self, n_features: int, horizon_steps: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon_steps))
        self.head_1h = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        deltas = self.head(last_h)
        delta_1h = self.head_1h(last_h)
        last_do = x[:, -1, 0:1]
        return last_do + deltas, last_do + delta_1h


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, feats, lookback: int, horizon: int):
        self.feats = feats
        self.lookback = lookback
        self.horizon = horizon
        self.n = len(feats) - lookback - horizon + 1
        if self.n <= 0:
            raise ValueError("not enough rows to build training windows")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = self.feats[idx: idx + self.lookback]
        y = self.feats[idx + self.lookback: idx + self.lookback + self.horizon, 0]
        return torch.from_numpy(x), torch.from_numpy(y)
