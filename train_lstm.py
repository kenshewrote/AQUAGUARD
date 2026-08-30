"""
Train an LSTM DO forecaster on the Nile Tilapia 5-minute aggregated dataset.

Usage (from predictive_wq/):
  python train_lstm.py
  python train_lstm.py --data "C:/path/to/aggregated_data.csv"
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lstm_forecast import (
    FEATURE_COLS,
    HORIZON_STEPS,
    LOOKBACK_STEPS,
    DOForecastLSTM,
    rows_to_feature_frame,
)

DEFAULT_DATA = os.environ.get("AQUAGUARD_DATA_PATH", os.path.join("data", "aggregated_data.csv"))


class WindowDataset(Dataset):
    def __init__(self, feats: np.ndarray, lookback: int, horizon: int):
        self.feats = feats
        self.lookback = lookback
        self.horizon = horizon
        self.n = len(feats) - lookback - horizon + 1
        if self.n <= 0:
            raise ValueError("not enough rows to build training windows")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = self.feats[idx : idx + self.lookback]
        # target: future DO only (feature index 0), already scaled with features
        y = self.feats[idx + self.lookback : idx + self.lookback + self.horizon, 0]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_features(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = rows_to_feature_frame(raw)
    # drop invalid windows if any
    return frame.dropna().reset_index(drop=True)


def train(args: argparse.Namespace) -> str:
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Pass --data pointing at aggregated_data.csv from DO-Forecasting-Tilapia-Dataset."
        )

    frame = load_features(args.data)
    print(f"Loaded {len(frame)} rows from {args.data}")

    values = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
    feature_mean = values.mean(axis=0)
    feature_std = values.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    scaled = (values - feature_mean) / feature_std

    # chronological split
    split = int(len(scaled) * 0.8)
    train_arr = scaled[:split]
    val_arr = scaled[split - LOOKBACK_STEPS :]  # overlap lookback into train for continuity

    train_ds = WindowDataset(train_arr, LOOKBACK_STEPS, HORIZON_STEPS)
    val_ds = WindowDataset(val_arr, LOOKBACK_STEPS, HORIZON_STEPS)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu")
    model = DOForecastLSTM(n_features=len(FEATURE_COLS), hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    residual_std = 0.35

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        abs_err = []
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * len(xb)
                # unscale DO channel for residual_std in mg/L
                pred_do = pred.cpu().numpy() * feature_std[0] + feature_mean[0]
                true_do = yb.cpu().numpy() * feature_std[0] + feature_mean[0]
                abs_err.append(np.abs(pred_do - true_do).ravel())
        val_loss /= max(len(val_ds), 1)
        if abs_err:
            residual_std = float(np.concatenate(abs_err).std())

        print(f"epoch {epoch:02d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  residual_std={residual_std:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"lstm_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "model.pt"))
    meta = {
        "feature_cols": FEATURE_COLS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "lookback_steps": LOOKBACK_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "hidden": args.hidden,
        "layers": args.layers,
        "residual_std": residual_std,
        "val_mse": best_val,
        "train_rows": int(split),
        "data_path": os.path.abspath(args.data),
        "created_utc": stamp,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # keep a local copy of training features for app history padding / series charts
    hist_out = os.path.join(args.out, "history_aggregated.csv")
    frame.to_csv(hist_out, index=False)
    print(f"Saved model to {out_dir}")
    print(f"Saved history copy to {hist_out}")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AquaGuard LSTM DO forecaster")
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to aggregated_data.csv")
    p.add_argument("--out", default="models/lstm", help="Output root for checkpoints")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    out = train(parse_args())
    print(out)
