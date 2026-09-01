"""
Train the dual-head LSTM on Ponds_data.csv, either as one combined model
(all 3 stations pooled) or as 3 separate per-station models - per
aquaguard_switch_training_data_prompt.md step 2, both are trained and
compared, not chosen arbitrarily.

Usage:
  python train_pondsdata.py --mode combined
  python train_pondsdata.py --mode per_station
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
from torch.utils.data import ConcatDataset, DataLoader

from pondsdata_model import PondsDOForecastLSTM, WindowDataset
from pondsdata_prep import add_do_slope, get_holdout_window, load_clean

FEATURE_COLS = ["do_mean", "temp_mean", "ph_mean", "do_slope"]
STEP_MINUTES = 20
LOOKBACK_STEPS = 24  # 8 hours at 20-min cadence
HORIZON_STEPS = 18  # 6 hours at 20-min cadence
STEPS_PER_HOUR = 3
CRASH_END = "2022-03-02"  # station1 crash: 2022-02-20 -> 2022-03-01; split must be at/after this


def station_frame(all_df: pd.DataFrame, station: str) -> pd.DataFrame:
    sub = all_df[all_df["station"] == station].sort_values("timestamp").reset_index(drop=True)
    return add_do_slope(sub)


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def train_one(train_ds, val_ds, n_features: int, epochs: int, batch_size: int, lr: float):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PondsDOForecastLSTM(n_features=n_features, horizon_steps=HORIZON_STEPS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            y_1h = yb[:, :STEPS_PER_HOUR].mean(dim=1, keepdim=True)
            opt.zero_grad()
            pred_full, pred_1h = model(xb)
            loss = loss_fn(pred_full, yb) + loss_fn(pred_1h, y_1h)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                y_1h = yb[:, :STEPS_PER_HOUR].mean(dim=1, keepdim=True)
                pred_full, pred_1h = model(xb)
                val_loss += (loss_fn(pred_full, yb) + loss_fn(pred_1h, y_1h)).item() * len(xb)
        val_loss /= max(len(val_ds), 1)

        print(f"  epoch {epoch:02d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def save_checkpoint(model, feature_mean, feature_std, out_dir: str, extra_meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    meta = {
        "feature_cols": FEATURE_COLS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "lookback_steps": LOOKBACK_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "step_minutes": STEP_MINUTES,
        **extra_meta,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["combined", "per_station"], required=True)
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    raw = load_clean()
    stations = sorted(raw["station"].unique())
    frames = {st: station_frame(raw, st) for st in stations}
    windows = {st: get_holdout_window(frames[st], LOOKBACK_STEPS, HORIZON_STEPS, min_split_date=CRASH_END) for st in stations}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.mode == "per_station":
        for st in stations:
            frame = frames[st]
            split = windows[st]["split"]
            values = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
            feature_mean, feature_std = fit_scaler(values[:split])
            scaled = (values - feature_mean) / feature_std

            train_arr = scaled[:split]
            val_arr = scaled[split - LOOKBACK_STEPS:]
            train_ds = WindowDataset(train_arr, LOOKBACK_STEPS, HORIZON_STEPS)
            val_ds = WindowDataset(val_arr, LOOKBACK_STEPS, HORIZON_STEPS)

            print(f"=== {st}: train_rows={split} ===")
            model, best_val = train_one(train_ds, val_ds, len(FEATURE_COLS), args.epochs, 32, 1e-3)
            out_dir = f"models/pondsdata/{st}_{stamp}"
            save_checkpoint(model, feature_mean, feature_std, out_dir, {
                "station": st, "mode": "per_station", "val_mse": best_val,
                "train_rows": int(split), "crash_in_training": True,
            })

    else:  # combined
        train_values = np.concatenate(
            [frames[st][FEATURE_COLS].to_numpy(dtype=np.float32)[: windows[st]["split"]] for st in stations],
            axis=0,
        )
        feature_mean, feature_std = fit_scaler(train_values)

        train_datasets, val_datasets = [], []
        for st in stations:
            frame = frames[st]
            split = windows[st]["split"]
            values = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
            scaled = (values - feature_mean) / feature_std
            train_datasets.append(WindowDataset(scaled[:split], LOOKBACK_STEPS, HORIZON_STEPS))
            val_datasets.append(WindowDataset(scaled[split - LOOKBACK_STEPS:], LOOKBACK_STEPS, HORIZON_STEPS))

        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
        print(f"=== combined: pooled train_rows={sum(windows[st]['split'] for st in stations)} across {stations} ===")
        model, best_val = train_one(train_ds, val_ds, len(FEATURE_COLS), args.epochs, 32, 1e-3)
        out_dir = f"models/pondsdata/combined_{stamp}"
        save_checkpoint(model, feature_mean, feature_std, out_dir, {
            "stations": stations, "mode": "combined", "val_mse": best_val,
            "train_rows": int(sum(windows[st]["split"] for st in stations)), "crash_in_training": True,
        })


if __name__ == "__main__":
    main()
