"""
Evaluate a Pondsdata checkpoint on its station's held-out window (same
gap-free discipline as holdout.py/eval_lstm.py). For the combined model,
evaluate against EACH station's own held-out window so it's directly
comparable to that station's per-station model, hour for hour.

Usage:
  python eval_pondsdata.py --model models/pondsdata/station1_<stamp>
  python eval_pondsdata.py --model models/pondsdata/combined_<stamp> --station station1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pondsdata_model import PondsDOForecastLSTM
from pondsdata_prep import get_holdout_window, load_clean
from train_pondsdata import FEATURE_COLS, HORIZON_STEPS, LOOKBACK_STEPS, STEPS_PER_HOUR, CRASH_END, station_frame

HOURLY_POINTS = 6


def to_hourly_blocks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    assert len(values) == HORIZON_STEPS
    return values.reshape(HOURLY_POINTS, STEPS_PER_HOUR).mean(axis=1)


def directional_accuracy(anchor: float, actual: np.ndarray, pred: np.ndarray) -> float:
    actual_seq = np.concatenate([[anchor], actual])
    pred_seq = np.concatenate([[anchor], pred])
    correct, total = 0, 0
    for i in range(1, len(actual_seq)):
        td = actual_seq[i] - actual_seq[i - 1]
        pd_ = pred_seq[i] - pred_seq[i - 1]
        if abs(td) < 1e-6:
            continue
        total += 1
        if np.sign(td) == np.sign(pd_):
            correct += 1
    return correct / total if total else float("nan")


def evaluate(model_dir: str, station: str) -> dict:
    with open(f"{model_dir}/meta.json") as f:
        meta = json.load(f)
    feature_mean = np.asarray(meta["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(meta["feature_std"], dtype=np.float32)

    model = PondsDOForecastLSTM(n_features=len(FEATURE_COLS), horizon_steps=HORIZON_STEPS)
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu", weights_only=True))
    model.eval()

    raw = load_clean()
    frame = station_frame(raw, station)
    window = get_holdout_window(frame, LOOKBACK_STEPS, HORIZON_STEPS, min_split_date=CRASH_END)
    context, target = window["context"], window["target"]

    ctx_feats = context[FEATURE_COLS].to_numpy(dtype=np.float32)[-LOOKBACK_STEPS:]
    scaled = (ctx_feats - feature_mean) / feature_std
    x = torch.from_numpy(scaled).unsqueeze(0)
    with torch.inference_mode():
        pred_full, _ = model(x)
    pred = pred_full.numpy()[0] * feature_std[0] + feature_mean[0]
    pred_hourly = to_hourly_blocks(pred)

    actual_hourly = to_hourly_blocks(target["do_mean"].to_numpy(dtype=float))
    anchor = float(context["do_mean"].tail(STEPS_PER_HOUR).mean())

    mae = float(np.mean(np.abs(pred_hourly - actual_hourly)))
    rmse = float(np.sqrt(np.mean((pred_hourly - actual_hourly) ** 2)))
    dir_acc = directional_accuracy(anchor, actual_hourly, pred_hourly)
    return {
        "station": station, "model_dir": model_dir, "mae": mae, "rmse": rmse, "dir_acc": dir_acc,
        "window": f"{window['window_start']} -> {window['window_end']}",
        "actual_hourly": actual_hourly.tolist(), "pred_hourly": pred_hourly.tolist(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--station", required=True)
    args = p.parse_args()
    result = evaluate(args.model, args.station)
    print(f"{result['station']} ({result['model_dir']}): MAE={result['mae']:.3f}  RMSE={result['rmse']:.3f}  "
          f"dir_acc={result['dir_acc']:.0%}  window={result['window']}")
    for h, (a, pr) in enumerate(zip(result["actual_hourly"], result["pred_hourly"]), 1):
        print(f"  hour {h}: actual={a:.2f}  pred={pr:.2f}")
