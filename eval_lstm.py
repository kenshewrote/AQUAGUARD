"""
LSTM-only evaluation on the shared gap-free held-out window (see holdout.py).
Used to compare before/after each step of aquaguard_research_backed_improvements_prompt.md
against the original baseline (MAE 0.98, RMSE 1.09) without re-loading Chronos each time.

Usage:
  python eval_lstm.py                       # evaluates the latest checkpoint
  python eval_lstm.py --model models/lstm/lstm_20260830_202921
"""
from __future__ import annotations

import argparse

import numpy as np

from holdout import DATA_PATH, directional_accuracy, get_holdout_window, load_frame, to_hourly_blocks
from lstm_forecast import LSTMForecaster, STEPS_PER_HOUR, find_latest_model_dir


def evaluate(model_dir: str) -> dict:
    frame = load_frame(DATA_PATH)
    window = get_holdout_window(frame)
    context, target = window["context"], window["target"]

    actual_do = target["do_mean"].to_numpy(dtype=float)
    actual_hourly = to_hourly_blocks(actual_do)
    anchor = float(context["do_mean"].tail(STEPS_PER_HOUR).mean())

    lstm = LSTMForecaster(model_dir)
    forecast = lstm.predict_frame(context)
    pred_hourly = np.asarray(forecast["values"], dtype=float)

    mae = float(np.mean(np.abs(pred_hourly - actual_hourly)))
    rmse = float(np.sqrt(np.mean((pred_hourly - actual_hourly) ** 2)))
    dir_acc = directional_accuracy(anchor, actual_hourly, pred_hourly)

    result = {"model_dir": model_dir, "mae": mae, "rmse": rmse, "dir_acc": dir_acc,
              "actual_hourly": actual_hourly, "pred_hourly": pred_hourly}

    # Auxiliary 1-hour head, once it exists (research-improvements step 1).
    if "one_hour" in forecast:
        actual_1h = actual_hourly[0]
        pred_1h = float(forecast["one_hour"]["value"])
        result["one_hour_abs_err"] = abs(pred_1h - actual_1h)
        result["one_hour_actual"] = actual_1h
        result["one_hour_pred"] = pred_1h

    return result


def report(result: dict, label: str = "") -> None:
    tag = f" ({label})" if label else ""
    print(f"6-hour trajectory{tag}: MAE={result['mae']:.3f}  RMSE={result['rmse']:.3f}  dir_acc={result['dir_acc']:.0%}")
    print("  hour  actual  pred")
    for h, (a, p) in enumerate(zip(result["actual_hourly"], result["pred_hourly"]), 1):
        print(f"  {h:>4}  {a:>6.2f}  {p:>6.2f}")
    if "one_hour_pred" in result:
        print(f"1-hour aux head{tag}: actual={result['one_hour_actual']:.2f}  pred={result['one_hour_pred']:.2f}  "
              f"abs_err={result['one_hour_abs_err']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="model checkpoint dir; defaults to latest under models/lstm")
    args = p.parse_args()
    model_dir = args.model or find_latest_model_dir("models/lstm")
    result = evaluate(model_dir)
    report(result, label=model_dir)
