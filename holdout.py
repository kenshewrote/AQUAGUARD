"""
Shared held-out-window discipline for evaluating AquaGuard's DO forecaster.

Factored out of benchmark_lstm_vs_chronos.py so every evaluation script
(the Chronos comparison, and the research-backed improvement steps) uses
the exact same window, chosen the exact same way:

- The target (ground truth to score against) must be a real, gap-free run
  of HORIZON_STEPS 5-minute readings, entirely past the same chronological
  80/20 split train_lstm.py uses — so nothing in it was ever a training
  label. This dataset has genuine multi-day sensor-downtime gaps; a naive
  frame[split:split+72] slice can silently span one of them (an earlier
  version of this project did exactly that, producing a "6-hour window"
  that was actually 2.5 real days).
- The context is real, contiguous history immediately before the nearest
  long-enough clean run preceding the target, so the LSTM's 5-min grid
  resampler never has to interpolate fake data across a multi-day hole.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_forecast import HORIZON_STEPS, HOURLY_POINTS, LOOKBACK_STEPS, STEP_MINUTES, STEPS_PER_HOUR

DATA_PATH = "data/aggregated_data.csv"


def to_hourly_blocks(values: np.ndarray) -> np.ndarray:
    """Average consecutive STEPS_PER_HOUR-sized blocks -> HOURLY_POINTS values."""
    values = np.asarray(values, dtype=float)
    assert len(values) == HORIZON_STEPS, f"expected {HORIZON_STEPS} steps, got {len(values)}"
    return values.reshape(HOURLY_POINTS, STEPS_PER_HOUR).mean(axis=1)


def directional_accuracy(anchor: float, actual: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of steps where predicted direction (vs. the previous point)
    matches the true direction. Ties (|true delta| < 1e-6) are excluded."""
    actual_seq = np.concatenate([[anchor], actual])
    pred_seq = np.concatenate([[anchor], pred])
    correct, total = 0, 0
    for i in range(1, len(actual_seq)):
        true_delta = actual_seq[i] - actual_seq[i - 1]
        pred_delta = pred_seq[i] - pred_seq[i - 1]
        if abs(true_delta) < 1e-6:
            continue
        total += 1
        if np.sign(true_delta) == np.sign(pred_delta):
            correct += 1
    return correct / total if total else float("nan")


def find_contiguous_runs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """(start, end) index pairs (inclusive) of runs with exact 5-min spacing."""
    deltas = frame["timestamp"].diff().dt.total_seconds()
    gap_starts = deltas[deltas > STEP_MINUTES * 60].index.tolist()
    bounds = [0] + gap_starts + [len(frame)]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def load_frame(data_path: str = DATA_PATH) -> pd.DataFrame:
    from lstm_forecast import rows_to_feature_frame

    raw = pd.read_csv(data_path)
    return rows_to_feature_frame(raw).dropna().drop_duplicates("timestamp").reset_index(drop=True)


def get_holdout_window(frame: pd.DataFrame) -> dict:
    """Returns dict with context (DataFrame), target (DataFrame), split index,
    and the real elapsed-time gap between context end and target start."""
    split = int(len(frame) * 0.8)

    runs = find_contiguous_runs(frame)
    target_run = next((r for r in runs if r[0] >= split and (r[1] - r[0] + 1) >= HORIZON_STEPS), None)
    if target_run is None:
        raise SystemExit("no gap-free run of HORIZON_STEPS length found after the split")
    target = frame.iloc[target_run[0] : target_run[0] + HORIZON_STEPS].reset_index(drop=True)

    target_pos = runs.index(target_run)
    context_run = next(
        (r for r in reversed(runs[:target_pos]) if (r[1] - r[0] + 1) >= LOOKBACK_STEPS), None
    )
    if context_run is None:
        raise SystemExit("no run of LOOKBACK_STEPS length found before the target run")
    context = frame.iloc[: context_run[1] + 1].reset_index(drop=True)

    context_end_ts = context["timestamp"].iloc[-1]
    real_gap_hours = (target["timestamp"].iloc[0] - context_end_ts).total_seconds() / 3600

    return {
        "context": context,
        "target": target,
        "split": split,
        "context_end_ts": context_end_ts,
        "real_gap_hours": real_gap_hours,
        "window_start": str(target["timestamp"].iloc[0]),
        "window_end": str(target["timestamp"].iloc[-1]),
    }
