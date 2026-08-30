"""
Head-to-head: LSTM (models/lstm) vs Chronos, on ONE held-out window that
neither model has trained on.

Data: data/aggregated_data.csv (the Nile Tilapia 5-minute dataset the LSTM
was trained on). The target window is chosen from the same chronological
80/20 region train_lstm.py holds out — nothing in it was ever used as a
training label — AND is required to be a real, gap-free run of 6 hours,
because this dataset has genuine multi-day sensor-downtime gaps. (An
earlier version of this script picked a naive frame[split:split+72] slice
that silently spanned a 2.5-day gap — see find_contiguous_runs.)

Both models get the same context and are scored against the same target,
downsampled to 6 hourly points (matching what the dashboard actually shows).

Caveat (stated up front, not buried): Chronos here is univariate (do_mean
only). The LSTM is multivariate (do_mean + temp_mean + ph_mean). That is a
structural handicap for Chronos as currently wired into this repo, not a
reflection of what a properly multivariate Chronos/Chronos-Bolt setup could
do. Numbers below should be read with that in mind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lstm_forecast import (
    FEATURE_COLS,
    HORIZON_STEPS,
    LOOKBACK_STEPS,
    STEP_MINUTES,
    STEPS_PER_HOUR,
    HOURLY_POINTS,
    LSTMForecaster,
    find_latest_model_dir,
    rows_to_feature_frame,
)

DATA_PATH = "data/aggregated_data.csv"


def to_hourly_blocks(values: np.ndarray) -> np.ndarray:
    """Average consecutive STEPS_PER_HOUR-sized blocks -> HOURLY_POINTS values."""
    values = np.asarray(values, dtype=float)
    assert len(values) == HORIZON_STEPS, f"expected {HORIZON_STEPS} steps, got {len(values)}"
    return values.reshape(HOURLY_POINTS, STEPS_PER_HOUR).mean(axis=1)


def directional_accuracy(anchor: float, actual_hourly: np.ndarray, pred_hourly: np.ndarray) -> float:
    """Fraction of the 6 hourly steps where predicted direction (vs. the
    previous point) matches the true direction. Ties (no real movement,
    |delta| < 1e-6) are excluded from both numerator and denominator."""
    actual_seq = np.concatenate([[anchor], actual_hourly])
    pred_seq = np.concatenate([[anchor], pred_hourly])
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
    """(start, end) index pairs (inclusive) of runs with exact 5-min spacing.
    This dataset has real sensor-downtime gaps (some 25-32 hours long) — a
    'held-out window' spanning one of those gaps would compare an hour-4
    prediction to a ground-truth reading from a day and a half later, which
    is not a meaningful metric. We only score against a run with no gaps."""
    deltas = frame["timestamp"].diff().dt.total_seconds()
    gap_starts = deltas[deltas > STEP_MINUTES * 60].index.tolist()
    bounds = [0] + gap_starts + [len(frame)]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    frame = rows_to_feature_frame(raw).dropna().drop_duplicates("timestamp").reset_index(drop=True)

    split = int(len(frame) * 0.8)  # identical split point to train_lstm.py: nothing at or
    # past this index was ever used as a training LABEL, so any target drawn from here on
    # is genuinely held out for the LSTM. (Chronos/AutoGluon-bolt are zero-shot here, so
    # they were never trained on this data at all.)

    runs = find_contiguous_runs(frame)
    target_run = next((r for r in runs if r[0] >= split and (r[1] - r[0] + 1) >= HORIZON_STEPS), None)
    if target_run is None:
        raise SystemExit("no gap-free run of HORIZON_STEPS length found after the split")
    target = frame.iloc[target_run[0] : target_run[0] + HORIZON_STEPS].reset_index(drop=True)

    # Context: real, contiguous history immediately before the *nearest* run of at
    # least LOOKBACK_STEPS length that precedes the target. The dataset has small
    # stub runs (5-6 rows) squeezed between multi-day gaps right before the target
    # run; using those as-is would force the LSTM's 5-min grid resampler to
    # interpolate clean-looking fake data across a >24h hole. Walking back to the
    # last long enough real run avoids inventing history that never happened.
    target_pos = runs.index(target_run)
    context_run = next(
        (r for r in reversed(runs[:target_pos]) if (r[1] - r[0] + 1) >= LOOKBACK_STEPS), None
    )
    if context_run is None:
        raise SystemExit("no run of LOOKBACK_STEPS length found before the target run")
    context = frame.iloc[: context_run[1] + 1].reset_index(drop=True)

    window_start = str(target["timestamp"].iloc[0])
    window_end = str(target["timestamp"].iloc[-1])
    context_end_ts = context["timestamp"].iloc[-1]
    real_gap_hours = (target["timestamp"].iloc[0] - context_end_ts).total_seconds() / 3600

    print(f"Target (ground truth, gap-free): {window_start} -> {window_end}  ({HORIZON_STEPS} x 5-min steps = 6h)")
    print(f"Context ends: {context_end_ts}  (last {LOOKBACK_STEPS} rows the LSTM actually sees are real & contiguous)")
    if real_gap_hours > 1:
        print(
            f"NOTE: the source dataset has a real ~{real_gap_hours:.1f}h sensor gap between context end and "
            "target start (no clean run bridges them). This is a next-72-real-readings forecast test, not a "
            "literal next-calendar-6-hours test — see the docstring at the top of this file."
        )
    print()

    actual_do = target["do_mean"].to_numpy(dtype=float)
    actual_hourly = to_hourly_blocks(actual_do)
    anchor = float(context["do_mean"].tail(STEPS_PER_HOUR).mean())  # last real hour, same anchor for both models

    # ---------------- LSTM ----------------
    model_dir = find_latest_model_dir("models/lstm")
    lstm = LSTMForecaster(model_dir)
    lstm_forecast = lstm.predict_frame(context)
    lstm_hourly = np.asarray(lstm_forecast["values"], dtype=float)

    lstm_mae = float(np.mean(np.abs(lstm_hourly - actual_hourly)))
    lstm_rmse = float(np.sqrt(np.mean((lstm_hourly - actual_hourly) ** 2)))
    lstm_dir_acc = directional_accuracy(anchor, actual_hourly, lstm_hourly)

    # ---------------- Chronos (zero-shot, univariate: do_mean only) ----------------
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    chronos_context = torch.tensor(context["do_mean"].to_numpy(dtype=float))
    chronos_out = pipeline.predict(chronos_context, prediction_length=HORIZON_STEPS)
    chronos_5min = chronos_out.median(dim=1).values[0].numpy()  # (HORIZON_STEPS,)
    chronos_hourly = to_hourly_blocks(chronos_5min)

    chronos_mae = float(np.mean(np.abs(chronos_hourly - actual_hourly)))
    chronos_rmse = float(np.sqrt(np.mean((chronos_hourly - actual_hourly) ** 2)))
    chronos_dir_acc = directional_accuracy(anchor, actual_hourly, chronos_hourly)

    # ---------------- AutoGluon / Chronos-Bolt (finetune_chronos.py's bolt_small preset) ----------------
    bolt_hourly = None
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        lookback_ctx = context.tail(LOOKBACK_STEPS).reset_index(drop=True)
        # AutoGluon's TimeSeriesDataFrame requires one unbroken regular-frequency
        # index per item. The real target run starts ~58h after context ends (a
        # genuine sensor gap — see the note above), so we re-stamp the target's
        # timestamps to continue immediately after the context, same as the LSTM's
        # own predict_frame already does internally. We still score against the
        # real DO VALUES; only the clock labels used to satisfy AutoGluon are fake.
        synth_target_ts = pd.date_range(
            lookback_ctx["timestamp"].iloc[-1] + pd.Timedelta(minutes=STEP_MINUTES),
            periods=HORIZON_STEPS,
            freq=f"{STEP_MINUTES}min",
        )
        train_tsdf = TimeSeriesDataFrame.from_data_frame(
            pd.DataFrame({"item_id": "pond1", "timestamp": lookback_ctx["timestamp"], "do_mean": lookback_ctx["do_mean"]}),
            id_column="item_id", timestamp_column="timestamp",
        )
        full_tsdf = TimeSeriesDataFrame.from_data_frame(
            pd.DataFrame({
                "item_id": "pond1",
                "timestamp": list(lookback_ctx["timestamp"]) + list(synth_target_ts),
                "do_mean": list(lookback_ctx["do_mean"]) + list(actual_do),
            }),
            id_column="item_id", timestamp_column="timestamp",
        )

        bolt_predictor = TimeSeriesPredictor(
            prediction_length=HORIZON_STEPS, target="do_mean", eval_metric="MASE", freq=f"{STEP_MINUTES}min",
        )
        bolt_predictor.fit(train_data=train_tsdf, presets="bolt_small", time_limit=180)
        leaderboard = bolt_predictor.leaderboard(full_tsdf)
        print("AutoGluon leaderboard (scored on the same held-out window):")
        print(leaderboard.to_string(index=False))
        print()

        bolt_preds = bolt_predictor.predict(train_tsdf)
        bolt_5min = bolt_preds["mean"].to_numpy(dtype=float)[:HORIZON_STEPS]
        bolt_hourly = to_hourly_blocks(bolt_5min)
        bolt_mae = float(np.mean(np.abs(bolt_hourly - actual_hourly)))
        bolt_rmse = float(np.sqrt(np.mean((bolt_hourly - actual_hourly) ** 2)))
        bolt_dir_acc = directional_accuracy(anchor, actual_hourly, bolt_hourly)
    except Exception as e:
        print(f"AutoGluon/Chronos-Bolt comparison skipped: {e}")
        bolt_mae = bolt_rmse = bolt_dir_acc = None

    # ---------------- Report ----------------
    header = f"{'hour':>5} {'actual':>8} {'LSTM':>8} {'chronos':>8}"
    if bolt_hourly is not None:
        header += f" {'ag-bolt':>8}"
    print(header)
    for h in range(HOURLY_POINTS):
        row = f"{h+1:>5} {actual_hourly[h]:>8.2f} {lstm_hourly[h]:>8.2f} {chronos_hourly[h]:>8.2f}"
        if bolt_hourly is not None:
            row += f" {bolt_hourly[h]:>8.2f}"
        print(row)
    print()
    print(f"{'model':<12} {'MAE (mg/L)':>12} {'RMSE (mg/L)':>13} {'dir. accuracy':>14}")
    print(f"{'LSTM':<12} {lstm_mae:>12.3f} {lstm_rmse:>13.3f} {lstm_dir_acc:>13.0%}")
    print(f"{'Chronos':<12} {chronos_mae:>12.3f} {chronos_rmse:>13.3f} {chronos_dir_acc:>13.0%}")
    if bolt_mae is not None:
        print(f"{'AG-Bolt':<12} {bolt_mae:>12.3f} {bolt_rmse:>13.3f} {bolt_dir_acc:>13.0%}")
    print()
    print("NOTE: both Chronos variants here are univariate (do_mean only) — neither was given")
    print("temp_mean or ph_mean, which the LSTM was. That is a structural handicap for Chronos")
    print("as currently wired into this repo, not a fair reflection of a multivariate setup.")
    print("NOTE: raw Chronos was fed the full available context (up to its own max window),")
    print("while the LSTM architecturally only ever looks at its fixed last-96-step lookback —")
    print("so 'context length seen' is not identical across models, only the scored target is.")
    print(f"Reproducibility: target {window_start} -> {window_end}, context ends {context_end_ts}, "
          f"split index {split} of {len(frame)}, data={DATA_PATH}")


if __name__ == "__main__":
    main()
