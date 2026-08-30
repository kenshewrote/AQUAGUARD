"""Tests for the LSTM DO forecast adapter (Approach B: 5-min model, hourly API)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_forecast import (
    FEATURE_COLS,
    HORIZON_STEPS,
    LOOKBACK_STEPS,
    STEP_MINUTES,
    downsample_to_hourly,
    ensure_five_minute_grid,
    rows_to_feature_frame,
)


def test_step_and_horizon_match_six_hour_api():
    assert STEP_MINUTES == 5
    assert HORIZON_STEPS * STEP_MINUTES == 6 * 60
    assert LOOKBACK_STEPS * STEP_MINUTES >= 4 * 60


def test_downsample_to_hourly_returns_six_points():
    times = pd.date_range("2026-04-01", periods=HORIZON_STEPS, freq="5min")
    values = np.linspace(7.0, 4.0, HORIZON_STEPS)
    out = downsample_to_hourly(times, values, lower=values - 0.2, upper=values + 0.2)
    assert len(out["times"]) == 6
    assert len(out["values"]) == 6
    assert len(out["lower"]) == 6
    assert len(out["upper"]) == 6
    assert out["values"][0] == pytest.approx(float(values[:12].mean()), rel=1e-5)
    assert out["values"][-1] == pytest.approx(float(values[-12:].mean()), rel=1e-5)


def test_rows_to_feature_frame_accepts_dashboard_aliases():
    rows = [
        {"timestamp": "2026-04-01T00:00:00", "do": 6.5, "temp": 28.0, "ph": 7.2},
        {"timestamp": "2026-04-01T00:05:00", "Dissolved Oxygen (mg/L)": 6.4, "Temperature (C)": 28.1, "pH": 7.1},
    ]
    frame = rows_to_feature_frame(rows)
    assert list(frame.columns) == ["timestamp"] + FEATURE_COLS
    assert frame["do_mean"].iloc[0] == pytest.approx(6.5)
    assert frame["temp_mean"].iloc[0] == pytest.approx(28.0)
    assert frame["ph_mean"].iloc[0] == pytest.approx(7.2)


def test_ensure_five_minute_grid_upsamples_hourly():
    rows = [
        {"timestamp": f"2026-04-01T{h:02d}:00:00", "do": 7.0 - 0.1 * h, "temp": 28.0, "ph": 7.2}
        for h in range(6)
    ]
    grid = ensure_five_minute_grid(rows)
    # 6 hourly anchors → 5-min grid spanning 5 hours = 61 points (inclusive), at least > 48
    assert len(grid) >= 48
    assert (grid["timestamp"].diff().dt.total_seconds().dropna() == STEP_MINUTES * 60).all()
