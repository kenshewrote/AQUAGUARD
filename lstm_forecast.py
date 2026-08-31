"""
LSTM dissolved-oxygen forecaster for AquaGuard (Approach B).

Trains/infers on 5-minute multivariate windows (DO, temperature, pH),
then downsamples the 6-hour horizon to 6 hourly points for the existing API.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FEATURE_COLS = ["do_mean", "temp_mean", "ph_mean", "do_slope"]
STEP_MINUTES = 5
LOOKBACK_STEPS = 96  # 8 hours of 5-min history
HORIZON_STEPS = 72  # 6 hours ahead at 5-min
MIN_LOOKBACK_STEPS = 48  # require ~4 hours before padding is mandatory
HOURLY_POINTS = 6
STEPS_PER_HOUR = 60 // STEP_MINUTES

DO_ALIASES = ("do_mean", "do", "do_mgl", "dissolved oxygen (mg/l)", "dissolved_oxygen")
TEMP_ALIASES = ("temp_mean", "temp", "temp_c", "temperature", "temperature (c)")
PH_ALIASES = ("ph_mean", "ph", "pH")
TS_ALIASES = ("timestamp", "datetime", "time", "date")


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch in ("_", " "))


def _pick_column(columns, aliases) -> str | None:
    norm_map = {_norm(c): c for c in columns}
    for alias in aliases:
        key = _norm(alias)
        if key in norm_map:
            return norm_map[key]
    # substring fallback for dissolved oxygen style names
    for alias in aliases:
        key = _norm(alias).replace(" ", "").replace("_", "")
        for ncol, orig in norm_map.items():
            compact = ncol.replace(" ", "").replace("_", "")
            if key and key in compact:
                return orig
    return None


def _coalesce_numeric(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    """Combine all columns matching aliases (row-wise first non-null)."""
    matched = []
    for col in df.columns:
        ncol = _norm(col)
        compact = ncol.replace(" ", "").replace("_", "")
        for alias in aliases:
            a = _norm(alias)
            ac = a.replace(" ", "").replace("_", "")
            if ncol == a or (ac and ac in compact):
                matched.append(col)
                break
    if not matched:
        return pd.Series(np.nan, index=df.index, dtype=float)
    # preserve alias priority: earlier aliases win when multiple cols present
    prioritized = []
    for alias in aliases:
        a = _norm(alias)
        ac = a.replace(" ", "").replace("_", "")
        for col in matched:
            ncol = _norm(col)
            compact = ncol.replace(" ", "").replace("_", "")
            if ncol == a or compact == ac:
                if col not in prioritized:
                    prioritized.append(col)
    for col in matched:
        if col not in prioritized:
            prioritized.append(col)
    block = df[prioritized].apply(pd.to_numeric, errors="coerce")
    return block.bfill(axis=1).iloc[:, 0]


def rows_to_feature_frame(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed dashboard/sensor row dicts into FEATURE_COLS + timestamp."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if df.empty:
        raise ValueError("no rows provided")

    ts_col = _pick_column(df.columns, TS_ALIASES)

    out = pd.DataFrame(index=df.index)
    out["do_mean"] = _coalesce_numeric(df, DO_ALIASES)
    out["temp_mean"] = _coalesce_numeric(df, TEMP_ALIASES)
    out["ph_mean"] = _coalesce_numeric(df, PH_ALIASES)

    if out["do_mean"].isna().all():
        raise ValueError("could not find a dissolved-oxygen column")

    if ts_col:
        out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        out["timestamp"] = pd.date_range("2026-01-01", periods=len(out), freq=f"{STEP_MINUTES}min")

    out = out.dropna(subset=["do_mean", "timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # fill missing covariates with series medians (or sensible defaults)
    if out["temp_mean"].isna().all():
        out["temp_mean"] = 28.0
    else:
        out["temp_mean"] = out["temp_mean"].fillna(out["temp_mean"].median())
    if out["ph_mean"].isna().all():
        out["ph_mean"] = 7.5
    else:
        out["ph_mean"] = out["ph_mean"].fillna(out["ph_mean"].median())

    # Rate of change of DO, computed (not parsed from raw columns, so it works
    # for any input source: demo file, upload, or manual entry) so the model
    # gets an explicit "how fast is this falling right now" signal instead of
    # having to infer slope implicitly from the raw level trajectory. Added
    # after finding the model badly lagged sharp DO declines — it only flagged
    # danger once the CURRENT reading was already below critical, not ahead
    # of it, on a fast-crash demo file.
    out["do_slope"] = out["do_mean"].diff().fillna(0.0)

    return out[["timestamp"] + FEATURE_COLS]


def ensure_five_minute_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Resample arbitrary sensor cadences onto a regular 5-minute grid.
    Hourly AquaGuard demo/upload files become 12 interpolated steps per hour
    so the LSTM lookback (4–8 hours) has enough timesteps.
    """
    frame = rows_to_feature_frame(frame)
    if len(frame) == 0:
        return frame
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    if len(frame) == 1:
        return frame.reset_index()

    deltas = frame.index.to_series().diff().dt.total_seconds().dropna()
    median_sec = float(deltas.median()) if len(deltas) else STEP_MINUTES * 60

    # already near 5-minute cadence
    if 60 <= median_sec <= 10 * 60:
        grid = frame.resample(f"{STEP_MINUTES}min").mean().interpolate(limit_direction="both")
    else:
        # hourly (or coarser/irregular): build 5-min grid and interpolate
        grid = frame.resample(f"{STEP_MINUTES}min").mean().interpolate(limit_direction="both")

    grid = grid.dropna(subset=FEATURE_COLS).reset_index()
    return grid[["timestamp"] + FEATURE_COLS]


def downsample_to_hourly(
    times: pd.DatetimeIndex | np.ndarray | list,
    values: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, list]:
    """Collapse 5-minute horizon arrays into 6 hourly means for the dashboard API."""
    values = np.asarray(values, dtype=float)
    if len(values) != HORIZON_STEPS:
        raise ValueError(f"expected {HORIZON_STEPS} forecast steps, got {len(values)}")

    times = pd.to_datetime(pd.Index(times))
    if lower is None:
        lower = values
    if upper is None:
        upper = values
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    hourly_values, hourly_times, hourly_lower, hourly_upper = [], [], [], []
    for h in range(HOURLY_POINTS):
        start = h * STEPS_PER_HOUR
        end = start + STEPS_PER_HOUR
        hourly_values.append(float(values[start:end].mean()))
        hourly_lower.append(float(lower[start:end].mean()))
        hourly_upper.append(float(upper[start:end].mean()))
        # stamp each hour at the end of that hour window (matches +1h..+6h UX)
        hourly_times.append(str(times[end - 1]))

    return {
        "times": hourly_times,
        "values": hourly_values,
        "lower": hourly_lower,
        "upper": hourly_upper,
    }


class DOForecastLSTM(nn.Module):
    """Two heads share one LSTM backbone (single forward pass, not two trained
    models): `head` predicts the full 6-hour/72-step trajectory as before,
    and `head_1h` is a dedicated near-term output for the 1-hour-ahead value.

    Why a dedicated head instead of just reading the first 12 steps of
    `head`'s output: with a single 72-way MSE loss, gradient signal for the
    near-term steps gets averaged in with 71 other steps, so nothing forces
    the model to prioritize 1-hour accuracy specifically. `head_1h` gets its
    own loss term in train_lstm.py, giving the near-term signal dedicated
    gradient — this is what makes it an honestly higher-confidence signal
    rather than just a relabeled slice of the same trajectory.

    Tried and reverted: additive (Bahdanau-style) attention over the LSTM's
    per-timestep outputs, meant to let the model weight less-recent parts of
    the 8-hour lookback more heavily (aquaguard_research_backed_improvements_
    prompt.md step 3, citing Yang/Liu/Gao 2023 and the IPSO-CNN-GRU-TAM
    Eagle Mountain Lake study). Across 3 independent retrains on this ~900-row
    dataset it measured WORSE than this dual-head-only version (MAE 0.577/
    0.587/0.652, mean ~0.61) with higher run-to-run variance, versus this
    version's tight 0.567/0.572 (mean ~0.57) on the same held-out window —
    the extra attention parameters add capacity this small a dataset can't
    reliably fit. Reverted per the project's own discipline: don't keep an
    addition that measures worse. Revisit if the training set grows a lot
    (more seasons/ponds) — the working attention code is in git history.
    """

    def __init__(self, n_features: int = 3, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, HORIZON_STEPS),
        )
        self.head_1h = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, lookback, features) — feature 0 is scaled DO
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        # Predict residual from the latest DO so short crash windows stay grounded
        deltas = self.head(last_h)
        delta_1h = self.head_1h(last_h)
        last_do = x[:, -1, 0:1]
        return last_do + deltas, last_do + delta_1h


class LSTMForecaster:
    """Load a trained checkpoint and produce dashboard-compatible hourly forecasts."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        meta_path = os.path.join(model_dir, "meta.json")
        weights_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(meta_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"LSTM model not found in {model_dir} — run train_lstm.py first")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feature_mean = np.asarray(self.meta["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(self.meta["feature_std"], dtype=np.float32)
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        self.residual_std = float(self.meta.get("residual_std", 0.35))
        # Checkpoints trained before the dual-head change won't have this key or the
        # head_1h.* weights — fall back gracefully instead of serving a garbage 1h value.
        self.has_one_hour_head = "residual_std_1h" in self.meta
        self.residual_std_1h = float(self.meta.get("residual_std_1h", self.residual_std))
        self.lookback = int(self.meta.get("lookback_steps", LOOKBACK_STEPS))
        self.device = torch.device("cpu")

        self.model = DOForecastLSTM(
            n_features=len(FEATURE_COLS),
            hidden=int(self.meta.get("hidden", 64)),
            layers=int(self.meta.get("layers", 2)),
        )
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=self.has_one_hour_head)
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.feature_mean) / self.feature_std

    def _prepare_window(self, frame: pd.DataFrame, history_pad: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.Timestamp]:
        frame = ensure_five_minute_grid(frame)
        if history_pad is not None and not (isinstance(history_pad, pd.DataFrame) and history_pad.empty):
            pad = ensure_five_minute_grid(history_pad)
            need = max(0, self.lookback - len(frame))
            if need and len(pad):
                if len(frame):
                    first_ts = pd.Timestamp(frame["timestamp"].iloc[0])
                    chrono = pad[pad["timestamp"] < first_ts].tail(need)
                else:
                    chrono = pad.tail(need)
                    first_ts = pd.Timestamp(chrono["timestamp"].iloc[-1]) + pd.Timedelta(minutes=STEP_MINUTES)

                if len(chrono) < need:
                    # Demo/manual files often use a different calendar than the
                    # training history — stitch the latest history rows immediately
                    # before the user window so lookback is still dense.
                    extra = pad.tail(need).copy()
                    start = first_ts - pd.Timedelta(minutes=STEP_MINUTES * len(extra))
                    extra["timestamp"] = pd.date_range(start, periods=len(extra), freq=f"{STEP_MINUTES}min")
                    chrono = extra

                frame = pd.concat([chrono.tail(need), frame], ignore_index=True)

        if len(frame) < MIN_LOOKBACK_STEPS:
            raise ValueError(
                f"need at least ~{MIN_LOOKBACK_STEPS * STEP_MINUTES // 60} hours of history "
                f"({MIN_LOOKBACK_STEPS} steps), got {len(frame)}"
            )

        window = frame.tail(self.lookback).reset_index(drop=True)
        if len(window) < self.lookback:
            # left-pad by repeating the earliest row (cold start)
            pad_n = self.lookback - len(window)
            pad_rows = pd.concat([window.iloc[[0]]] * pad_n, ignore_index=True)
            window = pd.concat([pad_rows, window], ignore_index=True)

        feats = window[FEATURE_COLS].to_numpy(dtype=np.float32)
        last_ts = pd.Timestamp(window["timestamp"].iloc[-1])
        return self._scale(feats), last_ts

    @torch.inference_mode()
    def predict_frame(
        self,
        frame: pd.DataFrame | list[dict],
        history_pad: pd.DataFrame | None = None,
    ) -> dict[str, list]:
        x, last_ts = self._prepare_window(frame, history_pad=history_pad)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with self._lock:
            pred_full, pred_1h = self.model(tensor)
            pred = pred_full.cpu().numpy()[0]
            pred_1h_scaled = float(pred_1h.cpu().numpy()[0, 0])

        # predictions are in scaled DO space (feature 0); invert using DO mean/std
        do_mean = float(self.feature_mean[0])
        do_std = float(self.feature_std[0])
        pred_do = pred * do_std + do_mean

        future_times = pd.date_range(
            last_ts + pd.Timedelta(minutes=STEP_MINUTES),
            periods=HORIZON_STEPS,
            freq=f"{STEP_MINUTES}min",
        )
        # Widen the band the further out we forecast — a 1-hour-ahead point and a
        # 6-hour-ahead point shouldn't claim the same confidence. sqrt(step_number)
        # is the standard cheap proxy for how error accumulates over a horizon.
        band_floor = max(self.residual_std, 0.15)
        step_numbers = np.arange(1, HORIZON_STEPS + 1, dtype=np.float32)
        band = band_floor * np.sqrt(step_numbers)
        lower = np.maximum(pred_do - band, 0.0)  # dissolved oxygen can't be negative
        result = downsample_to_hourly(future_times, pred_do, lower, pred_do + band)

        if self.has_one_hour_head:
            # Dedicated near-term signal (see DOForecastLSTM docstring) — its own
            # band comes from its own validation residual, not a slice of the
            # 6-hour trajectory's band, so it's honestly tighter, not just relabeled.
            pred_1h_do = pred_1h_scaled * do_std + do_mean
            band_1h = max(self.residual_std_1h, 0.15)
            result["one_hour"] = {
                "time": str(future_times[STEPS_PER_HOUR - 1]),
                "value": pred_1h_do,
                "lower": max(pred_1h_do - band_1h, 0.0),
                "upper": pred_1h_do + band_1h,
            }
        return result


def find_latest_model_dir(root: str = "models/lstm") -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No LSTM model directory at {root}")
    candidates = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "model.pt"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No trained LSTM checkpoints in {root} — run train_lstm.py first")
    return sorted(candidates)[-1]
