"""
Loading/cleaning for Ponds_data.csv (~74,700 real rows, 3 stations, 20-min
cadence, Feb 2022-Jan 2023, includes a genuine 9-day severe anoxia crash at
station1 Feb 20 - Mar 1 2022).

Kept entirely separate from lstm_forecast.py / train_lstm.py (the Oman-
dataset pipeline behind the live app) per aquaguard_switch_training_data_
prompt.md: this is an ADDITIONAL model for comparison, not a replacement.

Data-quality findings (see conversation for the investigation):
- NITRATE(PPM)/PH/DO/MANGANESE(mg/l) are stored as strings because of a
  small number of Excel '#VALUE!' error cells (44-63 rows out of 74,796,
  i.e. well under 0.1%) - coerced to numeric, those rows become NaN and
  are dropped along with 38 fully-blank trailer rows at the file's end.
- `label` does NOT mean "crash" or "poor quality": label=1 rows have a
  HIGHER mean DO (11.96) than label=0 (10.50), and 87% of all label=1
  rows are concentrated in Station3 alone (station1/Station2 are ~4%
  label=1, Station3 is ~46%) rather than being spread proportionally -
  it looks like a station/sub-pond category, not a water-quality signal.
  Its meaning could not be confidently determined, so it is EXCLUDED.
- Station names are inconsistently cased ("station1" vs "Station2") -
  normalized to lowercase.
- 20-minute native cadence per station; gaps exist (largest ~20h) but
  are small relative to the dataset - see find_contiguous_runs.
"""
from __future__ import annotations

import pandas as pd

RAW_PATH = "Ponds_data.csv"
NATIVE_STEP_MINUTES = 20

RAW_NUMERIC_COLS = ["NITRATE(PPM)", "PH", "AMMONIA(mg/l)", "TEMP", "DO", "TURBIDITY", "MANGANESE(mg/l)"]


def load_clean() -> pd.DataFrame:
    """Returns one cleaned frame with columns:
    station, timestamp, do_mean, temp_mean, ph_mean (+ raw nitrate/turbidity/
    manganese/ammonia kept for reference, not used as model features here to
    keep feature parity with the Oman-trained model for a fair comparison)."""
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = df.dropna(subset=["station"]).copy()
    df["station"] = df["station"].str.strip().str.lower()

    for col in RAW_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    before = len(df)
    df = df.dropna(subset=["timestamp", "DO", "TEMP", "PH"]).copy()
    dropped = before - len(df)
    print(f"pondsdata_prep: dropped {dropped}/{before} rows (parse failures / #VALUE! / blank trailer rows)")

    df = df.rename(columns={"DO": "do_mean", "TEMP": "temp_mean", "PH": "ph_mean"})
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    return df[["station", "timestamp", "do_mean", "temp_mean", "ph_mean"]]


def add_do_slope(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-station rate of change, same feature as lstm_forecast.py's do_slope."""
    frame = frame.copy()
    frame["do_slope"] = frame.groupby("station")["do_mean"].diff().fillna(0.0)
    return frame


def find_contiguous_runs(frame: pd.DataFrame, step_minutes: int = NATIVE_STEP_MINUTES) -> list[tuple[int, int]]:
    """(start, end) index pairs (inclusive) of runs with exact native-cadence
    spacing, for a single-station frame already sorted by timestamp. Same
    discipline as holdout.py's find_contiguous_runs - don't trust a held-out
    window without confirming it's real and gap-free."""
    deltas = frame["timestamp"].diff().dt.total_seconds()
    gap_starts = deltas[deltas > step_minutes * 60].index.tolist()
    frame = frame.reset_index(drop=True)
    deltas = frame["timestamp"].diff().dt.total_seconds()
    gap_starts = deltas[deltas > step_minutes * 60].index.tolist()
    bounds = [0] + gap_starts + [len(frame)]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def get_holdout_window(frame: pd.DataFrame, lookback_steps: int, horizon_steps: int, min_split_date=None) -> dict:
    """Same discipline as holdout.py's get_holdout_window, generalized for
    arbitrary lookback/horizon/cadence. `frame` must already be one station's
    data, sorted by timestamp, reset_index(drop=True).

    min_split_date: if given, the split index is pushed forward to at least
    this date - used to guarantee a known event (e.g. the station1 crash)
    falls in the TRAINING portion, never in the held-out target."""
    split = int(len(frame) * 0.8)
    if min_split_date is not None:
        min_idx = int((frame["timestamp"] >= pd.Timestamp(min_split_date)).idxmax())
        split = max(split, min_idx)

    # A run doesn't need to *start* at/after split - it just needs enough
    # room after split within it. (A long run spanning the split point is
    # common here, unlike the shorter Oman dataset this logic was first
    # written for - a run starting well before split but ending well after
    # it should still be usable for the target.)
    runs = find_contiguous_runs(frame)
    target_run = None
    target_start = None
    for r in runs:
        candidate_start = max(split, r[0])
        if r[1] - candidate_start + 1 >= horizon_steps:
            target_run = r
            target_start = candidate_start
            break
    if target_run is None:
        raise SystemExit(f"no gap-free run with {horizon_steps} steps of room after split {split}")
    target = frame.iloc[target_start: target_start + horizon_steps].reset_index(drop=True)

    # Context: real, contiguous history immediately before the target start -
    # either earlier in the same run (if target_start > run start), or the
    # nearest earlier run long enough, same discipline as holdout.py.
    if target_start - target_run[0] >= lookback_steps:
        context = frame.iloc[:target_start].reset_index(drop=True)
    else:
        target_pos = runs.index(target_run)
        context_run = next(
            (r for r in reversed(runs[:target_pos]) if (r[1] - r[0] + 1) >= lookback_steps), None
        )
        if context_run is None:
            raise SystemExit(f"no run of {lookback_steps} steps found before the target")
        context = frame.iloc[: context_run[1] + 1].reset_index(drop=True)

    return {
        "context": context,
        "target": target,
        "split": split,
        "window_start": str(target["timestamp"].iloc[0]),
        "window_end": str(target["timestamp"].iloc[-1]),
        "context_end_ts": context["timestamp"].iloc[-1],
    }


if __name__ == "__main__":
    df = load_clean()
    print(df.groupby("station").agg(rows=("timestamp", "size"), start=("timestamp", "min"), end=("timestamp", "max")))
    df = add_do_slope(df)
    for st in df["station"].unique():
        sub = df[df["station"] == st].reset_index(drop=True)
        runs = find_contiguous_runs(sub)
        longest = max(runs, key=lambda r: r[1] - r[0])
        print(f"{st}: {len(runs)} contiguous runs, longest = {longest[1]-longest[0]+1} rows "
              f"({sub['timestamp'].iloc[longest[0]]} -> {sub['timestamp'].iloc[longest[1]]})")
