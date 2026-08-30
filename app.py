from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import re
import time
import threading

from lstm_forecast import (
    FEATURE_COLS,
    LSTMForecaster,
    MIN_LOOKBACK_STEPS,
    ensure_five_minute_grid,
    find_latest_model_dir,
    rows_to_feature_frame,
)

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# Load the trained LSTM model once at startup.
# models/lstm/ is gitignored (regenerable via train_lstm.py) so a fresh clone
# or deployment has nothing there — fall back to deploy_model/, a small
# (~290KB) stable-named checkpoint that IS committed specifically so a
# deployed instance has something to load without needing to retrain.
# ============================================================
MODEL_PATH = None
for _root in ("models/lstm", "deploy_model"):
    try:
        MODEL_PATH = find_latest_model_dir(_root)
        break
    except FileNotFoundError:
        continue
if MODEL_PATH is None:
    raise RuntimeError("No trained LSTM checkpoint in models/lstm or deploy_model — run train_lstm.py first")
predictor = LSTMForecaster(MODEL_PATH)
print(f"Loaded LSTM model from {MODEL_PATH}")

predict_lock = threading.Lock()


def predict_with_retry(frame, history_pad=None, retries=2):
    """
    Serializes access to the model and retries once after reload if needed.
    `frame` is a DataFrame/list of sensor rows; returns hourly forecast dict.
    """
    global predictor
    last_err = None
    for attempt in range(retries + 1):
        try:
            with predict_lock:
                return predictor.predict_frame(frame, history_pad=history_pad)
        except RuntimeError as e:
            last_err = e
            if attempt < retries:
                print(f"predict failed ({e}), reloading model and retrying...")
                predictor = LSTMForecaster(MODEL_PATH)
                time.sleep(0.3)
                continue
            raise last_err


# ============================================================
# Historical pond series (Nile Tilapia aggregated, preferred)
# Falls back to IoTMLCQ workbook if present.
# ============================================================
HISTORY_CANDIDATES = [
    os.path.join("models", "lstm", "history_aggregated.csv"),
    os.path.join("deploy_model", "history_aggregated.csv"),
    os.environ.get("AQUAGUARD_DATA_PATH", os.path.join("data", "aggregated_data.csv")),
    "Data_Model_IoTMLCQ_2024.xlsx",
]

DATA_PATH = next((p for p in HISTORY_CANDIDATES if os.path.exists(p)), None)
if not DATA_PATH:
    raise RuntimeError(
        "No history dataset found. Run train_lstm.py first, or place "
        "aggregated_data.csv / Data_Model_IoTMLCQ_2024.xlsx in predictive_wq/."
    )

if DATA_PATH.endswith((".xlsx", ".xls")):
    df_raw = pd.read_excel(DATA_PATH)
else:
    df_raw = pd.read_csv(DATA_PATH)

df = rows_to_feature_frame(df_raw)
print(f"Loaded history from {DATA_PATH} ({len(df)} rows)")

TARGET_COL = "do_mean"
TIMESTAMP_COL = "timestamp"

SPECIES_THRESHOLDS = {
    "catfish": {"caution": 5.0, "critical": 3.0},
    "bass": {"caution": 6.0, "critical": 3.0},
    "tilapia": {"caution": 6.0, "critical": 5.0},
}

# in-memory ledger - resets when the server restarts (no database needed for the demo)
ledger = []


@app.route("/")
def index():
    return send_from_directory(".", "aquaguard_dashboard.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("assets", filename)


@app.route("/api/series")
def get_series():
    """
    Returns a window of actual history plus the model's forecast
    (mean + confidence band) for the dashboard chart.
    """
    lookback_hours = int(request.args.get("lookback", 72))
    with predict_lock:
        payload = predictor.predict_series_bundle(df, lookback_hours=lookback_hours)
    return jsonify(payload)


@app.route("/api/predict_latest")
def predict_latest():
    """
    Runs a fresh prediction on the most recent data, checks it against
    the selected species threshold, logs a ledger entry, and returns
    the decision - this is what the frontend polls for the live demo.
    """
    species = request.args.get("species", "catfish")
    th = SPECIES_THRESHOLDS.get(species, SPECIES_THRESHOLDS["catfish"])

    forecast = predict_with_retry(df)
    predicted_value = float(forecast["values"][0])
    predicted_time = forecast["times"][0]

    if predicted_value < th["critical"]:
        status = "danger"
        action = "Aerator ON - Feed reduced 50%"
    elif predicted_value < th["caution"]:
        status = "warn"
        action = "Monitoring"
    else:
        status = "safe"
        action = "No action"

    entry = {
        "time": predicted_time,
        "predicted": round(predicted_value, 2),
        "threshold": th["critical"],
        "status": status,
        "action": action,
        "species": species,
    }
    ledger.insert(0, entry)
    del ledger[8:]  # keep only the most recent 8 rows, matching the frontend

    return jsonify({"latest": entry, "ledger": ledger})


def find_column(columns, exact_names, contains_names=None):
    """
    Fuzzy-matches an uploaded file's column names against known aliases,
    e.g. "Dissolved Oxygen (mg/L)", "DO_mg_L", and "do" should all resolve
    to the same field. Exact (normalized) matches are tried first so a short
    alias like "do" or "ph" doesn't accidentally match a longer, unrelated
    column via substring search.
    """
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    norm_map = {c: norm(c) for c in columns}
    exact_norm = {norm(n) for n in exact_names}
    for orig, n in norm_map.items():
        if n in exact_norm:
            return orig
    if contains_names:
        contains_norm = [norm(n) for n in contains_names]
        for orig, n in norm_map.items():
            if any(cn in n for cn in contains_norm):
                return orig
    return None


def run_do_forecast(manual_rows, species, source, log_to_ledger=True):
    """
    manual_rows: chronologically-sorted list of dicts with at least
    timestamp + DO (+ optional temp/ph). Forecasts the next 6 hours and
    flags the first forecasted hour that crosses the species threshold.
    """
    th = SPECIES_THRESHOLDS.get(species, SPECIES_THRESHOLDS["catfish"])

    frame = rows_to_feature_frame(manual_rows)
    # Pad with pond history only when the user's own window is too short.
    # Longer sequences (e.g. 5 hourly readings) must drive the forecast alone
    # so a developing crash is not diluted by healthy historical padding.
    grid = ensure_five_minute_grid(frame)
    history_pad = df if len(grid) < MIN_LOOKBACK_STEPS else None
    # Pure LSTM forecast only — thresholds are applied afterward to label risk,
    # not mixed into the predicted DO values.
    forecast = predict_with_retry(frame, history_pad=history_pad)

    forecast_values = forecast["values"]
    forecast_times = forecast["times"]
    lower = forecast["lower"]
    upper = forecast["upper"]

    # Risk uses FUTURE predicted DO vs species lines (not "current DO already low").
    hours_until = None
    crash_value = None
    status = "safe"
    for i, v in enumerate(forecast_values):
        if v < th["critical"]:
            status = "danger"
            hours_until = i + 1
            crash_value = v
            break
        if v < th["caution"] and status == "safe":
            status = "warn"
            hours_until = i + 1
            crash_value = v

    if status == "danger":
        action = "Aerator ON - Feed reduced 50%"
    elif status == "warn":
        action = "Monitoring"
    else:
        action = "No action"

    # The 1-hour head is a dedicated, higher-confidence near-term signal (see
    # DOForecastLSTM in lstm_forecast.py) — used to CONFIRM a 6-hour crash
    # warning before the full aerator/feed response fires, not to replace the
    # 6-hour early-warning value. If the 6h trajectory says danger but the
    # confident near-term read doesn't yet show it, hold at standby instead of
    # firing the full action immediately; the 6h warning itself stays visible.
    one_hour = forecast.get("one_hour")
    if status == "danger" and one_hour is not None and one_hour["value"] >= th["caution"]:
        action = "Standby - 6h warning active, 1h signal not yet confirming (monitor closely)"

    entry = {
        "time": forecast_times[(hours_until - 1) if hours_until else 0],
        "predicted": round(crash_value if crash_value is not None else forecast_values[0], 2),
        "threshold": th["critical"],
        "status": status,
        "action": action,
        "species": species,
        "source": source,
    }
    if log_to_ledger:
        ledger.insert(0, entry)
        del ledger[8:]

    forecast_out = {"times": forecast_times, "values": forecast_values, "lower": lower, "upper": upper}
    if one_hour is not None:
        forecast_out["one_hour"] = one_hour
    crash_out = {
        "status": status,
        "hours_until": hours_until,
        "predicted_value": round(crash_value, 2) if crash_value is not None else None,
    }
    return forecast_out, crash_out


def find_episodes(rows, do_col, ts_col, th):
    """
    Groups consecutive non-safe readings (gap <= 2h) into distinct episodes,
    the same logic used to find real crash/caution windows in a dataset.
    """
    episodes = []
    cur = None
    for _, row in rows.iterrows():
        v = float(row[do_col])
        ts = row[ts_col]
        lvl = "danger" if v < th["critical"] else ("warn" if v < th["caution"] else "safe")
        if lvl != "safe":
            if cur and (ts - cur["end_ts"]) <= pd.Timedelta(hours=2):
                cur["end_ts"] = ts
                cur["min_do"] = min(cur["min_do"], v)
                cur["status"] = "danger" if (cur["status"] == "danger" or lvl == "danger") else "warn"
                cur["count"] += 1
            else:
                if cur:
                    episodes.append(cur)
                cur = {"start_ts": ts, "end_ts": ts, "min_do": v, "status": lvl, "count": 1}
        elif cur:
            episodes.append(cur)
            cur = None
    if cur:
        episodes.append(cur)

    return [{
        "start": str(e["start_ts"]),
        "end": str(e["end_ts"]),
        "min_do": round(e["min_do"], 2),
        "status": e["status"],
        "duration_hours": e["count"],
    } for e in episodes]


@app.route("/api/predict_manual", methods=["POST"])
def predict_manual():
    """
    Takes one or more manually-entered DO readings (treated as the most
    recent readings for the pond) and forecasts the next 6 hours.
    Temperature and pH are fed into the LSTM when provided.
    """
    body = request.get_json(force=True) or {}
    species = body.get("species", "catfish")
    readings = body.get("readings") or []

    if not readings:
        return jsonify({"error": "at least one reading is required"}), 400

    last_ts = df["timestamp"].max()
    manual_rows = []
    input_readings = []
    for i, r in enumerate(readings):
        try:
            do_val = float(r["do"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": f"reading {i} is missing a valid 'do' value"}), 400
        # manual entries are treated as successive 5-min steps ending "now"
        ts = last_ts + pd.Timedelta(minutes=5 * (i + 1))
        row = {
            "timestamp": ts,
            "do_mean": do_val,
            "temp_mean": r.get("temp"),
            "ph_mean": r.get("ph"),
        }
        manual_rows.append(row)
        input_readings.append({"time": str(ts), "do": do_val, "temp": r.get("temp"), "ph": r.get("ph")})

    forecast_out, crash_out = run_do_forecast(manual_rows, species, source="manual")

    return jsonify({
        "species": species,
        "input_readings": input_readings,
        "forecast": forecast_out,
        "crash": crash_out,
        "ledger": ledger,
    })


def predict_from_sensor_df(upload_df, species, filename, source, cutoff=None):
    """
    Shared core for /api/predict_upload and /api/predict_sample: given a
    raw sensor DataFrame, auto-detects the DO / timestamp / temperature /
    pH columns, forecasts the next 6 hours from the tail of the file, and
    scans the file for real historical crash/caution episodes.
    """
    th = SPECIES_THRESHOLDS.get(species, SPECIES_THRESHOLDS["catfish"])

    do_col = find_column(upload_df.columns, ["do", "domgl", "do_mean", "do_mgl"], ["dissolvedoxygen", "oxygen"])
    if not do_col:
        return jsonify({"error": "couldn't find a dissolved-oxygen column in this file"}), 400
    ts_col = find_column(upload_df.columns, ["datetime", "timestamp", "date", "time"], ["datetime", "timestamp"])
    temp_col = find_column(upload_df.columns, ["temp", "temperature", "temp_c", "temp_mean"], ["temperature"])
    ph_col = find_column(upload_df.columns, ["ph", "ph_mean"])

    upload_df[do_col] = pd.to_numeric(upload_df[do_col], errors="coerce")
    upload_df = upload_df.dropna(subset=[do_col]).copy()
    if upload_df.empty:
        return jsonify({"error": "no usable dissolved-oxygen readings found in this file"}), 400

    if ts_col:
        upload_df["_ts"] = pd.to_datetime(upload_df[ts_col], errors="coerce")
        upload_df = upload_df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    else:
        base = df["timestamp"].max()
        upload_df = upload_df.reset_index(drop=True)
        upload_df["_ts"] = [base + pd.Timedelta(minutes=5 * (i + 1)) for i in range(len(upload_df))]

    total_rows = len(upload_df)
    if cutoff is not None:
        cutoff = max(1, min(cutoff, total_rows))
        upload_df = upload_df.head(cutoff).reset_index(drop=True)

    upload_df = upload_df.tail(500).reset_index(drop=True)
    episodes = find_episodes(upload_df, do_col, "_ts", th)

    # Prefer ~8h of history when available (96 x 5-min); fall back to whatever remains.
    tail = upload_df.tail(96)
    manual_rows = []
    for _, row in tail.iterrows():
        manual_rows.append({
            "timestamp": row["_ts"],
            "do_mean": float(row[do_col]),
            "temp_mean": float(row[temp_col]) if temp_col and pd.notna(row[temp_col]) else None,
            "ph_mean": float(row[ph_col]) if ph_col and pd.notna(row[ph_col]) else None,
        })

    forecast_out, crash_out = run_do_forecast(manual_rows, species, source=source, log_to_ledger=(cutoff is None))

    input_readings = [{
        "time": str(row["_ts"]),
        "do": float(row[do_col]),
        "temp": float(row[temp_col]) if temp_col and pd.notna(row[temp_col]) else None,
        "ph": float(row[ph_col]) if ph_col and pd.notna(row[ph_col]) else None,
    } for _, row in tail.iterrows()]

    return jsonify({
        "species": species,
        "filename": filename,
        "rows_parsed": len(upload_df),
        "total_rows": total_rows,
        "cutoff_applied": cutoff,
        "columns_detected": {"do": do_col, "timestamp": ts_col, "temp": temp_col, "ph": ph_col},
        "input_readings": input_readings,
        "forecast": forecast_out,
        "crash": crash_out,
        "episodes": episodes,
        "ledger": ledger,
    })


@app.route("/api/predict_upload", methods=["POST"])
def predict_upload():
    """Accepts an uploaded CSV/XLSX of sensor readings instead of hand-typed rows."""
    species = request.form.get("species", "catfish")
    cutoff = request.form.get("cutoff", type=int)

    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "no file uploaded"}), 400
    f = request.files["file"]

    try:
        if f.filename.lower().endswith((".xlsx", ".xls")):
            upload_df = pd.read_excel(f)
        else:
            upload_df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"could not read file: {e}"}), 400

    return predict_from_sensor_df(upload_df, species, f.filename, source="upload", cutoff=cutoff)


SAMPLE_DIR = "sample_data"
SAMPLE_FILES = {
    "demo_01_safe_baseline.csv": {
        "label": "Safe Baseline",
        "description": "Steady healthy DO throughout — no crash predicted for any species.",
    },
    "demo_02_severe_crash_all_species.csv": {
        "label": "Severe Crash",
        "description": "Deep dip to ~1.6 mg/L — crashes catfish, bass, and tilapia alike.",
    },
    "demo_03_tilapia_only_crash.csv": {
        "label": "Tilapia-Only Crash",
        "description": "Dip to ~4.2 mg/L — crashes tilapia; only caution for catfish/bass.",
    },
    "demo_04_early_warning_caution.csv": {
        "label": "Early-Warning Caution",
        "description": "Mild dip to ~4.8 mg/L — caution for catfish/bass, a real crash for tilapia.",
    },
    "test_sensor_log.csv": {
        "label": "Original Crash Demo",
        "description": "The first crash-and-recover test file — dips to 2.0 mg/L.",
    },
}


@app.route("/api/sample_files")
def sample_files():
    """Lists the server-side demo CSVs available to test against, for the frontend's sample picker."""
    available = [
        {"filename": fname, **meta}
        for fname, meta in SAMPLE_FILES.items()
        if os.path.exists(os.path.join(SAMPLE_DIR, fname))
    ]
    return jsonify(available)


@app.route("/api/predict_sample", methods=["POST"])
def predict_sample():
    """Runs the same forecast/episode pipeline as an upload, but against a known server-side demo file."""
    body = request.get_json(force=True) or {}
    species = body.get("species", "catfish")
    filename = body.get("filename", "")
    cutoff = body.get("cutoff")

    if filename not in SAMPLE_FILES:
        return jsonify({"error": "unknown sample file"}), 400
    path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "sample file not found on server"}), 404

    upload_df = pd.read_csv(path)
    return predict_from_sensor_df(upload_df, species, filename, source="sample", cutoff=cutoff)


@app.route("/api/ledger")
def get_ledger():
    return jsonify(ledger)


if __name__ == "__main__":
    # NEVER let this default to True in a shared/production environment — Flask's
    # debug mode exposes an interactive Python debugger console over HTTP.
    # Set FLASK_DEBUG=1 in your local dev environment to turn it on.
    # host 0.0.0.0 + $PORT: required for hosting platforms (Render, Railway, etc.)
    # that assign the listen port via env var and proxy from outside the container.
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        use_reloader=False,
    )
