"""
Step 2 of aquaguard_research_backed_improvements_prompt.md: before spending
effort sourcing solar radiation data (the "close spec gaps" prompt), check
whether the features already sitting in aggregated_data.csv (DO, temp, pH,
is_daylight, hour-of-day) actually predict DO for this specific pond's data.

Reference: "A hybrid XGBoost-ISSA-LSTM model for accurate short-term and
long-term dissolved oxygen prediction in ponds" — uses XGBoost feature
ranking to select inputs before feeding the LSTM, rather than assuming more
features automatically help.

Task: predict do_mean 1 hour ahead (12 x 5-min steps) from the CURRENT row's
features. This mirrors the near-term prediction the new 1-hour head targets.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from holdout import DATA_PATH
from lstm_forecast import STEPS_PER_HOUR

CANDIDATE_FEATURES = ["do_mean", "temp_mean", "ph_mean", "is_daylight", "hour", "hour_sin", "hour_cos"]


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=CANDIDATE_FEATURES + ["do_mean"]).sort_values("timestamp").reset_index(drop=True)

    horizon = STEPS_PER_HOUR  # 1 hour ahead at 5-min resolution
    X = raw[CANDIDATE_FEATURES].iloc[: -horizon].to_numpy(dtype=float)
    y = raw["do_mean"].iloc[horizon:].to_numpy(dtype=float)

    # chronological split, same discipline as train_lstm.py — no shuffling across time
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    print(f"XGBoost 1h-ahead DO prediction on held-out tail: MAE={mae:.3f} mg/L (sanity check, not the real benchmark)")
    print()

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    print("Feature importance for predicting DO 1 hour ahead (gain-based):")
    for i in order:
        bar = "#" * int(importances[i] * 60)
        print(f"  {CANDIDATE_FEATURES[i]:<12} {importances[i]:.3f}  {bar}")

    print()
    top = CANDIDATE_FEATURES[order[0]]
    daylight_rank = [CANDIDATE_FEATURES[i] for i in order].index("is_daylight") + 1
    hour_rank = min(
        [CANDIDATE_FEATURES[i] for i in order].index(f) + 1 for f in ("hour", "hour_sin", "hour_cos")
    )
    print(f"Most important feature: {top}")
    print(f"is_daylight rank: {daylight_rank} of {len(CANDIDATE_FEATURES)}")
    print(f"best hour-of-day proxy rank: {hour_rank} of {len(CANDIDATE_FEATURES)}")
    print(
        "Interpretation: is_daylight/hour-of-day are the cheap, already-available proxies for "
        "solar radiation's main effect (daytime photosynthesis raising DO). If they rank low here, "
        "that's evidence against prioritizing new solar-radiation sensor/data work for this pond — "
        "if they rank high, it's a case FOR it (a direct radiation reading would likely beat a "
        "day/night proxy)."
    )


if __name__ == "__main__":
    main()
