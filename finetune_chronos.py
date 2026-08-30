import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

CSV_PATH = "Data_Model_IoTMLCQ_2024.xlsx"
TARGET_COL = "Dissolved Oxygen (mg/L)"
TIMESTAMP_COL = "Datetime"
PREDICTION_LENGTH = 6

if CSV_PATH.endswith(".xlsx"):
    df = pd.read_excel(CSV_PATH)
else:
    df = pd.read_csv(CSV_PATH)

df["item_id"] = "pond1"

if TIMESTAMP_COL and TIMESTAMP_COL in df.columns:
    df["timestamp"] = pd.to_datetime(df[TIMESTAMP_COL])
else:
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="h")

ts_df = TimeSeriesDataFrame.from_data_frame(
    df[["item_id", "timestamp", TARGET_COL]],
    id_column="item_id",
    timestamp_column="timestamp",
)

train_data, test_data = ts_df.train_test_split(prediction_length=PREDICTION_LENGTH)

predictor = TimeSeriesPredictor(
    prediction_length=PREDICTION_LENGTH,
    target=TARGET_COL,
    eval_metric="MASE",
)

predictor.fit(
    train_data=train_data,
    presets="bolt_small",
    time_limit=600,
)

print("\nTraining complete.")
print(predictor.leaderboard(test_data))

predictor.save()
print("\nModel saved. Ready to wire into the dashboard.")