from chronos import ChronosPipeline
import torch
import numpy as np

print("Loading pretrained Chronos model...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.float32,
)
print("Model loaded successfully!")

# quick fake DO-like sequence: oscillating values, similar shape to real pond data
context = torch.tensor(6.5 + 1.5 * np.sin(np.linspace(0, 8 * np.pi, 168)))  # 7 days hourly

print("Running a forecast...")
forecast = pipeline.predict(
    context,
    prediction_length=6,  # predict 6 hours ahead
)

print("Forecast shape:", forecast.shape)
print("Predicted next 6 values:", forecast.median(dim=1).values)