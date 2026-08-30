import numpy as np
import pandas as pd

np.random.seed(42)
hours = 24 * 60
time = np.arange(hours)

do_baseline = 6.5 + 1.5 * np.sin(2 * np.pi * time / 24)
temp = 26 + 3 * np.sin(2 * np.pi * time / (24 * 30)) + np.random.normal(0, 0.3, hours)

crash_points = np.random.choice(hours, size=8, replace=False)
do = do_baseline.copy()
for cp in crash_points:
    dip_length = np.random.randint(6, 15)
    for i in range(dip_length):
        if cp + i < hours:
            do[cp + i] -= (dip_length - i) * 0.35

do += np.random.normal(0, 0.15, hours)
ph = 7.5 + 0.3 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 0.05, hours)

df = pd.DataFrame({"hour": time, "temperature": temp, "ph": ph, "dissolved_oxygen": do})
df.to_csv("pond_data.csv", index=False)
print(df.head())
print(f"\nSaved {len(df)} rows to pond_data.csv")