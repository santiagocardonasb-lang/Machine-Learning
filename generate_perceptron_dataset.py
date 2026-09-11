import numpy as np
import pandas as pd

rng = np.random.default_rng(13)

N_RECORDS = 600
BREAK_EVEN = 75

# Independent variables (X)
budget = rng.uniform(1.0, 40.0, N_RECORDS).round(2)   # millions of COP
duration = rng.integers(5, 61, N_RECORDS)             # days the campaign runs
channels = rng.integers(1, 7, N_RECORDS)              # number of channels used

# Every variable contributes to the result, plus factors nobody controls
noise = rng.normal(0, 12, N_RECORDS)
score = 1.8 * budget + 0.5 * duration + 6 * channels + noise

# Dependent variable (Y): 1 if the campaign was profitable, 0 if it was not
profitable = (score >= BREAK_EVEN).astype(int)

df = pd.DataFrame({
    "budget": budget,
    "duration_days": duration,
    "channels": channels,
    "profitable": profitable,
})

df.to_csv("data/campaign_profit.csv", index=False)

print(f"Dataset created: {len(df)} records in data/campaign_profit.csv")
print()
print("Class balance:")
print(df["profitable"].value_counts().sort_index())