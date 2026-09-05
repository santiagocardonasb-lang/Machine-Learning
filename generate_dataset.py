"""
Generates the advertising investment vs. units sold dataset.

Data source: synthetic data generated with numpy using a fixed seed (42),
simulating the monthly behavior of a consumer product. The relationship
between the variables is linear with random noise, which reproduces the
pattern observed in real marketing data.

Run this script once:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

# Fixed seed: guarantees the CSV is always identical
rng = np.random.default_rng(42)

N_RECORDS = 600

# Independent variable (X): monthly advertising investment, millions of COP
investment = rng.uniform(0.5, 50.0, N_RECORDS).round(2)

# Dependent variable (Y): units sold during the month
# True model: 120 base units + 38 extra units per million invested
noise = rng.normal(0, 150, N_RECORDS)
units = 120 + 38 * investment + noise

# Sales cannot be negative and are counted as whole units
units = np.maximum(units, 0).round().astype(int)

df = pd.DataFrame({
    "advertising_investment": investment,
    "units_sold": units,
})

df.to_csv("data/advertising_sales.csv", index=False)

print(f"Dataset created: {len(df)} records in data/advertising_sales.csv")
print(df.head())