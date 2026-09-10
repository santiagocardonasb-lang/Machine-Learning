import numpy as np
import pandas as pd

# Fixed seed: guarantees the CSV is always identical
rng = np.random.default_rng(7)

N_RECORDS = 600

# Units the product must sell in one month for the goal to be reached
MONTHLY_GOAL = 1000

# Independent variable (X): monthly advertising investment, millions of COP
investment = rng.uniform(0.5, 50.0, N_RECORDS).round(2)

# Same generative process as Activity 1: 120 base units + 38 units per million invested
noise = rng.normal(0, 150, N_RECORDS)
units = 120 + 38 * investment + noise

# Dependent variable (Y): 1 if the month reached the goal, 0 if it did not
goal_reached = (units >= MONTHLY_GOAL).astype(int)

df = pd.DataFrame({
    "advertising_investment": investment,
    "goal_reached": goal_reached,
})

df.to_csv("data/sales_goal.csv", index=False)

print(f"Dataset created: {len(df)} records in data/sales_goal.csv")
print()
print("Class balance:")
print(df["goal_reached"].value_counts().sort_index())