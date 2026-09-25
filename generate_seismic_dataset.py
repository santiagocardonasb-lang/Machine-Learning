import numpy as np
import pandas as pd

rng = np.random.default_rng(21)

N_SHALLOW, N_NEST, N_DEEP = 500, 450, 250

# Shallow crustal seismicity: very frequent, small magnitude
magnitude_shallow = rng.normal(2.70, 0.42, N_SHALLOW)
depth_shallow = rng.gamma(2.5, 7.5, N_SHALLOW) + 2.0

# Bucaramanga seismic nest: intermediate depth, moderate magnitude
magnitude_nest = rng.normal(3.90, 0.45, N_NEST)
depth_nest = rng.normal(155, 20, N_NEST)

# Deep subduction: only the larger events are detected at that depth
magnitude_deep = rng.normal(4.85, 0.55, N_DEEP)
depth_deep = rng.normal(450, 45, N_DEEP)

magnitude = np.clip(
    np.concatenate([magnitude_shallow, magnitude_nest, magnitude_deep]), 1.5, 7.5).round(1)
depth = np.clip(
    np.concatenate([depth_shallow, depth_nest, depth_deep]), 1.0, 700).round(1)

dataset = pd.DataFrame({"magnitude": magnitude, "depth_km": depth})

# Shuffled so the three populations are not stored in blocks: the order of the file must
# not carry information the algorithm could take advantage of
dataset = dataset.sample(frac=1, random_state=21).reset_index(drop=True)
dataset.to_csv("data/seismic_events.csv", index=False)

# A 100 record sample for the manual exercise, over the same phenomenon
sample = dataset.sample(n=100, random_state=7).reset_index(drop=True)
sample.to_csv("data/seismic_sample.csv", index=False)

print(f"Full dataset : {len(dataset)} records in data/seismic_events.csv")
print(f"Manual sample: {len(sample)} records in data/seismic_sample.csv")
print()
print(dataset.describe().round(2))