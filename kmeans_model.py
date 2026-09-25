from pathlib import Path

import classification  # imported first: it sets the headless matplotlib backend
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).parent / "data" / "seismic_events.csv"

FEATURES = ["magnitude", "depth_km"]

N_CLUSTERS = 3      # chosen by the silhouette score, see K_SELECTION below
RANDOM_STATE = 42   # fixes the initial centroids so the result is reproducible
N_INIT = 10         # runs the algorithm 10 times and keeps the best one
K_RANGE = range(2, 9)

CLUSTER_COLORS = ["#2f6f4e", "#c0562c", "#3a5f8a", "#8a6d3b", "#6b4e7d"]

CLUSTER_MEANING = {
    0: "Shallow crustal seismicity, associated with the fault systems. The most frequent "
       "and the least energetic.",
    1: "Deep subduction events. Few and far between, and only the larger ones are recorded, "
       "because seismic networks cannot detect small earthquakes at that depth.",
    2: "Intermediate depth seismicity, consistent with the Bucaramanga seismic nest.",
}

dataset = pd.read_csv(DATASET_PATH)

# ---- preprocessing ----
PREPROCESSING = {
    "initial_records": len(dataset),
    "missing_values": int(dataset.isna().sum().sum()),
    "duplicates_removed": int(dataset.duplicated().sum()),
}
# ---- preprocessing ----
PREPROCESSING = {
    "initial_records": len(dataset),
    "missing_values": int(dataset.isna().sum().sum()),
    "repeated_pairs": int(dataset.duplicated().sum()),
    "repeated_pairs_note": (
        "Rows sharing the same magnitude and depth were kept. In a seismic catalogue each "
        "event also has a date, a time and a location, so two distinct earthquakes can share "
        "these two values. Removing them would discard real observations."
    ),
    "final_records": len(dataset),
}
PREPROCESSING["final_records"] = len(dataset)

scaler = StandardScaler().fit(dataset[FEATURES])
points = scaler.transform(dataset[FEATURES])

PREPROCESSING["scaling"] = (
    "StandardScaler: each variable is transformed to mean 0 and standard deviation 1. "
    "Depth has a standard deviation of 162 km while magnitude has 0.95, and Euclidean "
    "distance adds both differences without weighting them, so without this step depth "
    "would decide every assignment on its own."
)

# ---- how many clusters ----
K_SELECTION = []
for k in K_RANGE:
    candidate = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(points)
    K_SELECTION.append({
        "k": k,
        "silhouette": round(float(silhouette_score(points, candidate.labels_)), 3),
        "inertia": round(float(candidate.inertia_), 1),
    })

BEST_K = max(K_SELECTION, key=lambda row: row["silhouette"])["k"]

# ---- final model ----
model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
labels = model.fit_predict(points)

dataset["cluster"] = labels

SILHOUETTE = round(float(silhouette_score(points, labels)), 3)

centroids_real = scaler.inverse_transform(model.cluster_centers_)

SUMMARY = []
for cluster in range(N_CLUSTERS):
    members = dataset[dataset["cluster"] == cluster]
    SUMMARY.append({
        "cluster": cluster,
        "records": int(len(members)),
        "share": round(100 * len(members) / len(dataset), 1),
        "centroid_magnitude": round(float(centroids_real[cluster][0]), 2),
        "centroid_depth": round(float(centroids_real[cluster][1]), 1),
        "magnitude_min": round(float(members["magnitude"].min()), 1),
        "magnitude_max": round(float(members["magnitude"].max()), 1),
        "depth_min": round(float(members["depth_km"].min()), 1),
        "depth_max": round(float(members["depth_km"].max()), 1),
        "meaning": CLUSTER_MEANING[cluster],
    })

INFO = {
    "records": len(dataset),
    "features": FEATURES,
    "n_clusters": N_CLUSTERS,
    "random_state": RANDOM_STATE,
    "n_init": N_INIT,
    "best_k": BEST_K,
    "silhouette": SILHOUETTE,
    "inertia": round(float(model.inertia_), 1),
    "source": (
        "Synthetic dataset generated with numpy using a fixed seed (21) by the "
        "generate_seismic_dataset.py script included in this repository. It reproduces "
        "three seismic populations documented in Colombia."
    ),
}


def _build_cluster_chart():
    figure, axes = plt.subplots(figsize=(9, 5.5))

    for cluster in range(N_CLUSTERS):
        members = dataset[dataset["cluster"] == cluster]
        axes.scatter(members["magnitude"], members["depth_km"],
                     s=18, alpha=0.45, color=CLUSTER_COLORS[cluster],
                     label=f"Cluster {cluster} ({len(members)} events)")

    for cluster, centroid in enumerate(centroids_real):
        axes.scatter(centroid[0], centroid[1], marker="X", s=300,
                     color=CLUSTER_COLORS[cluster], edgecolors="black", linewidths=1.4)

    axes.set_title(f"Seismic events grouped by K-Means (k = {N_CLUSTERS})")
    axes.set_xlabel("Magnitude (Richter scale)")
    axes.set_ylabel("Depth (km)")
    axes.invert_yaxis()   # depth grows downwards, as in a seismic cross section
    axes.legend(fontsize=9)
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return classification.figure_to_base64(figure)


def _build_selection_chart():
    figure, axes = plt.subplots(figsize=(9, 4.2))
    ks = [row["k"] for row in K_SELECTION]

    axes.plot(ks, [row["silhouette"] for row in K_SELECTION],
              marker="o", color="#2f6f4e", linewidth=2.2, label="Silhouette score")
    axes.axvline(BEST_K, color="#c0562c", linestyle="--", linewidth=1.4,
                 label=f"Best k = {BEST_K}")
    axes.set_title("Choosing the number of clusters")
    axes.set_xlabel("Number of clusters (k)")
    axes.set_ylabel("Silhouette score")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return classification.figure_to_base64(figure)


CLUSTER_CHART = _build_cluster_chart()
SELECTION_CHART = _build_selection_chart()

# Every record with its assigned cluster: the page shows the full table, not a sample
RECORD_ROWS = dataset.to_dict("records")

if __name__ == "__main__":
    print("Preprocessing:", PREPROCESSING)
    print("Best k       :", BEST_K)
    print("Silhouette   :", SILHOUETTE)
    print()
    for row in K_SELECTION:
        print(f'  k={row["k"]}  silhouette {row["silhouette"]}  inertia {row["inertia"]}')
    print()
    for row in SUMMARY:
        print(f'  cluster {row["cluster"]}: {row["records"]:4d} events '
              f'| centroid {row["centroid_magnitude"]} / {row["centroid_depth"]} km')