from pathlib import Path

import classification  # imported first: it sets the headless matplotlib backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).parent / "data" / "seismic_sample.csv"

FEATURES = ["magnitude", "depth_km"]
N_CLUSTERS = 3
N_ITERATIONS = 3

# Deliberately poor starting points, given in real units so they can be read directly.
# Good starting points would converge in a single iteration and there would be nothing
# to observe across the three required iterations.
INITIAL_CENTROIDS = np.array([
    [2.5, 50.0],
    [4.0, 100.0],
    [5.5, 300.0],
])

CLUSTER_COLORS = ["#2f6f4e", "#c0562c", "#3a5f8a"]

sample = pd.read_csv(DATASET_PATH)

# Preprocessing: depth spans 1 to 700 km while magnitude spans 2 to 6. Euclidean distance
# adds both differences, so without standardising, depth would decide every assignment and
# magnitude would be ignored
scaler = StandardScaler().fit(sample[FEATURES])
points = scaler.transform(sample[FEATURES])

def scale(values):
    """Standardises an array of points, keeping the feature names scikit-learn expects."""
    return scaler.transform(pd.DataFrame(values, columns=FEATURES))

def euclidean_distances(points, centroids):
    """Distance from every point to every centroid: sqrt of the sum of squared differences."""
    return np.sqrt(((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))


def within_cluster_variance(points, labels, centroids):
    """Sum of squared distances from each record to the centroid of its own cluster."""
    per_cluster = []
    for cluster in range(len(centroids)):
        members = points[labels == cluster]
        per_cluster.append(
            float(((members - centroids[cluster]) ** 2).sum()) if len(members) else 0.0)
    return sum(per_cluster), per_cluster


def build_chart(labels, centroids, title):
    """Scatter of the sample in real units, coloured by cluster, with the centroids."""
    figure, axes = plt.subplots(figsize=(8, 5))
    real_centroids = scaler.inverse_transform(centroids)

    if labels is None:
        axes.scatter(sample["magnitude"], sample["depth_km"],
                     s=45, color="#777777", alpha=0.6, label="Seismic events")
    else:
        for cluster in range(N_CLUSTERS):
            members = sample[labels == cluster]
            axes.scatter(members["magnitude"], members["depth_km"],
                         s=45, alpha=0.7, color=CLUSTER_COLORS[cluster],
                         label=f"Cluster {cluster}")

    for cluster, centroid in enumerate(real_centroids):
        axes.scatter(centroid[0], centroid[1], marker="X", s=300,
                     color=CLUSTER_COLORS[cluster], edgecolors="black", linewidths=1.4,
                     label=f"Centroid {cluster}")

    axes.set_title(title)
    axes.set_xlabel("Magnitude (Richter scale)")
    axes.set_ylabel("Depth (km)")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return classification.figure_to_base64(figure)


def run():
    """Runs the three iterations and collects everything the page has to display."""
    centroids = scale(INITIAL_CENTROIDS)
    iterations = []

    for number in range(1, N_ITERATIONS + 1):
        distances = euclidean_distances(points, centroids)
        labels = distances.argmin(axis=1)
        total_variance, variance_per_cluster = within_cluster_variance(points, labels, centroids)

        new_centroids = np.array([
            points[labels == cluster].mean(axis=0) if (labels == cluster).any()
            else centroids[cluster]
            for cluster in range(N_CLUSTERS)
        ])
        shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1))

        # One row per record: the three distances, the assignment, and the original values
        rows = []
        for index in range(len(sample)):
            rows.append({
                "id": index + 1,
                "magnitude": float(sample.loc[index, "magnitude"]),
                "depth_km": float(sample.loc[index, "depth_km"]),
                "d0": round(float(distances[index, 0]), 3),
                "d1": round(float(distances[index, 1]), 3),
                "d2": round(float(distances[index, 2]), 3),
                "cluster": int(labels[index]),
            })

        iterations.append({
            "number": number,
            "centroids_before": scaler.inverse_transform(centroids).round(2).tolist(),
            "centroids_after": scaler.inverse_transform(new_centroids).round(2).tolist(),
            "shift": shift.round(3).tolist(),
            "sizes": [int((labels == cluster).sum()) for cluster in range(N_CLUSTERS)],
            "variance_total": round(total_variance, 3),
            "variance_per_cluster": [round(value, 2) for value in variance_per_cluster],
            "rows": rows,
            "chart": build_chart(labels, new_centroids, f"Iteration {number}"),
        })

        centroids = new_centroids

    return {
        "records": len(sample),
        "features": FEATURES,
        "n_clusters": N_CLUSTERS,
        "n_iterations": N_ITERATIONS,
        "initial_centroids": INITIAL_CENTROIDS.tolist(),
        "initial_chart": build_chart(None, scale(INITIAL_CENTROIDS),
                                     "Initial state: the 100 events and the three starting centroids"),
        "iterations": iterations,
        "final_centroids": scaler.inverse_transform(centroids).round(2).tolist(),
        "source": (
            "100 record sample of data/seismic_events.csv, taken with a fixed seed by the "
            "generate_seismic_dataset.py script included in this repository."
        ),
    }


MANUAL = run()


if __name__ == "__main__":
    print("Initial centroids:", MANUAL["initial_centroids"])
    for iteration in MANUAL["iterations"]:
        print(f'--- iteration {iteration["number"]} ---')
        print("  sizes           :", iteration["sizes"])
        print("  within variance :", iteration["variance_total"], iteration["variance_per_cluster"])
        print("  centroids after :", iteration["centroids_after"])
        print("  centroid shift  :", iteration["shift"])