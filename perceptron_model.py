from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend: required to run on a server

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import classification

DATASET_PATH = Path(__file__).parent / "data" / "campaign_profit.csv"

FEATURES = ["budget", "duration_days", "channels"]
TARGET = "profitable"

CLASSES = {
    0: "Not profitable",
    1: "Profitable",
}

CLASS_MEANING = {
    0: "The campaign returned less than it cost, so the investment was not recovered.",
    1: "The campaign returned more than it cost, so the investment paid off.",
}

FEATURE_LABELS = {
    "budget": ("Budget", "millions of COP"),
    "duration_days": ("Duration", "days"),
    "channels": ("Channels", "number of channels used"),
}

dataset = pd.read_csv(DATASET_PATH)

X = dataset[FEATURES]  # three independent variables
y = dataset[TARGET]

X_train, X_test, y_train, y_test = classification.split(X, y)

# StandardScaler first: the Perceptron is sensitive to the scale of each variable
model = make_pipeline(StandardScaler(), Perceptron(random_state=42))
model.fit(X_train, y_train)

METRICS = classification.evaluate(model, X_test, y_test)

INFO = {
    "records": len(dataset),
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "features": [
        {
            "column": column,
            "name": FEATURE_LABELS[column][0],
            "unit": FEATURE_LABELS[column][1],
            "minimum": float(dataset[column].min()),
            "maximum": float(dataset[column].max()),
        }
        for column in FEATURES
    ],
    "target_name": "Campaign profitability",
    "classes": CLASSES,
    "class_meaning": CLASS_MEANING,
    "class_counts": {
        int(key): int(value)
        for key, value in dataset[TARGET].value_counts().sort_index().items()
    },
    "source": (
        "Synthetic dataset generated with numpy using a fixed seed (13) by the "
        "generate_perceptron_dataset.py script included in this repository."
    ),
}


def classify(budget, duration_days, channels):
    """Classifies one campaign from its three independent variables."""
    new_data = pd.DataFrame([{
        "budget": budget,
        "duration_days": duration_days,
        "channels": channels,
    }])
    predicted = int(model.predict(new_data)[0])

    return {
        "value": predicted,
        "label": CLASSES[predicted],
        "meaning": CLASS_MEANING[predicted],
    }


def _build_chart():
    """Scatter of budget against duration, with each class in its own colour."""
    figure, axes = plt.subplots(figsize=(9, 5))

    for value, color, marker in ((0, "#c0562c", "o"), (1, "#2f6f4e", "s")):
        subset = dataset[dataset[TARGET] == value]
        axes.scatter(
            subset["budget"], subset["duration_days"],
            alpha=0.5, s=32, color=color, marker=marker,
            label=f"{value} - {CLASSES[value]}",
        )

    axes.set_title("Campaign budget and duration, grouped by profitability")
    axes.set_xlabel("Budget (millions of COP)")
    axes.set_ylabel("Duration (days)")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()

    return classification.figure_to_base64(figure)


CHART = _build_chart()


if __name__ == "__main__":
    print("Records:", INFO["records"])
    print("Training:", INFO["training_records"], "| Testing:", INFO["testing_records"])
    print("Class counts:", INFO["class_counts"])
    print("Metrics:", METRICS)
    print("classify(30, 45, 5) ->", classify(30, 45, 5))
    print("classify(3, 10, 1)  ->", classify(3, 10, 1))