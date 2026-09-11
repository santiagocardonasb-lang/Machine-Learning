from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend: required to run on a server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import classification

DATASET_PATH = Path(__file__).parent / "data" / "sales_goal.csv"

FEATURE = "advertising_investment"
TARGET = "goal_reached"

CLASSES = {
    0: "Goal not reached",
    1: "Goal reached",
}

CLASS_MEANING = {
    0: "The month sold fewer than 1000 units, so the sales goal was missed.",
    1: "The month sold 1000 units or more, so the sales goal was met.",
}

dataset = pd.read_csv(DATASET_PATH)

X = dataset[[FEATURE]]  # 2-D: scikit-learn expects a table of features
y = dataset[TARGET]     # 1-D: a single column of labels

X_train, X_test, y_train, y_test = classification.split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

METRICS = classification.evaluate(model, X_test, y_test)

INFO = {
    "records": len(dataset),
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "feature_name": "Advertising investment",
    "feature_unit": "millions of COP",
    "target_name": "Sales goal reached",
    "classes": CLASSES,
    "class_meaning": CLASS_MEANING,
    "class_counts": {
        int(key): int(value)
        for key, value in dataset[TARGET].value_counts().sort_index().items()
    },
    "monthly_goal": 1000,
    "min_investment": float(dataset[FEATURE].min()),
    "max_investment": float(dataset[FEATURE].max()),
    "source": (
        "Synthetic dataset generated with numpy using a fixed seed (7) by the "
        "generate_logistic_dataset.py script included in this repository."
    ),
}


def classify(investment):
    """Classifies one advertising investment and returns the class and its probability."""
    new_data = pd.DataFrame({FEATURE: [investment]})
    predicted = int(model.predict(new_data)[0])
    probability = float(model.predict_proba(new_data)[0][1])

    return {
        "value": predicted,
        "label": CLASSES[predicted],
        "meaning": CLASS_MEANING[predicted],
        "probability": probability,
    }


def _build_chart():
    """Scatter of both classes plus the sigmoid curve learned by the model."""
    figure, axes = plt.subplots(figsize=(9, 5))

    for value, color, marker in ((0, "#c0562c", "o"), (1, "#2f6f4e", "s")):
        subset = dataset[dataset[TARGET] == value]
        axes.scatter(
            subset[FEATURE], subset[TARGET],
            alpha=0.35, s=30, color=color, marker=marker,
            label=f"{value} - {CLASSES[value]}",
        )

    # The sigmoid: probability of reaching the goal across the whole range of investment
    grid = pd.DataFrame({
        FEATURE: np.linspace(dataset[FEATURE].min(), dataset[FEATURE].max(), 300)
    })
    axes.plot(
        grid[FEATURE], model.predict_proba(grid)[:, 1],
        color="#1f3d2b", linewidth=2.5, label="Probability of reaching the goal",
    )

    axes.axhline(
        0.5, color="#888888", linestyle="--", linewidth=1.2,
        label="Classification threshold (0.5)",
    )

    axes.set_title("Advertising investment vs. sales goal achievement")
    axes.set_xlabel("Advertising investment (millions of COP)")
    axes.set_ylabel("Class / probability of reaching the goal")
    axes.legend(loc="center right", fontsize=9)
    axes.grid(alpha=0.2)
    figure.tight_layout()

    return classification.figure_to_base64(figure)


CHART = _build_chart()


if __name__ == "__main__":
    print("Records:", INFO["records"])
    print("Training:", INFO["training_records"], "| Testing:", INFO["testing_records"])
    print("Class counts:", INFO["class_counts"])
    print("Metrics:", METRICS)
    print("classify(5)  ->", classify(5))
    print("classify(40) ->", classify(40))