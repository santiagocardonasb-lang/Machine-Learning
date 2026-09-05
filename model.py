import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend: required to run on a server

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Absolute path: works no matter which directory the app is started from
DATASET_PATH = Path(__file__).parent / "data" / "advertising_sales.csv"

FEATURE = "advertising_investment"
TARGET = "units_sold"

dataset = pd.read_csv(DATASET_PATH)

X = dataset[[FEATURE]]  # 2-D: scikit-learn expects a table of features
y = dataset[TARGET]     # 1-D: a single column of answers

model = LinearRegression()
model.fit(X, y)

SLOPE = float(model.coef_[0])
INTERCEPT = float(model.intercept_)
R2 = float(model.score(X, y))

# Everything the Application page needs to describe the dataset and the model
INFO = {
    "records": len(dataset),
    "feature_name": "Advertising investment",
    "feature_unit": "millions of COP",
    "target_name": "Units sold",
    "target_unit": "units per month",
    "source": (
        "Synthetic dataset generated with numpy using a fixed seed (42) by the "
        "generate_dataset.py script included in this repository. It simulates "
        "the monthly behavior of a consumer product."
    ),
    "slope": SLOPE,
    "intercept": INTERCEPT,
    "r2": R2,
    "min_investment": float(dataset[FEATURE].min()),
    "max_investment": float(dataset[FEATURE].max()),
}


def predict(investment):
    """Predicts the units sold for a given advertising investment."""
    new_data = pd.DataFrame({FEATURE: [investment]})
    return float(model.predict(new_data)[0])


def _build_chart():
    """Draws the scatter plot and the regression line, returns it as base64."""
    figure, axes = plt.subplots(figsize=(9, 5))

    axes.scatter(
        dataset[FEATURE], dataset[TARGET],
        alpha=0.4, s=25, color="#2f6f4e", label="Observed data",
    )

    # Two points are enough to draw a straight line: the minimum and the maximum
    line_x = pd.DataFrame({FEATURE: [dataset[FEATURE].min(), dataset[FEATURE].max()]})
    axes.plot(
        line_x[FEATURE], model.predict(line_x),
        color="#c0562c", linewidth=2.5, label="Regression line",
    )

    axes.set_title("Advertising investment vs. units sold")
    axes.set_xlabel("Advertising investment (millions of COP)")
    axes.set_ylabel("Units sold (units per month)")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


CHART = _build_chart()


if __name__ == "__main__":
    print(INFO)
    print("Prediction for 10 million COP:", predict(10))