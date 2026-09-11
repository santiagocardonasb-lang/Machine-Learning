import base64
import io

import matplotlib
matplotlib.use("Agg")  # Headless backend: required to run on a server

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split(X, y):
    """Splits the data into 80% training and 20% testing, keeping the class balance."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def evaluate(model, X_test, y_test):
    """Runs the model on the test set and returns its classification metrics."""
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total": int(tn + fp + fn + tp),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def figure_to_base64(figure):
    """Turns a matplotlib figure into a base64 string ready for an <img> tag."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")