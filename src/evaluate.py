"""Evaluate saved model predictions in data/predictions.csv and apply the quality gate."""

import csv
import sys
from pathlib import Path

from src.metrics import accuracy, f1_score, precision, quality_gate, recall

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions.csv"


def load_predictions(path=DATA_PATH):
    """Read y_true and y_pred columns from a CSV file."""
    y_true, y_pred = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return y_true, y_pred


def main():
    y_true, y_pred = load_predictions()
    print(f"Samples   : {len(y_true)}")
    print(f"Accuracy  : {accuracy(y_true, y_pred):.3f}")
    print(f"Precision : {precision(y_true, y_pred):.3f}")
    print(f"Recall    : {recall(y_true, y_pred):.3f}")
    print(f"F1 score  : {f1_score(y_true, y_pred):.3f}")
    result = quality_gate(y_true, y_pred)
    print("Quality gate:", "PASSED" if result["passed"] else "FAILED")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
