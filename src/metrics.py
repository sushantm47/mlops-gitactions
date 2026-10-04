"""
Model evaluation metrics implemented from scratch.

Modification of MLOps Lab 1 (Github_Labs/Lab1): the original calculator
functions (fun1-fun4) are replaced with ML evaluation metrics that are
tested with pytest and unittest and verified against scikit-learn in CI.
"""

import math


def _validate(y_true, y_pred):
    """Check that inputs are non-empty lists/tuples of equal length."""
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise TypeError("y_true and y_pred must be lists or tuples")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")


def _validate_numeric(y_true, y_pred):
    _validate(y_true, y_pred)
    for v in list(y_true) + list(y_pred):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("Regression inputs must be numeric")


# ---------------- Classification metrics ----------------

def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true labels."""
    _validate(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, positive=1):
    """Return binary confusion counts as a dict: tp, fp, tn, fn."""
    _validate(y_true, y_pred)
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            counts["tp"] += 1
        elif p == positive:
            counts["fp"] += 1
        elif t == positive:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def precision(y_true, y_pred, positive=1):
    """TP / (TP + FP). Returns 0.0 when there are no positive predictions."""
    cm = confusion_matrix(y_true, y_pred, positive)
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom else 0.0


def recall(y_true, y_pred, positive=1):
    """TP / (TP + FN). Returns 0.0 when there are no positive labels."""
    cm = confusion_matrix(y_true, y_pred, positive)
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom else 0.0


def f1_score(y_true, y_pred, positive=1):
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ---------------- Regression metrics ----------------

def mae(y_true, y_pred):
    """Mean absolute error."""
    _validate_numeric(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def mse(y_true, y_pred):
    """Mean squared error."""
    _validate_numeric(y_true, y_pred)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return math.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    """Coefficient of determination. Raises if y_true has zero variance."""
    _validate_numeric(y_true, y_pred)
    mean_t = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_t) ** 2 for t in y_true)
    if ss_tot == 0:
        raise ValueError("R^2 is undefined when y_true has zero variance")
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot


# ---------------- MLOps quality gate ----------------

def quality_gate(y_true, y_pred, min_accuracy=0.8, min_f1=0.75):
    """
    Decide whether a classifier is good enough to deploy.

    Returns a dict with the computed metrics and a boolean 'passed'.
    """
    if not (0 <= min_accuracy <= 1 and 0 <= min_f1 <= 1):
        raise ValueError("Thresholds must be between 0 and 1")
    acc = accuracy(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "f1": f1,
        "passed": acc >= min_accuracy and f1 >= min_f1,
    }
