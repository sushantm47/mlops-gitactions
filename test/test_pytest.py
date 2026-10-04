"""Pytest suite for src/metrics.py, cross-checked against scikit-learn."""

import math

import pytest
from sklearn import metrics as sk

from src import metrics
from src.evaluate import load_predictions

Y_TRUE = [1, 0, 1, 1, 0, 0, 1, 0]
Y_PRED = [1, 0, 0, 1, 0, 1, 1, 0]


# ---------- classification: compare with scikit-learn ----------

@pytest.mark.parametrize(
    "ours, theirs",
    [
        (metrics.accuracy, sk.accuracy_score),
        (metrics.precision, sk.precision_score),
        (metrics.recall, sk.recall_score),
        (metrics.f1_score, sk.f1_score),
    ],
)
def test_classification_matches_sklearn(ours, theirs):
    assert ours(Y_TRUE, Y_PRED) == pytest.approx(theirs(Y_TRUE, Y_PRED))


def test_confusion_matrix_counts():
    assert metrics.confusion_matrix(Y_TRUE, Y_PRED) == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}


def test_confusion_matrix_matches_sklearn():
    tn, fp, fn, tp = sk.confusion_matrix(Y_TRUE, Y_PRED).ravel()
    cm = metrics.confusion_matrix(Y_TRUE, Y_PRED)
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (tn, fp, fn, tp)


def test_perfect_predictions():
    assert metrics.accuracy(Y_TRUE, Y_TRUE) == 1.0
    assert metrics.f1_score(Y_TRUE, Y_TRUE) == 1.0


def test_no_positive_predictions_returns_zero():
    assert metrics.precision([1, 1, 0], [0, 0, 0]) == 0.0
    assert metrics.f1_score([1, 1, 0], [0, 0, 0]) == 0.0


def test_string_labels_with_custom_positive():
    y_true = ["spam", "ham", "spam"]
    y_pred = ["spam", "spam", "spam"]
    assert metrics.precision(y_true, y_pred, positive="spam") == pytest.approx(2 / 3)
    assert metrics.recall(y_true, y_pred, positive="spam") == 1.0


# ---------- regression: compare with scikit-learn ----------

R_TRUE = [3.0, -0.5, 2.0, 7.0]
R_PRED = [2.5, 0.0, 2.0, 8.0]


@pytest.mark.parametrize(
    "ours, theirs",
    [
        (metrics.mae, sk.mean_absolute_error),
        (metrics.mse, sk.mean_squared_error),
        (metrics.r2_score, sk.r2_score),
    ],
)
def test_regression_matches_sklearn(ours, theirs):
    assert ours(R_TRUE, R_PRED) == pytest.approx(theirs(R_TRUE, R_PRED))


def test_rmse_is_sqrt_of_mse():
    assert metrics.rmse(R_TRUE, R_PRED) == pytest.approx(math.sqrt(metrics.mse(R_TRUE, R_PRED)))


# ---------- error handling ----------

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.accuracy([1, 0], [1])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        metrics.mae([], [])


@pytest.mark.parametrize("bad", ["1010", 42, None])
def test_non_list_input_raises(bad):
    with pytest.raises(TypeError):
        metrics.accuracy(bad, [1, 0])


def test_regression_rejects_non_numeric():
    with pytest.raises(TypeError):
        metrics.mse([1, "a"], [1, 2])


def test_r2_zero_variance_raises():
    with pytest.raises(ValueError):
        metrics.r2_score([5, 5, 5], [4, 5, 6])


# ---------- MLOps quality gate ----------

def test_quality_gate_passes_good_model():
    assert metrics.quality_gate(Y_TRUE, Y_TRUE)["passed"] is True


def test_quality_gate_fails_bad_model():
    flipped = [1 - y for y in Y_TRUE]
    assert metrics.quality_gate(Y_TRUE, flipped)["passed"] is False


def test_quality_gate_invalid_threshold():
    with pytest.raises(ValueError):
        metrics.quality_gate(Y_TRUE, Y_PRED, min_accuracy=1.5)


def test_saved_predictions_pass_gate():
    """The committed model predictions must meet the deployment threshold."""
    y_true, y_pred = load_predictions()
    assert metrics.quality_gate(y_true, y_pred)["passed"]


def test_evaluate_main_prints_report(capsys):
    from src.evaluate import main
    assert main() == 0
    out = capsys.readouterr().out
    assert "Quality gate: PASSED" in out
