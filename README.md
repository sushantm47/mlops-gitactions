# MLOps Lab 1 – Model Evaluation Metrics with CI

![Pytest](https://github.com/sushantm47/mlops-gitactions/actions/workflows/pytest_action.yml/badge.svg)
![Unittest](https://github.com/sushantm47/mlops-gitactions/actions/workflows/unittest_action.yml/badge.svg)

Modified version of **Github_Labs/Lab1** from the IE-7374 MLOps course (Prof. Ramin Mohammadi, Northeastern University).
The lab teaches virtual environments, project structure, unit testing with **pytest** and **unittest**, and
running those tests automatically with **GitHub Actions**.

## What I changed from the original lab

| Area | Original lab | This repo |
| --- | --- | --- |
| Code under test | `calculator.py` with `fun1`–`fun4` (add, subtract, multiply, combine) | `metrics.py`: ML evaluation metrics written from scratch – accuracy, precision, recall, F1, confusion matrix, MAE, MSE, RMSE, R² |
| MLOps angle | none | `quality_gate()` decides whether a model is good enough to deploy; `evaluate.py` runs it on saved predictions |
| Data | empty `data/` folder | `data/predictions.csv` – predictions from a toy spam classifier |
| Test correctness | hand-computed expected values | pytest cross-checks every metric against **scikit-learn** |
| Tests | 4 pytest + 4 unittest | 25 pytest (parametrized, error cases, `capsys`) + 11 unittest (`setUp`, `subTest`) |
| Error handling | basic type checks | type, empty-input, length-mismatch, non-numeric and zero-variance checks |
| CI | single Python version, pytest/unittest only | Python 3.10/3.11/3.12 matrix, flake8 linting, 90% coverage minimum, quality-gate step, JUnit report uploaded as an artifact, manual trigger |

## Project structure

```
.
├── .github/workflows/
│   ├── pytest_action.yml      # lint + pytest + coverage + quality gate
│   └── unittest_action.yml    # unittest suite
├── data/
│   └── predictions.csv        # y_true, y_pred from a toy classifier
├── src/
│   ├── __init__.py
│   ├── metrics.py             # metric functions + quality_gate
│   └── evaluate.py            # loads predictions, prints report, exits 1 if gate fails
├── test/
│   ├── __init__.py
│   ├── test_pytest.py
│   └── test_unittest.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Running locally

```bash
python3 -m venv lab_01
source lab_01/bin/activate          # Windows: lab_01\Scripts\activate
pip install -r requirements.txt

pytest --cov=src --cov-report=term-missing
python -m unittest discover -s test -p "test_unittest.py" -t . -v
flake8 src test --max-line-length=100
python -m src.evaluate
```

Sample output of `python -m src.evaluate`:

```
Samples   : 20
Accuracy  : 0.900
Precision : 0.900
Recall    : 0.900
F1 score  : 0.900
Quality gate: PASSED
```

## GitHub Actions

Both workflows run on every push and pull request to `main` and can be started manually from the Actions tab.

- **Testing with Pytest** – lints with flake8, runs pytest on three Python versions with a 90% coverage
  minimum, runs the model quality gate, and uploads `pytest-report.xml` as an artifact.
- **Python Unittests** – runs the unittest suite.

If a metric is wrong, coverage drops, or the saved model predictions fall below the deployment
threshold (accuracy ≥ 0.80, F1 ≥ 0.75), the workflow fails and the change is blocked.

## Screenshots

<!-- Add screenshots of the green Actions runs here, e.g. ![Actions](assets/actions.png) -->
