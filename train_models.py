"""Train the five classifiers and save them with their test metrics.

Run it with:  python train_models.py
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = "Class"
SEED = 20261106


def build_models():
    # key is the file name, value is (display name, pipeline)
    return {
        "logistic_regression": (
            "Logistic Regression",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        ),
        "decision_tree": (
            "Decision Tree",
            make_pipeline(DecisionTreeClassifier(random_state=SEED)),
        ),
        "knn": (
            "KNN",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        ),
        "gaussian_naive_bayes": (
            "Naive Bayes (Gaussian)",
            make_pipeline(GaussianNB()),
        ),
        # SVC has no probabilities by default, so calibrate it to get them for AUC
        "support_vector_machine": (
            "Support Vector Machine",
            make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(SVC(random_state=SEED), method="sigmoid", cv=5),
            ),
        ),
    }


def evaluate_model(model, X, y):
    """Return the six metrics. Macro averaging because there are 26 classes."""
    # one pass only: predict() would repeat the expensive predict_proba work
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    # one-vs-rest AUC per letter, then averaged. Skip letters that are not in
    # y (possible when someone uploads only part of the test set).
    aucs = []
    for i, letter in enumerate(model.classes_):
        is_letter = (y == letter).to_numpy()
        if is_letter.any() and not is_letter.all():
            aucs.append(roc_auc_score(is_letter, proba[:, i]))
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": sum(aucs) / len(aucs) if aucs else float("nan"),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": matthews_corrcoef(y, pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=Path("train_data.csv"))
    parser.add_argument("--test-data", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--metrics-file", type=Path, default=Path("model_metrics.csv"))
    args = parser.parse_args()

    train = pd.read_csv(args.train_data)
    test = pd.read_csv(args.test_data)
    features = [c for c in train.columns if c != TARGET]
    if features != [c for c in test.columns if c != TARGET]:
        raise SystemExit("train and test files have different feature columns")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, (name, model) in build_models().items():
        model.fit(train[features], train[TARGET])
        metrics = evaluate_model(model, test[features], test[TARGET])
        joblib.dump(model, args.model_dir / f"{key}.joblib")
        rows.append({"ML Model Name": name, **metrics})
        print(name, {k: round(v, 4) for k, v in metrics.items()})

    pd.DataFrame(rows).to_csv(args.metrics_file, index=False)
    print("saved", args.metrics_file)


if __name__ == "__main__":
    main()
