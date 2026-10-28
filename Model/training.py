"""Train LR, Random Forest, and XGBoost on the contracted feature set.

Drops loan_status (leakage). Recomputes log_loan_amount with log1p.
Saves the best pipeline by test ROC-AUC as best_model.pkl.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "credit_risk_dataset_updated.csv"
OUT_DIR = ROOT / "training_outputs"
OUT_DIR.mkdir(exist_ok=True)

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
]
NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "credit_history_length",
    "log_loan_amount",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "default_status"
LEAKAGE_COLUMNS = ["loan_status"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_contracted_frame() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df["log_loan_amount"] = np.log1p(df["loan_amnt"])
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing contracted columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)
    return X, y


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("scaler", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }
    print(f"\n=== {name} (test) ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Brier:   {metrics['brier']:.4f}")
    print(f"Precision / Recall / F1: {metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}")
    print("Confusion matrix [tn fp; fn tp]:")
    print(cm)
    print(classification_report(y_test, y_pred, zero_division=0))
    return metrics, y_prob


def plot_calibration(curves: dict[str, np.ndarray], y_test: pd.Series) -> None:
    plt.figure(figsize=(7.5, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for name, y_prob in curves.items():
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default rate")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "calibration.png", dpi=150)
    plt.close()


def search_model(name: str, estimator, param_distributions, X_train, y_train, cv, n_iter: int):
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"\n{name} best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"{name} best params: {search.best_params_}")
    return search


def main() -> None:
    X, y = load_contracted_frame()
    print(f"Rows: {len(X)}")
    print(f"Positive rate: {y.mean():.4f}")
    print(f"Features: {FEATURE_COLUMNS}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"Train pos={pos} neg={neg} scale_pos_weight={scale_pos_weight:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    searches = {}

    searches["logistic_regression"] = search_model(
        "logistic_regression",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        {
            "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
        },
        X_train,
        y_train,
        cv,
        n_iter=5,
    )

    searches["random_forest"] = search_model(
        "random_forest",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_leaf": [1, 4, 8],
            "model__max_features": ["sqrt", 0.5],
        },
        X_train,
        y_train,
        cv,
        n_iter=12,
    )

    searches["xgboost"] = search_model(
        "xgboost",
        xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        ),
        {
            "model__n_estimators": [200, 400, 600],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__reg_lambda": [1, 5, 10],
            "model__gamma": [0, 0.1, 0.5],
        },
        X_train,
        y_train,
        cv,
        n_iter=18,
    )

    comparison = []
    curves = {}
    for name, search in searches.items():
        metrics, y_prob = evaluate(name, search.best_estimator_, X_test, y_test)
        metrics["cv_roc_auc"] = float(search.best_score_)
        metrics["best_params"] = {k: _jsonable(v) for k, v in search.best_params_.items()}
        comparison.append(metrics)
        curves[name] = y_prob

    plot_calibration(curves, y_test)

    winner = max(comparison, key=lambda m: m["roc_auc"])
    winner_name = winner["model"]
    best_pipeline = searches[winner_name].best_estimator_

    joblib.dump(best_pipeline, ROOT / "best_model.pkl")
    X_test.to_csv(ROOT / "X_test.csv", index=False)
    y_test.to_csv(ROOT / "y_test.csv", index=False)

    summary = {
        "target": TARGET,
        "dropped_columns": LEAKAGE_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "log_loan_amount": "log1p(loan_amnt)",
        "winner": winner_name,
        "winner_test_roc_auc": winner["roc_auc"],
        "comparison": comparison,
    }
    (OUT_DIR / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = [
        {
            "model": m["model"],
            "cv_roc_auc": m["cv_roc_auc"],
            "test_roc_auc": m["roc_auc"],
            "brier": m["brier"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        }
        for m in comparison
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "model_comparison.csv", index=False)

    (ROOT / "feature_columns.json").write_text(
        json.dumps(
            {
                "categorical": CATEGORICAL_FEATURES,
                "numeric": NUMERIC_FEATURES,
                "all": FEATURE_COLUMNS,
                "target": TARGET,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWinner: {winner_name} (test ROC-AUC {winner['roc_auc']:.4f})")
    print(f"Saved pipeline -> {ROOT / 'best_model.pkl'}")
    print(f"Comparison -> {OUT_DIR / 'model_comparison.csv'}")
    print(f"Calibration plot -> {OUT_DIR / 'calibration.png'}")


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
