"""Generate background sample and update global SHAP metrics for the contracted model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

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
BACKGROUND_SIZE = 100


def main() -> None:
    print("Loading pipeline and dataset...")
    pipeline = joblib.load(ROOT / "best_model.pkl")
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df["log_loan_amount"] = np.log1p(df["loan_amnt"])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 1. Transform training set and sample background rows
    X_train_enc = preprocessor.transform(X_train)
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()

    rng = np.random.default_rng(RANDOM_STATE)
    bg_indices = rng.choice(len(X_train_enc), size=BACKGROUND_SIZE, replace=False)
    background_encoded = X_train_enc[bg_indices]
    background_raw = X_train.iloc[bg_indices].copy().reset_index(drop=True)

    # Save background sample artifact
    bg_payload = {
        "encoded": background_encoded,
        "raw": background_raw,
        "feature_names": list(feature_names),
        "source": "train_split",
        "random_state": RANDOM_STATE,
        "size": BACKGROUND_SIZE,
    }
    joblib.dump(bg_payload, ROOT / "background_sample.pkl")
    print(f"Saved background sample -> {ROOT / 'background_sample.pkl'}")

    # 2. Build TreeExplainer
    print("Initializing TreeExplainer with training background...")
    explainer = shap.TreeExplainer(
        xgb_model,
        data=background_encoded,
        model_output="probability",
        feature_perturbation="interventional",
    )
    print(f"Explainer base value (expected p): {float(explainer.expected_value):.4f}")

    # 3. Compute global SHAP metrics on test set
    print("Computing SHAP on test set for global importance metrics...")
    X_test_enc = preprocessor.transform(X_test)
    if hasattr(X_test_enc, "toarray"):
        X_test_enc = X_test_enc.toarray()

    explanation = explainer(X_test_enc)
    shap_values = explanation.values

    # Encoded feature metrics
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed_shap = shap_values.mean(axis=0)
    total_importance = mean_abs_shap.sum()
    importance_pct = (
        mean_abs_shap / total_importance * 100 if total_importance > 0 else np.zeros_like(mean_abs_shap)
    )

    shap_metrics = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_signed_shap": mean_signed_shap,
            "importance_pct": importance_pct,
            "positive_contribution_pct": (shap_values > 0).mean(axis=0) * 100,
            "negative_contribution_pct": (shap_values < 0).mean(axis=0) * 100,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_metrics.insert(0, "rank", np.arange(1, len(shap_metrics) + 1))

    shap_metrics.to_csv(ROOT / "shap_global_metrics.csv", index=False)
    shap_metrics.to_csv(OUT_DIR / "shap_global_metrics.csv", index=False)
    print(f"Saved {ROOT / 'shap_global_metrics.csv'}")

    # 4. Grouped feature metrics (for clean dashboard visualization)
    grouped_shap = {}
    for i, col in enumerate(feature_names):
        clean = col.replace("encoder__", "").replace("scaler__", "")
        if clean.startswith("person_home_ownership_"):
            group = "person_home_ownership"
        elif clean.startswith("loan_intent_"):
            group = "loan_intent"
        elif clean.startswith("loan_grade_"):
            group = "loan_grade"
        elif clean == "log_loan_amount":
            group = "loan_amnt"
        else:
            group = clean

        if group not in grouped_shap:
            grouped_shap[group] = np.zeros(len(shap_values))
        grouped_shap[group] += shap_values[:, i]

    grouped_df = pd.DataFrame(grouped_shap)
    grouped_mean_abs = np.abs(grouped_df).mean(axis=0)
    grouped_mean_signed = grouped_df.mean(axis=0)
    grouped_total = grouped_mean_abs.sum()
    grouped_pct = grouped_mean_abs / grouped_total * 100

    grouped_metrics = pd.DataFrame(
        {
            "feature": grouped_mean_abs.index,
            "mean_abs_shap": grouped_mean_abs.values,
            "mean_signed_shap": grouped_mean_signed.values,
            "importance_pct": grouped_pct.values,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    grouped_metrics.insert(0, "rank", np.arange(1, len(grouped_metrics) + 1))

    grouped_metrics.to_csv(ROOT / "shap_global_grouped_metrics.csv", index=False)
    grouped_metrics.to_csv(OUT_DIR / "shap_global_grouped_metrics.csv", index=False)
    print(f"Saved {ROOT / 'shap_global_grouped_metrics.csv'}")

    print("\nTop Grouped Features by Mean |SHAP|:")
    print(grouped_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
