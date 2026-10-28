"""Real-time credit risk scoring and single-applicant SHAP explanation module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.pkl"
BACKGROUND_PATH = ROOT / "background_sample.pkl"
RISK_BANDS_PATH = ROOT / "risk_bands.json"

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

FEATURE_LABELS = {
    "loan_grade": "Loan Grade",
    "loan_int_rate": "Interest Rate",
    "person_income": "Annual Income",
    "loan_percent_income": "Debt-to-Income (Loan % of Income)",
    "credit_history_length": "Credit History Length",
    "loan_intent": "Loan Intent / Purpose",
    "loan_amnt": "Loan Amount",
    "person_emp_length": "Employment Length",
    "person_age": "Applicant Age",
    "person_home_ownership": "Home Ownership",
    "log_loan_amount": "Log Loan Amount",
}


def assign_band(p: float) -> str:
    """Map probability of default to frozen risk tiers."""
    if p < 0.20:
        return "Low"
    if p < 0.80:
        return "Medium"
    if p < 0.83:
        return "High"
    return "Very High"


def format_applicant_value(feature: str, val: Any) -> str:
    """Format raw applicant inputs into clean strings for human display."""
    if feature == "loan_grade":
        return f"Grade {val}"
    if feature == "loan_int_rate":
        return f"{float(val):.2f}%"
    if feature in ("person_income", "loan_amnt"):
        return f"${int(round(float(val))):,}"
    if feature == "loan_percent_income":
        pct = float(val) * 100 if float(val) <= 1.0 else float(val)
        return f"{pct:.1f}%"
    if feature == "person_emp_length":
        return f"{float(val):.1f} yrs"
    if feature == "credit_history_length":
        return f"{int(round(float(val)))} yrs"
    if feature == "person_age":
        return f"{int(round(float(val)))} yrs"
    if feature == "person_home_ownership":
        return str(val).title()
    if feature == "loan_intent":
        return str(val).replace("_", " ").title()
    return str(val)


def generate_plain_text_reason(
    feature: str,
    shap_val: float,
    applicant_val: Any,
) -> str:
    """Generate clear, compliance-friendly English explanation for a feature's SHAP impact."""
    increases = shap_val > 0
    val_disp = format_applicant_value(feature, applicant_val)
    impact_pct = abs(shap_val) * 100

    if feature == "loan_grade":
        if increases:
            return (
                f"Assigned loan grade '{applicant_val}' carries higher default risk (+{impact_pct:.1f}% risk)."
            )
        return (
            f"Favorable loan grade '{applicant_val}' substantially lowers estimated risk (-{impact_pct:.1f}% risk)."
        )

    if feature == "loan_int_rate":
        if increases:
            return (
                f"Elevated interest rate of {val_disp} increases repayment burden and default risk (+{impact_pct:.1f}%)."
            )
        return (
            f"Competitive interest rate of {val_disp} lowers borrowing costs and risk (-{impact_pct:.1f}%)."
        )

    if feature == "loan_percent_income":
        if increases:
            return (
                f"Debt burden ratio ({val_disp} of annual income) elevates repayment strain (+{impact_pct:.1f}%)."
            )
        return (
            f"Conservative loan-to-income ratio ({val_disp}) demonstrates strong coverage (-{impact_pct:.1f}%)."
        )

    if feature == "person_income":
        if increases:
            return (
                f"Annual income of {val_disp} is tight relative to total requested exposure (+{impact_pct:.1f}%)."
            )
        return (
            f"Robust annual income of {val_disp} provides strong repayment support (-{impact_pct:.1f}%)."
        )

    if feature == "credit_history_length":
        if increases:
            return (
                f"Shorter credit history ({val_disp}) provides less track record of repayment (+{impact_pct:.1f}%)."
            )
        return (
            f"Established credit history of {val_disp} indicates proven borrowing stability (-{impact_pct:.1f}%)."
        )

    if feature == "person_emp_length":
        if increases:
            return (
                f"Limited employment length ({val_disp}) indicates potential income variability (+{impact_pct:.1f}%)."
            )
        return (
            f"Stable employment tenure ({val_disp}) demonstrates career stability (-{impact_pct:.1f}%)."
        )

    if feature == "loan_amnt":
        if increases:
            return (
                f"Higher loan principal of {val_disp} increases total capital exposure (+{impact_pct:.1f}%)."
            )
        return (
            f"Modest loan principal of {val_disp} limits lender exposure (-{impact_pct:.1f}%)."
        )

    if feature == "person_home_ownership":
        if increases:
            return (
                f"Housing status ({val_disp}) is statistically associated with higher risk profile (+{impact_pct:.1f}%)."
            )
        return (
            f"Housing status ({val_disp}) indicates residential stability and reduces risk (-{impact_pct:.1f}%)."
        )

    if feature == "loan_intent":
        if increases:
            return (
                f"Loan purpose ({val_disp}) contributes to a higher expected risk category (+{impact_pct:.1f}%)."
            )
        return (
            f"Loan purpose ({val_disp}) has historically lower risk tendencies (-{impact_pct:.1f}%)."
        )

    if feature == "person_age":
        if increases:
            return f"Applicant age profile ({val_disp}) correlates with elevated credit risk (+{impact_pct:.1f}%)."
        return f"Applicant age profile ({val_disp}) correlates favorably with lower default risk (-{impact_pct:.1f}%)."

    # Default fallback
    direction = "increases" if increases else "decreases"
    return f"{FEATURE_LABELS.get(feature, feature)} ({val_disp}) {direction} default risk by {impact_pct:.1f}%."


class CreditRiskExplainer:
    """Loads pipeline, background distribution, and provides single-row scoring + TreeExplainer SHAP."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        background_path: Path = BACKGROUND_PATH,
        risk_bands_path: Path = RISK_BANDS_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not background_path.exists():
            raise FileNotFoundError(f"Background sample not found at {background_path}. Run prepare_shap_artifacts.py first.")

        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.xgb_model = self.pipeline.named_steps["model"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())

        bg_payload = joblib.load(background_path)
        self.background_encoded = bg_payload["encoded"]

        self.explainer = shap.TreeExplainer(
            self.xgb_model,
            data=self.background_encoded,
            model_output="probability",
            feature_perturbation="interventional",
        )
        self.base_value = float(self.explainer.expected_value)

        # Load risk band metadata
        if risk_bands_path.exists():
            self.risk_bands_config = json.loads(risk_bands_path.read_text(encoding="utf-8"))
        else:
            self.risk_bands_config = {}

    def prepare_row(self, applicant: dict[str, Any]) -> pd.DataFrame:
        """Validate, cast, and order input features into a 1-row DataFrame."""
        row: dict[str, Any] = {}

        # 1. Enums / Categoricals (unwrap enum .value if passed as Enum instance)
        def _to_str(val: Any) -> str:
            if hasattr(val, "value"):
                return str(val.value).upper()
            return str(val).upper()

        row["person_home_ownership"] = _to_str(applicant["person_home_ownership"])
        row["loan_intent"] = _to_str(applicant["loan_intent"])
        row["loan_grade"] = _to_str(applicant["loan_grade"])

        # 2. Numerics
        row["person_age"] = int(applicant["person_age"])
        row["person_income"] = float(applicant["person_income"])
        row["person_emp_length"] = float(applicant.get("person_emp_length", 0.0))
        row["credit_history_length"] = int(applicant.get("credit_history_length", 0))
        row["loan_amnt"] = float(applicant["loan_amnt"])
        row["loan_int_rate"] = float(applicant["loan_int_rate"])

        # loan_percent_income: if provided as percentage > 1.0 (e.g. 25 for 25%), convert to fraction
        lpi = float(applicant.get("loan_percent_income", 0.0))
        if lpi > 1.0:
            lpi = lpi / 100.0
        if lpi <= 0.0 and row["person_income"] > 0:
            lpi = row["loan_amnt"] / row["person_income"]
        row["loan_percent_income"] = round(lpi, 4)

        # Derived on server
        row["log_loan_amount"] = float(np.log1p(row["loan_amnt"]))

        df_row = pd.DataFrame([row])[FEATURE_COLUMNS]
        return df_row

    def score_and_explain(self, applicant: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
        """Compute predicted probability of default, risk category, and top SHAP explanations."""
        df_row = self.prepare_row(applicant)

        # Model prediction
        p_default = float(self.pipeline.predict_proba(df_row)[0, 1])
        risk_category = assign_band(p_default)

        # Preprocess features
        row_enc = self.preprocessor.transform(df_row)
        if hasattr(row_enc, "toarray"):
            row_enc = row_enc.toarray()

        # SHAP explanation
        exp = self.explainer(row_enc)
        raw_shap_values = exp.values[0]

        # Group one-hot encoded columns back to domain features
        grouped_shap: dict[str, float] = {}
        for feat_name, shap_val in zip(self.feature_names, raw_shap_values):
            clean = feat_name.replace("encoder__", "").replace("scaler__", "")
            if clean.startswith("person_home_ownership_"):
                group = "person_home_ownership"
            elif clean.startswith("loan_intent_"):
                group = "loan_intent"
            elif clean.startswith("loan_grade_"):
                group = "loan_grade"
            elif clean == "log_loan_amount":
                # Combine log_loan_amount with loan_amnt for clean domain reporting
                group = "loan_amnt"
            else:
                group = clean

            grouped_shap[group] = grouped_shap.get(group, 0.0) + float(shap_val)

        # Order grouped features by absolute SHAP contribution
        sorted_features = sorted(grouped_shap.items(), key=lambda item: abs(item[1]), reverse=True)

        reasons = []
        for feat, val in sorted_features[:top_n]:
            applicant_val = df_row[feat].iloc[0]
            direction = "increases" if val > 0 else "decreases"
            plain_text = generate_plain_text_reason(feat, val, applicant_val)
            reasons.append(
                {
                    "feature": feat,
                    "feature_name": FEATURE_LABELS.get(feat, feat),
                    "applicant_value": format_applicant_value(feat, applicant_val),
                    "shap_value": round(val, 4),
                    "direction": direction,
                    "plain_text": plain_text,
                }
            )

        sum_shap = float(raw_shap_values.sum())
        expected_prob = float(self.base_value + sum_shap)

        # Verification check
        diff = abs(expected_prob - p_default)
        is_valid = bool(diff < 1e-4)

        return {
            "p_default": round(p_default, 4),
            "risk_category": risk_category,
            "base_value": round(self.base_value, 4),
            "reasons": reasons,
            "explanation_check": {
                "base_value": round(self.base_value, 4),
                "sum_shap": round(sum_shap, 4),
                "reconstructed_prob": round(expected_prob, 4),
                "model_prob": round(p_default, 4),
                "absolute_diff": round(diff, 6),
                "is_valid": is_valid,
            },
            "applicant_summary": {
                feat: format_applicant_value(feat, df_row[feat].iloc[0])
                for feat in FEATURE_COLUMNS
                if feat != "log_loan_amount"
            },
        }


# Global singleton instance for module-level usage / FastAPI
_EXPLAINER_INSTANCE: CreditRiskExplainer | None = None


def get_explainer() -> CreditRiskExplainer:
    """Lazy loader for the singleton explainer instance."""
    global _EXPLAINER_INSTANCE
    if _EXPLAINER_INSTANCE is None:
        _EXPLAINER_INSTANCE = CreditRiskExplainer()
    return _EXPLAINER_INSTANCE


def predict_and_explain(applicant: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    """Convenience function to score and explain an applicant dictionary."""
    return get_explainer().score_and_explain(applicant, top_n=top_n)


def get_global_importance(grouped: bool = True) -> list[dict[str, Any]]:
    """Retrieve global SHAP importance rankings for dashboard visualizations."""
    filename = "shap_global_grouped_metrics.csv" if grouped else "shap_global_metrics.csv"
    path = ROOT / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    print("Initializing CreditRiskExplainer test run...")
    explainer = get_explainer()

    sample_applicant = {
        "person_age": 28,
        "person_income": 45000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "credit_history_length": 4,
        "loan_intent": "EDUCATION",
        "loan_grade": "C",
        "loan_amnt": 12000,
        "loan_int_rate": 13.5,
        "loan_percent_income": 0.27,
    }

    result = explainer.score_and_explain(sample_applicant)
    print("\nScore and SHAP Output:")
    print(json.dumps(result, indent=2))
