"""Pydantic schemas for Credit Risk API requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HomeOwnership(str, Enum):
    RENT = "RENT"
    MORTGAGE = "MORTGAGE"
    OWN = "OWN"
    OTHER = "OTHER"


class LoanIntent(str, Enum):
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    PERSONAL = "PERSONAL"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"


class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class ApplicantRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=120, description="Applicant age in years (>= 18)")
    person_income: float = Field(..., gt=0, description="Annual income in USD (> 0)")
    person_home_ownership: HomeOwnership = Field(..., description="Housing ownership status")
    person_emp_length: float = Field(default=0.0, ge=0.0, le=70.0, description="Employment length in years (>= 0)")
    credit_history_length: int = Field(default=0, ge=0, le=70, description="Credit history length in years (>= 0)")
    loan_intent: LoanIntent = Field(..., description="Purpose / intent of the loan")
    loan_grade: LoanGrade = Field(..., description="Officer-assigned loan grade (A-G)")
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount in USD (> 0)")
    loan_int_rate: float = Field(..., ge=0.0, le=100.0, description="Loan interest rate percentage")
    loan_percent_income: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Loan amount as fraction of annual income (optional, auto-derived if omitted)",
    )

    @field_validator("person_home_ownership", "loan_intent", "loan_grade", mode="before")
    @classmethod
    def uppercase_enum(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().upper()
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class ShapReason(BaseModel):
    feature: str = Field(..., description="Feature identifier")
    feature_name: str = Field(..., description="Human-readable feature name")
    applicant_value: str = Field(..., description="Formatted input value for this applicant")
    shap_value: float = Field(..., description="Signed contribution to default probability")
    direction: str = Field(..., description="'increases' or 'decreases' risk")
    plain_text: str = Field(..., description="Plain English compliance-ready explanation")


class ExplanationCheck(BaseModel):
    base_value: float = Field(..., description="Expected baseline probability")
    sum_shap: float = Field(..., description="Sum of all individual SHAP contributions")
    reconstructed_prob: float = Field(..., description="base_value + sum_shap")
    model_prob: float = Field(..., description="Output from pipeline.predict_proba")
    absolute_diff: float = Field(..., description="Difference between reconstructed and model prob")
    is_valid: bool = Field(..., description="True if difference is within numerical tolerance (< 1e-4)")


class PredictionResponse(BaseModel):
    p_default: float = Field(..., description="Predicted probability of default [0.0, 1.0]")
    risk_category: str = Field(..., description="Risk tier: Low, Medium, High, or Very High")
    base_value: float = Field(..., description="Expected probability from background baseline")
    reasons: list[ShapReason] = Field(..., description="Top 5 SHAP drivers with plain-language explanations")
    explanation_check: ExplanationCheck = Field(..., description="Mathematical verification of explanation fidelity")
    applicant_summary: dict[str, str] = Field(..., description="Summary of normalized applicant inputs")


class HealthResponse(BaseModel):
    status: str
    model: str
    base_value: float
    risk_thresholds: dict[str, Any]
    version: str


class GlobalImportanceItem(BaseModel):
    rank: int
    feature: str
    mean_abs_shap: float
    mean_signed_shap: float
    importance_pct: float
