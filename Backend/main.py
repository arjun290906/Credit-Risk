"""FastAPI backend service for Explainable Credit Risk Scoring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure Backend/, Model/, and Project Root are on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "Model"

for d in [str(BACKEND_DIR), str(MODEL_DIR), str(PROJECT_ROOT)]:
    if d not in sys.path:
        sys.path.insert(0, d)

try:
    from Backend.schemas import (
        ApplicantRequest,
        GlobalImportanceItem,
        HealthResponse,
        PredictionResponse,
    )
except ImportError:
    from schemas import (  # noqa: E402
        ApplicantRequest,
        GlobalImportanceItem,
        HealthResponse,
        PredictionResponse,
    )

try:
    from Model.explainer import get_explainer, get_global_importance
except ImportError:
    from explainer import get_explainer, get_global_importance  # noqa: E402

app = FastAPI(
    title="Explainable Credit Risk Scoring API",
    description=(
        "REST API for predicting probability of default, mapping scores to frozen risk bands, "
        "and explaining outcomes via real-time SHAP analysis."
    ),
    version="1.0.0",
)

# Enable CORS for Vite frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root health & metadata")
def root() -> dict[str, Any]:
    """Root endpoint returning service identity and documentation pointers."""
    return {
        "service": "Explainable Credit Risk API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get("/health", response_model=HealthResponse, summary="Service health status")
def health() -> HealthResponse:
    """Return health status, loaded model identity, base value, and risk band thresholds."""
    try:
        explainer = get_explainer()
        thresholds = explainer.risk_bands_config.get("thresholds", {
            "Low": {"min": 0.0, "max": 0.20},
            "Medium": {"min": 0.20, "max": 0.80},
            "High": {"min": 0.80, "max": 0.83},
            "Very High": {"min": 0.83, "max": 1.00},
        })
        return HealthResponse(
            status="healthy",
            model="XGBoost (contracted, leakage-free)",
            base_value=round(explainer.base_value, 4),
            risk_thresholds=thresholds,
            version="1.0.0",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Service initialization error: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse, summary="Score applicant and generate SHAP reasons")
def predict(applicant: ApplicantRequest) -> PredictionResponse:
    """
    Score an applicant's default risk:
    - Derives log_loan_amount on the server via log1p.
    - Predicts probability of default using the winning XGBoost pipeline.
    - Categorizes risk into Low / Medium / High / Very High tiers.
    - Computes real-time TreeExplainer SHAP contributions.
    - Translates top 5 drivers into clear, compliance-ready plain English explanations.
    - Verifies explanation additivity (base_value + sum(SHAP) = p_default).
    """
    try:
        explainer = get_explainer()
        result = explainer.score_and_explain(applicant.model_dump(mode="json"), top_n=5)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.get(
    "/shap/global",
    response_model=list[GlobalImportanceItem],
    summary="Get global SHAP feature importance",
)
def global_shap(
    grouped: bool = Query(
        default=True,
        description="Whether to return aggregated domain features (true) or raw one-hot features (false)",
    ),
) -> list[GlobalImportanceItem]:
    """Return offline global SHAP importance rankings for dashboard visualizations."""
    items = get_global_importance(grouped=grouped)
    return [GlobalImportanceItem(**item) for item in items]


@app.get("/model/metrics", summary="Model bake-off and population validation metrics")
def model_metrics() -> dict[str, Any]:
    """Return model comparison bake-off metrics and population risk band statistics."""
    import json
    comp_path = MODEL_DIR / "training_outputs" / "model_comparison.json"
    risk_path = MODEL_DIR / "risk_bands.json"
    comp_data = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else {}
    risk_data = json.loads(risk_path.read_text(encoding="utf-8")) if risk_path.exists() else {}

    # Simplify comparison list for dashboard display
    comparison_summary = []
    for m in comp_data.get("comparison", []):
        comparison_summary.append({
            "model": m.get("model"),
            "cv_roc_auc": round(m.get("cv_roc_auc", 0.0), 4),
            "test_roc_auc": round(m.get("roc_auc", 0.0), 4),
            "brier": round(m.get("brier", 0.0), 4),
            "precision": round(m.get("precision", 0.0), 4),
            "recall": round(m.get("recall", 0.0), 4),
            "f1": round(m.get("f1", 0.0), 4),
        })

    return {
        "comparison": comparison_summary,
        "winner": comp_data.get("winner", "xgboost"),
        "winner_test_roc_auc": round(comp_data.get("winner_test_roc_auc", 0.8994), 4),
        "risk_bands": risk_data.get("test_set", {}).get("bands", {}),
        "total_test_samples": risk_data.get("test_set", {}).get("n", 6305),
        "base_rate": round(risk_data.get("test_set", {}).get("base_rate", 0.1781), 4),
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI development server on http://127.0.0.1:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
