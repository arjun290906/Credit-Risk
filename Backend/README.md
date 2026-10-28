# Explainable Credit Risk API (FastAPI)

Production-grade REST backend for real-time credit risk scoring, risk band categorization, and SHAP-based plain-language explanations.

---

## 1. Quick Start

Activate the virtual environment and launch Uvicorn:

```bash
# From the project root
.\Model\.venv\Scripts\python.exe -m uvicorn Backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from the `Backend/` directory:

```bash
cd Backend
..\Model\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000 --reload
```

API Documentation:
- **Swagger Interactive UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 2. API Endpoints

### `GET /health`
Returns service status, loaded model type, baseline probability, and frozen risk band thresholds.

**Response:**
```json
{
  "status": "healthy",
  "model": "XGBoost (contracted, leakage-free)",
  "base_value": 0.2981,
  "risk_thresholds": {
    "Low": {"min": 0.0, "max": 0.20},
    "Medium": {"min": 0.20, "max": 0.80},
    "High": {"min": 0.80, "max": 0.83},
    "Very High": {"min": 0.83, "max": 1.00}
  },
  "version": "1.0.0"
}
```

---

### `POST /predict`
Scores a live applicant, categorizes risk into Low / Medium / High / Very High, computes real-time SHAP values, and translates the top 5 contributors into plain English sentences.

**Request Payload:**
```json
{
  "person_age": 28,
  "person_income": 45000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "credit_history_length": 4,
  "loan_intent": "EDUCATION",
  "loan_grade": "C",
  "loan_amnt": 12000,
  "loan_int_rate": 13.5,
  "loan_percent_income": 0.27
}
```

**Response Payload:**
```json
{
  "p_default": 0.8037,
  "risk_category": "High",
  "base_value": 0.2981,
  "reasons": [
    {
      "feature": "loan_grade",
      "feature_name": "Loan Grade",
      "applicant_value": "Grade C",
      "shap_value": 0.4166,
      "direction": "increases",
      "plain_text": "Assigned loan grade 'C' carries higher default risk (+41.7% risk)."
    },
    {
      "feature": "loan_int_rate",
      "feature_name": "Interest Rate",
      "applicant_value": "13.50%",
      "shap_value": 0.1002,
      "direction": "increases",
      "plain_text": "Elevated interest rate of 13.50% increases repayment burden and default risk (+10.0%)."
    },
    {
      "feature": "person_income",
      "feature_name": "Annual Income",
      "applicant_value": "$45,000",
      "shap_value": -0.0061,
      "direction": "decreases",
      "plain_text": "Robust annual income of $45,000 provides strong repayment support (-0.6%)."
    },
    {
      "feature": "person_emp_length",
      "feature_name": "Employment Length",
      "applicant_value": "3.0 yrs",
      "shap_value": 0.0041,
      "direction": "increases",
      "plain_text": "Limited employment length (3.0 yrs) indicates potential income variability (+0.4%)."
    },
    {
      "feature": "loan_intent",
      "feature_name": "Loan Intent / Purpose",
      "applicant_value": "Education",
      "shap_value": -0.0038,
      "direction": "decreases",
      "plain_text": "Loan purpose (Education) has historically lower risk tendencies (-0.4%)."
    }
  ],
  "explanation_check": {
    "base_value": 0.2981,
    "sum_shap": 0.5056,
    "reconstructed_prob": 0.8037,
    "model_prob": 0.8037,
    "absolute_diff": 0.0,
    "is_valid": true
  },
  "applicant_summary": {
    "person_home_ownership": "Rent",
    "loan_intent": "Education",
    "loan_grade": "Grade C",
    "person_age": "28 yrs",
    "person_income": "$45,000",
    "person_emp_length": "3.0 yrs",
    "loan_amnt": "$12,000",
    "loan_int_rate": "13.50%",
    "loan_percent_income": "27.0%",
    "credit_history_length": "4 yrs"
  }
}
```

---

### `GET /shap/global?grouped=true`
Returns the global feature importance ranking for dashboard charts.

---

## 3. Automated Testing

Run the HTTP server integration test suite:

```bash
.\Model\.venv\Scripts\python.exe .\Backend\test_server.py
```
