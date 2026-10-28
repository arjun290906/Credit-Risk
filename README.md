# Explainable Credit Risk Scoring Engine

An end-to-end, full-stack machine learning application that predicts loan default probability, categorizes borrowers into calibrated risk tiers, generates real-time plain-language SHAP explanations, and provides interactive what-if counterfactual simulations with regulatory adverse action notices.

---

## 1. System Architecture & Workflow

```text
                     +---------------------------------------+
                     |         React 19 Frontend (Vite)      |
                     |  - Scoring Form (No loan_status)      |
                     |  - Explainable Assessment Report      |
                     |  - What-If Counterfactual Simulator   |
                     |  - Portfolio Model Dashboard          |
                     +-------------------+-------------------+
                                         |
                                HTTP POST /predict
                                         v
                     +---------------------------------------+
                     |         FastAPI REST Backend          |
                     |  - Pydantic v2 Schema Validation      |
                     |  - Derives log_loan_amount = log1p()  |
                     |  - Auto-computes DTI ratio            |
                     +-------------------+-------------------+
                                         |
                  +----------------------+----------------------+
                  |                                             |
                  v                                             v
     +-------------------------+                   +-------------------------+
     |     XGBoost Pipeline    |                   |    SHAP TreeExplainer   |
     | - OneHot + StandardScaler|                  | - 100 Train Baseline Ref |
     | - Test ROC-AUC: 0.8994  |                   | - ~3.5ms Local Latency  |
     | - Predicts p_default    |                   | - One-Hot Aggregation   |
     +------------+------------+                   +------------+------------+
                  |                                             |
                  +----------------------+----------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |          Decisioning Layer            |
                     |  - Frozen Risk Tiers (Low/Med/Hi/VH)  |
                     |  - Top 5 Plain-Language Explanations  |
                     |  - Additive Verification Check        |
                     |  - FCRA/ECOA Adverse Action Advisory  |
                     +---------------------------------------+
```

---

## 2. Frozen Product Contract

The system enforces a strict input/output contract established in [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md).

### Applicant Inputs (User-Entered)
| Field | Type | Range / Options | UI Description |
| :--- | :--- | :--- | :--- |
| `person_age` | Integer | $\ge 18$ | Applicant age in years |
| `person_income` | Float | $> 0$ | Annual income in USD |
| `person_home_ownership` | Enum | `RENT`, `MORTGAGE`, `OWN`, `OTHER` | Housing status |
| `person_emp_length` | Float | $\ge 0.0$ | Employment tenure in years |
| `credit_history_length` | Integer | $\ge 0$ | Borrowing track record in years |
| `loan_intent` | Enum | `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` | Loan purpose |
| `loan_grade` | Enum | `A`, `B`, `C`, `D`, `E`, `F`, `G` | Officer-assigned risk grade |
| `loan_amnt` | Float | $> 0$ | Requested principal in USD |
| `loan_int_rate` | Float | $\ge 0.0\%$ | Interest rate percentage |
| `loan_percent_income` | Float | $0.0 - 1.0$ (or $0 - 100\%$) | Loan % of income (auto-derived) |

### Server-Derived Feature (Leakage Prevention)
- **`log_loan_amount`**: Calculated server-side as $\ln(1 + \text{loan\_amnt})$ (`np.log1p`). Never trusted from client input.
- **`loan_status` Dropped**: The current loan outcome column from Kaggle was explicitly removed from training and the form to prevent target leakage.

---

## 3. Calibrated Risk Bands

Because the model was trained with class-balancing (`scale_pos_weight = 4.615`), probabilities are shifted upward by $+1.53$ log-odds ($\sigma(1.53) \approx 0.82$ for a 50% unweighted risk). Empirical analysis of the 6,305 test-set applicants established the following frozen risk thresholds:

$$\text{Low: } p < 0.20 \quad\Big|\quad \text{Medium: } 0.20 \le p < 0.80 \quad\Big|\quad \text{High: } 0.80 \le p < 0.83 \quad\Big|\quad \text{Very High: } p \ge 0.83$$

| Risk Category | Probability Range | Test Population ($n$) | Share | Mean Predicted $P$ | Observed Historical Default Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low** | $[0.00, 0.20)$ | 4,105 | 65.11% | 0.0012 | **0.00%** |
| **Medium** | $[0.20, 0.80)$ | 402 | 6.38% | 0.7777 | **47.01%** |
| **High** | $[0.80, 0.83)$ | 906 | 14.37% | 0.8156 | **50.66%** |
| **Very High** | $[0.83, 1.00]$ | 892 | 14.15% | 0.8458 | **53.25%** |

*Observed historical default rates increase strictly monotonically across all four tiers ($0.0\% \to 47.0\% \to 50.7\% \to 53.3\%$).*

---

## 4. Model Bake-Off Results

Three candidate models were evaluated on the identical 80/20 stratified split:

| Model Architecture | 5-Fold CV ROC-AUC | Test ROC-AUC | Brier Calibration Loss | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | **0.8965** | **0.8994** | **0.1208** | **0.5107** | **1.0000** | **0.6761** | **★ Production Model** |
| **Logistic Regression** | 0.8966 | 0.8993 | 0.1215 | 0.5105 | 1.0000 | 0.6759 | Evaluated Candidate |
| **Random Forest** | 0.8963 | 0.8978 | 0.1201 | 0.5107 | 1.0000 | 0.6761 | Evaluated Candidate |

---

## 5. Quick Start & Run Instructions

### Prerequisites
- Python 3.12 (virtual environment provided in `Model/.venv`)
- Node.js 18+ and npm

### Terminal 1 — Start the FastAPI Backend
```powershell
# From the project root
.\Model\.venv\Scripts\python.exe -m uvicorn Backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Service Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Terminal 2 — Start the React Frontend
```powershell
# In a separate terminal
npm --prefix Frontend run dev
```
- Open your browser to [http://localhost:5173](http://localhost:5173).

---

## 6. Automated Test Suites

### 1. HTTP Server Integration Test
Spins up Uvicorn on localhost and tests `/health`, `/predict`, and `/shap/global`:
```powershell
.\Model\.venv\Scripts\python.exe .\Backend\test_server.py
```

### 2. SHAP Explanation Layer Mathematical Validation Suite
Evaluates additivity across 500 test cases, sensitivity monotonicity, and underwriting personas:
```powershell
.\Model\.venv\Scripts\python.exe .\Model\validate_shap_layer.py
```
Outputs:
- `Model/training_outputs/shap_validation_results.json`
- `Model/training_outputs/shap_validation_report.md`

### 3. Frontend Lint & Production Build
```powershell
npm --prefix Frontend run lint
npm --prefix Frontend run build
```

---

## 7. Interactive Features & Demo Walkthrough

1. **Applicant Scoring Form (`/`)**:
   - Collects 10 inputs with real-time DTI auto-calculation and server log preview.
   - Disables target leakage fields.
2. **Assessment Results Report (`/results`)**:
   - Large probability gauge ($p_{\text{default}}$) with adaptive score meter.
   - Calibrated Risk Tier Badge (**Low**, **Medium**, **High**, **Very High**) and underwriting guidance.
   - Top 5 plain-language SHAP driver cards with directional badges (`▲ Increases Risk` vs `▼ Decreases Risk`) and impact divergence bars.
   - **Mathematical Explanation Fidelity Card**: Proves that $\text{Base Value} + \sum \text{SHAP} = \text{Model Probability}$ down to $10^{-6}$ precision.
3. **What-If Counterfactual Simulator (`/results`)**:
   - Live sliders for Loan Amount, Annual Income, and Interest Rate, plus Loan Grade selector.
   - Sub-5ms recalculation displaying live delta score (e.g. `▼ -24.5% Risk Reduction`).
   - **FCRA / ECOA Adverse Action Notice**: Explains the top 3 principal denial factors and provides actionable counterfactual recommendations for borrowers.
4. **Portfolio Model Dashboard (`/dashboard`)**:
   - Horizontal bar chart of Global SHAP Importance with Domain vs One-Hot toggle.
   - 3-model bake-off comparison table.
   - Risk band population distribution cards.

---

## 8. Honest Limitations & Ethical Considerations

- **Kaggle Dataset Nature**: Uses a synthetic credit risk table. In a real-world deployment, this must be augmented with live credit bureau APIs (e.g., Experian, FICO, Plaid).
- **Target Semantics**: The target `default_status` represents historical default-on-file (`cb_person_default_on_file = Y`), rather than a prospective "will this new loan default" outcome.
- **Loan Grade Signal**: In this Kaggle dataset, `loan_grade` A & B have 0 defaults on file, while C–G have ~50% default rates. While realistic as an officer input, treating it as an independent causal feature would introduce bias. We document it as an officer-assigned synthesis.
- **Class Imbalance**: Positive defaults represent 17.8% of the data. Training with `scale_pos_weight = 4.615` shifts output probabilities upward, which is why risk thresholds were empirically calibrated on the probability histogram rather than standard 0.50 cuts.

---

## 9. Repository Structure

```text
Credit-Risk/
├── Backend/
│   ├── main.py                     # FastAPI service with CORS and route handlers
│   ├── schemas.py                  # Pydantic v2 input/output schemas
│   ├── test_server.py              # Automated HTTP integration test
│   └── README.md                   # Backend documentation
├── Frontend/
│   ├── src/
│   │   ├── App.jsx                 # Applicant scoring form
│   │   ├── Results.jsx             # Results, SHAP cards, and What-If simulator
│   │   ├── Dashboard.jsx           # Portfolio dashboard & global SHAP charts
│   │   ├── main.jsx                # React Router v7 configuration
│   │   └── App.css                 # Burgundy/rose/Inter styling
│   └── package.json
├── Model/
│   ├── best_model.pkl              # Fitted winning XGBoost pipeline
│   ├── explainer.py                # Real-time single-applicant SHAP engine
│   ├── background_sample.pkl       # 100-sample training background reference
│   ├── choose_risk_bands.py        # Risk band threshold selection script
│   ├── prepare_shap_artifacts.py   # Global SHAP generator
│   ├── validate_shap_layer.py      # Mathematical validation test suite
│   ├── risk_bands.json             # Frozen risk tier configuration
│   ├── requirements.txt            # Python package specifications
│   └── training_outputs/           # Calibration plots, histograms, metrics
├── PRODUCT_CONTRACT.md             # Frozen product specification
├── requirements.txt                # Root Python dependencies
└── README.md                       # Main project documentation
```
