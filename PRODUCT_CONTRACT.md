# Credit Risk Scoring — Product Contract

This is the frozen interface for the lab system. The report, form, model, and API must match this document. Later implementation (retrain, SHAP, FastAPI, results UI) follows these fields only.

**Target:** `default_status` — 1 if the applicant has a historical default on file (`cb_person_default_on_file` = Y), else 0. This is **not** “will this new loan default”; it is the label available in the Kaggle credit-risk table after cleaning.

**Do not collect or send `loan_status`.** In this dataset it is the current-loan outcome. Using it as a feature leaks the decision we are trying to support. Drop it from training `X`, from the form, and from SHAP.

---

## Applicant inputs (user-entered)

| Field | Type | Allowed values / range | UI label |
|---|---|---|---|
| `person_age` | number | ≥ 18, integer | Person age |
| `person_income` | number | > 0, annual | Person income |
| `person_home_ownership` | enum | `OWN`, `MORTGAGE`, `RENT`, `OTHER` | Home ownership |
| `person_emp_length` | number | ≥ 0, years | Employment length (years) |
| `credit_history_length` | number | ≥ 0, years | Credit history length (years) |
| `loan_intent` | enum | `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` | Loan intent |
| `loan_grade` | enum | `A`–`G` | Loan grade (officer-assigned; see limitations) |
| `loan_amnt` | number | > 0 | Loan amount |
| `loan_int_rate` | number | ≥ 0, percent | Loan interest rate (%) |
| `loan_percent_income` | number | 0–1 (or 0–100% displayed, stored as fraction matching training) | Loan percent of income |

**Derived on the server (never typed, never trusted from the client):**

| Field | Rule |
|---|---|
| `log_loan_amount` | `log1p(loan_amnt)` — same as the cleaning notebook, **not** `ln(loan_amnt)` |

Optional UI helper: pre-fill `loan_percent_income` as `loan_amnt / person_income` when both are present; user may override.

---

## Fields we will **not** claim

The proposal mentioned **existing debt** and **number of previous loans**. They are **not in the dataset**. Do not add them to the form, model, or report as if they exist.

Report wording: use **loan percent of income** as the debt-to-income style indicator. Credit history length stands in for credit maturity. Say this explicitly in the problem/solution sections.

---

## Model outputs (API + results page)

Every successful score returns:

| Field | Meaning |
|---|---|
| `p_default` | Probability that `default_status = 1`, from `predict_proba` |
| `risk_category` | `Low` / `Medium` / `High` / `Very High` from frozen thresholds (established in Step 3 on test-set probability distribution):<br>• **Low**: `p < 0.20` (65.1% share, 0.0% observed default rate)<br>• **Medium**: `0.20 <= p < 0.80` (6.4% share, 47.0% observed default rate)<br>• **High**: `0.80 <= p < 0.83` (14.4% share, 50.7% observed default rate)<br>• **Very High**: `p >= 0.83` (14.1% share, 53.3% observed default rate) |
| `reasons` | Top **5** SHAP contributions for this applicant: feature name, signed SHAP value, `increases` or `decreases` risk, one plain-language sentence |
| `base_value` | SHAP expected probability (for explanation checks) |

Global SHAP importance is a **dashboard** artifact, not part of the per-applicant contract.

---

## Alignment checklist

| Surface | Must match this contract |
|---|---|
| Mini-project report | Inputs, target, and outputs as above; drop “previous loans” / generic “existing debt” |
| `Frontend` form | Collect only the applicant-input table; hide `loan_status`; show log amount as derived |
| `training.py` | `y = default_status`; `X` = inputs + `log_loan_amount`; no `loan_status` |
| `/predict` (Step 5) | Same JSON field names as the tables above |
| Results page | Show `p_default`, badge, five reasons — not a raw form dump |

---

## Explicit limitations (keep in the report)

- Public Kaggle-style table, not a live bureau feed.
- Target is historical default-on-file, not a prospective “this loan will default” label.
- `loan_grade` is often assigned using similar risk information; treat it as an available officer input, not an independent causal factor.
- Class imbalance and missing original `loan_int_rate` / `person_emp_length` rows were handled in cleaning (median rate; drop missing employment length).
