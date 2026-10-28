import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const navigate = useNavigate();
  const location = useLocation();

  // If navigated back with existing applicant data, prefill the form
  const initialData = location.state?.applicant || {
    person_age: "",
    person_income: "",
    person_home_ownership: "",
    person_emp_length: "",
    credit_history_length: "",
    loan_intent: "",
    loan_grade: "",
    loan_amnt: "",
    loan_int_rate: "",
    loan_percent_income: "",
  };

  const [formData, setFormData] = useState(initialData);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => {
      const next = { ...prev, [name]: value };

      // Auto-compute DTI ratio if loan amount or income changes
      if (name === "loan_amnt" || name === "person_income") {
        const amt = Number(name === "loan_amnt" ? value : next.loan_amnt);
        const inc = Number(name === "person_income" ? value : next.person_income);
        if (amt > 0 && inc > 0) {
          const ratio = amt / inc;
          next.loan_percent_income = ratio <= 1 ? (ratio * 100).toFixed(1) : "100.0";
        }
      }
      return next;
    });

    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const age = Number(formData.person_age);
    const income = Number(formData.person_income);
    const loanAmount = Number(formData.loan_amnt);
    const intRate = Number(formData.loan_int_rate);

    if (age < 18) {
      setError("Applicant must be at least 18 years old.");
      return;
    }
    if (income <= 0 || !Number.isFinite(income)) {
      setError("Annual income must be greater than zero.");
      return;
    }
    if (loanAmount <= 0 || !Number.isFinite(loanAmount)) {
      setError("Loan amount must be greater than zero.");
      return;
    }
    if (intRate < 0 || !Number.isFinite(intRate)) {
      setError("Interest rate cannot be negative.");
      return;
    }

    // Convert loan_percent_income from percentage (0-100) to fraction (0-1)
    const rawLpi = Number(formData.loan_percent_income);
    const lpiFraction = rawLpi > 1.0 ? rawLpi / 100.0 : rawLpi > 0 ? rawLpi : loanAmount / income;

    const payload = {
      person_age: age,
      person_income: income,
      person_home_ownership: formData.person_home_ownership,
      person_emp_length: Number(formData.person_emp_length || 0),
      credit_history_length: Number(formData.credit_history_length || 0),
      loan_intent: formData.loan_intent,
      loan_grade: formData.loan_grade,
      loan_amnt: loanAmount,
      loan_int_rate: intRate,
      loan_percent_income: Number(lpiFraction.toFixed(4)),
    };

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with status ${response.status}`);
      }

      const prediction = await response.json();

      navigate("/results", {
        state: {
          prediction,
          applicant: formData,
        },
      });
    } catch (err) {
      console.error("Prediction request failed:", err);
      setError(
        err.message.includes("Failed to fetch")
          ? "Cannot connect to scoring service. Please ensure the backend is running at http://127.0.0.1:8000."
          : `Prediction error: ${err.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  const renderInput = (name, label, type = "number", step = "any", placeholder = "") => (
    <div className="form-group" key={name}>
      <label htmlFor={name}>{label}</label>
      <input
        id={name}
        name={name}
        type={type}
        step={step}
        value={formData[name]}
        onChange={handleChange}
        placeholder={placeholder || `Enter ${label.toLowerCase()}`}
        min={type === "number" ? "0" : undefined}
        required
      />
    </div>
  );

  const renderDropdown = (name, label, options) => (
    <div className="form-group" key={name}>
      <label htmlFor={name}>{label}</label>
      <select
        id={name}
        name={name}
        value={formData[name]}
        onChange={handleChange}
        required
      >
        <option value="">Select {label}</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand-icon">
          <span className="brand-dot"></span>
        </div>

        <div>
          <h1>Credit Risk Predictor</h1>
          <p>Explainable AI lending decisions</p>
        </div>

        <nav className="header-nav">
          <button className="nav-link nav-active">
            Scoring Form
          </button>
          <button onClick={() => navigate("/dashboard")} className="nav-link">
            Portfolio Dashboard
          </button>
        </nav>
      </header>

      <main className="main-content">
        <div className="page-heading">
          <span className="eyebrow">CREDIT ASSESSMENT</span>
          <h2>
            Applicant <span>Information</span>
          </h2>
          <p>
            Enter financial details to evaluate probability of default, assign a calibrated risk tier,
            and inspect the top 5 plain-language SHAP drivers.
          </p>
        </div>

        <form className="form-card" onSubmit={handleSubmit}>
          {/* Section 01: Personal Details */}
          <div className="section-heading">
            <span className="section-number">01</span>
            <div>
              <h3>Personal Details</h3>
              <p>Applicant age, income, employment, and credit history</p>
            </div>
          </div>

          <div className="form-grid">
            {renderInput("person_age", "Person Age", "number", "1", "e.g. 28")}
            {renderInput("person_income", "Annual Income ($)", "number", "100", "e.g. 55000")}
            {renderDropdown("person_home_ownership", "Home Ownership", [
              "RENT",
              "MORTGAGE",
              "OWN",
              "OTHER",
            ])}
            {renderInput("person_emp_length", "Employment Length (years)", "number", "0.5", "e.g. 3.5")}
            {renderInput("credit_history_length", "Credit History Length (years)", "number", "1", "e.g. 5")}
          </div>

          <div className="form-divider" />

          {/* Section 02: Loan Details */}
          <div className="section-heading">
            <span className="section-number">02</span>
            <div>
              <h3>Loan Details</h3>
              <p>Requested loan parameters and officer-assigned grade</p>
            </div>
          </div>

          <div className="form-grid">
            {renderDropdown("loan_intent", "Loan Intent", [
              "EDUCATION",
              "MEDICAL",
              "VENTURE",
              "PERSONAL",
              "HOMEIMPROVEMENT",
              "DEBTCONSOLIDATION",
            ])}

            {renderDropdown("loan_grade", "Loan Grade (Officer Assigned)", [
              "A",
              "B",
              "C",
              "D",
              "E",
              "F",
              "G",
            ])}

            {renderInput("loan_amnt", "Loan Amount ($)", "number", "100", "e.g. 10000")}
            {renderInput("loan_int_rate", "Loan Interest Rate (%)", "number", "0.01", "e.g. 11.50")}
            {renderInput("loan_percent_income", "Loan % of Income (DTI Indicator, %)", "number", "0.1", "e.g. 18.2")}
          </div>

          <div className="form-divider" />

          {/* Section 03: Calculated Features */}
          <div className="section-heading">
            <span className="section-number">03</span>
            <div>
              <h3>Calculated Features</h3>
              <p>Server-derived values matching model training specification</p>
            </div>
          </div>

          <div className="calculated-grid">
            <div className="calculated-field">
              <div>
                <label>Log Loan Amount</label>
                <p>Natural log formula: ln(1 + loan_amnt)</p>
              </div>
              <div className="calculated-value">
                {Number(formData.loan_amnt) > 0
                  ? Math.log1p(Number(formData.loan_amnt)).toFixed(4)
                  : "Auto-calculated"}
              </div>
            </div>

            <div className="calculated-field">
              <div>
                <label>Debt-to-Income (DTI) Ratio</label>
                <p>Requested loan relative to annual income</p>
              </div>
              <div className="calculated-value">
                {Number(formData.person_income) > 0 && Number(formData.loan_amnt) > 0
                  ? `${((Number(formData.loan_amnt) / Number(formData.person_income)) * 100).toFixed(1)}%`
                  : "Auto-calculated"}
              </div>
            </div>
          </div>

          {error && <div className="error-message">{error}</div>}

          <div className="form-footer">
            <p>
              <span className="required-star">*</span> All fields required.
              Explanations computed in real-time via SHAP TreeExplainer.
            </p>

            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? (
                <>Evaluating Risk & SHAP...</>
              ) : (
                <>
                  Score & Explain Risk <span aria-hidden="true">→</span>
                </>
              )}
            </button>
          </div>
        </form>

        <p className="bottom-note">
          Credit Risk Predictor · Contracted Features Only (No Loan Status Leakage)
        </p>
      </main>
    </div>
  );
}

export default App;
