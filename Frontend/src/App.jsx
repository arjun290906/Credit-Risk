
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

function App() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    person_age: "",
    person_income: "",
    person_home_ownership: "",
    person_emp_length: "",
    loan_intent: "",
    loan_grade: "",
    loan_amnt: "",
    loan_int_rate: "",
    loan_status: "",
    loan_percent_income: "",
    credit_history_length: "",
  });

  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    setError("");
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const loanAmount = Number(formData.loan_amnt);

    if (loanAmount <= 0 || !Number.isFinite(loanAmount)) {
      setError("Loan amount must be greater than zero.");
      return;
    }

    const numericFields = [
      "person_age",
      "person_income",
      "person_emp_length",
      "loan_amnt",
      "loan_int_rate",
      "loan_percent_income",
      "credit_history_length",
    ];

    const data = { ...formData };

    numericFields.forEach((field) => {
      data[field] = Number(data[field]);
    });

    data.loan_status = Number(data.loan_status);

    // Natural logarithm of loan amount
    data.log_loan_amount = Math.log(loanAmount);

    // Navigate to the results page with the submitted data
    navigate("/results", {
      state: { formData: data },
    });
  };

  const renderInput = (name, label, type = "number", step = "any") => (
    <div className="form-group" key={name}>
      <label htmlFor={name}>{label}</label>
      <input
        id={name}
        name={name}
        type={type}
        step={step}
        value={formData[name]}
        onChange={handleChange}
        placeholder={`Enter ${label.toLowerCase()}`}
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
    <p>Intelligent lending decisions</p>
  </div>

  <div className="header-badge">
    ● CREDIT ANALYSIS
  </div>
</header>


      <main className="main-content">
        
      <div className="page-heading">
        <span className="eyebrow">CREDIT ASSESSMENT</span>

        <h2>
          Applicant <span>Information</span>
        </h2>

        <p>
          Enter the applicant's financial and loan details to prepare
          the information for credit risk assessment.
        </p>
      </div>


        <form className="form-card" onSubmit={handleSubmit}>
          <div className="section-heading">
            <span className="section-number">01</span>
            <div>
              <h3>Personal Details</h3>
              <p>Information about the loan applicant</p>
            </div>
          </div>

          <div className="form-grid">
            {renderInput("person_age", "Person Age", "number", "1")}
            {renderInput("person_income", "Person Income")}
            {renderDropdown("person_home_ownership", "Home Ownership", [
              "OWN",
              "MORTGAGE",
              "RENT",
              "OTHER",
            ])}
            {renderInput("person_emp_length", "Employment Length (years)")}
            {renderInput(
              "credit_history_length",
              "Credit History Length (years)"
            )}
          </div>

          <div className="form-divider" />

          <div className="section-heading">
            <span className="section-number">02</span>
            <div>
              <h3>Loan Details</h3>
              <p>Enter the requested loan information</p>
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

            {renderDropdown("loan_grade", "Loan Grade", [
              "B",
              "C",
              "A",
              "D",
              "E",
              "F",
              "G",
            ])}

            {renderInput("loan_amnt", "Loan Amount")}

            {renderInput(
              "loan_int_rate",
              "Loan Interest Rate (%)"
            )}

            {renderInput(
              "loan_percent_income",
              "Loan Percent Income"
            )}

            {renderDropdown("loan_status", "Loan Status", ["0", "1"])}
          </div>

          <div className="form-divider" />

          <div className="section-heading">
            <span className="section-number">03</span>
            <div>
              <h3>Calculated Feature</h3>
              <p>Automatically derived from the loan amount</p>
            </div>
          </div>

          <div className="calculated-field">
            <div>
              <label>Log Loan Amount</label>
              <p>Natural logarithm of the entered loan amount</p>
            </div>
            <div className="calculated-value">
              {Number(formData.loan_amnt) > 0
                ? Math.log(Number(formData.loan_amnt)).toFixed(4)
                : "Auto-calculated"}
            </div>
          </div>

          {error && <p className="error-message">{error}</p>}

          <div className="form-footer">
            <p>
              <span className="required-star">*</span> All fields are
              required.
            </p>

            <button type="submit" className="submit-button">
              Predict Credit Risk <span aria-hidden="true">→</span>
            </button>
          </div>
        </form>

        <p className="bottom-note">
          Credit Risk Predictor · Data-driven credit assessment
        </p>
      </main>
    </div>
  );
}

export default App;
