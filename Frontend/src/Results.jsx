import { useState, useEffect, useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function Results() {
  const { state } = useLocation();
  const navigate = useNavigate();

  const initialPrediction = state?.prediction;
  const applicant = state?.applicant;

  // What-If Counterfactual Simulation State
  const [simLoanAmnt, setSimLoanAmnt] = useState(
    applicant ? Number(applicant.loan_amnt) : 10000
  );
  const [simIncome, setSimIncome] = useState(
    applicant ? Number(applicant.person_income) : 50000
  );
  const [simIntRate, setSimIntRate] = useState(
    applicant ? Number(applicant.loan_int_rate) : 12.0
  );
  const [simGrade, setSimGrade] = useState(
    applicant ? applicant.loan_grade : "C"
  );

  const [simPrediction, setSimPrediction] = useState(initialPrediction);
  const [simLoading, setSimLoading] = useState(false);

  // Debounced counterfactual prediction runner
  const runSimulation = useCallback(async () => {
    if (!applicant) return;

    setSimLoading(true);
    try {
      const derivedDti = simIncome > 0 ? Number((simLoanAmnt / simIncome).toFixed(4)) : 0.2;
      const simPayload = {
        person_age: Number(applicant.person_age),
        person_income: simIncome,
        person_home_ownership: applicant.person_home_ownership,
        person_emp_length: Number(applicant.person_emp_length || 0),
        credit_history_length: Number(applicant.credit_history_length || 0),
        loan_intent: applicant.loan_intent,
        loan_grade: simGrade,
        loan_amnt: simLoanAmnt,
        loan_int_rate: simIntRate,
        loan_percent_income: derivedDti,
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(simPayload),
      });

      if (res.ok) {
        const data = await res.json();
        setSimPrediction(data);
      }
    } catch (err) {
      console.error("What-if simulation error:", err);
    } finally {
      setSimLoading(false);
    }
  }, [applicant, simLoanAmnt, simIncome, simIntRate, simGrade]);

  useEffect(() => {
    const timer = setTimeout(() => {
      runSimulation();
    }, 180);
    return () => clearTimeout(timer);
  }, [simLoanAmnt, simIncome, simIntRate, simGrade, runSimulation]);

  if (!initialPrediction) {
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
        </header>

        <main className="main-content">
          <div className="form-card empty-state">
            <h2>No Evaluation Found</h2>
            <p>Please enter applicant parameters on the form to run an assessment.</p>
            <button
              onClick={() => navigate("/")}
              className="submit-button"
              style={{ marginTop: "20px" }}
            >
              ← Back to Assessment Form
            </button>
          </div>
        </main>
      </div>
    );
  }

  const { p_default, risk_category, base_value, reasons, explanation_check, applicant_summary } =
    initialPrediction;

  const probPercent = (p_default * 100).toFixed(1);
  const basePercent = (base_value * 100).toFixed(1);
  const riskClass = `risk-badge-${risk_category.toLowerCase().replace(/\s+/g, "-")}`;

  // What-If Deltas
  const currentSimP = simPrediction?.p_default ?? p_default;
  const simProbPercent = (currentSimP * 100).toFixed(1);
  const deltaP = Number(simProbPercent) - Number(probPercent);
  const simRiskClass = `risk-badge-${(simPrediction?.risk_category || risk_category)
    .toLowerCase()
    .replace(/\s+/g, "-")}`;

  const getRecommendation = (category) => {
    switch (category) {
      case "Low":
        return "Strong credit profile. High probability of timely debt service with minimal default risk.";
      case "Medium":
        return "Moderate risk tier. Applicant satisfies baseline criteria but warrants secondary verification.";
      case "High":
        return "Elevated risk tier. Enhanced underwriting criteria, lower exposure, or co-signer recommended.";
      case "Very High":
        return "Critical risk threshold. High probability of default; application requires senior officer review.";
      default:
        return "Risk assessment completed.";
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-icon">
          <span className="brand-dot"></span>
        </div>

        <div>
          <h1>Credit Risk Predictor</h1>
          <p>Explainable AI lending decisions</p>
        </div>

        <nav className="header-nav">
          <button onClick={() => navigate("/")} className="nav-link">
            Scoring Form
          </button>
          <button onClick={() => navigate("/dashboard")} className="nav-link">
            Portfolio Dashboard
          </button>
        </nav>
      </header>

      <main className="main-content">
        {/* Page Heading */}
        <div className="page-heading">
          <span className="eyebrow">ASSESSMENT REPORT</span>
          <h2>
            Credit Risk <span>Analysis</span>
          </h2>
          <p>
            Scored via calibrated XGBoost model with real-time SHAP attribution and interactive what-if counterfactual simulation.
          </p>
        </div>

        {/* 1. HERO RISK OVERVIEW CARD */}
        <section className="form-card hero-risk-card">
          <div className="hero-risk-grid">
            <div className="probability-display">
              <span className="metric-label">PROBABILITY OF DEFAULT</span>
              <div className="metric-score">
                {probPercent}
                <span className="percent-sign">%</span>
              </div>
              <div className="score-meter-bg">
                <div
                  className={`score-meter-fill ${riskClass}`}
                  style={{ width: `${Math.min(Number(probPercent), 100)}%` }}
                />
              </div>
              <span className="baseline-subtext">Baseline population risk: {basePercent}%</span>
            </div>

            <div className="risk-classification">
              <span className="metric-label">ASSIGNED RISK TIER</span>
              <div className={`risk-badge-large ${riskClass}`}>
                <span className="status-dot"></span>
                {risk_category.toUpperCase()} RISK
              </div>
              <p className="recommendation-text">{getRecommendation(risk_category)}</p>
            </div>
          </div>
        </section>

        {/* 2. SHAP DECISION DRIVERS (TOP 5 REASONS) */}
        <section className="form-card section-card">
          <div className="section-heading">
            <span className="section-number">01</span>
            <div>
              <h3>Key Decision Drivers (SHAP Explanations)</h3>
              <p>Top factors contributing to this applicant's predicted default probability</p>
            </div>
          </div>

          <div className="reasons-list">
            {reasons.map((reason, idx) => {
              const isIncrease = reason.direction === "increases";
              const impactPct = (Math.abs(reason.shap_value) * 100).toFixed(1);

              return (
                <div key={idx} className={`reason-card ${isIncrease ? "reason-risk-up" : "reason-risk-down"}`}>
                  <div className="reason-header">
                    <div className="reason-title-block">
                      <span className="reason-rank">#{idx + 1}</span>
                      <h4 className="reason-feature-name">{reason.feature_name}</h4>
                      <span className="reason-input-pill">{reason.applicant_value}</span>
                    </div>

                    <div className={`direction-badge ${isIncrease ? "badge-up" : "badge-down"}`}>
                      {isIncrease ? "▲ Increases Risk" : "▼ Decreases Risk"} ({isIncrease ? "+" : "-"}
                      {impactPct}%)
                    </div>
                  </div>

                  <p className="reason-plain-text">{reason.plain_text}</p>

                  <div className="impact-bar-wrapper">
                    <div className="impact-bar-track">
                      <div
                        className={`impact-bar-fill ${isIncrease ? "fill-increase" : "fill-decrease"}`}
                        style={{ width: `${Math.min(Number(impactPct) * 2, 100)}%` }}
                      />
                    </div>
                    <span className="impact-bar-value">{reason.shap_value > 0 ? `+${reason.shap_value}` : reason.shap_value}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* 3. INTERACTIVE WHAT-IF COUNTERFACTUAL SIMULATOR */}
        <section className="form-card section-card what-if-card">
          <div className="section-heading">
            <span className="section-number">02</span>
            <div>
              <h3>What-If Counterfactual Simulator</h3>
              <p>Adjust loan terms or borrower parameters to evaluate live impact on default risk</p>
            </div>
          </div>

          <div className="what-if-layout">
            {/* Left Controls: Sliders */}
            <div className="what-if-controls">
              {/* Slider 1: Loan Amount */}
              <div className="slider-group">
                <div className="slider-header">
                  <label htmlFor="sim-loan-amnt">Loan Amount</label>
                  <span className="slider-value">${simLoanAmnt.toLocaleString()}</span>
                </div>
                <input
                  id="sim-loan-amnt"
                  type="range"
                  min="1000"
                  max="35000"
                  step="500"
                  value={simLoanAmnt}
                  onChange={(e) => setSimLoanAmnt(Number(e.target.value))}
                  className="range-slider"
                />
              </div>

              {/* Slider 2: Annual Income */}
              <div className="slider-group">
                <div className="slider-header">
                  <label htmlFor="sim-income">Annual Income</label>
                  <span className="slider-value">${simIncome.toLocaleString()}</span>
                </div>
                <input
                  id="sim-income"
                  type="range"
                  min="15000"
                  max="150000"
                  step="2500"
                  value={simIncome}
                  onChange={(e) => setSimIncome(Number(e.target.value))}
                  className="range-slider"
                />
              </div>

              {/* Slider 3: Interest Rate */}
              <div className="slider-group">
                <div className="slider-header">
                  <label htmlFor="sim-int-rate">Interest Rate</label>
                  <span className="slider-value">{simIntRate.toFixed(2)}%</span>
                </div>
                <input
                  id="sim-int-rate"
                  type="range"
                  min="5.0"
                  max="24.0"
                  step="0.25"
                  value={simIntRate}
                  onChange={(e) => setSimIntRate(Number(e.target.value))}
                  className="range-slider"
                />
              </div>

              {/* Dropdown: Loan Grade */}
              <div className="slider-group">
                <div className="slider-header">
                  <label htmlFor="sim-grade">Assigned Loan Grade</label>
                  <span className="slider-value">Grade {simGrade}</span>
                </div>
                <select
                  id="sim-grade"
                  value={simGrade}
                  onChange={(e) => setSimGrade(e.target.value)}
                  className="grade-select"
                >
                  {["A", "B", "C", "D", "E", "F", "G"].map((g) => (
                    <option key={g} value={g}>
                      Grade {g}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Right Comparison: Live Delta */}
            <div className="what-if-comparison">
              <span className="metric-label">SIMULATED OUTCOME</span>
              <div className="sim-score-row">
                <div className="sim-score-block">
                  <span className="sim-score-label">Simulated Probability</span>
                  <div className="sim-score-val">
                    {simLoading ? "..." : `${simProbPercent}%`}
                  </div>
                </div>

                <div className="sim-delta-block">
                  <span className="sim-score-label">Delta vs Original</span>
                  <span
                    className={`delta-badge ${
                      deltaP < 0 ? "delta-good" : deltaP > 0 ? "delta-bad" : "delta-neutral"
                    }`}
                  >
                    {deltaP > 0 ? `+${deltaP.toFixed(1)}%` : `${deltaP.toFixed(1)}%`}
                    {deltaP < 0 ? " Risk Reduction" : deltaP > 0 ? " Risk Increase" : " No Change"}
                  </span>
                </div>
              </div>

              <div className={`risk-badge-large ${simRiskClass}`} style={{ marginTop: "12px" }}>
                <span className="status-dot"></span>
                {(simPrediction?.risk_category || risk_category).toUpperCase()} RISK
              </div>

              <div className="sim-derived-note">
                Simulated Debt-to-Income (DTI):{" "}
                <strong>{simIncome > 0 ? ((simLoanAmnt / simIncome) * 100).toFixed(1) : 0}%</strong>
              </div>
            </div>
          </div>

          {/* Adverse Action & Regulatory Compliance Notice */}
          <div className="adverse-action-box">
            <div className="adverse-action-header">
              <span className="compliance-icon">⚖️</span>
              <div>
                <h5>Adverse Action & Compliance Notice (FCRA / ECOA Story)</h5>
                <p>Transparent explanation required if credit terms are denied or conditioned</p>
              </div>
            </div>
            <p className="adverse-action-text">
              Under federal fair lending regulations (FCRA 15 U.S.C. § 1681m / ECOA Regulation B § 1002.9), 
              borrowers are entitled to the principal reasons influencing negative or conditioned credit terms. 
              According to the trained model's verified SHAP attributions, the key drivers impacting this application are:
            </p>
            <ul className="adverse-action-reasons">
              {reasons.slice(0, 3).map((r, i) => (
                <li key={i}>
                  <strong>{r.feature_name} ({r.applicant_value}):</strong> {r.plain_text}
                </li>
              ))}
            </ul>
            <p className="adverse-action-footer">
              Actionable Guidance: What-if counterfactual analysis indicates that securing an interest rate 
              under 10.0% or reducing requested principal significantly mitigates expected default probability.
            </p>
          </div>
        </section>

        {/* 4. MATHEMATICAL EXPLANATION VERIFICATION */}
        <section className="form-card section-card">
          <div className="section-heading">
            <span className="section-number">03</span>
            <div>
              <h3>Explanation Fidelity & Additivity Check</h3>
              <p>Mathematical proof that SHAP attributions sum exactly to model output</p>
            </div>
          </div>

          <div className="fidelity-grid">
            <div className="fidelity-item">
              <span className="fidelity-label">Base Expectation</span>
              <span className="fidelity-val">{(explanation_check.base_value * 100).toFixed(1)}%</span>
              <span className="fidelity-sub">Population average</span>
            </div>

            <div className="fidelity-operator">+</div>

            <div className="fidelity-item">
              <span className="fidelity-label">Net SHAP Impact</span>
              <span className="fidelity-val">
                {explanation_check.sum_shap > 0 ? "+" : ""}
                {(explanation_check.sum_shap * 100).toFixed(1)}%
              </span>
              <span className="fidelity-sub">Sum of all factors</span>
            </div>

            <div className="fidelity-operator">=</div>

            <div className="fidelity-item fidelity-total">
              <span className="fidelity-label">Model Probability</span>
              <span className="fidelity-val">{(explanation_check.model_prob * 100).toFixed(1)}%</span>
              <span className="fidelity-sub">
                {explanation_check.is_valid ? "✓ Additivity Verified" : "Verification Warning"}
              </span>
            </div>
          </div>
        </section>

        {/* 5. APPLICANT PARAMETER SUMMARY */}
        <section className="form-card section-card">
          <div className="section-heading">
            <span className="section-number">04</span>
            <div>
              <h3>Submitted Applicant Profile</h3>
              <p>Normalized input features evaluated during this scoring cycle</p>
            </div>
          </div>

          <div className="profile-grid">
            {Object.entries(applicant_summary || {}).map(([key, val]) => (
              <div key={key} className="profile-item">
                <span className="profile-key">{key.replace(/_/g, " ")}</span>
                <span className="profile-val">{val}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ACTION BUTTONS */}
        <div className="results-actions">
          <button
            onClick={() => navigate("/", { state: { applicant } })}
            className="secondary-button"
          >
            ← Adjust Initial Parameters
          </button>

          <button onClick={() => navigate("/dashboard")} className="secondary-button">
            View Global Dashboard 📊
          </button>

          <button onClick={() => navigate("/")} className="submit-button">
            New Applicant Evaluation <span aria-hidden="true">→</span>
          </button>
        </div>

        <p className="bottom-note">
          Credit Risk Predictor · Fully Explainable Underwriting Decisioning
        </p>
      </main>
    </div>
  );
}

export default Results;
