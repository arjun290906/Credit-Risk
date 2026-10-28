import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const FEATURE_DESCRIPTIONS = {
  loan_grade: "Officer-assigned risk rating based on preliminary borrower review",
  loan_int_rate: "Lender interest rate reflecting cost of capital and default risk",
  person_income: "Gross annual income supporting debt service obligations",
  loan_percent_income: "Debt-to-income (DTI) ratio of requested loan to annual income",
  credit_history_length: "Tenure of borrowing track record in years",
  loan_intent: "Stated purpose of borrowing (Debt consolidation, education, etc.)",
  loan_amnt: "Total requested principal exposure",
  person_emp_length: "Borrower employment tenure indicating career stability",
  person_age: "Applicant demographic age profile",
  person_home_ownership: "Residential stability status (Mortgage, rent, own)",
};

function Dashboard() {
  const navigate = useNavigate();

  const [globalShap, setGlobalShap] = useState([]);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [useGrouped, setUseGrouped] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let ignore = false;

    Promise.all([
      fetch(`${API_BASE}/shap/global?grouped=${useGrouped}`),
      fetch(`${API_BASE}/model/metrics`),
    ])
      .then(async ([shapRes, metricsRes]) => {
        if (!shapRes.ok || !metricsRes.ok) {
          throw new Error("Failed to load dashboard data from scoring service.");
        }
        const [shapData, metricsData] = await Promise.all([
          shapRes.json(),
          metricsRes.json(),
        ]);
        if (!ignore) {
          setGlobalShap(shapData);
          setModelMetrics(metricsData);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!ignore) {
          console.error(err);
          setError(
            "Cannot connect to backend service. Please ensure Uvicorn is running on port 8000."
          );
          setLoading(false);
        }
      });

    return () => {
      ignore = true;
    };
  }, [useGrouped]);

  return (
    <div className="app-container">
      {/* Header with Navigation */}
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
          <button className="nav-link nav-active">
            Portfolio Dashboard
          </button>
        </nav>
      </header>

      <main className="main-content">
        <div className="page-heading">
          <span className="eyebrow">PORTFOLIO INSIGHTS</span>
          <h2>
            Model <span>Dashboard</span> & SHAP Drivers
          </h2>
          <p>
            Offline population-level feature importance, model comparison bake-off, and calibrated risk tier distributions.
          </p>
        </div>

        {error && (
          <div className="error-message" style={{ marginBottom: "24px" }}>
            {error}
          </div>
        )}

        {/* 1. GLOBAL SHAP FEATURE IMPORTANCE */}
        <section className="form-card section-card">
          <div className="section-header-row">
            <div className="section-heading" style={{ marginBottom: 0 }}>
              <span className="section-number">01</span>
              <div>
                <h3>Global Feature Importance (SHAP)</h3>
                <p>Mean absolute SHAP value across the 6,305 test-set applicants</p>
              </div>
            </div>

            <div className="toggle-group">
              <button
                className={`toggle-btn ${useGrouped ? "toggle-btn-active" : ""}`}
                onClick={() => setUseGrouped(true)}
              >
                Domain Features
              </button>
              <button
                className={`toggle-btn ${!useGrouped ? "toggle-btn-active" : ""}`}
                onClick={() => setUseGrouped(false)}
              >
                Raw One-Hot
              </button>
            </div>
          </div>

          <div className="form-divider" style={{ margin: "20px 0 24px" }} />

          {loading ? (
            <p className="loading-text">Loading SHAP importance data...</p>
          ) : (
            <div className="shap-bar-chart">
              {globalShap.map((item) => {
                const maxPct = globalShap[0]?.importance_pct || 100;
                const relativeWidth = Math.max((item.importance_pct / maxPct) * 100, 3);

                return (
                  <div key={item.feature} className="shap-chart-row">
                    <div className="shap-label-col">
                      <span className="shap-rank-badge">#{item.rank}</span>
                      <span className="shap-feature-name">
                        {item.feature.replace(/^encoder__|^scaler__/, "").replace(/_/g, " ")}
                      </span>
                    </div>

                    <div className="shap-bar-col">
                      <div className="shap-track">
                        <div
                          className="shap-fill"
                          style={{ width: `${relativeWidth}%` }}
                        />
                      </div>
                      <span className="shap-pct-label">{item.importance_pct.toFixed(1)}%</span>
                    </div>

                    <div className="shap-desc-col">
                      {FEATURE_DESCRIPTIONS[item.feature] || `Encoded indicator for ${item.feature}`}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* 2. MODEL COMPARISON BAKE-OFF */}
        {modelMetrics && (
          <section className="form-card section-card">
            <div className="section-heading">
              <span className="section-number">02</span>
              <div>
                <h3>Model Bake-Off Comparison</h3>
                <p>Evaluated on the exact same 80/20 stratified split with contracted features</p>
              </div>
            </div>

            <div className="table-responsive">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Model Candidate</th>
                    <th>5-Fold CV ROC-AUC</th>
                    <th>Test ROC-AUC</th>
                    <th>Brier Loss</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {modelMetrics.comparison.map((m) => {
                    const isWinner = m.model === modelMetrics.winner;
                    const modelName =
                      m.model === "xgboost"
                        ? "XGBoost Classifier"
                        : m.model === "random_forest"
                        ? "Random Forest"
                        : "Logistic Regression";

                    return (
                      <tr key={m.model} className={isWinner ? "winner-row" : ""}>
                        <td className="model-name-cell">
                          <strong>{modelName}</strong>
                        </td>
                        <td>{m.cv_roc_auc.toFixed(4)}</td>
                        <td>
                          <strong>{m.test_roc_auc.toFixed(4)}</strong>
                        </td>
                        <td>{m.brier.toFixed(4)}</td>
                        <td>{m.precision.toFixed(4)}</td>
                        <td>{m.recall.toFixed(2)}</td>
                        <td>{m.f1.toFixed(4)}</td>
                        <td>
                          {isWinner ? (
                            <span className="winner-badge">★ Production Model</span>
                          ) : (
                            <span className="candidate-badge">Evaluated</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* 3. CALIBRATED RISK TIERS */}
        {modelMetrics?.risk_bands && (
          <section className="form-card section-card">
            <div className="section-heading">
              <span className="section-number">03</span>
              <div>
                <h3>Test Population Risk Tier Distribution</h3>
                <p>Empirical segmentation of 6,305 applicants under frozen probability cutoffs</p>
              </div>
            </div>

            <div className="risk-bands-grid">
              {Object.entries(modelMetrics.risk_bands).map(([bandName, bandData]) => {
                const bandClass = `risk-badge-${bandName.toLowerCase().replace(/\s+/g, "-")}`;
                const sharePct = (bandData.share * 100).toFixed(1);
                const defaultPct = (bandData.default_rate * 100).toFixed(1);
                const meanPPct = (bandData.mean_p * 100).toFixed(1);

                return (
                  <div key={bandName} className={`band-summary-card ${bandClass}`}>
                    <div className="band-card-header">
                      <span className="band-title">{bandName.toUpperCase()} RISK</span>
                      <span className="band-n">{bandData.n.toLocaleString()} applicants</span>
                    </div>

                    <div className="band-metrics">
                      <div className="band-metric-item">
                        <span className="band-metric-label">Population Share</span>
                        <span className="band-metric-val">{sharePct}%</span>
                      </div>
                      <div className="band-metric-item">
                        <span className="band-metric-label">Mean Predicted P</span>
                        <span className="band-metric-val">{meanPPct}%</span>
                      </div>
                      <div className="band-metric-item">
                        <span className="band-metric-label">Observed Default Rate</span>
                        <span className="band-metric-val default-rate-val">{defaultPct}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* Navigation Action */}
        <div className="results-actions" style={{ marginTop: "32px" }}>
          <button onClick={() => navigate("/")} className="submit-button">
            Launch Applicant Scoring Form <span aria-hidden="true">→</span>
          </button>
        </div>

        <p className="bottom-note">
          Credit Risk Predictor · Fully Calibrated & Explainable Lending Intelligence
        </p>
      </main>
    </div>
  );
}

export default Dashboard;
