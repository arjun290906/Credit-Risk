
import { useLocation, useNavigate } from "react-router-dom";

function Results() {
  const { state } = useLocation();
  const navigate = useNavigate();

  const formData = state?.formData;

  if (!formData) {
    return (
      <div style={{ padding: "40px", fontFamily: "Arial" }}>
        <h2>No submitted data found</h2>
        <button onClick={() => navigate("/")}>
          Back to Form
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1 style={{ color: "#741e39" }}>Credit Risk Predictor</h1>
      <h2>Results Page</h2>
      <p>Your details were submitted successfully.</p>

      <h3>Submitted Data</h3>
      <pre
        style={{
          padding: "20px",
          background: "#faf3f6",
          borderRadius: "10px",
          overflowX: "auto",
        }}
      >
        {JSON.stringify(formData, null, 2)}
      </pre>

      <p>
        Log Loan Amount: {formData.log_loan_amount.toFixed(4)}
      </p>

      <button
        onClick={() => navigate("/")}
        style={{
          background: "#741e39",
          color: "white",
          padding: "12px 20px",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
        }}
      >
        Back to Form
      </button>
    </div>
  );
}

export default Results;
