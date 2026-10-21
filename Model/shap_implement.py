
import pandas as pd
import numpy as np
import joblib
import shap
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score

# 1. Load the saved pipeline and test data
best_model = joblib.load("best_model.pkl")

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze("columns")

# 2. Extract the fitted preprocessing and XGBoost model
preprocessor = best_model.named_steps["preprocessor"]
xgb_model = best_model.named_steps["model"]

# 3. Transform the test data
X_test_encoded = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

if issparse(X_test_encoded):
    X_test_encoded = X_test_encoded.toarray()

# 4. Select a background sample from the test data
# For a strict evaluation workflow, use training data instead.
rng = np.random.default_rng(42)
background_size = min(100, len(X_test_encoded))

background_indices = rng.choice(
    len(X_test_encoded),
    size=background_size,
    replace=False
)
background = X_test_encoded[background_indices]

# 5. Create the SHAP explainer
explainer = shap.TreeExplainer(
    xgb_model,
    data=background,
    model_output="probability",
    feature_perturbation="interventional"
)

# 6. Calculate SHAP values
explanation = explainer(X_test_encoded)
shap_values = explanation.values

# 7. Calculate numerical global metrics
mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_signed_shap = shap_values.mean(axis=0)

total_importance = mean_abs_shap.sum()

importance_pct = (
    mean_abs_shap / total_importance * 100
    if total_importance > 0
    else np.zeros_like(mean_abs_shap)
)

shap_metrics = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
    "mean_signed_shap": mean_signed_shap,
    "importance_pct": importance_pct,
    "positive_contribution_pct":
        (shap_values > 0).mean(axis=0) * 100,
    "negative_contribution_pct":
        (shap_values < 0).mean(axis=0) * 100
})

shap_metrics = shap_metrics.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

shap_metrics.insert(
    0, "rank", np.arange(1, len(shap_metrics) + 1)
)

# 8. Save metrics as CSV
shap_metrics.to_csv("shap_global_metrics.csv", index=False)

print("\nTop 15 important features:")
print(shap_metrics.head(15).to_string(index=False))

# 9. Evaluate the model on the saved test set
y_prob = best_model.predict_proba(X_test)[:, 1]
print("\nTest ROC-AUC:", roc_auc_score(y_test, y_prob))

# 10. Save individual SHAP explanations
local_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)
local_df.insert(0, "row_index", X_test.index)

local_df.to_csv("shap_individual_values.csv", index=False)

print("\nSHAP analysis complete.")
