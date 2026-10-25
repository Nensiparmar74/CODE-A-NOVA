"""
app.py
------
ChurnShield — Customer Churn Prediction Dashboard (Streamlit)

Loads the PRE-TRAINED pipeline from models/churn_model.joblib and NEVER
retrains at runtime. Predictions typically complete in well under 1 second
on CPU (see outputs/reports/model_metrics.json -> single_prediction_latency_seconds).

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.utils import risk_category, retention_suggestions
from src.explainability import SHAP_AVAILABLE

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "outputs" / "reports" / "model_metrics.json"
DATASET_META_PATH = ROOT / "outputs" / "reports" / "dataset_metadata.json"
PROCESSED_PATH = ROOT / "data" / "processed" / "telco_clean.csv"

st.set_page_config(page_title="ChurnShield", page_icon="🛡️", layout="wide")


# ----------------------------- Cached loaders -----------------------------
@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact["threshold"], artifact["feature_columns"], artifact["model_name"], artifact.get("shap_background")


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_metadata():
    if DATASET_META_PATH.exists():
        with open(DATASET_META_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_training_sample():
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        return df
    return None


pipeline, threshold, feature_columns, model_name, shap_background = load_model()
metrics_report = load_metrics()
train_sample = load_training_sample()
dataset_meta = load_dataset_metadata()

# ----------------------------- Header -----------------------------
st.title("🛡️ ChurnShield")
st.subheader("Customer Churn Prediction Dashboard")
st.markdown(
    "Estimate a telecom customer's probability of churning using a trained "
    f"**{model_name}** model, and see which factors are driving the prediction. "
    "Enter a customer's profile in the sidebar to get an instant, explainable prediction."
)
if dataset_meta.get("dataset_type") == "synthetic_demo":
    st.warning("Development dataset detected. Replace the raw CSV with the official IBM/Kaggle dataset and rerun training before final internship submission.")

tabs = st.tabs(["🔮 Prediction", "📊 Churn Overview", "📈 Key Churn Drivers", "ℹ️ About This Model"])

# ----------------------------- Sidebar inputs -----------------------------
st.sidebar.header("Customer Profile")

def sidebar_inputs():
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

    st.sidebar.markdown("---")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

    st.sidebar.markdown("---")
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    if phone_service == "Yes":
        multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"

    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    if internet_service != "No":
        online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = online_backup = device_protection = tech_support = "No internet service"
        streaming_tv = streaming_movies = "No internet service"

    st.sidebar.markdown("---")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    default_total = round(monthly_charges * tenure, 2)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(default_total), step=1.0)

    return {
        "gender": gender, "SeniorCitizen": "1" if senior_citizen == "Yes" else "0",
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }


customer_input = sidebar_inputs()
predict_clicked = st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ----------------------------- Prediction tab -----------------------------
with tabs[0]:
    if predict_clicked:
        import time
        X_input = pd.DataFrame([customer_input])[feature_columns]

        t0 = time.time()
        proba = pipeline.predict_proba(X_input)[0, 1]
        pred = int(proba >= threshold)
        latency = time.time() - t0

        risk = risk_category(proba)
        risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}[risk]

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{proba:.1%}")
        col2.metric("Prediction", "Will Churn" if pred else "Will Stay",
                    delta=f"threshold={threshold:.2f}", delta_color="off")
        col3.metric("Risk Category", f"{risk_color} {risk}")
        st.caption(f"Prediction served in {latency*1000:.1f} ms on CPU (target: under 3 seconds).")

        # Probability gauge
        st.markdown("#### Churn Probability")
        if PLOTLY_AVAILABLE:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1d3557"},
                    "steps": [
                        {"range": [0, 30], "color": "#a8dadc"},
                        {"range": [30, 60], "color": "#f4a261"},
                        {"range": [60, 100], "color": "#e63946"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
                },
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.progress(min(int(proba * 100), 100))

        st.markdown("---")
        st.markdown("#### Why this prediction? (Explainability)")

        if SHAP_AVAILABLE and train_sample is not None:
            try:
                from src.explainability import explain_local
                X_bg = shap_background[feature_columns] if shap_background is not None else train_sample.drop(columns=["Churn"])[feature_columns].sample(
                    min(100, len(train_sample)), random_state=42
                )
                top_pos, top_neg, _ = explain_local(pipeline, X_input, X_bg)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**⬆️ Increases churn risk:**")
                    if len(top_pos):
                        for _, row in top_pos.iterrows():
                            st.write(f"- {row['feature']} (impact: +{row['shap_value']:.3f})")
                    else:
                        st.write("No strong positive contributors.")
                with c2:
                    st.markdown("**⬇️ Decreases churn risk:**")
                    if len(top_neg):
                        for _, row in top_neg.iterrows():
                            st.write(f"- {row['feature']} (impact: {row['shap_value']:.3f})")
                    else:
                        st.write("No strong negative contributors.")
            except Exception as e:
                st.info(f"Per-customer SHAP explanation unavailable ({e}). Showing rule-based drivers instead below.")
        else:
            st.warning(
                "SHAP is unavailable in this environment. Install dependencies from requirements.txt "
                "and restart Streamlit to enable the required live SHAP explanation."
            )

        st.markdown("#### 💡 Retention Suggestions")
        for tip in retention_suggestions(customer_input):
            st.write(f"- {tip}")

    else:
        st.info("👈 Fill in the customer profile in the sidebar and click **Predict Churn Risk**.")

# ----------------------------- Overview tab -----------------------------
with tabs[1]:
    st.markdown("### Churn Overview (Training Data)")
    if train_sample is not None:
        churn_rate = train_sample["Churn"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", f"{len(train_sample):,}")
        c2.metric("Overall Churn Rate", f"{churn_rate:.1%}")
        c3.metric("Retained Customers", f"{(1-churn_rate):.1%}")

        if PLOTLY_AVAILABLE:
            import plotly.express as px
            fig = px.histogram(train_sample, x="Churn", color="Churn",
                                labels={"Churn": "Churn (0=No, 1=Yes)"},
                                title="Churn Distribution", color_discrete_sequence=["#2E86AB", "#E63946"])
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(train_sample, x="Churn", y="MonthlyCharges", color="Churn",
                          title="Monthly Charges by Churn Status",
                          color_discrete_sequence=["#2E86AB", "#E63946"])
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Processed dataset not found. Run the training pipeline first.")

# ----------------------------- Drivers tab -----------------------------
with tabs[2]:
    st.markdown("### Key Churn Drivers")
    shap_img = ROOT / "outputs" / "shap" / "global_shap_importance.png"
    perm_img = ROOT / "outputs" / "shap" / "permutation_importance.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="Global SHAP Feature Importance")
    elif perm_img.exists():
        st.image(str(perm_img), caption="Permutation importance (fallback artifact only; final submission requires SHAP)")
    else:
        st.info("Run `python src/explainability.py` to generate the global feature-importance chart.")

    if train_sample is not None:
        st.markdown("#### Churn Rate by Contract Type")
        rate = train_sample.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
        st.bar_chart(rate)
        st.markdown("#### Churn Rate by Internet Service")
        rate2 = train_sample.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
        st.bar_chart(rate2)

# ----------------------------- About tab -----------------------------
with tabs[3]:
    st.markdown("### Model Information")
    if metrics_report:
        final = metrics_report["final_test_metrics_tuned_threshold"]
        st.write(f"**Selected model:** {metrics_report['best_model_name']}")
        st.write(f"**Decision threshold:** {final['threshold']:.2f} (tuned to prioritize recall)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{final['accuracy']:.1%}")
        c2.metric("Precision", f"{final['precision']:.1%}")
        c3.metric("Recall", f"{final['recall']:.1%}")
        c4.metric("F1 Score", f"{final['f1']:.1%}")
        c5.metric("ROC-AUC", f"{final['roc_auc']:.3f}")
        st.caption("Metrics computed on a held-out test set (20% of data, never seen during training or tuning).")

        st.markdown("#### Target vs Actual")
        targets = metrics_report["targets"]
        met = metrics_report["targets_met"]
        st.write(f"- Accuracy > 80%: **{final['accuracy']:.1%}** — {'✅ met' if met['accuracy'] else '❌ not met'}")
        st.write(f"- Recall > 75%: **{final['recall']:.1%}** — {'✅ met' if met['recall'] else '❌ not met'}")
        st.write(f"- ROC-AUC > 0.85: **{final['roc_auc']:.3f}** — {'✅ met' if met['roc_auc'] else '❌ not met'}")

        if not metrics_report.get("xgboost_available_in_this_environment", False):
            st.error("XGBoost is unavailable in the current trained artifact. Reinstall requirements and rerun training before final submission.")
    else:
        st.warning("Metrics report not found. Run `python src/model_training.py` first.")

    st.markdown("---")
    st.markdown(
        "**Disclaimer:** All relationships described in this dashboard (e.g., contract type, "
        "tenure, monthly charges) are statistical **associations** learned from historical data, "
        "not proven causal effects."
    )
