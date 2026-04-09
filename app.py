import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from src import load_model, get_explainer, get_top_factors, plot_waterfall, predict_single, get_default_customer
from src.preprocess import load_and_clean, split_data
# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── load model (cached) ───────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    pipeline = load_model("xgboost")
    data_path = Path(__file__).resolve().parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    from src.preprocess import load_and_clean, split_data
    df = load_and_clean(data_path)
    X_train, _, _, _ = split_data(df)
    explainer, _ = get_explainer(pipeline, X_train)
    return pipeline, explainer

pipeline, explainer = load_artifacts()

# ── sidebar — customer inputs ─────────────────────────────────────────────────

st.sidebar.header("Customer Details")
defaults = get_default_customer()

with st.sidebar:
    st.subheader("Demographics")
    gender          = st.selectbox("Gender",       ["Male", "Female"], index=0)
    partner         = st.selectbox("Partner",      ["Yes", "No"],      index=1)
    dependents      = st.selectbox("Dependents",   ["Yes", "No"],      index=1)

    st.subheader("Services")
    phone_service   = st.selectbox("Phone Service",     ["Yes", "No"],                          index=0)
    multiple_lines  = st.selectbox("Multiple Lines",    ["Yes", "No", "No phone service"],      index=1)
    internet        = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"],           index=1)
    online_security = st.selectbox("Online Security",   ["Yes", "No", "No internet service"],   index=1)
    online_backup   = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"],   index=1)
    device_protect  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"],   index=1)
    tech_support    = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"],   index=1)
    streaming_tv    = st.selectbox("Streaming TV",      ["Yes", "No", "No internet service"],   index=1)
    streaming_movies= st.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"],   index=1)

    st.subheader("Account")
    contract        = st.selectbox("Contract",        ["Month-to-month", "One year", "Two year"], index=0)
    paperless       = st.selectbox("Paperless Billing",["Yes", "No"],                             index=0)
    payment         = st.selectbox("Payment Method",  [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))


# ── build customer dict ───────────────────────────────────────────────────────

customer = {
    "gender": gender, "Partner": partner, "Dependents": dependents,
    "PhoneService": phone_service, "MultipleLines": multiple_lines,
    "InternetService": internet, "OnlineSecurity": online_security,
    "OnlineBackup": online_backup, "DeviceProtection": device_protect,
    "TechSupport": tech_support, "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
}

# ── main panel ────────────────────────────────────────────────────────────────

st.title("📡 Telco Churn Predictor")
st.caption("Adjust the customer profile in the sidebar to see real-time churn risk and SHAP explanations.")

result = predict_single(pipeline, customer)
prob   = result["churn_probability"]
label  = result["label"]

# risk metric + colour
col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{prob:.1%}")
col2.metric("Risk Level", label)
col3.metric("Prediction", "Will Churn" if result["churn_prediction"] else "Will Stay")

# colour-coded risk banner
colour = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
st.markdown(f"### {colour[label]} {label}")
st.progress(prob)

st.divider()

# ── SHAP panels ───────────────────────────────────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("Top factors driving this prediction")
    factors = get_top_factors(explainer, pipeline, pd.DataFrame([customer]))
    for _, row in factors.iterrows():
        st.markdown(f"**{row['feature']}** — `{row['raw_value']}` &nbsp; {row['direction']}")

with right:
    st.subheader("SHAP waterfall")
    fig, ax = plt.subplots()
    plot_waterfall(explainer, pipeline, pd.DataFrame([customer]))
    st.pyplot(plt.gcf())
    plt.close()

st.divider()

# ── global feature importance ─────────────────────────────────────────────────

with st.expander("Global feature importance (test set)", expanded=False):
    plots_dir = Path(__file__).resolve().parent / "models" / "plots"
    shap_bar = plots_dir / "shap_bar.png"
    shap_summary = plots_dir / "shap_summary.png"

    if shap_bar.exists():
        st.image(str(shap_bar), caption="Mean absolute SHAP values")
    if shap_summary.exists():
        st.image(str(shap_summary), caption="SHAP beeswarm summary")
    if not shap_bar.exists():
        st.info("Run `python run_pipeline.py` first to generate global SHAP plots.")
