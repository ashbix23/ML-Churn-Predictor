import pandas as pd
import numpy as np
from pathlib import Path


# ── inference ─────────────────────────────────────────────────────────────────

def predict_single(pipeline, customer: dict) -> dict:
    """
    Run inference on a single customer.
    Accepts a dict of raw feature values (as the user would enter in the UI).
    Returns churn probability and a binary prediction.
    """
    X = pd.DataFrame([customer])
    prob = pipeline.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": pred,
        "label": "High Risk" if prob >= 0.7 else "Medium Risk" if prob >= 0.4 else "Low Risk",
    }


def predict_batch(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on a full DataFrame.
    Returns original df with churn_probability and label columns appended.
    """
    probs = pipeline.predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_probability"] = probs.round(4)
    out["label"] = pd.cut(
        probs,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return out


# ── default input template ────────────────────────────────────────────────────

def get_default_customer() -> dict:
    """
    Returns a sample customer dict with sensible defaults.
    Used to pre-populate the Streamlit UI.
    """
    return {
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from train import load_model

    pipeline = load_model("xgboost")
    customer = get_default_customer()
    result = predict_single(pipeline, customer)

    print(f"Churn probability : {result['churn_probability']:.2%}")
    print(f"Prediction        : {'Churn' if result['churn_prediction'] else 'No Churn'}")
    print(f"Risk label        : {result['label']}")