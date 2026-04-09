import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ── column groups ─────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"


# ── main function ─────────────────────────────────────────────────────────────

def load_and_clean(data_path: str) -> pd.DataFrame:
    """Load raw CSV and apply cleaning steps."""
    df = pd.read_csv(data_path)

    # TotalCharges is read as object due to whitespace entries — fix that
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop the handful of rows where TotalCharges couldn't be parsed
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    # Encode target: Yes → 1, No → 0
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test sets, stratified on the target."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def get_preprocessing_summary(df: pd.DataFrame) -> None:
    """Print a quick sanity check after loading."""
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df[TARGET_COL].mean():.2%}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


# ── entry point (for quick testing) ──────────────────────────────────────────

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean(data_path)
    get_preprocessing_summary(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")