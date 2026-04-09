import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from pathlib import Path


# ── core metrics ──────────────────────────────────────────────────────────────

def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Print and return key classification metrics.
    Uses 0.5 threshold for classification report, full curve for AUC scores.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Avg Precision:     {avg_precision:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(pipeline, X_test: pd.DataFrame, y_test: pd.Series, save_dir: Path = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.show()


def plot_roc_curve(pipeline, X_test: pd.DataFrame, y_test: pd.Series, save_dir: Path = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("ROC Curve")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "roc_curve.png", dpi=150)
    plt.show()


def plot_precision_recall(pipeline, X_test: pd.DataFrame, y_test: pd.Series, save_dir: Path = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "precision_recall.png", dpi=150)
    plt.show()


def plot_all(pipeline, X_test: pd.DataFrame, y_test: pd.Series, save_dir: Path = None):
    """Convenience wrapper — runs all three plots."""
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(pipeline, X_test, y_test, save_dir)
    plot_roc_curve(pipeline, X_test, y_test, save_dir)
    plot_precision_recall(pipeline, X_test, y_test, save_dir)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocess import load_and_clean, split_data
    from train import load_model
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean(data_path)
    _, X_test, _, y_test = split_data(df)

    pipeline = load_model("xgboost")
    metrics = evaluate(pipeline, X_test, y_test)
    plot_all(pipeline, X_test, y_test)