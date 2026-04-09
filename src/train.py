import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import joblib
from pathlib import Path

from .features import build_pipeline, get_feature_names

# ── models to compare ─────────────────────────────────────────────────────────

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    ),
    "xgboost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=3,       # handles class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    ),
}


# ── cross validation ──────────────────────────────────────────────────────────

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> dict[str, float]:
    """
    Run stratified k-fold CV on all models.
    Returns a dict of model_name → mean ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        results[name] = scores.mean()
        print(f"{name:<25} ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


# ── train best model ──────────────────────────────────────────────────────────

def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_results: dict[str, float],
) -> tuple:
    """
    Retrain the best model (by CV ROC-AUC) on the full training set.
    Returns (model_name, fitted_pipeline).
    """
    best_name = max(cv_results, key=cv_results.get)
    print(f"\nBest model: {best_name} (ROC-AUC: {cv_results[best_name]:.4f})")

    best_pipeline = build_pipeline(MODELS[best_name])
    best_pipeline.fit(X_train, y_train)

    return best_name, best_pipeline


# ── save / load ───────────────────────────────────────────────────────────────

def save_model(pipeline, model_name: str) -> Path:
    """Serialise fitted pipeline to models/."""
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / f"{model_name}.pkl"
    joblib.dump(pipeline, out_path)
    print(f"Model saved → {out_path}")
    return out_path


def load_model(model_name: str):
    """Load a saved pipeline by name."""
    path = Path(__file__).resolve().parents[1] / "models" / f"{model_name}.pkl"
    return joblib.load(path)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocess import load_and_clean, split_data
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    print("── Cross-validation results ──")
    cv_results = compare_models(X_train, y_train)

    model_name, pipeline = train_best_model(X_train, y_train, cv_results)
    save_model(pipeline, model_name)
