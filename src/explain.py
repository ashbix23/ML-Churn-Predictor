import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from .features import get_feature_names


# ── build explainer ───────────────────────────────────────────────────────────

def get_explainer(pipeline, X_train: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_train_transformed = preprocessor.transform(X_train)

    if any(
        name in type(classifier).__name__.lower()
        for name in ["forest", "boost", "tree", "xgb", "lgbm", "gbm"]
    ):
        explainer = shap.TreeExplainer(classifier, data=X_train_transformed)
    else:
        explainer = shap.LinearExplainer(classifier, X_train_transformed)

    return explainer, X_train_transformed

def get_shap_values(explainer, pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for a dataset.
    Returns array of shape (n_samples, n_features).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    shap_values = explainer.shap_values(X_transformed)

    # TreeExplainer on binary classifiers returns list of [class0, class1]
    # we want class 1 (churn)
    if isinstance(shap_values, list):
        return shap_values[1]

    return shap_values


# ── global plots ──────────────────────────────────────────────────────────────

def plot_summary(explainer, pipeline, X: pd.DataFrame, save_dir: Path = None):
    """Beeswarm summary plot — shows feature importance + direction of effect."""
    shap_values = get_shap_values(explainer, pipeline, X)
    feature_names = get_feature_names()

    shap.summary_plot(
        shap_values,
        features=pipeline.named_steps["preprocessor"].transform(X),
        feature_names=feature_names,
        show=False,
    )
    plt.title("SHAP Summary — Feature Impact on Churn")
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_bar(explainer, pipeline, X: pd.DataFrame, save_dir: Path = None):
    """Bar plot of mean absolute SHAP values — clean global importance view."""
    shap_values = get_shap_values(explainer, pipeline, X)
    feature_names = get_feature_names()

    shap.summary_plot(
        shap_values,
        features=pipeline.named_steps["preprocessor"].transform(X),
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── single prediction explanation ─────────────────────────────────────────────

def plot_waterfall(explainer, pipeline, X_row: pd.DataFrame, save_dir: Path = None):
    """
    Waterfall plot for a single customer — shows exactly why the model
    predicted churn or no churn for that individual.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_row)
    feature_names = get_feature_names()

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    explanation = shap.Explanation(
        values=sv,
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
        data=X_transformed[0],
        feature_names=feature_names,
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.show()


def get_top_factors(explainer, pipeline, X_row: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return top n SHAP drivers for a single customer as a clean DataFrame.
    Used by the Streamlit app to display human-readable explanations.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_row)
    feature_names = get_feature_names()

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    factors = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "raw_value": X_row.iloc[0][feature_names].values,
    })

    factors["abs_shap"] = factors["shap_value"].abs()
    factors = factors.sort_values("abs_shap", ascending=False).head(n)
    factors["direction"] = factors["shap_value"].apply(
        lambda x: "↑ increases churn risk" if x > 0 else "↓ decreases churn risk"
    )

    return factors[["feature", "raw_value", "shap_value", "direction"]].reset_index(drop=True)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .preprocess import load_and_clean, split_data
    from .train import load_model
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = load_model("xgboost")
    explainer, _ = get_explainer(pipeline, X_train)

    plot_summary(explainer, pipeline, X_test)
    plot_bar(explainer, pipeline, X_test)

    sample = X_test.iloc[[0]]
    plot_waterfall(explainer, pipeline, sample)
    print(get_top_factors(explainer, pipeline, sample))
