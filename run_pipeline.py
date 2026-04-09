import argparse
from pathlib import Path

from src import (
    load_and_clean,
    split_data,
    compare_models,
    train_best_model,
    save_model,
    evaluate,
    plot_all,
    get_explainer,
    plot_summary,
    plot_bar,
)

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PLOTS_DIR = ROOT / "models" / "plots"


# ── pipeline steps ────────────────────────────────────────────────────────────

def run(skip_cv: bool = False):
    print("\n── 1. Loading & cleaning data ───────────────────────────────────────")
    df = load_and_clean(DATA_PATH)
    print(f"Rows: {len(df)} | Churn rate: {df['Churn'].mean():.2%}")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n── 2. Model comparison (CV) ─────────────────────────────────────────")
    if skip_cv:
        print("Skipped — using xgboost by default")
        cv_results = {"xgboost": 0.0}
    else:
        cv_results = compare_models(X_train, y_train)

    print("\n── 3. Training best model ───────────────────────────────────────────")
    model_name, pipeline = train_best_model(X_train, y_train, cv_results)

    print("\n── 4. Evaluating on test set ────────────────────────────────────────")
    evaluate(pipeline, X_test, y_test)
    plot_all(pipeline, X_test, y_test, save_dir=PLOTS_DIR)

    print("\n── 5. Generating SHAP explanations ──────────────────────────────────")
    explainer, _ = get_explainer(pipeline, X_train)
    plot_summary(explainer, pipeline, X_test, save_dir=PLOTS_DIR)
    plot_bar(explainer, pipeline, X_test, save_dir=PLOTS_DIR)

    print("\n── 6. Saving model ──────────────────────────────────────────────────")
    save_model(pipeline, model_name)

    print("\n── Done ─────────────────────────────────────────────────────────────")
    print(f"Model : {model_name}")
    print(f"Plots : {PLOTS_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the churn prediction pipeline end to end.")
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation and go straight to training XGBoost.",
    )
    args = parser.parse_args()
    run(skip_cv=args.skip_cv)