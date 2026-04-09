from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

from .preprocess import CATEGORICAL_COLS, NUMERIC_COLS

# ── individual transformers ───────────────────────────────────────────────────

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])


# ── combined preprocessor ─────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ],
    remainder="drop",
)


def build_pipeline(model) -> Pipeline:
    """
    Wrap any sklearn-compatible model with the full preprocessing pipeline.
    Usage:
        pipeline = build_pipeline(RandomForestClassifier())
        pipeline.fit(X_train, y_train)
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def get_feature_names() -> list[str]:
    """Return ordered feature names post-transformation (numeric first, then categorical)."""
    return NUMERIC_COLS + CATEGORICAL_COLS
