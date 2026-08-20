"""Train churn prediction models and save the best full pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    PROJECT_ROOT,
    build_preprocessor,
    clean_dataset,
    get_feature_columns,
    load_dataset,
    split_features_target,
)
from sklearn.model_selection import train_test_split


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def get_models() -> dict[str, object]:
    """Define candidate classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    }


def evaluate_pipeline(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Calculate classification metrics for a trained pipeline."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC Score": round(roc_auc_score(y_test, y_proba), 4),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "ROC Curve": {
            "fpr": fpr.round(4).tolist(),
            "tpr": tpr.round(4).tolist(),
            "thresholds": thresholds.round(4).tolist(),
        },
    }


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract transformed feature names from the fitted preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    """Return sorted feature importance for tree models or coefficients for logistic regression."""
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        return []

    importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
    importance["Feature"] = (
        importance["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    importance = importance.sort_values("Importance", ascending=False).head(20)
    return importance.round(5).to_dict(orient="records")


def train_and_select_best_model() -> tuple[Pipeline, dict[str, object]]:
    """Train all candidate models, select the best by ROC-AUC then F1, and save artifacts."""
    MODELS_DIR.mkdir(exist_ok=True)

    raw_df = load_dataset()
    print("Dataset shape:", raw_df.shape)
    print("Columns:", list(raw_df.columns))
    print("Data types:")
    print(raw_df.dtypes)
    print("Missing values before cleaning:")
    print(raw_df.isna().sum())
    print("Duplicate rows:", raw_df.duplicated().sum())

    df = clean_dataset(raw_df)
    print("Dataset shape after cleaning:", df.shape)
    print("Missing values after cleaning:")
    print(df.isna().sum())

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results: dict[str, dict[str, object]] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(df)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        results[model_name] = evaluate_pipeline(pipeline, X_test, y_test)
        trained_pipelines[model_name] = pipeline

    comparison = pd.DataFrame(results).T
    best_model_name = (
        comparison.sort_values(["ROC-AUC Score", "F1 Score"], ascending=False)
        .index[0]
    )
    best_pipeline = trained_pipelines[best_model_name]
    feature_importance = get_feature_importance(best_pipeline)

    artifact = {
        "pipeline": best_pipeline,
        "feature_columns": get_feature_columns(df),
        "best_model_name": best_model_name,
        "feature_importance": feature_importance,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(artifact, file)

    metrics = {
        "best_model_name": best_model_name,
        "model_comparison": results,
        "feature_importance": feature_importance,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nModel comparison:")
    print(comparison[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC Score"]])
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    return best_pipeline, metrics


if __name__ == "__main__":
    train_and_select_best_model()

