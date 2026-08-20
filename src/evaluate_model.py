"""Evaluate the saved churn model on a held-out test set."""

from __future__ import annotations

import pickle

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from data_preprocessing import clean_dataset, load_dataset, split_features_target
from train_model import MODEL_PATH, evaluate_pipeline


def evaluate_saved_model() -> dict[str, object]:
    """Load the saved model artifact and print evaluation metrics."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run: python src/train_model.py")

    with MODEL_PATH.open("rb") as file:
        artifact = pickle.load(file)

    df = clean_dataset(load_dataset())
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = artifact["pipeline"]
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    y_pred = pipeline.predict(X_test)

    print(f"Best model: {artifact['best_model_name']}")
    print(pd.Series(metrics).drop(labels=["ROC Curve"]).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    return metrics


if __name__ == "__main__":
    evaluate_saved_model()

