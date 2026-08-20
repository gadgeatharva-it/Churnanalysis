"""Data loading, cleaning, preprocessing, and business-analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the churn dataset from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicates, whitespace, data types, and missing values."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    object_columns = cleaned.select_dtypes(include="object").columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)

    numeric_columns = cleaned.select_dtypes(include=np.number).columns
    categorical_columns = cleaned.select_dtypes(exclude=np.number).columns

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in categorical_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].mode()[0])

    return cleaned


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding target and ID fields."""
    return [col for col in df.columns if col not in {TARGET_COLUMN, ID_COLUMN}]


def get_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns."""
    feature_columns = get_feature_columns(df)
    numeric_features = df[feature_columns].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[feature_columns].select_dtypes(exclude=np.number).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable preprocessing pipeline for ML models."""
    numeric_features, categorical_features = get_column_groups(df)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and binary target variable."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    X = df[get_feature_columns(df)]
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Target column contains values other than 'Yes' and 'No'.")

    return X, y.astype(int)


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Clean data, split it, and return a preprocessor fitted later in model pipelines."""
    cleaned = clean_dataset(df)
    X, y = split_features_target(cleaned)
    preprocessor = build_preprocessor(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, preprocessor


def dataset_quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Return beginner-friendly dataset diagnostics."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate churn rate by a categorical feature."""
    grouped = (
        df.groupby(column, dropna=False)[TARGET_COLUMN]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="Churn Rate (%)")
        .sort_values("Churn Rate (%)", ascending=False)
    )
    grouped["Churn Rate (%)"] = grouped["Churn Rate (%)"].round(2)
    return grouped


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Calculate core BI KPIs for the dashboard."""
    churned = int((df[TARGET_COLUMN] == "Yes").sum())
    total = int(len(df))
    return {
        "Total Customers": total,
        "Total Churned Customers": churned,
        "Overall Churn Rate": round(churned / total * 100, 2),
        "Average Monthly Charges": round(float(df["MonthlyCharges"].mean()), 2),
        "Average Customer Tenure": round(float(df["tenure"].mean()), 1),
    }


def generate_business_recommendations(df: pd.DataFrame, top_features: Iterable[str] | None = None) -> list[str]:
    """Create simple recommendations from observed churn patterns and model drivers."""
    recommendations: list[str] = []

    contract_rates = churn_rate_by(df, "Contract")
    highest_contract = contract_rates.iloc[0]
    if highest_contract["Churn Rate (%)"] > df[TARGET_COLUMN].eq("Yes").mean() * 100:
        recommendations.append(
            f"Prioritize retention offers for {highest_contract['Contract']} contract customers, "
            f"where churn is {highest_contract['Churn Rate (%)']:.1f}%."
        )

    payment_rates = churn_rate_by(df, "PaymentMethod")
    highest_payment = payment_rates.iloc[0]
    recommendations.append(
        f"Review payment friction for customers using {highest_payment['PaymentMethod']}; "
        f"this segment has the highest churn rate at {highest_payment['Churn Rate (%)']:.1f}%."
    )

    tenure_cutoff = df["tenure"].median()
    low_tenure_churn = df.loc[df["tenure"] <= tenure_cutoff, TARGET_COLUMN].eq("Yes").mean() * 100
    high_tenure_churn = df.loc[df["tenure"] > tenure_cutoff, TARGET_COLUMN].eq("Yes").mean() * 100
    if low_tenure_churn > high_tenure_churn:
        recommendations.append(
            "Launch onboarding and early-life engagement campaigns because newer customers churn more often "
            f"({low_tenure_churn:.1f}% vs {high_tenure_churn:.1f}%)."
        )

    monthly_cutoff = df["MonthlyCharges"].median()
    high_charge_churn = df.loc[df["MonthlyCharges"] > monthly_cutoff, TARGET_COLUMN].eq("Yes").mean() * 100
    low_charge_churn = df.loc[df["MonthlyCharges"] <= monthly_cutoff, TARGET_COLUMN].eq("Yes").mean() * 100
    if high_charge_churn > low_charge_churn:
        recommendations.append(
            "Monitor customers with above-median monthly charges for pricing dissatisfaction "
            f"({high_charge_churn:.1f}% churn vs {low_charge_churn:.1f}%)."
        )

    if top_features:
        readable = ", ".join(feature.replace("_", " ") for feature in list(top_features)[:3])
        recommendations.append(f"Use the strongest model drivers in retention playbooks: {readable}.")

    return recommendations[:5]

