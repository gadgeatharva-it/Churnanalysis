"""Create synthetic Telco-style churn rows and update the working dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
ORIGINAL_PATH = PROJECT_ROOT / "data" / "customer_churn_original.csv"


YES_NO = ["Yes", "No"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source rows before synthetic sampling."""
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["tenure"] * cleaned["MonthlyCharges"])
    return cleaned


def choose_weighted(rng: np.random.Generator, values: list[str], weights: list[float]) -> str:
    """Pick one categorical value from weighted options."""
    return str(rng.choice(values, p=np.array(weights) / np.sum(weights)))


def generate_churn_label(row: pd.Series, rng: np.random.Generator) -> str:
    """Generate a realistic churn label from business-risk factors."""
    score = -1.45
    score += 1.05 if row["Contract"] == "Month-to-month" else -0.45 if row["Contract"] == "Two year" else -0.2
    score += 0.65 if row["InternetService"] == "Fiber optic" else -0.25 if row["InternetService"] == "No" else 0.0
    score += 0.42 if row["PaymentMethod"] == "Electronic check" else -0.12
    score += 0.35 if row["PaperlessBilling"] == "Yes" else -0.08
    score += 0.35 if row["OnlineSecurity"] == "No" else -0.18 if row["OnlineSecurity"] == "Yes" else 0.0
    score += 0.30 if row["TechSupport"] == "No" else -0.15 if row["TechSupport"] == "Yes" else 0.0
    score += -0.035 * float(row["tenure"])
    score += 0.012 * max(float(row["MonthlyCharges"]) - 65.0, 0)
    probability = 1 / (1 + np.exp(-score))
    return "Yes" if rng.random() < probability else "No"


def mutate_row(row: pd.Series, index: int, rng: np.random.Generator) -> pd.Series:
    """Create one synthetic customer by perturbing a sampled real row."""
    synthetic = row.copy()
    synthetic["customerID"] = f"SYN-{index:06d}"

    if rng.random() < 0.18:
        synthetic["Contract"] = choose_weighted(rng, CONTRACTS, [0.58, 0.22, 0.20])
    if rng.random() < 0.18:
        synthetic["PaymentMethod"] = choose_weighted(rng, PAYMENT_METHODS, [0.40, 0.22, 0.20, 0.18])
    if rng.random() < 0.12:
        synthetic["InternetService"] = choose_weighted(rng, INTERNET_SERVICES, [0.34, 0.46, 0.20])
    if rng.random() < 0.10:
        synthetic["PaperlessBilling"] = choose_weighted(rng, YES_NO, [0.60, 0.40])
    if rng.random() < 0.08:
        synthetic["Partner"] = choose_weighted(rng, YES_NO, [0.48, 0.52])
    if rng.random() < 0.08:
        synthetic["Dependents"] = choose_weighted(rng, YES_NO, [0.30, 0.70])

    if synthetic["InternetService"] == "No":
        for service in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            synthetic[service] = "No internet service"
    else:
        for service in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            if synthetic[service] == "No internet service" or rng.random() < 0.12:
                synthetic[service] = choose_weighted(rng, YES_NO, [0.43, 0.57])

    if synthetic["PhoneService"] == "No":
        synthetic["MultipleLines"] = "No phone service"
    elif synthetic["MultipleLines"] == "No phone service" or rng.random() < 0.10:
        synthetic["MultipleLines"] = choose_weighted(rng, YES_NO, [0.42, 0.58])

    tenure_shift = int(rng.normal(0, 7))
    synthetic["tenure"] = int(np.clip(int(synthetic["tenure"]) + tenure_shift, 0, 72))

    charge_shift = float(rng.normal(0, 8))
    if synthetic["InternetService"] == "No":
        monthly = rng.normal(22, 5)
    elif synthetic["InternetService"] == "DSL":
        monthly = float(synthetic["MonthlyCharges"]) + charge_shift
    else:
        monthly = float(synthetic["MonthlyCharges"]) + charge_shift + 3
    synthetic["MonthlyCharges"] = round(float(np.clip(monthly, 18.0, 120.0)), 2)

    total_noise = rng.normal(0, max(synthetic["MonthlyCharges"] * 1.5, 10))
    synthetic["TotalCharges"] = round(max(synthetic["tenure"] * synthetic["MonthlyCharges"] + total_noise, 0), 2)
    synthetic["Churn"] = generate_churn_label(synthetic, rng)
    return synthetic


def augment_dataset(target_size: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Augment the working dataset up to target_size rows."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not ORIGINAL_PATH.exists():
        original = pd.read_csv(DATA_PATH)
        original.to_csv(ORIGINAL_PATH, index=False)
    else:
        original = pd.read_csv(ORIGINAL_PATH)

    original = clean_base(original)
    if target_size <= len(original):
        original.to_csv(DATA_PATH, index=False)
        return original

    rng = np.random.default_rng(seed)
    rows_needed = target_size - len(original)
    sampled = original.sample(n=rows_needed, replace=True, random_state=seed).reset_index(drop=True)
    synthetic_rows = [mutate_row(sampled.iloc[i], i + 1, rng) for i in range(rows_needed)]
    synthetic_df = pd.DataFrame(synthetic_rows, columns=original.columns)

    augmented = pd.concat([original, synthetic_df], ignore_index=True)
    augmented.to_csv(DATA_PATH, index=False)
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment the Telco churn dataset with synthetic rows.")
    parser.add_argument("--target-size", type=int, default=12000, help="Total row count after augmentation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    augmented = augment_dataset(target_size=args.target_size, seed=args.seed)
    synthetic_count = int(augmented["customerID"].astype(str).str.startswith("SYN-").sum())
    print(f"Dataset saved to {DATA_PATH}")
    print(f"Total customers: {len(augmented):,}")
    print(f"Synthetic customers: {synthetic_count:,}")
    print(f"Original backup: {ORIGINAL_PATH}")


if __name__ == "__main__":
    main()
