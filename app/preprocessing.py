
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/loan_approval_dataset.csv")
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "requested_loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CATEGORICAL_FEATURES = [
    "education",
    "self_employed",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    """Load and validate the dataset."""
    df = pd.read_csv(path)

    required = FEATURES + ["loan_status", "loan_amount"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[required].isnull().any().any():
        raise ValueError("Unexpected missing values in required columns.")

    if not set(df["loan_status"].unique()).issubset(
        {"Approved", "Rejected"}
    ):
        raise ValueError("Unexpected loan_status values.")

    return df


def create_preprocessor():
    """Create preprocessing for both Random Forest models."""
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def prepare_classifier_data(df):
    X = df[FEATURES].copy()
    y = df["loan_status"].copy()

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def prepare_regression_data(df):
    """Prepare sanctioned amount regression data."""
    approved = df.loc[df["loan_status"] == "Approved"].copy()

    if approved.empty:
        raise ValueError("No approved applications for regression.")

    X = approved[FEATURES].copy()
    y = approved["loan_amount"].copy()

    if (y <= 0).any():
        raise ValueError(
            "Approved applications must have positive loan amounts."
        )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )