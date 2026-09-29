
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = (
    BASE_DIR / "models" / "stage_1_rf_classifier_pipeline.pkl"
)
REGRESSOR_PATH = (
    BASE_DIR / "models" / "stage_2_rf_regression_pipeline.pkl"
)

_classifier = None
_regressor = None


def load_models():
    """Load both trained models once."""
    global _classifier, _regressor

    if _classifier is None:
        _classifier = joblib.load(CLASSIFIER_PATH)

    if _regressor is None:
        _regressor = joblib.load(REGRESSOR_PATH)

    return _classifier, _regressor


def predict_loan(application):
    """
    Predict approval status and, if approved,
    estimate the loan amount.

    application must be a dictionary containing
    all model input features.
    """
    classifier, regressor = load_models()

    input_df = pd.DataFrame([application])

    status = classifier.predict(input_df)[0]

    result = {
        "loan_status": status,
        "predicted_loan_amount": None,
    }

    if status == "Approved":
        amount = regressor.predict(input_df)[0]
        result["predicted_loan_amount"] = max(0, round(float(amount), 2))

    return result