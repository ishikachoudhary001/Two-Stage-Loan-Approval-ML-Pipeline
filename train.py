
from pathlib import Path

import joblib
import yaml
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from app.preprocessing import (
    load_data,
    create_preprocessor,
    prepare_classifier_data,
    prepare_regression_data,
)


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_classifier(df, config):
    X_train, X_test, y_train, y_test = prepare_classifier_data(df)

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=config["training"]["classifier"]["n_estimators"],
            max_depth=config["training"]["classifier"]["max_depth"],
            random_state=config["training"]["classifier"]["random_state"],
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n========== CLASSIFICATION ==========")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label='Approved', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, pos_label='Approved', zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, pos_label='Approved', zero_division=0):.4f}")
    print("Confusion matrix (Rejected, Approved):")
    print(confusion_matrix(y_test, predictions, labels=["Rejected", "Approved"]))

    return model


def train_regressor(df, config):
    X_train, X_test, y_train, y_test = prepare_regression_data(df)

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=config["training"]["regressor"]["n_estimators"],
            max_depth=config["training"]["regressor"]["max_depth"],
            random_state=config["training"]["regressor"]["random_state"],
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n========== REGRESSION ==========")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2_score(y_test, predictions):.4f}")

    return model


def main():
    config = load_config()
    df = load_data()

    classifier = train_classifier(df, config)
    regressor = train_regressor(df, config)

    classifier_path = Path(config["models"]["classifier"])
    regressor_path = Path(config["models"]["regressor"])

    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    regressor_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    print("\n========== MODELS SAVED ==========")
    print(classifier_path)
    print(regressor_path)


if __name__ == "__main__":
    main()