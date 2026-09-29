# Two-Stage Loan Approval & Amount Prediction Pipeline

A machine learning project that uses a two-stage Random Forest pipeline
to predict loan approval status and estimate the loan amount for
applications predicted as approved.

## Overview

This project separates loan prediction into two stages:

-   **Stage 1 --- Loan Approval Classification:** A Random Forest
    Classifier predicts whether a loan application is approved or
    rejected.
-   **Stage 2 --- Loan Amount Regression:** A Random Forest Regressor
    estimates the loan amount for applications predicted as approved.

The project includes data preprocessing, model training, evaluation,
prediction, and an interactive Streamlit interface.

## Features

-   Two-stage machine learning architecture
-   Numerical and categorical data preprocessing
-   Stratified train-test split for classification
-   Random Forest classification and regression
-   Five-fold cross-validation
-   Classification and regression evaluation metrics
-   Saved machine learning pipelines using Joblib
-   Configurable model parameters using YAML
-   Interactive Streamlit application

## Tech Stack

-   **Language:** Python
-   **Data Processing:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn
-   **Web Interface:** Streamlit
-   **Visualization:** Plotly
-   **Model Persistence:** Joblib
-   **Configuration:** PyYAML

## Project Structure

``` text
Two-Stage-Loan-Approval-ML-Pipeline/
├── app/
│   ├── __init__.py
│   ├── predict.py
│   └── preprocessing.py
├── data/
│   └── loan_approval_dataset.csv
├── models/
│   ├── stage_1_rf_classifier_pipeline.pkl
│   └── stage_2_rf_regression_pipeline.pkl
├── .gitignore
├── config.yaml
├── train.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## How It Works

### Stage 1: Loan Approval Classification

The first stage uses a Random Forest Classifier to predict the loan
status.

**Input features:** - Number of dependents - Education - Self-employment
status - Annual income - Requested loan amount - Loan term - CIBIL
score - Residential assets value - Commercial assets value - Luxury
assets value - Bank asset value

**Output:** - `Approved` - `Rejected`

The classification pipeline applies preprocessing and then predicts the
application status.

### Stage 2: Loan Amount Prediction

If Stage 1 predicts an application as approved, the second stage uses a
Random Forest Regressor to estimate the loan amount.

The regressor is trained only on approved applications, using the same
applicant features. It predicts a continuous loan amount.

## Model Configuration

The baseline configuration for both models is:

  Parameter                                    Value
  -------------------------- -----------------------
  Algorithm                            Random Forest
  Number of estimators                           100
  Maximum depth                                    8
  Minimum samples per leaf                         5
  Random state                                    42
  Class weighting              Balanced (classifier)

The configuration is stored in `config.yaml` and can be changed without
modifying the model creation code.

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/Two-Stage-Loan-Approval-ML-Pipeline.git
cd Two-Stage-Loan-Approval-ML-Pipeline
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

``` bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

``` powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

## Usage

### Train the models

``` bash
python train.py
```

This trains the classifier and regressor and saves the fitted pipelines
in the `models/` directory.

### Run the Streamlit application

``` bash
streamlit run streamlit_app.py
```

Enter the applicant details in the interface to get the predicted
approval status. If the application is predicted as approved, the app
also displays the estimated loan amount.

## Model Evaluation

The following results were obtained with the simplified baseline
configuration on the project's synthetic dataset.

### Classification

  Metric                           Result
  ------------------------------ --------
  Training Accuracy                94.23%
  Testing Accuracy                 93.60%
  Training F1-score (Approved)     87.17%
  Testing F1-score (Approved)      85.32%
  Mean 5-fold CV F1-score          83.08%
  CV F1 standard deviation         0.0149

### Regression

  Metric                             Result
  --------------------------- -------------
  Training MAE                  ₹109,261.96
  Testing MAE                   ₹175,023.65
  Testing R²                         0.9581
  Mean 5-fold CV MAE            ₹168,495.57
  CV MAE standard deviation      ₹10,294.07
  Mean 5-fold CV R²                  0.9594

These are baseline results from the current validation run. Results may
vary if the dataset or configuration changes.




